/*
The MIT License

Copyright (c) 2019-2021, Prominence AI, Inc.


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in-
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "Dsl.h"
#include "DslServices.h"
#include "DslPlayerBintr.h"

namespace DSL
{
    PlayerBintr::PlayerBintr(const char* name, 
        DSL_SOURCE_PTR pSource, DSL_SINK_PTR pSink)
        : Bintr(name, true) // Pipeline = true
        , PipelineStateMgr(m_pGstObj)
        , PipelineXWinMgr(m_pGstObj)
        , m_pSource(pSource)
        , m_pSink(pSink)
    {
        LOG_FUNC();

        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "queue");
        m_pConverter = DSL_ELEMENT_NEW(NVDS_ELEM_VIDEO_CONV, "video-converter");
        m_pConverterCapsFilter = DSL_ELEMENT_NEW(NVDS_ELEM_CAPS_FILTER, "converter-caps-filter");

        GstCaps* pCaps = gst_caps_from_string("video/x-raw(memory:NVMM), format=NV12");
        m_pConverterCapsFilter->SetAttribute("caps", pCaps);
        gst_caps_unref(pCaps);

        g_mutex_init(&m_asyncCommMutex);
        
        AddChild(m_pQueue);
        AddChild(m_pConverter);
        AddChild(m_pConverterCapsFilter);
        
        if (!AddChild(m_pSource))
        {
            LOG_ERROR("Failed to add SourceBintr '" << m_pSource->GetName() 
                << "' to PlayerBintr '" << GetName() << "'");
            throw;
        }
        if (!AddChild(m_pSink))
        {
            LOG_ERROR("Failed to add SinkBintr '" << m_pSink->GetName() 
                << "' to PlayerBintr '" << GetName() << "'");
            throw;
        }
        
        AddXWindowDeleteEventHandler(PlayerTerminate, this);
    }

    PlayerBintr::PlayerBintr(const char* name)
        : Bintr(name, true) // Pipeline = true
        , PipelineStateMgr(m_pGstObj)
        , PipelineXWinMgr(m_pGstObj)
    {
        LOG_FUNC();

        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "queue");
        m_pConverter = DSL_ELEMENT_NEW(NVDS_ELEM_VIDEO_CONV, "video-converter");
        m_pConverterCapsFilter = DSL_ELEMENT_NEW(NVDS_ELEM_CAPS_FILTER, "converter-caps-filter");

        GstCaps* pCaps = gst_caps_from_string("video/x-raw(memory:NVMM), format=NV12");
        m_pConverterCapsFilter->SetAttribute("caps", pCaps);
        gst_caps_unref(pCaps);

        g_mutex_init(&m_asyncCommMutex);

        AddChild(m_pQueue);
        AddChild(m_pConverter);
        AddChild(m_pConverterCapsFilter);
        
        AddXWindowDeleteEventHandler(PlayerTerminate, this);
    }

    PlayerBintr::~PlayerBintr()
    {
        LOG_FUNC();

        GstState state;
        GetState(state, 0);
        if (state == GST_STATE_PLAYING or state == GST_STATE_PAUSED)
        {
            Stop();
        }
        RemoveXWindowDeleteEventHandler(PlayerTerminate);
        g_mutex_clear(&m_asyncCommMutex);
    }

    bool PlayerBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_pSource == nullptr or m_pSink == nullptr)
        {
            LOG_ERROR("PlayerBintr '" << GetName() << "' missing required components");
            return false;
        }
        if (m_isLinked)
        {
            LOG_ERROR("PlayerBintr '" << GetName() << "' is already linked");
            return false;
        }
        if (!m_pSource->LinkAll() or ! m_pSink->LinkAll() or 
            !m_pSource->LinkToSink(m_pQueue) or
            !m_pQueue->LinkToSink(m_pConverter) or
            !m_pConverter->LinkToSink(m_pConverterCapsFilter) or
            !m_pConverterCapsFilter->LinkToSink(m_pSink))
        {
            LOG_ERROR("Failed link SourceBintr '" << m_pSource->GetName() 
                << "' to SinkBintr '" << m_pSink->GetName() << "'");
            return false;
        }
        m_isLinked = true;
        return true;
    }

    void PlayerBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (m_pSource == nullptr or m_pSink == nullptr)
        {
            LOG_ERROR("PlayerBintr '" << GetName() << "' missing required components");
            return;
        }
        if (!m_isLinked)
        {
            LOG_ERROR("PlayerBintr '" << GetName() << "' is not linked");
            return;
        }
        if (!m_pSource->UnlinkFromSink() or
            !m_pQueue->UnlinkFromSink() or
            !m_pConverter->UnlinkFromSink() or
            !m_pConverterCapsFilter->UnlinkFromSink())
        {
            LOG_ERROR("Failed unlink SourceBintr '" << m_pSource->GetName() 
                << "' to SinkBintr '" << m_pSink->GetName() << "'");
            return;
        }
        m_pSource->UnlinkAll();
        m_pSink->UnlinkAll();
        m_isLinked = false;
    }
    
    bool PlayerBintr::Play()
    {
        LOG_FUNC();
        LOG_WARN("DO PLAY");

        GstState currentState;
        GetState(currentState, 0);
        if (currentState == GST_STATE_PLAYING)
        {
            LOG_ERROR("Unable to play Pipeline '" << GetName() 
                << "' as it's already playing");
            return false;
        }
        AddEosListener(PlayerHandleEos, this);
        return HandlePlay();
    }
    
    bool PlayerBintr::HandlePlay()
    {
        LOG_FUNC();
        GstState currentState;
        GetState(currentState, 0);
        if (currentState == GST_STATE_NULL or currentState == GST_STATE_READY)
        {
            if (!LinkAll())
            {
                LOG_ERROR("Unable to prepare Pipeline '" << GetName() << "' for Play");
                return false;
            }
            if (!SetState(GST_STATE_PAUSED, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
            {
                LOG_ERROR("Failed to Pause before playing Player '" << GetName() << "'");
                return false;
            }
        }
        if (!SetState(GST_STATE_PLAYING, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
        {
            LOG_ERROR("Failed to play Player '" << GetName() << "'");
            return false;
        }
        return true;
    }
    
    bool PlayerBintr::Pause()
    {
        LOG_FUNC();
        LOG_WARN("DO PAUSE");
        
        GstState state;
        GetState(state, 0);
        if (state != GST_STATE_PLAYING)
        {
            LOG_WARN("Player '" << GetName() << "' is not in a state of Playing");
            return false;
        }
        // If the main loop is running -- normal case -- then we can't change the 
        // state of the Player in the Application's context. 
        if (g_main_loop_is_running(DSL::Services::GetServices()->GetMainLoopHandle()))
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_asyncCommMutex);
            g_timeout_add(1, PlayerPause, this);
            g_cond_wait(&m_asyncCondition, &m_asyncCommMutex);
        }
        // Else, we are running under test without the mainloop
        else
        {
            HandlePause();
        }
        return true;
    }

    void PlayerBintr::HandlePause()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_asyncCommMutex);
        LOG_WARN("HANDLE PAUSE");
        
        // Call the base class to Pause
        if (!SetState(GST_STATE_PAUSED, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
        {
            LOG_ERROR("Failed to Pause Player '" << GetName() << "'");
        }
        g_cond_signal(&m_asyncCondition);
    }

    bool PlayerBintr::Stop()
    {
        LOG_FUNC();
        LOG_WARN("DO STOP");
        
        if (!IsLinked())
        {
            LOG_INFO("PlayerBintr is not linked when called to stop");
            return false;
        }

        // Need to remove the Terminate on EOS handler or it will call (reenter) this 
        // Stop function When we send the EOS message.
        RemoveEosListener(PlayerHandleEos);

        // Call the source to disable its EOS consumer, before sending EOS
        m_pSource->DisableEosConsumer();

        SendEos();
        sleep(1);

        // If the main loop is running -- normal case -- then we can't change the 
        // state of the Player in the Application's context. 
        if (g_main_loop_is_running(DSL::Services::GetServices()->GetMainLoopHandle()))
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_asyncCommMutex);
            g_timeout_add(1, PlayerStop, this);
            g_cond_wait(&m_asyncCondition, &m_asyncCommMutex);
        }
        // Else, we are running under test without the mainloop
        else
        {
            HandleStop();
        }
        return true;
    }
    
    void PlayerBintr::HandleStop()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_asyncCommMutex);
        LOG_WARN("HANDLE STOP");

        if (!SetState(GST_STATE_READY, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
        {
            LOG_ERROR("Failed to Stop Player '" << GetName() << "'");
        }
      
        // Unlink All objects and elements
        UnlinkAll();
        
        // If we are running under the main loop, then this funtion was called from a timer
        // thread while the client is blocked in the Stop() function on the async GCond
        g_cond_signal(&m_asyncCondition);
        
        // iterate through the map of Termination event listeners calling each
        for(auto const& imap: m_terminationEventListeners)
        {
            try
            {
                imap.first(imap.second);
            }
            catch(...)
            {
                LOG_ERROR("Exception calling Client Termination event Listener");
            }
        }
    }
    
    void PlayerBintr::HandleEos()
    {
        LOG_FUNC();
        LOG_WARN("HANDLE EOS");

        // Do not lock mutext!
        HandleStop();
    }

    void PlayerBintr::HandleTermination()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_asyncCommMutex);
        LOG_WARN("HANDLE TERMINATION");

        // Start asyn Stop timer, do not wait or block as we are
        // in the State Manager's bus watcher context.
        g_timeout_add(1, PlayerStop, this);
    }

    
    bool PlayerBintr::AddTerminationEventListener(
        dsl_player_termination_event_listener_cb listener, void* clientData)
    {
        LOG_FUNC();

        if (m_terminationEventListeners.find(listener) != m_terminationEventListeners.end())
        {   
            LOG_ERROR("Player listener is not unique");
            return false;
        }
        m_terminationEventListeners[listener] = clientData;
        
        return true;
    }
    
    bool PlayerBintr::RemoveTerminationEventListener(
        dsl_player_termination_event_listener_cb listener)
    {
        LOG_FUNC();

        if (m_terminationEventListeners.find(listener) == m_terminationEventListeners.end())
        {   
            LOG_ERROR("Player listener was not found");
            return false;
        }
        m_terminationEventListeners.erase(listener);
        
        return true;
    }

    //----------------------------------------------------------------------------------

   const uint RenderPlayerBintr::m_displayId(0);
   const uint RenderPlayerBintr::m_depth(0);
    
    RenderPlayerBintr::RenderPlayerBintr(const char* name, uint renderType, 
        uint offsetX, uint offsetY, uint zoom)
        : PlayerBintr(name)
        , m_renderType(renderType)
        , m_zoom(zoom)
        , m_offsetX(offsetX)
        , m_offsetY(offsetY)
        , m_width(0)
        , m_height(0)
    {
        LOG_FUNC();

        g_mutex_init(&m_filePathQueueMutex);
    }
    
    RenderPlayerBintr::~RenderPlayerBintr()
    {
        LOG_FUNC();
            
        g_mutex_clear(&m_filePathQueueMutex);
    }
    
    const char* RenderPlayerBintr::GetFilePath()
    {
        LOG_FUNC();
        
        DSL_RESOURCE_SOURCE_PTR pResourceSource = 
            std::dynamic_pointer_cast<ResourceSourceBintr>(m_pSource);
        return pResourceSource->GetUri();
    }

    bool RenderPlayerBintr::SetFilePath(const char* filePath)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_filePathQueueMutex);
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set File Path for RenderPlayerBintr '" << GetName() 
                << "' as it's currently linked");
            return false;
        }
        
        DSL_RESOURCE_SOURCE_PTR pResourceSource = 
            std::dynamic_pointer_cast<ResourceSourceBintr>(m_pSource);

        if (!pResourceSource->SetUri(filePath))
        {
            LOG_ERROR("Unable to set File Path for RenderPlayerBintr '" << GetName());
            return false;
        }
        
        // update the Bintr's dimensions from the SourceBintr's dimensions
        m_pSource->GetDimensions(&m_width, &m_height);
        
        // everything we need to create the SinkBintr
        if (!SetDimensions())
        {
            LOG_ERROR("Failed to update RenderSink for RenderPlayerBintr '" 
                << GetName() << "'");
            return false;
        }
        return true;
    }
    
    bool RenderPlayerBintr::QueueFilePath(const char* filePath)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_filePathQueueMutex);

        std::ifstream streamUriFile(filePath);
        if (!streamUriFile.good())
        {
            LOG_ERROR("File'" << filePath << "' Not found");
            return false;
        }

        m_filePathQueue.push(filePath);
        return true;
    }

    void RenderPlayerBintr::GetOffsets(uint* offsetX, uint* offsetY)
    {
        LOG_FUNC();
        
        *offsetX = m_offsetX;
        *offsetY = m_offsetY;
    }
    
    bool RenderPlayerBintr::SetOffsets(uint offsetX, uint offsetY)
    {
        LOG_FUNC();
        

        m_offsetX = offsetX;
        m_offsetY = offsetY;

        DSL_RENDER_SINK_PTR pRenderSink = 
            std::dynamic_pointer_cast<RenderSinkBintr>(m_pSink);

        // If the RenderSink is a WindowSinkBintr
        if (GetXWindow())
        {
            SetXWindowOffsets(m_offsetX, m_offsetY);
            return true;
        }
        // Else, update the OverlaySinkBintr;
        return pRenderSink->SetOffsets(m_offsetX, m_offsetY);
    }

    bool RenderPlayerBintr::SetDimensions()
    {
        LOG_FUNC();

        // scale the width and hight based on zoom percentage
        uint width = std::round((m_zoom * m_width) / 100);
        uint height = std::round((m_zoom * m_height) / 100);
        
        DSL_RENDER_SINK_PTR pRenderSink = 
            std::dynamic_pointer_cast<RenderSinkBintr>(m_pSink);

        // If the RenderSink is a WindowSinkBintr
        if (GetXWindow())
        {
            SetXWindowDimensions(width, height);
            return true;
        }
        // Else, update the OverlaySinkBintr;
        return pRenderSink->SetDimensions(width, height);
    }

    uint RenderPlayerBintr::GetZoom()
    {
        LOG_FUNC();

        return m_zoom;
    }

    bool RenderPlayerBintr::SetZoom(uint zoom)
    {
        LOG_FUNC();

        m_zoom = zoom;
        return SetDimensions();
    }
    
    bool RenderPlayerBintr::CreateRenderSink()
    {
        LOG_FUNC();
        
        // scale the width and hight based on zoom percentage
        uint width = std::round((m_zoom * m_width) / 100);
        uint height = std::round((m_zoom * m_height) / 100);
        
        std::string sinkName = m_name + "-render-sink__";
        if (m_renderType == DSL_RENDER_TYPE_OVERLAY)
        {
            m_pSink = DSL_OVERLAY_SINK_NEW(sinkName.c_str(), 
                m_displayId, m_depth, m_offsetX, m_offsetY, width, height);
        }
        else
        {
            m_pSink = DSL_WINDOW_SINK_NEW(sinkName.c_str(), 
                m_offsetX, m_offsetY, width, height);
        }
        if (!AddChild(m_pSink))
        {
            LOG_ERROR("Failed to add SinkBintr '" << m_pSink->GetName() 
                << "' to PlayerBintr '" << GetName() << "'");
            return false;
        }
        return true;
    }

    void RenderPlayerBintr::HandleEos()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_filePathQueueMutex);

        HandleStop();
        
        if (m_filePathQueue.size())
        {
            std::string nextFilePath = m_filePathQueue.front();
            m_filePathQueue.pop();
            
            LOG_INFO("Playing next file = '" << nextFilePath);
            SetFilePath(nextFilePath.c_str());
            
            HandlePlay();
        }
    }

    //----------------------------------------------------------------------------------
    
    VideoRenderPlayerBintr::VideoRenderPlayerBintr(const char* name, const char* filePath, 
        uint renderType, uint offsetX, uint offsetY, uint zoom, bool repeatEnabled)
        : RenderPlayerBintr(name, renderType, offsetX, offsetY, zoom)
        , m_repeatEnabled(repeatEnabled)
    {
        LOG_FUNC();

        
        std::string sourceName = m_name + "-file-source";
        m_pSource = DSL_FILE_SOURCE_NEW(name, filePath, repeatEnabled);
            
        if (!AddChild(m_pSource))
        {
            LOG_ERROR("Failed to add SourceBintr '" << m_pSource->GetName() 
                << "' to PlayerBintr '" << GetName() << "'");
            throw;
        }

        // update the Bintr's dimensions from the SourceBintr's dimensions
        m_pSource->GetDimensions(&m_width, &m_height);
        
        // everything we need to create the SinkBintr
        if (!CreateRenderSink())
        {
            LOG_ERROR("Failed to create RenderSink for VideoRenderPlayerBintr '" 
                << GetName() << "'");
            throw;
        }
    }
    
    VideoRenderPlayerBintr::~VideoRenderPlayerBintr()
    {
        LOG_FUNC();
    }
    
    bool VideoRenderPlayerBintr::GetRepeatEnabled()
    {
        LOG_FUNC();
        
        return m_repeatEnabled;
    }
    
    bool VideoRenderPlayerBintr::SetRepeatEnabled(bool repeatEnabled)
    {
        LOG_FUNC();

        if (IsLinked())
        {
            LOG_ERROR("Unable to set Repeat Enabled for VideoRenderPlayerBintr '" 
                << GetName() << "' as it's currently Linked");
            return false;
        }
        m_repeatEnabled = repeatEnabled;
        
        DSL_FILE_SOURCE_PTR pFileSource = 
            std::dynamic_pointer_cast<FileSourceBintr>(m_pSource);
        
        pFileSource->SetRepeatEnabled(repeatEnabled);
        
        return true;
    }

    //--------------------------------------------------------------------------------

    ImageRenderPlayerBintr::ImageRenderPlayerBintr(const char* name, const char* filePath, 
        uint renderType, uint offsetX, uint offsetY, uint zoom, uint timeout)
        : RenderPlayerBintr(name, renderType, offsetX, offsetY, zoom)
        , m_timeout(timeout)
    {
        LOG_FUNC();
        
        const bool isLive(false);
        const uint fpsN(4), fpsD(1);
        
        std::string sourceName = m_name + "-image-source";
        m_pSource = DSL_IMAGE_SOURCE_NEW(sourceName.c_str(), 
            filePath, isLive, fpsN, fpsD, m_timeout);        

        if (!AddChild(m_pSource))
        {
            LOG_ERROR("Failed to add SourceBintr '" << m_pSource->GetName() 
                << "' to PlayerBintr '" << GetName() << "'");
            throw;
        }
        
        // get the image dimensions from Souce
        m_pSource->GetDimensions(&m_width, &m_height);

        // everything we need to create the SinkBintr
        if (!CreateRenderSink())
        {
            LOG_ERROR("Failed to create RenderSink for VideoRenderPlayerBintr '" 
                << GetName() << "'");
            throw;
        }
    }
    
    ImageRenderPlayerBintr::~ImageRenderPlayerBintr()
    {
        LOG_FUNC();
    }

    uint ImageRenderPlayerBintr::GetTimeout()
    {
        LOG_FUNC();
        
        return m_timeout;
    }
    
    bool ImageRenderPlayerBintr::SetTimeout(uint timeout)
    {
        LOG_FUNC();

        if (IsLinked())
        {
            LOG_ERROR("Unable to set Timeout for ImageRenderPlayerBintr '" 
                << GetName() << "' as it's currently Linked");
            return false;
        }
        m_timeout = timeout;
        
        DSL_IMAGE_SOURCE_PTR pImageSource = 
            std::dynamic_pointer_cast<ImageSourceBintr>(m_pSource);
        
        pImageSource->SetTimeout(timeout);
        return true;
    }
    
    //--------------------------------------------------------------------------------
    
    static int PlayerPause(gpointer pPlayer)
    {
        static_cast<PlayerBintr*>(pPlayer)->HandlePause();
        
        // Return false to self destroy timer - one shot.
        return false;
    }
    
    static int PlayerStop(gpointer pPlayer)
    {
        static_cast<PlayerBintr*>(pPlayer)->HandleStop();
        
        // Return false to self destroy timer - one shot.
        return false;
    }
    
    static void PlayerTerminate(void* pPlayer)
    {
        static_cast<PlayerBintr*>(pPlayer)->HandleTermination();
    }    
    
    static void PlayerHandleEos(void* pPlayer)
    {
        static_cast<PlayerBintr*>(pPlayer)->HandleEos();
    }    
    
} // DSL   