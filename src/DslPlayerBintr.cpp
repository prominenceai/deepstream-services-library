/*
The MIT License

Copyright (c) 2019-2023, Prominence AI, Inc.


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
#include "DslSourceBintr.h"
#include "DslSinkBintr.h"

namespace DSL
{
    PlayerBintr::PlayerBintr(const char* name, 
        DSL_BINTR_PTR pSource, DSL_BINTR_PTR pSink)
        : Bintr(name, true) // Pipeline = true
        , PipelineStateMgr(m_pGstObj)
        , PipelineXWinMgr(m_pGstObj)
        , m_pSource(pSource)
        , m_pSink(pSink)
        , m_inTermination(false)
        , m_clearPlayNextMutex(false)
    {
        LOG_FUNC();

//        m_pQueue = DSL_ELEMENT_NEW("queue", name);
//        m_pConverter = DSL_ELEMENT_NEW("nvvideoconvert", name);
//        m_pConverterCapsFilter = DSL_ELEMENT_NEW("capsfilter", name);
//
//        GstCaps* pCaps = gst_caps_from_string("video/x-raw(memory:NVMM), format=NV12");
//        m_pConverterCapsFilter->SetAttribute("caps", pCaps);
//        gst_caps_unref(pCaps);

        g_mutex_init(&m_asyncCommMutex);
        g_mutex_init(&m_playNextMutex);
        
//        AddChild(m_pQueue);
//        AddChild(m_pConverter);
//        AddChild(m_pConverterCapsFilter);
        
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
        , m_inTermination(false)
        , m_clearPlayNextMutex(false)
    {
        LOG_FUNC();

        m_pQueue = DSL_ELEMENT_NEW("queue", name);
        m_pConverter = DSL_ELEMENT_NEW("nvvideoconvert", name);
        m_pConverterCapsFilter = DSL_ELEMENT_NEW("capsfilter", name);

        GstCaps* pCaps = gst_caps_from_string("video/x-raw(memory:NVMM), format=NV12");
        m_pConverterCapsFilter->SetAttribute("caps", pCaps);
        gst_caps_unref(pCaps);

        g_mutex_init(&m_asyncCommMutex);
        g_mutex_init(&m_playNextMutex);

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
        g_mutex_clear(&m_playNextMutex);
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
//        if (!m_pSource->LinkAll() or ! m_pSink->LinkAll() or 
//            !m_pSource->LinkToSink(m_pQueue) or
//            !m_pQueue->LinkToSink(m_pConverter) or
//            !m_pConverter->LinkToSink(m_pConverterCapsFilter) or
//            !m_pConverterCapsFilter->LinkToSink(m_pSink))
        if (!m_pSource->LinkAll() or ! m_pSink->LinkAll() or 
            !m_pSource->LinkToSink(m_pSink))
        {
            LOG_ERROR("Failed to link SourceBintr '" << m_pSource->GetName() 
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
//        if (!m_pSource->UnlinkFromSink() or
//            !m_pQueue->UnlinkFromSink() or
//            !m_pConverter->UnlinkFromSink() or
//            !m_pConverterCapsFilter->UnlinkFromSink())
        if (!m_pSource->UnlinkFromSink())
        {
            LOG_ERROR("Failed ti unlink SourceBintr '" << m_pSource->GetName() 
                << "' from SinkBintr '" << m_pSink->GetName() << "'");
            return;
        }
        m_pSource->UnlinkAll();
        m_pSink->UnlinkAll();
        m_isLinked = false;
    }
    
    bool PlayerBintr::Play()
    {
        LOG_FUNC();

        GstState currentState;
        GetState(currentState, 0);
        if (currentState == GST_STATE_PLAYING)
        {
            LOG_ERROR("Unable to play Player '" << GetName() 
                << "' as it's already in a state of playing");
            return false;
        }
        // m_pSource is of type DSL_BINTR_PTR - need to cast to DSL_SOURCE_PTR
        // for the source to be used as such
        DSL_SOURCE_PTR pSourceBintr = 
            std::dynamic_pointer_cast<SourceBintr>(m_pSource);
        if (!pSourceBintr->IsLinkable())
        {
            LOG_ERROR("Unable to Play Player '" << GetName() 
                << "' as its Source is in an un-playable state");
            return false;
        }
        // If the main loop is running -- normal case -- then we can't change the 
        // state of the Player in the Application's context. 
        if (g_main_loop_is_running(DSL::Services::GetServices()->GetMainLoopHandle()))
        {
            g_timeout_add(1, PlayerPlay, this);
        }
        // Else, we are running under test without the mainloop
        else
        {
            HandlePlay();
        }
        return true;
    }
    
    bool PlayerBintr::HandlePlay()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_asyncCommMutex);

        // m_pSource is of type DSL_BINTR_PTR - need to cast to DSL_SOURCE_PTR
        // for the source to be used as such
        DSL_SOURCE_PTR pSourceBintr = 
            std::dynamic_pointer_cast<SourceBintr>(m_pSource);
        
        GstState currentState;
        GetState(currentState, 0);
        if (currentState == GST_STATE_NULL or currentState == GST_STATE_READY)
        {
            if (!LinkAll())
            {
                LOG_ERROR("Unable to prepare Player '" << GetName() << "' for Play");
                return false;
            }
            // For non-live sources we Pause to preroll before we play
            if (!pSourceBintr->IsLive())
            {
                if (!SetState(GST_STATE_PAUSED, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
                {
                    LOG_ERROR("Failed to Pause before playing Player '" << GetName() << "'");
                    return false;
                }
            }
        }
        if (!SetState(GST_STATE_PLAYING, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
        {
            LOG_ERROR("Failed to play Player '" << GetName() << "'");
            return false;
        }

        // conditionally add the EOS Listener as it may have been
        // removed by the client with a previous call to Stop()
        if (!IsEosListener(PlayerHandleEos))
        {
            AddEosListener(PlayerHandleEos, this);
        }

        // If the Play was invoked from a Play-Next cycle, clear the Mutex.
        if (m_clearPlayNextMutex)
        {
            g_mutex_unlock(&m_playNextMutex);
            m_clearPlayNextMutex = false;
        }

        return true;
    }
    
    bool PlayerBintr::Pause()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_asyncCommMutex);

        GstState state;
        GetState(state, 0);
        if (state != GST_STATE_PLAYING)
        {
            LOG_WARN("Player '" << GetName() << "' is not in a state of Playing");
            return false;
        }
        
        // Call the base class to Pause the Player - can be called from any context.
        if (!SetState(GST_STATE_PAUSED, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
        {
            LOG_ERROR("Failed to Pause Player '" << GetName() << "'");
            return false;
        }
        return true;
    }

    bool PlayerBintr::Stop()
    {
        LOG_FUNC();
        
        if (!IsLinked())
        {
            LOG_INFO("PlayerBintr is not linked when called to stop");
            return false;
        }
        GstState state;
        GetState(state, 0);
        if (state == GST_STATE_PAUSED)
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_asyncCommMutex);
            LOG_INFO("Setting Player '" << GetName() 
                << "' to PLAYING before setting to NULL");
            // Call the base class to Play the Player - can be called from any context.
            if (!SetState(GST_STATE_PLAYING, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
            {
                LOG_ERROR("Failed to set Player '" << GetName() 
                    << "' to PLAYING before setting to NULL");
                return false;
            }
        }
        
        // Disable the EOS management
        PrepareForEos();

        // Need to check the context to see if we're running from either
        // the XDisplay thread or the bus-watch fucntion
        
        // Try and lock the Display mutex first
        if (!g_mutex_trylock(&m_displayMutex))
        {
            // lock-failed which means we are already in the XWindow thread context
            // calling on a client handler function for Key release or xWindow delete. 
            // Safe to stop the Player in this context.
            LOG_INFO("dsl_player_stop called from XWindow display thread context");
            HandleStop();
            return true;
        }
        // Try the bus-watch mutex next
        if (!g_mutex_trylock(&m_busWatchMutex))
        {
            // lock-failed which means we're in the bus-watch function context
            // calling on a client listener or handler function. Safe to stop 
            // the Player in this context. 
            LOG_INFO("dsl_player_stop called from bus-watch-function thread context");
            HandleStop();
            g_mutex_unlock(&m_displayMutex);
            return true;
        }

        // If the main loop is running -- normal case -- then we can't change the 
        // state of the Player in the Application's context. 
        if (g_main_loop_is_running(DSL::Services::GetServices()->GetMainLoopHandle()))
        {
            LOG_INFO("Sending application message to stop the player");
            
            gst_element_post_message(GetGstElement(),
                gst_message_new_application(GetGstObject(),
                    gst_structure_new_empty("stop-pipline")));
        }
        // Else, client has stopped the main-loop or we are running under test 
        // without the mainloop running - can't send a message so handle stop now.
        else
        {
            HandleStop();
        }
        g_mutex_unlock(&m_displayMutex);
        g_mutex_unlock(&m_busWatchMutex);
        return true;
    }

    void PlayerBintr::PrepareForEos()
    {
        LOG_FUNC();

        // Need to remove the Terminate on EOS handler or it will call (reenter) this 
        // Stop function When we send the EOS message.
        RemoveEosListener(PlayerHandleEos);

        std::dynamic_pointer_cast<SourceBintr>(m_pSource)->DisableEosConsumer();

    }
    
    void PlayerBintr::HandleStop()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_asyncCommMutex);

        // If the client is not stoping due to EOS, we must EOS the Player 
        // to gracefully stop any recording in progress before changing the 
        // Player's state to NULL, 
        if (!m_eosFlag)
        {
            // Send an EOS event to the Pipline bin. 
            SendEos();
            
            // once the EOS event has been received on all sink pads of all
            // elements, an EOS message will be posted on the bus. We need to
            // discard all bus messages while waiting for the EOS message.
            GstMessage* msg = gst_bus_timed_pop_filtered(m_pGstBus, 
                DSL_DEFAULT_WAIT_FOR_EOS_TIMEOUT_IN_SEC * GST_SECOND,
                    (GstMessageType)(GST_MESSAGE_CLOCK_LOST | GST_MESSAGE_ERROR | 
                        GST_MESSAGE_EOS));

//            if (!msg or GST_MESSAGE_TYPE(msg) != GST_MESSAGE_EOS)
//            {
//                LOG_WARN("Player '" << GetName() 
//                    << "' failed to receive final EOS message on ");
//            }
        }

        if (!SetState(GST_STATE_NULL, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
        {
            LOG_ERROR("Failed to Stop Player '" << GetName() << "'");
        }
        
        m_eosFlag = false;
        UnlinkAll();
        
        if (m_inTermination)
        {
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
            m_inTermination = false;
        }
    }

    void PlayerBintr::HandleEos()
    {
        LOG_FUNC();

        // Do not lock mutext!
        HandleStop();
    }

    void PlayerBintr::HandleTermination()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_asyncCommMutex);
        
        // Set the termination flag so that the async HandleStop()
        // Can notifiy all Termination Listeners
        m_inTermination = true;

        // Remove the EOS Listener and disable the source's EOS handler, 
        PrepareForEos();
        
        // Start asyn Stop timer to complete the stop, do not wait or block as we are
        // in the State Manager's bus watcher context - i.e. main loop.
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
        
        pResourceSource->GetDimensions(&m_width, &m_height);
        
        // everything we need to create the SinkBintr
        if (!SetDimensions())
        {
            LOG_ERROR("Failed to update RenderSink dimensions for RenderPlayerBintr '" 
                << GetName() << "'");
            return false;
        }
        if (!SetOffsets(m_offsetX, m_offsetY))
        {
            LOG_ERROR("Failed to update RenderSink offsets for RenderPlayerBintr '" 
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
        if (OwnsXWindow())
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
    
    bool RenderPlayerBintr::Reset()
    {
        LOG_FUNC();

        DSL_RENDER_SINK_PTR pRenderSink = 
            std::dynamic_pointer_cast<RenderSinkBintr>(m_pSink);

        // If the RenderSink is a WindowSinkBintr
        if (OwnsXWindow())
        {
            DestroyXWindow();
        }
        
        return pRenderSink->Reset();
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

    bool RenderPlayerBintr::Next()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_asyncCommMutex);
        
        
        // Lock the Play-Next mutex and set the flag
        if (!g_mutex_trylock(&m_playNextMutex))
        {
            LOG_ERROR("Unable to Play next file path as the PlayerBintr '" 
                << GetName() << "' is currently in a Transitioning state");
        }
        m_clearPlayNextMutex = true;
        
        GstState state;
        GetState(state, 0);
        if (state == GST_STATE_NULL or state == GST_STATE_READY)
        {
            LOG_ERROR("Unable to Play next file path as the PlayerBintr '" 
                << GetName() << "' is not in a Paused or Playing state");

            g_mutex_unlock(&m_playNextMutex);
            m_clearPlayNextMutex = false;
            return false;
        }
        
        // Need to initiate the stop process
        PrepareForEos();
        
        // The process of playing next is the same as handling an EOS.
        HandleEos();
        return true;
    }
    
    void RenderPlayerBintr::HandleStopAndPlay()
    {
        LOG_FUNC();

        // Start by invoking the HandleStop routine to set the Player back to
        // the Ready state and unlink all so that the file path can be updated
        HandleStop();

        if (m_filePathQueue.empty())
        {
            LOG_ERROR("Called to handle Stop and Play for PlayerBintr '" 
                << GetName() << "' without queued files???");
            return;
        }

        // get the next file path from the queue
        std::string nextFilePath = m_filePathQueue.front();
        m_filePathQueue.pop();
        
        LOG_INFO("Playing next file = '" << nextFilePath);
        
        if (!SetFilePath(nextFilePath.c_str()))
        {
            LOG_ERROR("Failed to set next file path for PlayerBintr '" 
                << GetName() << "'");
            return;
        }
        
        // Finish by invoking Handle Play to re-link all and transition the 
        // Player back to a state of Playing.
        HandlePlay();
    }
    

    void RenderPlayerBintr::HandleEos()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_filePathQueueMutex);

        // If there are file paths queued to play next
        if (m_filePathQueue.size())
        {
            g_timeout_add(1, PlayerStopAndPlay, this);
        }
        else
        {   
            // need to set the async comm flag to tell the the Handle
            // to inform all registered Termination Listeners of EOS
            m_inTermination = true;
            g_timeout_add(1, PlayerStop, this);
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
        DSL_FILE_SOURCE_PTR pSource = DSL_FILE_SOURCE_NEW(name, filePath, repeatEnabled);
            
        if (!AddChild(pSource))
        {
            LOG_ERROR("Failed to add SourceBintr '" << pSource->GetName() 
                << "' to PlayerBintr '" << GetName() << "'");
            throw;
        }

        // update the Bintr's dimensions from the SourceBintr's dimensions
        pSource->GetDimensions(&m_width, &m_height);
        
        // everything we need to create the SinkBintr
        if (!CreateRenderSink())
        {
            LOG_ERROR("Failed to create RenderSink for VideoRenderPlayerBintr '" 
                << GetName() << "'");
            throw;
        }
        
        // save to member variable of type DSL_BINTR_PTR
        m_pSource = pSource;
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
        DSL_IMAGE_STREAM_SOURCE_PTR pSource = DSL_IMAGE_STREAM_SOURCE_NEW(sourceName.c_str(), 
            filePath, isLive, fpsN, fpsD, m_timeout);        

        if (!AddChild(pSource))
        {
            LOG_ERROR("Failed to add SourceBintr '" << pSource->GetName() 
                << "' to PlayerBintr '" << GetName() << "'");
            throw;
        }
        
        // get the image dimensions from the Source
        pSource->GetDimensions(&m_width, &m_height);

        // everything we need to create the SinkBintr
        if (!CreateRenderSink())
        {
            LOG_ERROR("Failed to create RenderSink for VideoRenderPlayerBintr '" 
                << GetName() << "'");
            throw;
        }
        
        // save to member variable of type DSL_BINTR_PTR
        m_pSource = pSource;
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
        
        DSL_IMAGE_STREAM_SOURCE_PTR pImageSource = 
            std::dynamic_pointer_cast<ImageStreamSourceBintr>(m_pSource);
        
        pImageSource->SetTimeout(timeout);
        return true;
    }
    
    //--------------------------------------------------------------------------------
    
    static int PlayerPlay(gpointer pPlayer)
    {
        static_cast<PlayerBintr*>(pPlayer)->HandlePlay();
        
        // Return false to self destroy timer - one shot.
        return false;
    }
    
    static int PlayerStop(gpointer pPlayer)
    {
        static_cast<PlayerBintr*>(pPlayer)->HandleStop();
        
        // Return false to self destroy timer - one shot.
        return false;
    }
    
    static int PlayerStopAndPlay(gpointer pPlayer)
    {
        static_cast<RenderPlayerBintr*>(pPlayer)->HandleStopAndPlay();
        
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