/*
The MIT License

Copyright (c)   2021, Prominence AI, Inc.

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
#include "DslApi.h"
#include "DslServices.h"
#include "DslServicesValidate.h"
#include "DslPlayerBintr.h"

namespace DSL
{
    DslReturnType Services::PlayerNew(const char* name, 
        const char* source, const char* sink)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, source);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, source);
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, sink);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SINK(m_components, sink);
        
            if (m_players.find(name) != m_players.end())
            {   
                LOG_ERROR("Player name '" << name << "' is not unique");
                return DSL_RESULT_PLAYER_NAME_NOT_UNIQUE;
            }
            DSL_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<SourceBintr>(m_components[source]);

            DSL_SINK_PTR pSinkBintr = 
                std::dynamic_pointer_cast<SinkBintr>(m_components[sink]);
            
            m_players[name] = std::shared_ptr<PlayerBintr>(new 
                PlayerBintr(name, pSourceBintr, pSinkBintr));
                
            LOG_INFO("New Player '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Player '" << name << "' threw exception on create");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerRenderVideoNew(const char* name, const char* filePath,
            uint renderType, uint offsetX, uint offsetY, uint zoom, boolean repeatEnabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
        
            if (renderType == DSL_RENDER_TYPE_OVERLAY)
                {
                // Get the Device properties
                cudaDeviceProp deviceProp;
                cudaGetDeviceProperties(&deviceProp, 0);
                
                if (!deviceProp.integrated)
                {
                    LOG_ERROR("Overlay Sink is not supported on dGPU x86_64 builds");
                    return DSL_RESULT_SINK_OVERLAY_NOT_SUPPORTED;
                }
            }
            if (m_players.find(name) != m_players.end())
            {   
                LOG_ERROR("Player name '" << name << "' is not unique");
                return DSL_RESULT_PLAYER_NAME_NOT_UNIQUE;
            }
            std::string pathString(filePath);
            if (pathString.size())
            {
                std::ifstream streamUriFile(filePath);
                if (!streamUriFile.good())
                {
                    LOG_ERROR("File Source'" << filePath << "' Not found");
                    return DSL_RESULT_SOURCE_FILE_NOT_FOUND;
                }
            }
            m_players[name] = std::shared_ptr<VideoRenderPlayerBintr>(new 
                VideoRenderPlayerBintr(name, filePath, renderType,
                    offsetX, offsetY, zoom, repeatEnabled));
                    
            LOG_INFO("New Render File Player '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Render File Player '" << name << "' threw exception on create");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerRenderImageNew(const char* name, const char* filePath,
            uint renderType, uint offsetX, uint offsetY, uint zoom, uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            if (m_players.find(name) != m_players.end())
            {   
                LOG_ERROR("Player name '" << name << "' is not unique");
                return DSL_RESULT_PLAYER_NAME_NOT_UNIQUE;
            }
            std::string pathString(filePath);
            if (pathString.size())
            {
                std::ifstream streamUriFile(filePath);
                if (!streamUriFile.good())
                {
                    LOG_ERROR("File Source'" << filePath << "' Not found");
                    return DSL_RESULT_SOURCE_FILE_NOT_FOUND;
                }
            }
            m_players[name] = std::shared_ptr<ImageRenderPlayerBintr>(new 
                ImageRenderPlayerBintr(name, filePath, renderType,
                    offsetX, offsetY, zoom, timeout));
                    
            LOG_INFO("New Render Image Player '" << name << "' created successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Render Image Player '" << name 
                << "' threw exception on create");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerRenderFilePathGet(const char* name, 
        const char** filePath)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_PLAYER_IS_NOT_RENDER_PLAYER(m_players, name);

            DSL_PLAYER_RENDER_BINTR_PTR pRenderPlayer = 
                std::dynamic_pointer_cast<RenderPlayerBintr>(m_players[name]);

            *filePath = pRenderPlayer->GetFilePath();
            
            LOG_INFO("Render Player '" << name << "' returned File Path = '" 
                << *filePath << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Render Player '" << name 
                << "' threw exception getting File Path");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }
            

    DslReturnType Services::PlayerRenderFilePathSet(const char* name, const char* filePath)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_PLAYER_IS_NOT_RENDER_PLAYER(m_players, name);

            DSL_PLAYER_RENDER_BINTR_PTR pRenderPlayer = 
                std::dynamic_pointer_cast<RenderPlayerBintr>(m_players[name]);

            if (!pRenderPlayer->SetFilePath(filePath))
            {
                LOG_ERROR("Failed to Set File Path '" << filePath 
                    << "' for Render Player '" << name << "'");
                return DSL_RESULT_PLAYER_SET_FAILED;
            }
            LOG_INFO("Render Player '" << name << "' set File Path = '" 
                << filePath << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Render Player '" << name 
                << "' threw exception setting File Path");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerRenderFilePathQueue(const char* name, 
        const char* filePath)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_PLAYER_IS_NOT_RENDER_PLAYER(m_players, name);

            DSL_PLAYER_RENDER_BINTR_PTR pRenderPlayer = 
                std::dynamic_pointer_cast<RenderPlayerBintr>(m_players[name]);

            if (!pRenderPlayer->QueueFilePath(filePath))
            {
                LOG_ERROR("Failed to Queue File Path '" << filePath 
                    << "' for Render Player '" << name << "'");
                return DSL_RESULT_PLAYER_SET_FAILED;
            }
            LOG_INFO("Render Player '" << name << "' queued File Path = '" 
                << filePath << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Render Player '" << name 
                << "' threw exception queuing File Path");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerRenderOffsetsGet(const char* name, uint* offsetX, uint* offsetY)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_PLAYER_IS_NOT_RENDER_PLAYER(m_players, name);

            DSL_PLAYER_RENDER_BINTR_PTR pRenderPlayer = 
                std::dynamic_pointer_cast<RenderPlayerBintr>(m_players[name]);

            pRenderPlayer->GetOffsets(offsetX, offsetY);
            
            LOG_INFO("Render Player '" << name << "' returned Offset X = " 
                << *offsetX << " and Offset Y = " << *offsetY << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Render Player '" << name << "' threw an exception getting offsets");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerRenderOffsetsSet(const char* name, uint offsetX, uint offsetY)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_PLAYER_IS_NOT_RENDER_PLAYER(m_players, name);

            DSL_PLAYER_RENDER_BINTR_PTR pRenderPlayer = 
                std::dynamic_pointer_cast<RenderPlayerBintr>(m_players[name]);

            if (!pRenderPlayer->SetOffsets(offsetX, offsetY))
            {
                LOG_ERROR("Render Player '" << name << "' failed to set offsets");
                return DSL_RESULT_PLAYER_SET_FAILED;
            }
            LOG_INFO("Render Sink '" << name << "' set Offset X = " 
                << offsetX << " and Offset Y = " << offsetY << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception setting Clock offsets");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerRenderZoomGet(const char* name, uint* zoom)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_PLAYER_IS_NOT_RENDER_PLAYER(m_players, name);

            DSL_PLAYER_RENDER_BINTR_PTR pRenderPlayer = 
                std::dynamic_pointer_cast<RenderPlayerBintr>(m_players[name]);

            *zoom = pRenderPlayer->GetZoom();
            
            LOG_INFO("Render Player '" << name << "' returned Zoom = " 
                << *zoom << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Render Player '" << name 
                << "' threw exception getting Zoom");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }
            

    DslReturnType Services::PlayerRenderZoomSet(const char* name, uint zoom)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_PLAYER_IS_NOT_RENDER_PLAYER(m_players, name);

            DSL_PLAYER_RENDER_BINTR_PTR pRenderPlayer = 
                std::dynamic_pointer_cast<RenderPlayerBintr>(m_players[name]);

            if (!pRenderPlayer->SetZoom(zoom))
            {
                LOG_ERROR("Failed to Set Zooom '" << zoom 
                    << "' for Render Player '" << name << "'");
                return DSL_RESULT_PLAYER_SET_FAILED;
            }
            LOG_INFO("Render Player '" << name << "' set Zoom = " 
                << zoom << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Render Player '" << name 
                << "' threw exception setting Zoom");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerRenderReset(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_PLAYER_IS_NOT_RENDER_PLAYER(m_players, name);

            DSL_PLAYER_RENDER_BINTR_PTR pRenderPlayer = 
                std::dynamic_pointer_cast<RenderPlayerBintr>(m_players[name]);

            if (!pRenderPlayer->Reset())
            {
                LOG_ERROR("Failed to Reset Render Player '" << name << "'");
                return DSL_RESULT_PLAYER_SET_FAILED;
            }
            LOG_INFO("Render Player '" << name << "' Reset successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Render Player '" << name 
                << "' threw exception on Reset");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerRenderImageTimeoutGet(const char* name, 
        uint* timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_players, 
                name, ImageRenderPlayerBintr);

            DSL_PLAYER_RENDER_IMAGE_BINTR_PTR pImageRenderPlayer = 
                std::dynamic_pointer_cast<ImageRenderPlayerBintr>(m_players[name]);

            *timeout = pImageRenderPlayer->GetTimeout();

            LOG_INFO("Image Render Player '" << name << "' returned Timeout = " 
                << *timeout << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Image Render Player '" << name 
                << "' threw exception getting Timeout");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }
            

    DslReturnType Services::PlayerRenderImageTimeoutSet(const char* name, 
        uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_players, 
                name, ImageRenderPlayerBintr);

            DSL_PLAYER_RENDER_IMAGE_BINTR_PTR pImageRenderPlayer = 
                std::dynamic_pointer_cast<ImageRenderPlayerBintr>(m_players[name]);

            if (!pImageRenderPlayer->SetTimeout(timeout))
            {
                LOG_ERROR("Failed to Set Timeout to '" << timeout 
                    << "s' for Image Render Player '" << name << "'");
                return DSL_RESULT_PLAYER_SET_FAILED;
            }
            LOG_INFO("Image Render Player '" << name << "' set Timeout = " 
                << timeout << "' successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Image Render Player '" << name 
                << "' threw exception setting Timeout");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerRenderVideoRepeatEnabledGet(const char* name, 
        boolean* repeatEnabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_players, 
                name, VideoRenderPlayerBintr);

            DSL_PLAYER_RENDER_VIDEO_BINTR_PTR pVideoRenderPlayer = 
                std::dynamic_pointer_cast<VideoRenderPlayerBintr>(m_players[name]);

            *repeatEnabled = pVideoRenderPlayer->GetRepeatEnabled();

            LOG_INFO("Video Render Player '" << name << "' returned Repeat Enabled = " 
                << *repeatEnabled << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Image Render Player '" << name 
                << "' threw exception getting Timeout");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerRenderVideoRepeatEnabledSet(const char* name, 
        boolean repeatEnabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_players, 
                name, VideoRenderPlayerBintr);

            DSL_PLAYER_RENDER_VIDEO_BINTR_PTR pVideoRenderPlayer = 
                std::dynamic_pointer_cast<VideoRenderPlayerBintr>(m_players[name]);

            if (!pVideoRenderPlayer->SetRepeatEnabled(repeatEnabled))
            {
                LOG_ERROR("Failed to Set Repeat Enabled to '" << repeatEnabled 
                    << "' for Video Render Player '" << name << "'");
                return DSL_RESULT_PLAYER_SET_FAILED;
            }
            LOG_INFO("Video Render Player '" << name << "' set Repeat Enabled = " 
                << repeatEnabled << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Video Render Player '" << name 
                << "' threw exception setting Repeat Enabled");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerTerminationEventListenerAdd(const char* name,
        dsl_player_termination_event_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);

        try
        {
            if (!m_players[name]->AddTerminationEventListener(listener, clientData))
            {
                LOG_ERROR("Player '" << name 
                    << "' failed to add Termination Event Listener");
                return DSL_RESULT_PLAYER_CALLBACK_ADD_FAILED;
            }
            LOG_INFO("Player '" << name 
                << "' added Termination Event Listener successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Player '" << name 
                << "' threw an exception adding Termination Event Listner");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::PlayerTerminationEventListenerRemove(const char* name,
        dsl_player_termination_event_listener_cb listener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            
            if (!m_players[name]->RemoveTerminationEventListener(listener))
            {
                LOG_ERROR("Player '" << name 
                    << "' failed to remove Termination Event Listener");
                return DSL_RESULT_PLAYER_CALLBACK_REMOVE_FAILED;
            }
            LOG_INFO("Player '" << name 
                << "' removed Termination Event Listener successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Player '" << name 
                << "' threw an exception adding Termination Event Listner");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerXWindowHandleGet(const char* name, uint64_t* xwindow) 
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            
            *xwindow = m_players[name]->GetXWindow();

            LOG_INFO("Player '" << name 
                << "' returned X Window Handle successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Player '" << name << "' threw an exception getting XWindow handle");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::PlayerXWindowHandleSet(const char* name, uint64_t xwindow)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            
            if (!m_players[name]->SetXWindow(xwindow))
            {
                LOG_ERROR("Failure setting XWindow handle for Player '" << name << "'");
                return DSL_RESULT_PLAYER_XWINDOW_SET_FAILED;
            }
            LOG_INFO("Player '" << name 
                << "' set X Window Handle successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Player '" << name << "' threw an exception setting XWindow handle");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerXWindowKeyEventHandlerAdd(const char* name, 
        dsl_xwindow_key_event_handler_cb handler, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);

            if (!m_players[name]->AddXWindowKeyEventHandler(handler, clientData))
            {
                LOG_ERROR("Player '" << name 
                    << "' failed to add XWindow Key Event Handler");
                return DSL_RESULT_PLAYER_CALLBACK_ADD_FAILED;
            }
            LOG_INFO("Player '" << name 
                << "' added X Window Key Event Handler successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Player '" << name 
                << "' threw an exception adding XWindow Key Event Handler");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerXWindowKeyEventHandlerRemove(const char* name, 
        dsl_xwindow_key_event_handler_cb handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);

            if (!m_players[name]->RemoveXWindowKeyEventHandler(handler))
            {
                LOG_ERROR("Player '" << name 
                    << "' failed to remove XWindow Key Event Handler");
                return DSL_RESULT_PLAYER_CALLBACK_REMOVE_FAILED;
            }
            LOG_INFO("Player '" << name 
                << "' removed X Window Key Event Handler successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Player '" << name 
                << "' threw an exception removing XWindow Key Event Handler");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerPlay(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);

            if (!m_players[name]->Play())
            {
                return DSL_RESULT_PLAYER_FAILED_TO_PLAY;
            }
            LOG_INFO("Player '" << name 
                << "' transitioned to a state of PLAYING successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Player '" << name 
                << "' threw an exception on Play");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::PlayerPause(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);

            if (!m_players[name]->Pause())
            {
                return DSL_RESULT_PLAYER_FAILED_TO_PAUSE;
            }
            LOG_INFO("Player '" << name 
                << "' transitioned to a state of PAUSED successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Player '" << name 
                << "' threw an exception on Pause");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerStop(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);

        if (!m_players[name]->Stop())
        {
            return DSL_RESULT_PLAYER_FAILED_TO_STOP;
        }

        LOG_INFO("Player '" << name 
            << "' transitioned to a state of READY successfully");
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PlayerRenderNext(const char* name)
    {
        LOG_FUNC();

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_PLAYER_IS_NOT_RENDER_PLAYER(m_players, name);

            DSL_PLAYER_RENDER_BINTR_PTR pRenderPlayer = 
                std::dynamic_pointer_cast<RenderPlayerBintr>(m_players[name]);

            if (!pRenderPlayer->Next())
            {
                LOG_ERROR("Player '" << name 
                    << "' failed to Play Next");
                return DSL_RESULT_PLAYER_RENDER_FAILED_TO_PLAY_NEXT;
            }
            LOG_INFO("Render Player '" << name 
                << "' was able to Render next file path successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Player '" << name 
                << "' threw an exception on Play Next");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerStateGet(const char* name, uint* state)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            GstState gstState;
            m_players[name]->GetState(gstState, 0);
            *state = (uint)gstState;
            
            LOG_INFO("Player '" << name 
                << "' returned a current state of '" << StateValueToString(*state) << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Player '" << name 
                << "' threw an exception getting state");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    boolean Services::PlayerExists(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            return (boolean)(m_players.find(name) != m_players.end());
        }
        catch(...)
        {
            LOG_ERROR("Player '" << name 
                << "' threw an exception on check for Exists");
            return false;
        }
    }

    DslReturnType Services::PlayerDelete(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);

            m_players.erase(name);

            LOG_INFO("Player '" << name << "' deleted successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Player '" << name 
                << "' threw an exception on Delete");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }

    }

    DslReturnType Services::PlayerDeleteAll(bool checkInUse)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            if (m_players.empty())
            {
                return DSL_RESULT_SUCCESS;
            }
            for (auto &imap: m_players)
            {
                // In the case of DSL Delete all - we don't check for in-use
                // as their can be a circular type ownership/relation that will
                // cause it to fail... i.e. players can own record sinks which 
                // can own players, and so on...
                if (checkInUse and imap.second.use_count() > 1)
                {
                    LOG_ERROR("Can't delete Player '" << imap.second->GetName() 
                        << "' as it is currently in use");
                    return DSL_RESULT_PLAYER_IN_USE;
                }

                imap.second->RemoveAllChildren();
                imap.second = nullptr;
            }

            m_players.clear();

            LOG_INFO("All Players deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception on PlayerDeleteAll");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    uint Services::PlayerListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_players.size();
    }
    
}