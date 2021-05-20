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
#include "DslRecordMgr.h"
#include "DslPlayerBintr.h"

namespace DSL
{

    //-------------------------------------------------------------------------
    
    RecordMgr::RecordMgr(const char* name, const char* outdir, 
        uint container, dsl_record_client_listener_cb clientListener)
        : m_name(name)
        , m_outdir(outdir)
        , m_pContext(NULL)
        , m_initParams{0}
        , m_clientListener(clientListener)
        , m_clientData(0)
        , m_currentSessionId(UINT32_MAX)
    {
        LOG_FUNC();

        switch (container)
        {
        case DSL_CONTAINER_MP4 :
            m_initParams.containerType = NVDSSR_CONTAINER_MP4;        
            break;
        case DSL_CONTAINER_MKV :
            m_initParams.containerType = NVDSSR_CONTAINER_MKV;        
            break;
        default:
            LOG_ERROR("Invalid container = '" << container << "' for new RecordMgr '" << name << "'");
            throw;
        }
        
        // Set single callback listener. Unique clients are identifed using client_data provided on Start session
        m_initParams.callback = RecordCompleteCallback;
        
        // Set both width and height params to zero = no-transcode
        m_initParams.width = 0;  
        m_initParams.height = 0; 
        
        // Filename prefix uses bintr name by default
        m_initParams.fileNamePrefix = const_cast<gchar*>(m_name.c_str());
        m_initParams.dirpath = const_cast<gchar*>(m_outdir.c_str());
        
        m_initParams.defaultDuration = DSL_DEFAULT_VIDEO_RECORD_DURATION_IN_SEC;
        m_initParams.videoCacheSize = DSL_DEFAULT_VIDEO_RECORD_CACHE_IN_SEC;
    }
    
    RecordMgr::~RecordMgr()
    {
        LOG_FUNC();

        if (m_pContext)
        {
            LOG_INFO("Destroying context");
            DestroyContext();
        }
    }
    
    bool RecordMgr::CreateContext()
    {
        LOG_FUNC();
        
        // Create the smart record context
        if (NvDsSRCreate(&m_pContext, &m_initParams) != NVDSSR_STATUS_OK)
        {
            LOG_ERROR("Failed to create Smart Record Context for new RecordMgr '" << m_name << "'");
            return false;
        }
    }

    void RecordMgr::DestroyContext()
    {
        LOG_FUNC();

        if (!m_pContext)
        {
            LOG_ERROR("There is no context to destroy for RecordMgr '" << m_name << "'");
            return;
        }
        if (IsOn())
        {
            StopSession();
        }
        NvDsSRDestroy(m_pContext);
        m_pContext = NULL;
    }


    const char* RecordMgr::GetOutdir()
    {
        LOG_FUNC();
        
        return m_outdir.c_str();
    }
    
    bool RecordMgr::SetOutdir(const char* outdir)
    {
        LOG_FUNC();

        if (m_pContext)
        {
            LOG_ERROR("Unable to set the Output for RecordMgr '" << m_name 
                << "' as it is currently in use");
            return false;
        }
        
        m_outdir.assign(outdir);
        return true;
    }

    uint RecordMgr::GetContainer()
    {
        LOG_FUNC();
        
        return m_initParams.containerType;
    }

    bool RecordMgr::SetContainer(uint container)
    {
        LOG_FUNC();
        
        if (m_pContext)
        {
            LOG_ERROR("Unable to set container type for RecordMgr '" << m_name 
                << "' as it is currently in use");
            return false;
        }

        switch (container)
        {
        case DSL_CONTAINER_MP4 :
            m_initParams.containerType = NVDSSR_CONTAINER_MP4;        
            break;
        case DSL_CONTAINER_MKV :
            m_initParams.containerType = NVDSSR_CONTAINER_MKV;        
            break;
        default:
            LOG_ERROR("Invalid container = '" << container << "' for RecordMgr '" << m_name << "'");
            return false;
        }
        
        return true;
    }

    uint RecordMgr::GetCacheSize()
    {
        LOG_FUNC();
        
        return m_initParams.videoCacheSize;
    }

    bool RecordMgr::SetCacheSize(uint videoCacheSize)
    {
        LOG_FUNC();
        
        if (m_pContext)
        {
            LOG_ERROR("Unable to set cache size for RecordMgr '" << m_name 
                << "' as it is currently in use");
            return false;
        }

        m_initParams.videoCacheSize = videoCacheSize;
        
        return true;
    }

    void RecordMgr::GetDimensions(uint* width, uint* height)
    {
        LOG_FUNC();
        
        *width = m_initParams.width;
        *height = m_initParams.height;
    }

    bool RecordMgr::SetDimensions(uint width, uint height)
    {
        LOG_FUNC();
        
        if (m_pContext)
        {
            LOG_ERROR("Unable to set Dimensions for RecordMgr '" << m_name 
                << "' as it is currently in use");
            return false;
        }

        m_initParams.width = width;
        m_initParams.height = height;
        
        return true;
    }
    
    bool RecordMgr::StartSession(uint start, uint duration, void* clientData)
    {
        LOG_FUNC();
        
        if (!m_pContext)
        {
            LOG_ERROR("Unable to Start Session for RecordMgr '" << m_name 
                << "' context has not been created");
            return false;
        }
        
        LOG_INFO("Starting record session for RecordMgr '" << m_name 
            << "' with start = " << start << " and durarion = " << duration);
        
        // Save the client data to return     
        m_clientData = clientData;
        
        if (NvDsSRStart(m_pContext, &m_currentSessionId, start, duration, this) 
            != NVDSSR_STATUS_OK)
        {
            LOG_ERROR("Failed to Start Session for RecordMgr '" << m_name << "'");
            return false;
        }
        dsl_recording_info dslInfo{0};
        
        dslInfo.session_id = m_currentSessionId;
        dslInfo.recording_event = DSL_RECORDING_EVENT_START;
        try
        {
            m_clientListener(&dslInfo, m_clientData);
        }
        catch(...)
        {
            LOG_ERROR("Client Listener for RecordMgr '" << m_name << "' threw an exception");
            return false;
        }
        return true;
    }
    
    bool RecordMgr::StopSession()
    {
        LOG_FUNC();
        
        if (!m_pContext)
        {
            LOG_ERROR("Unable to Stop Session for RecordMgr '" << m_name 
                << "' context has not been created");
            return false;
        }
        if (m_currentSessionId == UINT32_MAX)
        {
            LOG_ERROR("Unable to Stop Session for RecordMgr '" << m_name 
                << "' no session has been started");
            return false;
        }
        LOG_INFO("Stoping record session for RecordMgre '" << m_name << "'");
        bool retval = (NvDsSRStop(m_pContext, m_currentSessionId) == NVDSSR_STATUS_OK);
        
        m_currentSessionId = UINT32_MAX;
        return retval;
    }
    
    bool RecordMgr::GotKeyFrame()
    {
        LOG_FUNC();
        
        if (!m_pContext)
        {
            LOG_WARN("There is no Record Bin context to query as '" << m_name 
                << "' context has not been created");
            return false;
        }
        return m_pContext->gotKeyFrame;
    }
    
    bool RecordMgr::IsOn()
    {
        LOG_FUNC();
        
        if (!m_pContext)
        {
            LOG_WARN("There is no Record Bin context to query as '" << m_name 
                << "' context has not been created");
            return false;
        }
        return m_pContext->recordOn;
    }
    
    bool RecordMgr::ResetDone()
    {
        LOG_FUNC();
        
        if (!m_pContext)
        {
            LOG_WARN("There is no Record Bin context to query as '" << m_name 
                << "' context has not been created");
            return false;
        }
        return m_pContext->resetDone;
    }
    
    bool RecordMgr::AddVideoPlayer(DSL_BINTR_PTR pPlayer)
    {
        LOG_FUNC();
        
        if (m_videoPlayers.find(pPlayer->GetName()) != m_videoPlayers.end())
        {   
            LOG_ERROR("Video Player is not unique");
            return false;
        }
        m_videoPlayers[pPlayer->GetName()] = pPlayer;
        
        return true;
    }
    
    bool RecordMgr::RemoveVideoPlayer(DSL_BINTR_PTR pPlayer)
    {
        LOG_FUNC();
        
        if (m_videoPlayers.find(pPlayer->GetCStrName()) == m_videoPlayers.end())
        {   
            LOG_ERROR("Video Player not found");
            return false;
        }
        m_videoPlayers.erase(pPlayer->GetName());
        
        return true;
    }

    bool RecordMgr::AddMailer(DSL_MAILER_PTR pMailer,
        const char* subject)
    {
        LOG_FUNC();
        
        if (m_mailers.find(pMailer->GetName()) != m_mailers.end())
        {   
            LOG_ERROR("Record Manager - Mailer is not unique");
            return false;
        }
        // combine all input parameters as MailerSpecs and add
        std::shared_ptr<MailerSpecs> pMailerSpecs = 
            std::shared_ptr<MailerSpecs>(new MailerSpecs(pMailer, subject, false));
            
        m_mailers[pMailer->GetName()] = pMailerSpecs;
        
        return true;
    }
    
    bool RecordMgr::RemoveMailer(DSL_MAILER_PTR pMailer)
    {
        LOG_FUNC();
        
        if (m_mailers.find(pMailer->GetCStrName()) == m_mailers.end())
        {   
            LOG_ERROR("Record Manager- Mailer not found");
            return false;
        }
        m_mailers.erase(pMailer->GetName());
        
        return true;
    }
    
    void* RecordMgr::HandleRecordComplete(NvDsSRRecordingInfo* pNvDsInfo)
    {
        LOG_FUNC();

        // String and WString viersion of the filename and dirpath
        std::string cstrFilename(pNvDsInfo->filename);
        std::wstring wstrFilename(cstrFilename.begin(), cstrFilename.end());
        std::string cstrDirpath(pNvDsInfo->dirpath);
        std::wstring wstrDirpath(cstrDirpath.begin(), cstrDirpath.end());

        std::string filespec = cstrDirpath + "/" +  cstrFilename;

        // If there are Video Players for playing the completed recording
        for (auto const& iter: m_videoPlayers)
        {
            if (iter.second->IsType(typeid(VideoRenderPlayerBintr)))
            {
                DSL_PLAYER_RENDER_VIDEO_BINTR_PTR pVideoPlayer = 
                    std::dynamic_pointer_cast<VideoRenderPlayerBintr>(iter.second);

                GstState state;
                pVideoPlayer->GetState(state, 0);

                // Queue the filepath if the Player is currently Playing/Paused
                // otherwise, set the filepath and Play the Player
                if (state == GST_STATE_PLAYING or state == GST_STATE_PAUSED)
                {
                    pVideoPlayer->QueueFilePath(filespec.c_str());
                }
                else
                {
                    pVideoPlayer->SetFilePath(filespec.c_str());
                    pVideoPlayer->Play();
                    
                }
            }
            // TODO handle ImageRtspPlayerBintr
        }

        // If there are Mailers for mailing the recording details
        if (m_mailers.size())
        {
            std::vector<std::string> body;

            body.push_back(std::string("File Name  : " 
                + std::string(pNvDsInfo->filename) + "<br>"));
            body.push_back(std::string("Location   : " 
                + std::string(pNvDsInfo->dirpath) + "<br>"));
            body.push_back(std::string("Session Id : " 
                + std::to_string(pNvDsInfo->sessionId) + "<br>"));
            body.push_back(std::string("Duration   : " 
                + std::to_string(pNvDsInfo->duration) + "<br>"));
            body.push_back(std::string("Width      : " 
                + std::to_string(pNvDsInfo->width) + "<br>"));
            body.push_back(std::string("Height     : " 
                + std::to_string(pNvDsInfo->height) + "<br>"));
                
            for (auto const& iter: m_mailers)
            {
                iter.second->m_pMailer->QueueMessage(iter.second->m_subject, body);
            }
        }
        
        // new DSL info structure with unicode strings for python3 compatibility
        dsl_recording_info dslInfo{0};

        dslInfo.filename = wstrFilename.c_str();
        dslInfo.dirpath = wstrDirpath.c_str();

        switch (pNvDsInfo->containerType)
        {
        case NVDSSR_CONTAINER_MP4 :
            dslInfo.container_type = DSL_CONTAINER_MP4;
            break;
        case NVDSSR_CONTAINER_MKV :
            dslInfo.container_type = DSL_CONTAINER_MKV;        
            break;
        default:
            LOG_ERROR("Invalid container = '" << pNvDsInfo->containerType << "' received from NvDsSR for RecordMgr'" << m_name << "'");
        }
        
        // copy the remaining data received from the nvidia lib
        dslInfo.session_id = pNvDsInfo->sessionId;
        dslInfo.recording_event = DSL_RECORDING_EVENT_END;
        dslInfo.duration = pNvDsInfo->duration;
        dslInfo.width = pNvDsInfo->width;
        dslInfo.height = pNvDsInfo->height;
        
        try
        {
            return m_clientListener(&dslInfo, m_clientData);
        }
        catch(...)
        {
            LOG_ERROR("Client Listener for RecordMgr '" << m_name << "' threw an exception");
            return NULL;
        }
        
    }

    //******************************************************************************************

    static void* RecordCompleteCallback(NvDsSRRecordingInfo* pNvDsInfo, void* pRecordMgr)
    {
        return static_cast<RecordMgr*>(pRecordMgr)->
            HandleRecordComplete(pNvDsInfo);        
    }
    
}