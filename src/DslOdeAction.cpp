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

#include "DslServices.h"
#include "DslOdeTrigger.h"
#include "DslOdeAction.h"
#include "DslDisplayTypes.h"

#define DATE_BUFF_LENGTH 40

namespace DSL
{
    OdeAction::OdeAction(const char* name)
        : Base(name)
        , m_enabled(true)
    {
        LOG_FUNC();

        g_mutex_init(&m_propertyMutex);
    }

    OdeAction::~OdeAction()
    {
        LOG_FUNC();

        g_mutex_clear(&m_propertyMutex);
    }

    bool OdeAction::GetEnabled()
    {
        LOG_FUNC();
        
        return m_enabled;
    }
    
    void OdeAction::SetEnabled(bool enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_enabled = enabled;
    }
    
    std::string OdeAction::Ntp2Str(uint64_t ntp)
    {
        time_t secs = round(ntp/1000000000);
        time_t usecs = ntp%1000000000;  // gives us fraction of seconds
        usecs *= 1000000; // multiply by 1e6
        usecs >>= 32; // and divide by 2^32
        
        struct tm currentTm;
        localtime_r(&secs, &currentTm);        
        
        char dateTime[64] = {0};
        char dateTimeUsec[64];
        strftime(dateTime, sizeof(dateTime), "%Y-%m-%d %H:%M:%S", &currentTm);
        snprintf(dateTimeUsec, sizeof(dateTimeUsec), "%s.%06ld", dateTime, usecs);

        return std::string(dateTimeUsec);
    }

    // ********************************************************************

    FormatBBoxOdeAction::FormatBBoxOdeAction(const char* name, uint borderWidth,
        DSL_RGBA_COLOR_PTR pColor, bool hasBgColor, DSL_RGBA_COLOR_PTR pBgColor)
        : OdeAction(name)
        , m_borderWidth(borderWidth)
        , m_pBorderColor(pColor)
        , m_hasBgColor(hasBgColor)
        , m_pBgColor(pBgColor)
    {
        LOG_FUNC();
    }

    FormatBBoxOdeAction::~FormatBBoxOdeAction()
    {
        LOG_FUNC();

    }

    void FormatBBoxOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
        GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled and pObjectMeta)
        {   
            pObjectMeta->rect_params.border_width = m_borderWidth;
            pObjectMeta->rect_params.border_color = *m_pBorderColor;
            
            if (m_hasBgColor)
            {
                pObjectMeta->rect_params.has_bg_color = true;
                pObjectMeta->rect_params.bg_color = *m_pBgColor;
            }
        }
    }
    
    // ********************************************************************

    CustomOdeAction::CustomOdeAction(const char* name, 
        dsl_ode_handle_occurrence_cb clientHandler, void* clientData)
        : OdeAction(name)
        , m_clientHandler(clientHandler)
        , m_clientData(clientData)
    {
        LOG_FUNC();
    }

    CustomOdeAction::~CustomOdeAction()
    {
        LOG_FUNC();
    }
    
    void CustomOdeAction::HandleOccurrence(DSL_BASE_PTR pBase, 
        GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (!m_enabled)
        {
            return;
        }
        try
        {
            DSL_ODE_TRIGGER_PTR pTrigger = std::dynamic_pointer_cast<OdeTrigger>(pBase);
            m_clientHandler(pTrigger->s_eventCount, pTrigger->m_wName.c_str(), pBuffer,
                pFrameMeta, pObjectMeta, m_clientData);
        }
        catch(...)
        {
            LOG_ERROR("Custom ODE Action '" << GetName() 
                << "' threw exception calling client callback");
        }
    }

    // ********************************************************************

    // Initialize static Event Counter
    uint64_t CaptureOdeAction::s_captureId = 0;

    CaptureOdeAction::CaptureOdeAction(const char* name, 
        uint captureType, const char* outdir, bool annotate)
        : OdeAction(name)
        , m_captureType(captureType)
        , m_outdir(outdir)
        , m_annotate(annotate)
        , m_captureCompleteTimerId(0)
    {
        LOG_FUNC();

        g_mutex_init(&m_captureCompleteMutex);
    }

    CaptureOdeAction::~CaptureOdeAction()
    {
        LOG_FUNC();

        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_captureCompleteMutex);
            if (m_captureCompleteTimerId)
            {
                g_source_remove(m_captureCompleteTimerId);
            }
            RemoveAllChildren();
        }
        g_mutex_clear(&m_captureCompleteMutex);
    }

    bool CaptureOdeAction::AddCaptureCompleteListener(
        dsl_capture_complete_listener_cb listener, void* userdata)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_captureCompleteMutex);
        
        if (m_captureCompleteListeners.find(listener) != m_captureCompleteListeners.end())
        {   
            LOG_ERROR("ODE Capture Action '" << GetName() 
                << "' - Complete listener is not unique");
            return false;
        }
        m_captureCompleteListeners[listener] = userdata;
        
        return true;
    }

    bool CaptureOdeAction::RemoveCaptureCompleteListener(
        dsl_capture_complete_listener_cb listener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_captureCompleteMutex);
        
        if (m_captureCompleteListeners.find(listener) == m_captureCompleteListeners.end())
        {   
            LOG_ERROR("ODE Capture Action '" << GetName() 
                << "'  - Complete listener not found");
            return false;
        }
        m_captureCompleteListeners.erase(listener);
        
        return true;
    }
    
    bool CaptureOdeAction::AddImagePlayer(DSL_PLAYER_BINTR_PTR pPlayer)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_captureCompleteMutex);
        
        if (m_imagePlayers.find(pPlayer->GetName()) != m_imagePlayers.end())
        {   
            LOG_ERROR("ODE Capture Action '" << GetName() 
                << "'  - Image Player is not unique");
            return false;
        }
        m_imagePlayers[pPlayer->GetName()] = pPlayer;
        
        return true;
    }
    
    bool CaptureOdeAction::RemoveImagePlayer(DSL_PLAYER_BINTR_PTR pPlayer)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_captureCompleteMutex);
        
        if (m_imagePlayers.find(pPlayer->GetCStrName()) == m_imagePlayers.end())
        {   
            LOG_ERROR("ODE Capture Action '" << GetName() 
                << "' - Image Player not found");
            return false;
        }
        m_imagePlayers.erase(pPlayer->GetName());
        
        return true;
    }
    
    bool CaptureOdeAction::AddMailer(DSL_MAILER_PTR pMailer,
        const char* subject, bool attach)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_captureCompleteMutex);
        
        if (m_mailers.find(pMailer->GetName()) != m_mailers.end())
        {   
            LOG_ERROR("ODE Capture Action '" << GetName() 
                << "'  - Mailer is not unique");
            return false;
        }
        // combine all input parameters as MailerSpecs and add
        std::shared_ptr<MailerSpecs> pMailerSpecs = 
            std::shared_ptr<MailerSpecs>(new MailerSpecs(pMailer, subject, attach));
            
        m_mailers[pMailer->GetName()] = pMailerSpecs;
        
        return true;
    }
    
    bool CaptureOdeAction::RemoveMailer(DSL_MAILER_PTR pMailer)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_captureCompleteMutex);
        
        if (m_mailers.find(pMailer->GetCStrName()) == m_mailers.end())
        {   
            LOG_ERROR("ODE Capture Action '" << GetName() 
                << "' - Mailer not found");
            return false;
        }
        m_mailers.erase(pMailer->GetName());
        
        return true;
    }

    void CaptureOdeAction::RemoveAllChildren()
    {
        LOG_FUNC();
    }
    
    cv::Mat& CaptureOdeAction::AnnotateObject(NvDsObjectMeta* pObjectMeta, 
        cv::Mat& bgr_frame)
    {
        // rectangle params are in floats so convert
        int left((int)pObjectMeta->rect_params.left);
        int top((int)pObjectMeta->rect_params.top);
        int width((int)pObjectMeta->rect_params.width); 
        int height((int)pObjectMeta->rect_params.height);

        // add the bounding-box rectange
        cv::rectangle(bgr_frame,
            cv::Point(left, top),
            cv::Point(left+width, top+height),
            cv::Scalar(0, 0, 255, 0),
            2);
        
        // assemble the label based on the available information
        std::string label(pObjectMeta->obj_label);
        
        if(pObjectMeta->object_id)
        {
            label = label + " " + std::to_string(pObjectMeta->object_id); 
        }
        if(pObjectMeta->confidence > 0)
        {
            label = label + " " + std::to_string(pObjectMeta->confidence); 
        }
        
        // add a black background rectangle for the label as cv::putText does not 
        // support a background color the size of the bacground is just an approximation 
        //based on character count not their actual sizes
        cv::rectangle(bgr_frame,
            cv::Point(left, top-30),
            cv::Point(left+label.size()*10+2, top-2),
            cv::Scalar(0, 0, 0, 0),
            cv::FILLED);

        // add the label to the black background
        cv::putText(bgr_frame, 
            label.c_str(), 
            cv::Point(left+2, top-12),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(255, 255, 255, 0),
            1);
            
        return bgr_frame;
    }

    void CaptureOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
        GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (!m_enabled)
        {
            return;
        }
        // ensure that if we're capturing an Object, object data must be provided
        // i.e Object capture and Frame event action result in a NOP
        if ((m_captureType == DSL_CAPTURE_TYPE_OBJECT) and (!pObjectMeta))
        {
            return;
        }
        
        // surface index is derived from the batch_id for the frame that triggered the event
        int surfaceIndex = pFrameMeta->batch_id;
        
        // Coordinates and dimensions for our destination surface for RGBA to 
        // BGR conversion required for JPEG
        uint32_t left(0), top(0), width(0), height(0);

        // MapInfo for the current buffer
        DslMapInfo mapInfo(pBuffer);
        
        // Transforming only one frame in the batch, so create a copy of the single 
        // surface ... becoming our new source surface. This creates a new mono (non-batched) 
        // surface copied from the "batched frames" using the batch id as the index
        DslMonoSurface srcSurface(mapInfo, pFrameMeta->batch_id);

        // capturing full frame or object only?
        if (m_captureType == DSL_CAPTURE_TYPE_FRAME)
        {
            width = srcSurface.width;
            height = srcSurface.height;
        }
        else
        {
            left = pObjectMeta->rect_params.left;
            top = pObjectMeta->rect_params.top;
            width = pObjectMeta->rect_params.width; 
            height = pObjectMeta->rect_params.height;
        }
        
        // New "create params" for our destination surface. we only need one surface so set 
        // memory allocation (for the array of surfaces) size to 0
        DslSurfaceCreateParams surfaceCreateParams(srcSurface.gpuId, width, height, 0);
        
        // New Destination surface with a batch size of 1 for transforming the single surface 
        DslBufferSurface dstSurface(1, surfaceCreateParams);

        // New "transform params" for the surface transform, croping or (future?) scaling
        DslTransformParams transformParams(left, top, width, height);
        
        // New "Cuda stream" for the surface transform
        DslCudaStream dslCudaStream(srcSurface.gpuId);
        
        // New "Transform Session" config params using the new Cuda stream
        DslSurfaceTransformSessionParams dslTransformSessionParams(srcSurface.gpuId, 
            dslCudaStream);
        
        // Set the "Transform Params" for the current tranform session
        if (!dslTransformSessionParams.Set())
        {
            LOG_ERROR("Destination surface failed to set transform session params for Action '" 
                << GetName() << "'");
            return;
        }
        
        // We can now transform our Mono Source surface to the first (and only) 
        // surface in the batched buffer.
        if (!dstSurface.TransformMonoSurface(srcSurface, 0, transformParams))
        {
            LOG_ERROR("Destination surface failed to transform for Action '" 
                << GetName() << "'");
            return;
        }
        
        // Map the tranformed surface for read
        if (!dstSurface.Map())
        {
            LOG_ERROR("Destination surface failed to map for Action '" << GetName() << "'");
            return;
        }
        
        // Sync the surface for CPU access
        if (!dstSurface.SyncForCpu())
        {
            LOG_ERROR("Destination surface failed to Sync for '" << GetName() << "'");
            return;
        }

        // New background Mat for our image
        cv::Mat* pbgrFrame = new cv::Mat(cv::Size(width, height), CV_8UC3);

        // new forground Mat using the first (and only) bufer in the batch
        cv::Mat in_mat = cv::Mat(height, width, CV_8UC4, 
            (&dstSurface)->surfaceList[0].mappedAddr.addr[0],
            (&dstSurface)->surfaceList[0].pitch);

        // Convert the RGBA buffer to BGR
        cv::cvtColor(in_mat, *pbgrFrame, CV_RGBA2BGR);
        
        // if this is a frame capture and the client wants the image annotated.
        if (m_captureType == DSL_CAPTURE_TYPE_FRAME and m_annotate)
        {
            // if object meta is available, then occurrence was triggered 
            // on an object occurrence, so we only annotate the single object
            if (pObjectMeta)
            {
                *pbgrFrame = AnnotateObject(pObjectMeta, *pbgrFrame);
            }
            
            // otherwise, we iterate throught the object-list highlighting each object.
            else
            {
                for (NvDsMetaList* pMeta = pFrameMeta->obj_meta_list; 
                    pMeta != NULL; pMeta = pMeta->next)
                {
                    // not to be confussed with pObjectMeta
                    NvDsObjectMeta* _pObjectMeta_ = (NvDsObjectMeta*) (pMeta->data);
                    if (_pObjectMeta_ != NULL)
                    {
                        *pbgrFrame = AnnotateObject(_pObjectMeta_, *pbgrFrame);
                    }
                }
            }
        }
        // convert to shared pointer and queue for asyn file save and client notificaton
        std::shared_ptr<cv::Mat> pImageMat = std::shared_ptr<cv::Mat>(pbgrFrame);
        QueueCapturedImage(pImageMat);
    }

    void CaptureOdeAction::QueueCapturedImage(std::shared_ptr<cv::Mat> pImageMat)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_captureCompleteMutex);
        
        m_imageMats.push(pImageMat);
        
        // start the asynchronous notification timer if not currently running
        if (!m_captureCompleteTimerId)
        {
            m_captureCompleteTimerId = g_timeout_add(1, CompleteCaptureHandler, this);
        }
    }

    int CaptureOdeAction::CompleteCapture()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_captureCompleteMutex);
        
        
        while (m_imageMats.size())
        {
            std::shared_ptr<cv::Mat> pImageMat = m_imageMats.front();
            m_imageMats.pop();
            
            char dateTime[64] = {0};
            time_t seconds = time(NULL);
            struct tm currentTm;
            localtime_r(&seconds, &currentTm);

            std::strftime(dateTime, sizeof(dateTime), "%Y%m%d-%H%M%S", &currentTm);
            std::string dateTimeStr(dateTime);

            std::ostringstream fileNameStream;
            fileNameStream << GetName() << "_" 
                << std::setw(5) << std::setfill('0') << s_captureId
                << "_" << dateTimeStr << ".jpeg";
                
            std::string filespec = m_outdir + "/" + 
                fileNameStream.str();

            cv::imwrite(filespec.c_str(), *pImageMat);

            // If there are Image Players for playing the captured image
            for (auto const& iter: m_imagePlayers)
            {
                if (iter.second->IsType(typeid(ImageRenderPlayerBintr)))
                {
                    DSL_PLAYER_RENDER_IMAGE_BINTR_PTR pImagePlayer = 
                        std::dynamic_pointer_cast<ImageRenderPlayerBintr>(iter.second);

                    GstState state;
                    pImagePlayer->GetState(state, 0);

                    // Queue the filepath if the Player is currently Playing/Paused
                    // otherwise, set the filepath and Play the Player
                    if (state == GST_STATE_PLAYING or state == GST_STATE_PAUSED)
                    {
                        pImagePlayer->QueueFilePath(filespec.c_str());
                        
                        // If the Player's EOS timeout is not running, i.e the previous
                        // image is being shown indefinetely - show next (i.e. this image)
                        if (!pImagePlayer->GetTimeout())
                        {
                            pImagePlayer->Next();
                        }
                    }
                    else
                    {
                        pImagePlayer->SetFilePath(filespec.c_str());
                        pImagePlayer->Play();
                        
                    }
                }
                // TODO handle ImageRtspPlayerBintr
            }
            
            // If there are complete listeners to notify
            if (m_captureCompleteListeners.size())
            {
                // assemble the capture info
                dsl_capture_info info{0};

                info.captureId = s_captureId;
                
                std::string fileName = fileNameStream.str();
                
                // convert the filename and dirpath to wchar string types (client format)
                std::wstring wstrFilename(fileName.begin(), fileName.end());
                std::wstring wstrDirpath(m_outdir.begin(), m_outdir.end());
               
                info.dirpath = wstrDirpath.c_str();
                info.filename = wstrFilename.c_str();
                
                // get the dimensions from the image Mat
                cv::Size imageSize = pImageMat->size();
                info.width = imageSize.width;
                info.height = imageSize.height;
                    
                // iterate through the map of listeners calling each
                for(auto const& imap: m_captureCompleteListeners)
                {
                    try
                    {
                        imap.first(&info, imap.second);
                    }
                    catch(...)
                    {
                        LOG_ERROR("ODE Capture Action '" << GetName() 
                            << "' threw exception calling Client Capture Complete Listener");
                    }
                }
            }

            // If there are Mailers for mailing the capture detals and optional image
            if (m_mailers.size())
            {
                std::vector<std::string> body;
                
                body.push_back(std::string("Action     : " 
                    + GetName() + "<br>"));
                body.push_back(std::string("File Name  : " 
                    + fileNameStream.str() + "<br>"));
                body.push_back(std::string("Location   : " 
                    + m_outdir + "<br>"));
                body.push_back(std::string("Capture Id : " 
                    + std::to_string(s_captureId) + "<br>"));

                // get the dimensions from the image Mat
                cv::Size imageSize = pImageMat->size();

                body.push_back(std::string("Width      : " 
                    + std::to_string(imageSize.width) + "<br>"));
                body.push_back(std::string("Height     : " 
                    + std::to_string(imageSize.height) + "<br>"));
                    
                for (auto const& iter: m_mailers)
                {
                    std::string filepath;
                    if (iter.second->m_attach)
                    {
                        filepath.assign(filespec.c_str());
                    }
                    iter.second->m_pMailer->QueueMessage(iter.second->m_subject, 
                        body, filepath);
                }
            }
            // Increment the global capture count
            s_captureId++;
        }

        // clear the timer id and return false to self remove
        m_captureCompleteTimerId = 0;
        return false;
    }

    static int CompleteCaptureHandler(gpointer pAction)
    {
        return static_cast<CaptureOdeAction*>(pAction)->
            CompleteCapture();
    }

    // ********************************************************************

    DisableHandlerOdeAction::DisableHandlerOdeAction(const char* name, const char* handler)
        : OdeAction(name)
        , m_handler(handler)
    {
        LOG_FUNC();
    }

    DisableHandlerOdeAction::~DisableHandlerOdeAction()
    {
        LOG_FUNC();
    }
    
    void DisableHandlerOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
        GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->PphEnabledSet(m_handler.c_str(), false);
        }
    }

    // ********************************************************************

    CustomizeLabelOdeAction::CustomizeLabelOdeAction(const char* name, 
        const std::vector<uint>& contentTypes, uint mode)
        : OdeAction(name)
        , m_contentTypes(contentTypes)
        , m_mode(mode)
    {
        LOG_FUNC();
    }

    CustomizeLabelOdeAction::~CustomizeLabelOdeAction()
    {
        LOG_FUNC();
    }

    void CustomizeLabelOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
        GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled and pObjectMeta)
        {   
            std::string label;
            
            if (m_mode == DSL_WRITE_MODE_APPEND)
            {
                label.append(pObjectMeta->text_params.display_text);
            }
            // Free up the existing label memory, and reallocate to ensure suffcient size
            g_free(pObjectMeta->text_params.display_text);
            pObjectMeta->text_params.display_text = (gchar*) g_malloc0(MAX_DISPLAY_LEN);

            for (auto const &iter: m_contentTypes)
            {
                switch(iter)
                {
                case DSL_OBJECT_LABEL_CLASS :
                    label.append((label.size()) ? " | " : "");
                    label.append(pObjectMeta->obj_label);
                    break;
                case DSL_OBJECT_LABEL_TRACKING_ID:
                    label.append((label.size()) ? " | " : "");
                    label.append(std::to_string(pObjectMeta->object_id));
                    break;
                case DSL_OBJECT_LABEL_LOCATION :
                    label.append((label.size()) ? " | L:" : "L:");
                    label.append(std::to_string(lrint(pObjectMeta->rect_params.left)));
                    label.append(",");
                    label.append(std::to_string(lrint(pObjectMeta->rect_params.top)));
                    break;
                case DSL_OBJECT_LABEL_DIMENSIONS :
                    label.append(((label.size()) ? " | D:" : "D:"));
                    label.append(std::to_string(lrint(pObjectMeta->rect_params.width)));
                    label.append("x");
                    label.append(std::to_string(lrint(pObjectMeta->rect_params.height)));
                    break;
                case DSL_OBJECT_LABEL_CONFIDENCE :
                    label.append(((label.size()) ? " | C:" : "C:"));
                    label.append(std::to_string(pObjectMeta->confidence));
                    break;
                case DSL_OBJECT_LABEL_PERSISTENCE :
                    label.append(((label.size()) ? " | T:" : "T:"));
                    label.append(std::to_string(pObjectMeta->
                        misc_obj_info[DSL_OBJECT_INFO_PERSISTENCE]));
                    label.append("s");
                    break;
                default :
                    LOG_ERROR("Invalid 'object content type' for customize label action '" <<
                        GetName() << "'");
                }
            }
            label.copy(pObjectMeta->text_params.display_text, label.size(), 0);
        }
    }


    // ********************************************************************

    DisplayOdeAction::DisplayOdeAction(const char* name, 
        uint offsetX, uint offsetY, bool offsetYWithClassId, 
        DSL_RGBA_FONT_PTR pFont, bool hasBgColor, DSL_RGBA_COLOR_PTR pBgColor)
        : OdeAction(name)
        , m_offsetX(offsetX)
        , m_offsetY(offsetY)
        , m_offsetYWithClassId(offsetYWithClassId)
        , m_pFont(pFont)
        , m_hasBgColor(hasBgColor)
        , m_pBgColor(pBgColor)
    {
        LOG_FUNC();
    }

    DisplayOdeAction::~DisplayOdeAction()
    {
        LOG_FUNC();
    }

    void DisplayOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
        GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled)
        {
            DSL_ODE_TRIGGER_PTR pTrigger = std::dynamic_pointer_cast<OdeTrigger>(pOdeTrigger);
            
            NvOSD_TextParams *pTextParams = &pDisplayMeta->text_params[pDisplayMeta->num_labels++];
            pTextParams->display_text = (gchar*) g_malloc0(MAX_DISPLAY_LEN);
            
            if (pObjectMeta)
            {
                std::string text = pTrigger->GetName() + " = " 
                    + std::to_string(pObjectMeta->misc_obj_info[DSL_OBJECT_INFO_PRIMARY_METRIC]);
                text.copy(pTextParams->display_text, MAX_DISPLAY_LEN, 0);
            }
            else
            {
                std::string text = pTrigger->GetName() + " = " 
                    + std::to_string(pTrigger->m_occurrences);
                text.copy(pTextParams->display_text, MAX_DISPLAY_LEN, 0);
            }

            // Setup X and Y display offsets
            pTextParams->x_offset = m_offsetX;
            pTextParams->y_offset = m_offsetY;
            
            // Typically set if action is shared by multiple ODE Triggers/ClassId's 
            if (m_offsetYWithClassId)
            {
                pTextParams->y_offset += pTrigger->m_classId * 2 * m_pFont->font_size + 2;
            }

            // Font, font-size, font-color
            pTextParams->font_params = *m_pFont;
            pTextParams->font_params.font_name = (gchar*) g_malloc0(MAX_DISPLAY_LEN);
            m_pFont->m_fontName.copy(pTextParams->font_params.font_name, MAX_DISPLAY_LEN, 0);
            

            // Text background color
            pTextParams->set_bg_clr = m_hasBgColor;
            pTextParams->text_bg_clr = *m_pBgColor;
            
            nvds_add_display_meta_to_frame(pFrameMeta, pDisplayMeta);
        }
    }
    
    // ********************************************************************

    EmailOdeAction::EmailOdeAction(const char* name, 
        DSL_BASE_PTR pMailer, const char* subject)
        : OdeAction(name)
        , m_pMailer(pMailer)
        , m_subject(subject)
    {
        LOG_FUNC();
    }

    EmailOdeAction::~EmailOdeAction()
    {
        LOG_FUNC();
    }

    void EmailOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
        GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled)
        {
            DSL_ODE_TRIGGER_PTR pTrigger = std::dynamic_pointer_cast<OdeTrigger>(pOdeTrigger);
            
            std::vector<std::string> body;
            
            body.push_back(std::string("Trigger Name    : " 
                + pTrigger->GetName() + "<br>"));
            body.push_back(std::string("  Unique ODE Id : " 
                + std::to_string(pTrigger->s_eventCount) + "<br>"));
            body.push_back(std::string("  NTP Timestamp : " 
                +  Ntp2Str(pFrameMeta->ntp_timestamp) + "<br>"));
            body.push_back(std::string("  Source Data   : ------------------------<br>"));
            if (pFrameMeta->bInferDone)
            {
                body.push_back(std::string("    Inference   : Yes<br>"));
            }
            else
            {
                body.push_back(std::string("    Inference   : No<br>"));
            }
            body.push_back(std::string("    SourceId    : " 
                +  std::to_string(pFrameMeta->source_id) + "<br>"));
            body.push_back(std::string("    BatchId     : " 
                +  std::to_string(pFrameMeta->batch_id) + "<br>"));
            body.push_back(std::string("    PadIndex    : " 
                +  std::to_string(pFrameMeta->pad_index) + "<br>"));
            body.push_back(std::string("    Frame       : " 
                +  std::to_string(pFrameMeta->frame_num) + "<br>"));
            body.push_back(std::string("    Width       : " 
                +  std::to_string(pFrameMeta->source_frame_width) + "<br>"));
            body.push_back(std::string("    Heigh       : " 
                +  std::to_string(pFrameMeta->source_frame_height) + "<br>"));
            body.push_back(std::string("  Object Data   : ------------------------<br>"));
            body.push_back(std::string("    Occurrences : " 
                +  std::to_string(pTrigger->m_occurrences) + "<br>"));

            if (pObjectMeta)
            {
                body.push_back(std::string("    Obj ClassId : " 
                    +  std::to_string(pObjectMeta->class_id) + "<br>"));
                body.push_back(std::string("    Tracking Id : " 
                    +  std::to_string(pObjectMeta->object_id) + "<br>"));
                body.push_back(std::string("    Label       : " 
                    +  std::string(pObjectMeta->obj_label) + "<br>"));
                body.push_back(std::string("    Persistence       : " 
                    + std::to_string(pObjectMeta->
                        misc_obj_info[DSL_OBJECT_INFO_PERSISTENCE]) + "<br>"));
                body.push_back(std::string("    Confidence  : " 
                    +  std::to_string(pObjectMeta->confidence) + "<br>"));
                body.push_back(std::string("    Left        : " 
                    +  std::to_string(lrint(pObjectMeta->rect_params.left)) + "<br>"));
                body.push_back(std::string("    Top         : " 
                    +  std::to_string(lrint(pObjectMeta->rect_params.top)) + "<br>"));
                body.push_back(std::string("    Width       : " 
                    +  std::to_string(lrint(pObjectMeta->rect_params.width)) + "<br>"));
                body.push_back(std::string("    Height      : " 
                    +  std::to_string(lrint(pObjectMeta->rect_params.height)) + "<br>"));
            }

            body.push_back(std::string("  Criteria      : ------------------------<br>"));
            body.push_back(std::string("    Class Id    : " 
                +  std::to_string(pTrigger->m_classId) + "<br>"));
            body.push_back(std::string("    Frame Count : " 
                +  std::to_string(pTrigger->m_minFrameCountN) + " out of " 
                +  std::to_string(pTrigger->m_minFrameCountD) + "<br>"));
            body.push_back(std::string("    Min Width   : " 
                +  std::to_string(lrint(pTrigger->m_minWidth)) + "<br>"));
            body.push_back(std::string("    Min Height  : " 
                +  std::to_string(lrint(pTrigger->m_minHeight)) + "<br>"));
            body.push_back(std::string("    Max Width   : " 
                +  std::to_string(lrint(pTrigger->m_maxWidth)) + "<br>"));
            body.push_back(std::string("    Max Height  : " 
                +  std::to_string(lrint(pTrigger->m_maxHeight)) + "<br>"));
            body.push_back(std::string("    Confidence  : " 
                +  std::to_string(pTrigger->m_minConfidence) + "<br>"));

            if (pTrigger->m_inferDoneOnly)
            {
                body.push_back(std::string("    Inference   : Yes<br>"));
            }
            else
            {
                body.push_back(std::string("    Inference   : No<br>"));
            }
            
            std::dynamic_pointer_cast<Mailer>(m_pMailer)->QueueMessage(m_subject, body);
        }
    }

    // ********************************************************************

    FileOdeAction::FileOdeAction(const char* name,
        const char* filePath, uint mode, uint format, bool forceFlush)
        : OdeAction(name)
        , m_filePath(filePath)
        , m_mode(mode)
        , m_format(format)
        , m_forceFlush(forceFlush)
        , m_flushThreadFunctionId(0)
    {
        LOG_FUNC();

        // determine if new or existing file
        std::ifstream streamUriFile(filePath);
        bool fileExists(streamUriFile.good());
        
        // add the CSV header by default (if format == CSV)
        bool addCsvHeader(true);

        try
        {
            if (m_mode == DSL_WRITE_MODE_APPEND)
            {
                m_ostream.open(m_filePath, std::fstream::out | std::fstream::app);
                
                // don't add the header if we're appending to an existing file
                addCsvHeader = !fileExists;
            }
            else
            {
                m_ostream.open(m_filePath, std::fstream::out | std::fstream::trunc);
            }
        }
        catch(...) 
        {
            LOG_ERROR("New FileOdeAction '" << name << "' failed to open");
            throw;
        }
    
        if (m_format == DSL_EVENT_FILE_FORMAT_TEXT)
        {
            char dateTime[DATE_BUFF_LENGTH] = {0};
            time_t seconds = time(NULL);
            struct tm currentTm;
            localtime_r(&seconds, &currentTm);

            strftime(dateTime, DATE_BUFF_LENGTH, "%a, %d %b %Y %H:%M:%S %z", &currentTm);
            std::string dateTimeStr(dateTime);
            
            m_ostream << "-------------------------------------------------------------------" << "\n";
            m_ostream << " File opened: " << dateTimeStr.c_str() << "\n";
            m_ostream << "-------------------------------------------------------------------" << "\n";
        }
        
        // Else it's CSV format, so add header if new/empty file
        else if (addCsvHeader)
        {
            m_ostream << "Trigger Name,";
            m_ostream << "Event Id,";
            m_ostream << "NTP Timestamp,";
            m_ostream << "Inference Done,";
            m_ostream << "Source Id,";
            m_ostream << "Batch Idx,";
            m_ostream << "Pad Idx,";
            m_ostream << "Frame,";
            m_ostream << "Width,";
            m_ostream << "Height,";
            m_ostream << "Occurrences,";
            m_ostream << "Class Id,";
            m_ostream << "Object Id,";
            m_ostream << "Label,";
            m_ostream << "Persistence,";
            m_ostream << "Confidence,";
            m_ostream << "Left,";
            m_ostream << "Top,";
            m_ostream << "Width,";
            m_ostream << "Height,";
            m_ostream << "Class Id Filter,";
            m_ostream << "Min Width,";
            m_ostream << "Min Height,";
            m_ostream << "Max Width,";
            m_ostream << "Max Height,";
            m_ostream << "Min Confidence,";
            m_ostream << "Inference Done Only\n";
        }
    
        g_mutex_init(&m_ostreamMutex);
    }

    FileOdeAction::~FileOdeAction()
    {
        LOG_FUNC();
        
        if (!m_ostream.is_open())
        {
            return;
        }
        
        if (m_flushThreadFunctionId)
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_ostreamMutex);
            g_source_remove(m_flushThreadFunctionId);
        }

        if (m_format == DSL_EVENT_FILE_FORMAT_TEXT)
        {
            char dateTime[DATE_BUFF_LENGTH] = {0};
            time_t seconds = time(NULL);
            struct tm currentTm;
            localtime_r(&seconds, &currentTm);

            strftime(dateTime, DATE_BUFF_LENGTH, "%a, %d %b %Y %H:%M:%S %z", &currentTm);
            std::string dateTimeStr(dateTime);

            m_ostream << "-------------------------------------------------------------------" << "\n";
            m_ostream << " File closed: " << dateTimeStr.c_str() << "\n";
            m_ostream << "-------------------------------------------------------------------" << "\n";
        }
            
        m_ostream.close();
        g_mutex_clear(&m_ostreamMutex);
    }

    void FileOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
        GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        LOCK_2ND_MUTEX_FOR_CURRENT_SCOPE(&m_ostreamMutex);

        if (!m_enabled)
        {
            return;
        }
        DSL_ODE_TRIGGER_PTR pTrigger = std::dynamic_pointer_cast<OdeTrigger>(pOdeTrigger);
        
        if (m_format == DSL_EVENT_FILE_FORMAT_TEXT)
        {
            m_ostream << "Trigger Name    : " << pTrigger->GetName() << "\n";
            m_ostream << "  Unique ODE Id : " << pTrigger->s_eventCount << "\n";
            m_ostream << "  NTP Timestamp : " << Ntp2Str(pFrameMeta->ntp_timestamp) << "\n";
            m_ostream << "  Source Data   : ------------------------" << "\n";
            if (pFrameMeta->bInferDone)
            {
                m_ostream << "    Inference   : Yes\n";
            }
            else
            {
                m_ostream << "    Inference   : No\n";
            }
            m_ostream << "    SourceId    : " << pFrameMeta->source_id << "\n";
            m_ostream << "    BatchId     : " << pFrameMeta->batch_id << "\n";
            m_ostream << "    PadIndex    : " << pFrameMeta->pad_index << "\n";
            m_ostream << "    Frame       : " << pFrameMeta->frame_num << "\n";
            m_ostream << "    Width       : " << pFrameMeta->source_frame_width << "\n";
            m_ostream << "    Heigh       : " << pFrameMeta->source_frame_height << "\n";
            m_ostream << "  Object Data   : ------------------------" << "\n";
            m_ostream << "    Occurrences : " << pTrigger->m_occurrences << "\n";

            if (pObjectMeta)
            {
                m_ostream << "    Obj ClassId : " << pObjectMeta->class_id << "\n";
                m_ostream << "    Tracking Id : " << pObjectMeta->object_id << "\n";
                m_ostream << "    Label       : " << pObjectMeta->obj_label << "\n";
                m_ostream << "    Persistence : " << pObjectMeta->
                    misc_obj_info[DSL_OBJECT_INFO_PERSISTENCE] + "\n";
                m_ostream << "    Confidence  : " << pObjectMeta->confidence << "\n";
                m_ostream << "    Left        : " << lrint(pObjectMeta->rect_params.left) << "\n";
                m_ostream << "    Top         : " << lrint(pObjectMeta->rect_params.top) << "\n";
                m_ostream << "    Width       : " << lrint(pObjectMeta->rect_params.width) << "\n";
                m_ostream << "    Height      : " << lrint(pObjectMeta->rect_params.height) << "\n";
            }

            m_ostream << "  Criteria      : ------------------------" << "\n";
            m_ostream << "    Class Id    : " << pTrigger->m_classId << "\n";
            m_ostream << "    Confidence  : " << pTrigger->m_minConfidence << "\n";
            m_ostream << "    Frame Count : " << pTrigger->m_minFrameCountN
                << " out of " << pTrigger->m_minFrameCountD << "\n";
            m_ostream << "    Min Width   : " << lrint(pTrigger->m_minWidth) << "\n";
            m_ostream << "    Min Height  : " << lrint(pTrigger->m_minHeight) << "\n";
            m_ostream << "    Max Width   : " << lrint(pTrigger->m_maxWidth) << "\n";
            m_ostream << "    Max Height  : " << lrint(pTrigger->m_maxHeight) << "\n";

            if (pTrigger->m_inferDoneOnly)
            {
                m_ostream << "    Inference   : Yes\n\n";
            }
            else
            {
                m_ostream << "    Inference   : No\n\n";
            }
        }
        else
        {
            m_ostream << pTrigger->GetName() << ",";
            m_ostream << pTrigger->s_eventCount << ",";
            m_ostream << pFrameMeta->ntp_timestamp << ",";
            if (pFrameMeta->bInferDone)
            {
                m_ostream << "Yes,";
            }
            else
            {
                m_ostream << "No,";
            }
            m_ostream << pFrameMeta->source_id << ",";
            m_ostream << pFrameMeta->batch_id << ",";
            m_ostream << pFrameMeta->pad_index << ",";
            m_ostream << pFrameMeta->frame_num << ",";
            m_ostream << pFrameMeta->source_frame_width << ",";
            m_ostream << pFrameMeta->source_frame_height << ",";
            m_ostream << pTrigger->m_occurrences << ",";

            if (pObjectMeta)
            {
                m_ostream << pObjectMeta->class_id << ",";
                m_ostream << pObjectMeta->object_id << ",";
                m_ostream << pObjectMeta->obj_label << ",";
                m_ostream << pObjectMeta->confidence << ",";
                m_ostream << pObjectMeta->
                    misc_obj_info[DSL_OBJECT_INFO_PERSISTENCE] + ",";
                m_ostream << lrint(pObjectMeta->rect_params.left) << ",";
                m_ostream << lrint(pObjectMeta->rect_params.top) << ",";
                m_ostream << lrint(pObjectMeta->rect_params.width) << ",";
                m_ostream << lrint(pObjectMeta->rect_params.height) << ",";
            }
            else
            {
                m_ostream << "0,0,0,0,0,0,0,0";
            }

            m_ostream << pTrigger->m_classId << ",";
            m_ostream << lrint(pTrigger->m_minWidth) << ",";
            m_ostream << lrint(pTrigger->m_minHeight) << ",";
            m_ostream << lrint(pTrigger->m_maxWidth) << ",";
            m_ostream << lrint(pTrigger->m_maxHeight) << ",";
            m_ostream << pTrigger->m_minConfidence << ",";

            if (pTrigger->m_inferDoneOnly)
            {
                m_ostream << "Yes\n";
            }
            else
            {
                m_ostream << "No\n";
            }
        }
        
        // If we're force flushing the stream and the flush
        // handler is not currently added to the idle thread
        if (m_forceFlush and !m_flushThreadFunctionId)
        {
            m_flushThreadFunctionId = g_idle_add(FileActionFlush, this);
        }
    }
    
    bool FileOdeAction::Flush()
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_ostreamMutex);
        
        m_ostream.flush();
        
        // end the thread
        m_flushThreadFunctionId = 0;
        return false;
    }

    static gboolean FileActionFlush(gpointer pAction)
    {
        return static_cast<FileOdeAction*>(pAction)->Flush();
    }

    // ********************************************************************

    FillSurroundingsOdeAction::FillSurroundingsOdeAction(const char* name, 
        DSL_RGBA_COLOR_PTR pColor)
        : OdeAction(name)
        , m_pColor(pColor)
    {
        LOG_FUNC();
    }

    FillSurroundingsOdeAction::~FillSurroundingsOdeAction()
    {
        LOG_FUNC();

    }

    void FillSurroundingsOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
        GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (m_enabled and pObjectMeta)
        {
            
            uint x1(roundf(pObjectMeta->rect_params.left));
            uint y1(roundf(pObjectMeta->rect_params.top));
            uint x2(x1+roundf(pObjectMeta->rect_params.width)); 
            uint y2(y1+roundf(pObjectMeta->rect_params.height)); 
            uint rWidth = roundf(pObjectMeta->rect_params.width);
            
            std::string leftRectName("left-rect");
            
            DSL_RGBA_RECTANGLE_PTR pLeftRect = DSL_RGBA_RECTANGLE_NEW(leftRectName.c_str(), 
                0, 0, x1, pFrameMeta->source_frame_height, 0, m_pColor, true, m_pColor);
                
            pLeftRect->AddMeta(pDisplayMeta, pFrameMeta);

            std::string rightRectName("right-rect");
            
            DSL_RGBA_RECTANGLE_PTR pRightRect = DSL_RGBA_RECTANGLE_NEW(rightRectName.c_str(), 
                x2, 0, pFrameMeta->source_frame_width, pFrameMeta->source_frame_height, 
                    0, m_pColor, true, m_pColor);
    
            pRightRect->AddMeta(pDisplayMeta, pFrameMeta);

            std::string topRectName("top-rect");
            
            DSL_RGBA_RECTANGLE_PTR pTopRect = DSL_RGBA_RECTANGLE_NEW(topRectName.c_str(), 
                x1, 0, rWidth, y1, 0, m_pColor, true, m_pColor);
                
            pTopRect->AddMeta(pDisplayMeta, pFrameMeta);

            std::string bottomRectName("bottom-rect");
            
            DSL_RGBA_RECTANGLE_PTR pBottomRect = DSL_RGBA_RECTANGLE_NEW(bottomRectName.c_str(), 
                x1, y2, rWidth, pFrameMeta->source_frame_height, 0, m_pColor, true, m_pColor);
                
            pBottomRect->AddMeta(pDisplayMeta, pFrameMeta);
        }
    }

    // ********************************************************************

    FillFrameOdeAction::FillFrameOdeAction(const char* name, DSL_RGBA_COLOR_PTR pColor)
        : OdeAction(name)
        , m_pColor(pColor)
    {
        LOG_FUNC();
        
    }

    FillFrameOdeAction::~FillFrameOdeAction()
    {
        LOG_FUNC();

    }

    void FillFrameOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
        GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled)
        {
            NvOSD_RectParams rectParams{0};
            rectParams.left = 0;
            rectParams.top = 0;
            rectParams.width = pFrameMeta->source_frame_width;
            rectParams.height = pFrameMeta->source_frame_height;
            rectParams.border_width = 0;
            rectParams.has_bg_color = true;
            rectParams.bg_color = *m_pColor;
            
            pDisplayMeta->rect_params[pDisplayMeta->num_rects++] = rectParams;
        }
    }

    // ********************************************************************

    FillObjectOdeAction::FillObjectOdeAction(const char* name, DSL_RGBA_COLOR_PTR pColor)
        : OdeAction(name)
        , m_pColor(pColor)
    {
        LOG_FUNC();
    }

    FillObjectOdeAction::~FillObjectOdeAction()
    {
        LOG_FUNC();

    }

    void FillObjectOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
        GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled and pObjectMeta)
        {
            pObjectMeta->rect_params.has_bg_color = true;
            pObjectMeta->rect_params.bg_color = *m_pColor;
        }
    }

    // ********************************************************************

    LogOdeAction::LogOdeAction(const char* name)
        : OdeAction(name)
    {
        LOG_FUNC();
    }

    LogOdeAction::~LogOdeAction()
    {
        LOG_FUNC();
    }

    void LogOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
        GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled)
        {
            DSL_ODE_TRIGGER_PTR pTrigger = std::dynamic_pointer_cast<OdeTrigger>(pOdeTrigger);
            
            LOG_INFO("Trigger Name    : " << pTrigger->GetName());
            LOG_INFO("  Unique ODE Id : " << pTrigger->s_eventCount);
            LOG_INFO("  NTP Timestamp : " << Ntp2Str(pFrameMeta->ntp_timestamp));
            LOG_INFO("  Source Data   : ------------------------");
            
            if (pFrameMeta->bInferDone)
            {
                LOG_INFO("    Inference   : Yes");
            }
            else
            {
                LOG_INFO("    Inference   : No");
            }
            LOG_INFO("    SourceId    : " << pFrameMeta->source_id);
            LOG_INFO("    BatchId     : " << pFrameMeta->batch_id);
            LOG_INFO("    PadIndex    : " << pFrameMeta->pad_index);
            LOG_INFO("    Frame       : " << pFrameMeta->frame_num);
            LOG_INFO("    Width       : " << pFrameMeta->source_frame_width);
            LOG_INFO("    Heigh       : " << pFrameMeta->source_frame_height );
            LOG_INFO("  Object Data   : ------------------------");
            LOG_INFO("    Occurrences : " << pTrigger->m_occurrences );
            
            if (pObjectMeta)
            {
                LOG_INFO("    Obj ClassId : " << pObjectMeta->class_id);
                LOG_INFO("    Tracking Id : " << pObjectMeta->object_id);
                LOG_INFO("    Label       : " << pObjectMeta->obj_label);
                LOG_INFO("    Confidence  : " << pObjectMeta->confidence);
                LOG_INFO("    Persistence : " << pObjectMeta->
                    misc_obj_info[DSL_OBJECT_INFO_PERSISTENCE]);
                LOG_INFO("    Left        : " << pObjectMeta->rect_params.left);
                LOG_INFO("    Top         : " << pObjectMeta->rect_params.top);
                LOG_INFO("    Width       : " << pObjectMeta->rect_params.width);
                LOG_INFO("    Height      : " << pObjectMeta->rect_params.height);
            }
            LOG_INFO("  Criteria      : ------------------------");
            LOG_INFO("    Class Id    : " << pTrigger->m_classId );
            LOG_INFO("    Confidence  : " << pTrigger->m_minConfidence);
            LOG_INFO("    Frame Count : " << pTrigger->m_minFrameCountN
                << " out of " << pTrigger->m_minFrameCountD);
            LOG_INFO("    Min Width   : " << pTrigger->m_minWidth);
            LOG_INFO("    Min Height  : " << pTrigger->m_minHeight);
            LOG_INFO("    Max Width   : " << pTrigger->m_maxWidth);
            LOG_INFO("    Max Height  : " << pTrigger->m_maxHeight);
            
            if (pTrigger->m_inferDoneOnly)
            {
                LOG_INFO("    Inference   : Yes");
            }
            else
            {
                LOG_INFO("    Inference   : No");
            }
        }
    }

    // ********************************************************************

    FormatLabelOdeAction::FormatLabelOdeAction(const char* name, 
        DSL_RGBA_FONT_PTR pFont, bool hasBgColor, DSL_RGBA_COLOR_PTR pBgColor)
        : OdeAction(name)
        , m_pFont(pFont)
        , m_hasBgColor(hasBgColor)
        , m_pBgColor(pBgColor)
    {
        LOG_FUNC();
    }

    FormatLabelOdeAction::~FormatLabelOdeAction()
    {
        LOG_FUNC();

    }

    void FormatLabelOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
        GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled and pObjectMeta)
        {   
            pObjectMeta->text_params.font_params = *m_pFont;
            
            if (m_hasBgColor)
            {
                pObjectMeta->text_params.set_bg_clr = true;
                pObjectMeta->text_params.text_bg_clr = *m_pBgColor;
            }
        }
    }

    // ********************************************************************

    AddDisplayMetaOdeAction::AddDisplayMetaOdeAction(const char* name, 
        DSL_DISPLAY_TYPE_PTR pDisplayType)
        : OdeAction(name)
    {
        LOG_FUNC();

        m_pDisplayTypes.push_back(pDisplayType);
    }

    AddDisplayMetaOdeAction::~AddDisplayMetaOdeAction()
    {
        LOG_FUNC();
    }
    
    void AddDisplayMetaOdeAction::AddDisplayType(DSL_DISPLAY_TYPE_PTR pDisplayType)
    {
        LOG_FUNC();
        
        m_pDisplayTypes.push_back(pDisplayType);
    }

    
    void AddDisplayMetaOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
        GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled)
        {
            for (const auto &ivec: m_pDisplayTypes)
            {
                ivec->AddMeta(pDisplayMeta, pFrameMeta);
            }
        }
    }

    // ********************************************************************

    PauseOdeAction::PauseOdeAction(const char* name, const char* pipeline)
        : OdeAction(name)
        , m_pipeline(pipeline)
    {
        LOG_FUNC();
    }

    PauseOdeAction::~PauseOdeAction()
    {
        LOG_FUNC();
    }
    
    void PauseOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
        GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->PipelinePause(m_pipeline.c_str());
        }
    }

    // ********************************************************************

    PrintOdeAction::PrintOdeAction(const char* name,
        bool forceFlush)
        : OdeAction(name)
        , m_forceFlush(forceFlush)
        , m_flushThreadFunctionId(0)
    {
        LOG_FUNC();

        g_mutex_init(&m_ostreamMutex);
    }

    PrintOdeAction::~PrintOdeAction()
    {
        LOG_FUNC();

        if (m_flushThreadFunctionId)
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_ostreamMutex);
            g_source_remove(m_flushThreadFunctionId);
        }
    }

    void PrintOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
        GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (!m_enabled)
        {
            return;
        }
        DSL_ODE_TRIGGER_PTR pTrigger = std::dynamic_pointer_cast<OdeTrigger>(pOdeTrigger);
        
        std::cout << "Trigger Name    : " << pTrigger->GetName() << "\n";
        std::cout << "  Unique ODE Id : " << pTrigger->s_eventCount << "\n";
        std::cout << "  NTP Timestamp : " << Ntp2Str(pFrameMeta->ntp_timestamp) << "\n";
        std::cout << "  Source Data   : ------------------------" << "\n";
        if (pFrameMeta->bInferDone)
        {
            std::cout << "    Inference   : Yes\n";
        }
        else
        {
            std::cout << "    Inference   : No\n";
        }
        std::cout << "    SourceId    : " << pFrameMeta->source_id << "\n";
        std::cout << "    BatchId     : " << pFrameMeta->batch_id << "\n";
        std::cout << "    PadIndex    : " << pFrameMeta->pad_index << "\n";
        std::cout << "    Frame       : " << pFrameMeta->frame_num << "\n";
        std::cout << "    Width       : " << pFrameMeta->source_frame_width << "\n";
        std::cout << "    Heigh       : " << pFrameMeta->source_frame_height << "\n";
        std::cout << "  Object Data   : ------------------------" << "\n";
        std::cout << "    Occurrences : " << pTrigger->m_occurrences << "\n";

        if (pObjectMeta)
        {
            std::cout << "    Obj ClassId : " << pObjectMeta->class_id << "\n";
            std::cout << "    Tracking Id : " << pObjectMeta->object_id << "\n";
            std::cout << "    Label       : " << pObjectMeta->obj_label << "\n";
            std::cout << "    Confidence  : " << pObjectMeta->confidence << "\n";
            std::cout << "    Persistence : " << pObjectMeta->
                misc_obj_info[DSL_OBJECT_INFO_PERSISTENCE] << "\n";
            std::cout << "    Left        : " << lrint(pObjectMeta->rect_params.left) << "\n";
            std::cout << "    Top         : " << lrint(pObjectMeta->rect_params.top) << "\n";
            std::cout << "    Width       : " << lrint(pObjectMeta->rect_params.width) << "\n";
            std::cout << "    Height      : " << lrint(pObjectMeta->rect_params.height) << "\n";
        }

        std::cout << "  Criteria      : ------------------------" << "\n";
        std::cout << "    Class Id    : " << pTrigger->m_classId << "\n";
        std::cout << "    Frame Count : " << pTrigger->m_minFrameCountN
            << " out of " << pTrigger->m_minFrameCountD << "\n";
        std::cout << "    Min Width   : " << lrint(pTrigger->m_minWidth) << "\n";
        std::cout << "    Min Height  : " << lrint(pTrigger->m_minHeight) << "\n";
        std::cout << "    Max Width   : " << lrint(pTrigger->m_maxWidth) << "\n";
        std::cout << "    Max Height  : " << lrint(pTrigger->m_maxHeight) << "\n";
        std::cout << "    Confidence  : " << pTrigger->m_minConfidence << "\n";

        if (pTrigger->m_inferDoneOnly)
        {
            std::cout << "    Inference   : Yes\n\n";
        }
        else
        {
            std::cout << "    Inference   : No\n\n";
        }

        // If we're force flushing the stream and the flush
        // handler is not currently added to the idle thread
        if (m_forceFlush and !m_flushThreadFunctionId)
        {
            m_flushThreadFunctionId = g_idle_add(PrintActionFlush, this);
        }
        
    }

    bool PrintOdeAction::Flush()
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_ostreamMutex);
        
        std::cout << std::flush;
        
        // end the thread
        m_flushThreadFunctionId = 0;
        return false;
    }

    static gboolean PrintActionFlush(gpointer pAction)
    {
        return static_cast<PrintOdeAction*>(pAction)->Flush();
    }

    // ********************************************************************

    RedactOdeAction::RedactOdeAction(const char* name)
        : OdeAction(name)
    {
        LOG_FUNC();
    }

    RedactOdeAction::~RedactOdeAction()
    {
        LOG_FUNC();

    }

    void RedactOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled and pObjectMeta)
        {
            // hide the OSD display text
            if (pObjectMeta->text_params.display_text)
            {
                pObjectMeta->text_params.set_bg_clr = 0;
                pObjectMeta->text_params.font_params.font_size = 0;
            }
            // shade in the background
            pObjectMeta->rect_params.border_width = 0;
            pObjectMeta->rect_params.has_bg_color = 1;
            pObjectMeta->rect_params.bg_color.red = 0.0;
            pObjectMeta->rect_params.bg_color.green = 0.0;
            pObjectMeta->rect_params.bg_color.blue = 0.0;
            pObjectMeta->rect_params.bg_color.alpha = 1.0;
        }
    }

    // ********************************************************************

    AddSinkOdeAction::AddSinkOdeAction(const char* name, 
        const char* pipeline, const char* sink)
        : OdeAction(name)
        , m_pipeline(pipeline)
        , m_sink(sink)
    {
        LOG_FUNC();
    }

    AddSinkOdeAction::~AddSinkOdeAction()
    {
        LOG_FUNC();
    }
    
    void AddSinkOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled)
        {
            Services::GetServices()->PipelineComponentAdd(m_pipeline.c_str(), m_sink.c_str());
        }
    }

    // ********************************************************************

    RemoveSinkOdeAction::RemoveSinkOdeAction(const char* name, 
        const char* pipeline, const char* sink)
        : OdeAction(name)
        , m_pipeline(pipeline)
        , m_sink(sink)
    {
        LOG_FUNC();
    }

    RemoveSinkOdeAction::~RemoveSinkOdeAction()
    {
        LOG_FUNC();
    }
    
    void RemoveSinkOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->PipelineComponentRemove(m_pipeline.c_str(), m_sink.c_str());
        }
    }

    // ********************************************************************

    AddSourceOdeAction::AddSourceOdeAction(const char* name, 
        const char* pipeline, const char* source)
        : OdeAction(name)
        , m_pipeline(pipeline)
        , m_source(source)
    {
        LOG_FUNC();
    }

    AddSourceOdeAction::~AddSourceOdeAction()
    {
        LOG_FUNC();
    }
    
    void AddSourceOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->PipelineComponentAdd(m_pipeline.c_str(), m_source.c_str());
        }
    }

    // ********************************************************************

    RemoveSourceOdeAction::RemoveSourceOdeAction(const char* name, 
        const char* pipeline, const char* source)
        : OdeAction(name)
        , m_pipeline(pipeline)
        , m_source(source)
    {
        LOG_FUNC();
    }

    RemoveSourceOdeAction::~RemoveSourceOdeAction()
    {
        LOG_FUNC();
    }
    
    void RemoveSourceOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->PipelineComponentRemove(m_pipeline.c_str(), m_source.c_str());
        }
    }

    // ********************************************************************

    ResetTriggerOdeAction::ResetTriggerOdeAction(const char* name, const char* trigger)
        : OdeAction(name)
        , m_trigger(trigger)
    {
        LOG_FUNC();
    }

    ResetTriggerOdeAction::~ResetTriggerOdeAction()
    {
        LOG_FUNC();
    }
    
    void ResetTriggerOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->OdeTriggerReset(m_trigger.c_str());
        }
    }


    // ********************************************************************

    DisableTriggerOdeAction::DisableTriggerOdeAction(const char* name, const char* trigger)
        : OdeAction(name)
        , m_trigger(trigger)
    {
        LOG_FUNC();
    }

    DisableTriggerOdeAction::~DisableTriggerOdeAction()
    {
        LOG_FUNC();
    }
    
    void DisableTriggerOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->OdeTriggerEnabledSet(m_trigger.c_str(), false);
        }
    }

    // ********************************************************************

    EnableTriggerOdeAction::EnableTriggerOdeAction(const char* name, const char* trigger)
        : OdeAction(name)
        , m_trigger(trigger)
    {
        LOG_FUNC();
    }

    EnableTriggerOdeAction::~EnableTriggerOdeAction()
    {
        LOG_FUNC();
    }
    
    void EnableTriggerOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->OdeTriggerEnabledSet(m_trigger.c_str(), true);
        }
    }

    // ********************************************************************

    DisableActionOdeAction::DisableActionOdeAction(const char* name, const char* action)
        : OdeAction(name)
        , m_action(action)
    {
        LOG_FUNC();
    }

    DisableActionOdeAction::~DisableActionOdeAction()
    {
        LOG_FUNC();
    }
    
    void DisableActionOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->OdeActionEnabledSet(m_action.c_str(), false);
        }
    }

    // ********************************************************************

    EnableActionOdeAction::EnableActionOdeAction(const char* name, const char* action)
        : OdeAction(name)
        , m_action(action)
    {
        LOG_FUNC();
    }

    EnableActionOdeAction::~EnableActionOdeAction()
    {
        LOG_FUNC();
    }
    
    void EnableActionOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->OdeActionEnabledSet(m_action.c_str(), true);
        }
    }



    AddAreaOdeAction::AddAreaOdeAction(const char* name, 
        const char* trigger, const char* area)
        : OdeAction(name)
        , m_trigger(trigger)
        , m_area(area)
    {
        LOG_FUNC();
    }

    AddAreaOdeAction::~AddAreaOdeAction()
    {
        LOG_FUNC();
    }
    
    void AddAreaOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->OdeTriggerAreaAdd(m_trigger.c_str(), m_area.c_str());
        }
    }
    
    // ********************************************************************

    RemoveAreaOdeAction::RemoveAreaOdeAction(const char* name, 
        const char* trigger, const char* area)
        : OdeAction(name)
        , m_trigger(trigger)
        , m_area(area)
    {
        LOG_FUNC();
    }

    RemoveAreaOdeAction::~RemoveAreaOdeAction()
    {
        LOG_FUNC();
    }
    
    void RemoveAreaOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->OdeTriggerAreaRemove(m_trigger.c_str(), m_area.c_str());
        }
    }
    
    // ********************************************************************

    RecordSinkStartOdeAction::RecordSinkStartOdeAction(const char* name, 
         DSL_BASE_PTR pRecordSink, uint start, uint duration, void* clientData)
        : OdeAction(name)
        , m_pRecordSink(pRecordSink)
        , m_start(start)
        , m_duration(duration)
        , m_clientData(clientData)
    {
        LOG_FUNC();
    }

    RecordSinkStartOdeAction::~RecordSinkStartOdeAction()
    {
        LOG_FUNC();
    }
    
    void RecordSinkStartOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            std::dynamic_pointer_cast<RecordSinkBintr>(m_pRecordSink)->StartSession(
                m_start, m_duration, m_clientData);
        }
    }

    // ********************************************************************

    RecordSinkStopOdeAction::RecordSinkStopOdeAction(const char* name, 
        DSL_BASE_PTR pRecordSink)
        : OdeAction(name)
        , m_pRecordSink(pRecordSink)
    {
        LOG_FUNC();
    }

    RecordSinkStopOdeAction::~RecordSinkStopOdeAction()
    {
        LOG_FUNC();
    }
    
    void RecordSinkStopOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            std::dynamic_pointer_cast<RecordSinkBintr>(m_pRecordSink)->StopSession();
        }
    }

    // ********************************************************************

    RecordTapStartOdeAction::RecordTapStartOdeAction(const char* name, 
        DSL_BASE_PTR pRecordTap, uint start, uint duration, void* clientData)
        : OdeAction(name)
        , m_pRecordTap(pRecordTap)
        , m_start(start)
        , m_duration(duration)
        , m_clientData(clientData)
    {
        LOG_FUNC();
    }

    RecordTapStartOdeAction::~RecordTapStartOdeAction()
    {
        LOG_FUNC();
    }
    
    void RecordTapStartOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            std::dynamic_pointer_cast<RecordTapBintr>(m_pRecordTap)->StartSession(
                m_start, m_duration, m_clientData);
        }
    }

    // ********************************************************************

    RecordTapStopOdeAction::RecordTapStopOdeAction(const char* name, 
        DSL_BASE_PTR pRecordTap)
        : OdeAction(name)
        , m_pRecordTap(pRecordTap)
    {
        LOG_FUNC();
    }

    RecordTapStopOdeAction::~RecordTapStopOdeAction()
    {
        LOG_FUNC();
    }
    
    void RecordTapStopOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            std::dynamic_pointer_cast<RecordTapBintr>(m_pRecordTap)->StopSession();
        }
    }
    // ********************************************************************

    TilerShowSourceOdeAction::TilerShowSourceOdeAction(const char* name, const char* tiler, uint timeout, bool hasPrecedence)
        : OdeAction(name)
        , m_tiler(tiler)
        , m_timeout(timeout)
        , m_hasPrecedence(hasPrecedence)
    {
        LOG_FUNC();
    }

    TilerShowSourceOdeAction::~TilerShowSourceOdeAction()
    {
        LOG_FUNC();
    }
    
    void TilerShowSourceOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_enabled)
        {
            // Ignore the return value,
            Services::GetServices()->TilerSourceShowSet(m_tiler.c_str(), pFrameMeta->source_id, m_timeout, m_hasPrecedence);
        }
    }
}    

