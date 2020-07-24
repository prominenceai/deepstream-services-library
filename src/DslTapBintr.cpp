/*
The MIT License

Copyright (c) 2019-Present, ROBERT HOWELL

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

#include <nvbufsurftransform.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/highgui/highgui.hpp"

#include "Dsl.h"
#include "DslTapBintr.h"
#include "DslBranchBintr.h"

namespace DSL
{

    TapBintr::TapBintr(const char* name)
        : Bintr(name)
    {
        LOG_FUNC();

        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "tap-bin-queue");
        AddChild(m_pQueue);
        m_pQueue->AddGhostPadToParent("sink");
    }

    TapBintr::~TapBintr()
    {
        LOG_FUNC();
    }

    bool TapBintr::LinkToSource(DSL_NODETR_PTR pTee)
    {
        LOG_FUNC();

        std::string srcPadName = "src_" + std::to_string(m_uniqueId);

        LOG_INFO("Linking Tap '" << GetName() << "' to Pad '" << srcPadName
            << "' for Tee '" << pTee->GetName() << "'");
        
        m_pGstStaticSinkPad = gst_element_get_static_pad(GetGstElement(), "sink");
        if (!m_pGstStaticSinkPad)
        {
            LOG_ERROR("Failed to get Static Sink Pad for TapBintr '" << GetName() << "'");
            return false;
        }

        GstPad* pRequestedSourcePad(NULL);

        // NOTE: important to use the correct request pad name based on the element type
        // Cast the base DSL_BASE_PTR to DSL_ELEMENTR_PTR so we can query the factory type 
        DSL_ELEMENT_PTR pTeeElementr = 
            std::dynamic_pointer_cast<Elementr>(pTee);

        if (pTeeElementr->IsFactoryName("nvstreamdemux"))
        {
            pRequestedSourcePad = gst_element_get_request_pad(pTee->GetGstElement(), srcPadName.c_str());
        }
        else // standard "Tee"
        {
            pRequestedSourcePad = gst_element_get_request_pad(pTee->GetGstElement(), "src_%u");
        }
            
        if (!pRequestedSourcePad)
        {
            LOG_ERROR("Failed to get Tee source Pad for TapBintr '" << GetName() <<"'");
            return false;
        }

        m_pGstRequestedSourcePads[srcPadName] = pRequestedSourcePad;

        return Bintr::LinkToSource(pTee);
        
    }
    
    bool TapBintr::UnlinkFromSource()
    {
        LOG_FUNC();
        
        // If we're not currently linked to the Tee
        if (!IsLinkedToSource())
        {
            LOG_ERROR("TapBintr '" << GetName() << "' is not in a Linked state");
            return false;
        }

        std::string srcPadName = "src_" + std::to_string(m_uniqueId);

        LOG_INFO("Unlinking and releasing requested Source Pad for Decode Source Tee " << GetName());
        
        gst_pad_send_event(m_pGstStaticSinkPad, gst_event_new_eos());
        if (!gst_pad_unlink(m_pGstRequestedSourcePads[srcPadName], m_pGstStaticSinkPad))
        {
            LOG_ERROR("TapBintr '" << GetName() << "' failed to unlink from Decode Source Tee");
            return false;
        }
        gst_element_release_request_pad(GetSource()->GetGstElement(), m_pGstRequestedSourcePads[srcPadName]);
        gst_object_unref(m_pGstRequestedSourcePads[srcPadName]);
                
        m_pGstRequestedSourcePads.erase(srcPadName);
        
        return Nodetr::UnlinkFromSource();
    }

    //-------------------------------------------------------------------------
    
    RecordTapBintr::RecordTapBintr(const char* name, const char* outdir, 
        uint container, NvDsSRCallbackFunc clientListener)
        : TapBintr(name)
        , m_outdir(outdir)
        , m_pContext(NULL)
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
            LOG_ERROR("Invalid container = '" << container << "' for new RecordTapBintr '" << name << "'");
            throw;
        }
        
        // Set single callback listener. Unique clients are identifed using client_data provided on Start session
        m_initParams.callback = clientListener;
        
        // Set both width and height params to zero = no-transcode
        m_initParams.width = 0;  
        m_initParams.height = 0; 
        
        // Filename prefix uses bintr name by default
        m_initParams.fileNamePrefix = const_cast<gchar*>(GetCStrName());
        m_initParams.dirpath = const_cast<gchar*>(m_outdir.c_str());
        
        m_initParams.defaultDuration = DSL_DEFAULT_VIDEO_RECORD_DURATION_IN_SEC;
        m_initParams.videoCacheSize = DSL_DEFAULT_VIDEO_RECORD_CACHE_IN_SEC;
    }
    
    RecordTapBintr::~RecordTapBintr()
    {
        LOG_FUNC();
    
        if (IsLinked())
        {    
            UnlinkAll();
        }
    }

    bool RecordTapBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("RecordTapBintr '" << m_name << "' is already linked");
            return false;
        }

        // Create the smart record context
        if (NvDsSRCreate(&m_pContext, &m_initParams) != NVDSSR_STATUS_OK)
        {
            LOG_ERROR("Failed to create Smart Record Context for new RecordTapBintr '" << m_name << "'");
            return false;
        }
        
        m_pRecordBin = DSL_NODETR_NEW("record-bin");
        m_pRecordBin->SetGstObject(GST_OBJECT(m_pContext->recordbin));
            
        AddChild(m_pRecordBin);

        GstPad* srcPad = gst_element_get_static_pad(m_pQueue->GetGstElement(), "src");
        GstPad* sinkPad = gst_element_get_static_pad(m_pRecordBin->GetGstElement(), "sink");
        
        if (gst_pad_link(srcPad, sinkPad) != GST_PAD_LINK_OK)
        {
            LOG_ERROR("Failed to link parser to record-bin new RecordTapBintr '" << m_name << "'");
            return false;
        }
        m_isLinked = true;
        return true;
    }
    
    void RecordTapBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("RecordTapBintr '" << m_name << "' is not linked");
            return;
        }
        GstPad* srcPad = gst_element_get_static_pad(m_pQueue->GetGstElement(), "src");
        GstPad* sinkPad = gst_element_get_static_pad(m_pRecordBin->GetGstElement(), "sink");
        
        gst_pad_unlink(srcPad, sinkPad);

        RemoveChild(m_pRecordBin);
        
        m_pRecordBin = nullptr;
        NvDsSRDestroy(m_pContext);
        m_pContext = NULL;
        
        m_isLinked = false;
    }

    const char* RecordTapBintr::GetOutdir()
    {
        LOG_FUNC();
        
        return m_outdir.c_str();
    }
    
    bool RecordTapBintr::SetOutdir(const char* outdir)
    {
        LOG_FUNC();

        if (IsLinked())
        {
            LOG_ERROR("Unable to set the Output for RecordTapBintr '" << GetName() 
                << "' as it's currently Linked");
            return false;
        }
        
        m_outdir.assign(outdir);
        return true;
    }

    uint RecordTapBintr::GetCacheSize()
    {
        LOG_FUNC();
        
        return m_initParams.videoCacheSize;
    }

    bool RecordTapBintr::SetCacheSize(uint videoCacheSize)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set cache size for RecordTapBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_initParams.videoCacheSize = videoCacheSize;
        
        return true;
    }


    void RecordTapBintr::GetDimensions(uint* width, uint* height)
    {
        LOG_FUNC();
        
        *width = m_initParams.width;
        *height = m_initParams.height;
    }

    bool RecordTapBintr::SetDimensions(uint width, uint height)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set Dimensions for RecordTapBintr '" << GetName() 
                << "' as it's currently Linked");
            return false;
        }

        m_initParams.width = width;
        m_initParams.height = height;
        
        return true;
    }
    
    bool RecordTapBintr::StartSession(uint* session, uint start, uint duration, void* clientData)
    {
        LOG_FUNC();
        
        if (!IsLinked())
        {
            LOG_ERROR("Unable to Start Session for RecordTapBintr '" << GetName() 
                << "' as it is not currently Linked");
            return false;
        }
        return (NvDsSRStart(m_pContext, session, start, duration, clientData) == NVDSSR_STATUS_OK);
    }
    
    bool RecordTapBintr::StopSession(uint session)
    {
        LOG_FUNC();
        
        if (!IsLinked())
        {
            LOG_ERROR("Unable to Stop Session for RecordTapBintr '" << GetName() 
                << "' as it is not currently Linked");
            return false;
        }
        return (NvDsSRStop(m_pContext, session) == NVDSSR_STATUS_OK);
    }
    
    bool RecordTapBintr::GotKeyFrame()
    {
        LOG_FUNC();
        
        if (!m_pContext or !IsLinked())
        {
            LOG_WARN("There is no Record Bin context to query as '" << GetName() 
                << "' is not currently Linked");
            return false;
        }
        return m_pContext->gotKeyFrame;
    }
    
    bool RecordTapBintr::IsOn()
    {
        LOG_FUNC();
        
        if (!m_pContext or !IsLinked())
        {
            LOG_WARN("There is no Record Bin context to query as '" << GetName() 
                << "' is not currently Linked");
            return false;
        }
        return m_pContext->recordOn;
    }
    
    bool RecordTapBintr::ResetDone()
    {
        LOG_FUNC();
        
        if (!m_pContext or !IsLinked())
        {
            LOG_WARN("There is no Record Bin context to query as '" << GetName() 
                << "' is not currently Linked");
            return false;
        }
        return m_pContext->resetDone;
    }

}