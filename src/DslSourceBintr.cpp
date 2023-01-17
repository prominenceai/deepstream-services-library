/*
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
#include "DslSourceBintr.h"
#include "DslPipelineBintr.h"
#include "DslSurfaceTransform.h"
#include <nvdsgstutils.h>
#include <gst/app/gstappsrc.h>

namespace DSL
{
    static bool set_full_caps(DSL_ELEMENT_PTR pElement, 
        const char* media, const char* format, uint width, uint height, 
        uint fpsN, uint fpsD, bool isNvidia)
    {
        GstCaps * pCaps(NULL);
        if (width and height)
        {
            pCaps = gst_caps_new_simple(media, 
                "format", G_TYPE_STRING, format,
                "width", G_TYPE_INT, width, 
                "height", G_TYPE_INT, height, 
                "framerate", GST_TYPE_FRACTION, fpsN, fpsD, NULL);
        }
        else
        {
            pCaps = gst_caps_new_simple(media, 
                "format", G_TYPE_STRING, format,
                "framerate", GST_TYPE_FRACTION, fpsN, fpsD, NULL);
        }    
        if (!pCaps)
        {
            LOG_ERROR("Failed to create new Simple Capabilities for '" 
                << pElement->GetName() << "'");
            return false;  
        }

        // if the provided element is an NVIDIA plugin, then we need to add
        // the additional feature to enable buffer access via the NvBuffer API.
        if (isNvidia)
        {
            GstCapsFeatures *feature = NULL;
            feature = gst_caps_features_new("memory:NVMM", NULL);
            gst_caps_set_features(pCaps, 0, feature);
        }
        // Set the provided element's caps and unref caps structure.
        pElement->SetAttribute("caps", pCaps);
        gst_caps_unref(pCaps);  

        return true;
    }

    static bool set_format_caps(DSL_ELEMENT_PTR pElement, 
        const char* media, const char* format, bool isNvidia)
    {
        GstCaps * pCaps = gst_caps_new_simple(media, 
            "format", G_TYPE_STRING, format, NULL);
        if (!pCaps)
        {
            LOG_ERROR("Failed to create new Simple Capabilities for '" 
                << pElement->GetName() << "'");
            return false;  
        }

        // if the provided element is an NVIDIA plugin, then we need to add
        // the additional feature to enable buffer access via the NvBuffer API.
        if (isNvidia)
        {
            GstCapsFeatures *feature = NULL;
            feature = gst_caps_features_new("memory:NVMM", NULL);
            gst_caps_set_features(pCaps, 0, feature);
        }
        // Set the provided element's caps and unref caps structure.
        pElement->SetAttribute("caps", pCaps);
        gst_caps_unref(pCaps);  

        return true;
    }
    
    SourceBintr::SourceBintr(const char* name)
        : Bintr(name)
        , m_cudaDeviceProp{0}
        , m_isLive(true)
        , m_width(0)
        , m_height(0)
        , m_fpsN(0)
        , m_fpsD(0)
        , m_bufferOutWidth(0)
        , m_bufferOutHeight(0)
        , m_bufferOutFpsN(0)
        , m_bufferOutFpsD(0)
        , m_bufferOutOrientation(DSL_VIDEO_ORIENTATION_NONE)
    {
        LOG_FUNC();

        // Set the stream-id of the unique Source name
        SetId(Services::GetServices()->_sourceNameSet(name));
        
        // Get the Device properties
        cudaGetDeviceProperties(&m_cudaDeviceProp, m_gpuId);

        // Media type is fixed to "video/x-raw"
        std::wstring L_mediaType(DSL_MEDIA_TYPE_VIDEO_XRAW);
        m_mediaType.assign(L_mediaType.begin(), L_mediaType.end());

        // Set the buffer-out-format to the default video format
        std::wstring L_bufferOutFormat(DSL_VIDEO_FORMAT_DEFAULT);
        m_bufferOutFormat.assign(L_bufferOutFormat.begin(), 
            L_bufferOutFormat.end());
        
        // All SourceBintrs have a Video Converter with Caps Filter used
        // to control the buffer-out format, dimensions, crop values, etc.
        
        // ---- Video Converter Setup

        m_pBufferOutVidConv = DSL_ELEMENT_EXT_NEW("nvvideoconvert", 
            name, "buffer-out");
        
        // Get property defaults that aren't specifically set
        m_pBufferOutVidConv->GetAttribute("gpu-id", &m_gpuId);
        m_pBufferOutVidConv->GetAttribute("nvbuf-memory-type", &m_nvbufMemType);
        
        // ---- Caps Filter Setup

        m_pBufferOutCapsFilter = DSL_ELEMENT_NEW("capsfilter", name);
        
        SetBufferOutFormat(m_bufferOutFormat.c_str());
        
        // add both elementrs as children to this Bintr
        AddChild(m_pBufferOutVidConv);
        AddChild(m_pBufferOutCapsFilter);

        // buffer-out caps filter is "src" ghost-pad for all SourceBintrs
        m_pBufferOutCapsFilter->AddGhostPadToParent("src");
        
        std::string padProbeName = GetName() + "-src-pad-probe";
        m_pSrcPadProbe = DSL_PAD_BUFFER_PROBE_NEW(padProbeName.c_str(), 
            "src", m_pBufferOutCapsFilter);
    }
    
    SourceBintr::~SourceBintr()
    {
        LOG_FUNC();

        if (m_isLinked)
        {    
            UnlinkAll();
        }
        
        Services::GetServices()->_sourceNameErase(GetCStrName());
    }
    
    bool SourceBintr::AddToParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' Source to the Parent Pipeline 
        return std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
            AddSourceBintr(std::dynamic_pointer_cast<SourceBintr>(shared_from_this()));
    }

    bool SourceBintr::IsParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // check if 'this' Source is child of Parent Pipeline 
        return std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
            IsSourceBintrChild(std::dynamic_pointer_cast<SourceBintr>(shared_from_this()));
    }

    bool SourceBintr::RemoveFromParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        if (!IsParent(pParentBintr))
        {
            LOG_ERROR("Source '" << GetName() << "' is not a child of Pipeline '" 
                << pParentBintr->GetName() << "'");
            return false;
        }
        
        // remove 'this' Source from the Parent Pipeline 
        return std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
            RemoveSourceBintr(std::dynamic_pointer_cast<SourceBintr>(shared_from_this()));
    }

    void SourceBintr::GetDimensions(uint* width, uint* height)
    {
        LOG_FUNC();
        
        *width = m_width;
        *height = m_height;
    }

    void SourceBintr::GetFrameRate(uint* fpsN, uint* fpsD)
    {
        LOG_FUNC();
        
        *fpsN = m_fpsN;
        *fpsD = m_fpsD;
    }

    bool SourceBintr::SetBufferOutFormat(const char* format)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't set buffer-out-format for SourceBintr '" << GetName() 
                << "' as it is currently in a linked state");
            return false;
        }

        m_bufferOutFormat = format;
        
        updateCaps();

        return true;
    }
    
    void SourceBintr::GetBufferOutDimensions(uint* width, uint* height)
    {
        LOG_FUNC();
        
        *width = m_bufferOutWidth;
        *height = m_bufferOutHeight;
    }
    
    bool SourceBintr::SetBufferOutDimensions(uint width, uint height)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't set buffer-out-dimensions for SourceBintr '" << GetName() 
                << "' as it is currently in a linked state");
            return false;
        }
        m_bufferOutWidth = width;
        m_bufferOutHeight = height;
        
        updateCaps();
        
        return true;
    }
    
    void SourceBintr::GetBufferOutFrameRate(uint* fpsN, uint* fpsD)
    {
        LOG_FUNC();
        
        *fpsN = m_bufferOutFpsN;
        *fpsD = m_bufferOutFpsD;
    }
    
    bool SourceBintr::SetBufferOutFrameRate(uint fpsN, uint fpsD)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't set buffer-out-dimensions for SourceBintr '" << GetName() 
                << "' as it is currently in a linked state");
            return false;
        }
        m_bufferOutFpsN = fpsN;
        m_bufferOutFpsD = fpsD;
        
        return updateCaps();
    }
    
    void SourceBintr::GetBufferOutCropRectangle(uint when, 
        uint* left, uint* top, uint* width, uint* height)
    {
        LOG_FUNC();
        
        const char* cropCString;

        if (when == DSL_VIDEO_CROP_PRE_CONVERSION)
        {
            m_pBufferOutVidConv->GetAttribute("src-crop", &cropCString);
        }
        else
        {
            m_pBufferOutVidConv->GetAttribute("dest-crop", &cropCString);
        }
        std::string cropString(cropCString);
        std::string delimiter(":");
        std::string leftSubStr = cropString.substr(0, cropString.find(delimiter)); 
        std::string topSubStr = cropString.substr(1, cropString.find(delimiter)); 
        std::string widthSubStr = cropString.substr(2, cropString.find(delimiter)); 
        std::string heightSubStr = cropString.substr(3, cropString.find(delimiter)); 
        
        *left = std::stoul(leftSubStr.c_str());
        *top = std::stoul(topSubStr.c_str());
        *width = std::stoul(widthSubStr.c_str());
        *height = std::stoul(heightSubStr.c_str());
    }
    
    bool SourceBintr::SetBufferOutCropRectangle(uint when, 
        uint left, uint top, uint width, uint height)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR(
                "Unable to set buffer-out crop settings for SourceBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        
        std::string cropSettings( 
            std::to_string(left) + ":" +
            std::to_string(top) + ":" +
            std::to_string(width) + ":" +
            std::to_string(height));
        
        if (when == DSL_VIDEO_CROP_PRE_CONVERSION)
        {
            m_pBufferOutVidConv->SetAttribute("src-crop", cropSettings.c_str());
        }
        else
        {
            m_pBufferOutVidConv->SetAttribute("dest-crop", cropSettings.c_str());
        }

        return true;
    }

    uint SourceBintr::GetBufferOutOrientation()
    {
        LOG_FUNC();
        
        return m_bufferOutOrientation;
    }
    
    bool SourceBintr::SetBufferOutOrientation(uint orientation)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR(
                "Unable to set buffer-out-orientation for SourceBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_bufferOutOrientation = orientation;
        m_pBufferOutVidConv->SetAttribute("flip-method", m_bufferOutOrientation);

        return true;
    }

    bool SourceBintr::SetNvbufMemType(uint nvbufMemType)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR(
                "Unable to set NVIDIA buffer memory type for SourceBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_nvbufMemType = nvbufMemType;
        m_pBufferOutVidConv->SetAttribute("nvbuf-memory-type", m_nvbufMemType);

        return true;
    }
    
    bool SourceBintr::updateCaps()
    {
        LOG_FUNC();

        GstCaps* pCaps = gst_caps_new_simple(m_mediaType.c_str(), 
            "format", G_TYPE_STRING, m_bufferOutFormat.c_str(),
            "width", G_TYPE_INT, m_bufferOutWidth, 
            "height", G_TYPE_INT, m_bufferOutHeight,
            "framerate", GST_TYPE_FRACTION, m_bufferOutFpsN, m_bufferOutFpsD, 
            NULL);

        if (!pCaps)
        {
            LOG_ERROR("Failed to create new Simple Capabilities for SourceBintr '" 
                << GetName() << "'");
            return false;  
        }

        // The Video converter is an NVIDIA plugin so we need to add the
        // additional feature to enable buffer access via the NvBuffer API.
        GstCapsFeatures *feature = NULL;
        feature = gst_caps_features_new("memory:NVMM", NULL);
        gst_caps_set_features(pCaps, 0, feature);

        // Set the provided element's caps and unref caps structure.
        m_pBufferOutCapsFilter->SetAttribute("caps", pCaps);
        gst_caps_unref(pCaps); 
        
        return true;
    }

    //*********************************************************************************
    AppSourceBintr::AppSourceBintr(const char* name, bool isLive, 
            const char* bufferInFormat, uint width, uint height, uint fpsN, uint fpsD)
        : SourceBintr(name) 
        , m_doTimestamp(TRUE)
        , m_bufferInFormat(bufferInFormat)
        , m_needDataHandler(NULL)
        , m_enoughDataHandler(NULL)
        , m_clientData(NULL)
        , m_maxBytes(0)
// TODO support GST 1.20 properties        
//        , m_maxBuffers(0)
//        , m_maxTime(0)
//        , m_leakyType(0)
    {
        LOG_FUNC();
        
        m_isLive = isLive;
        m_width = width;
        m_height = height;
        m_fpsN = fpsN;
        m_fpsD = fpsD;
        
        // ---- Source Element Setup

        m_pSourceElement = DSL_ELEMENT_NEW("appsrc", name);

        // Set the full capabilities (format, dimensions, and framerate)
        // NVIDIA plugin = false... this is a GStreamer plugin
        if (!set_full_caps(m_pSourceElement, m_mediaType.c_str(), 
            m_bufferInFormat.c_str(), m_width, m_height, m_fpsN, m_fpsD, false))
        {
            throw;
        }
            
        // emit-signals are disabled by default... need to enable
        m_pSourceElement->SetAttribute("emit-signals", true);
        
        // register the data callbacks with the appsrc element
        g_signal_connect(m_pSourceElement->GetGObject(), "need-data", 
            G_CALLBACK(on_need_data_cb), this);
        g_signal_connect(m_pSourceElement->GetGObject(), "enough-data", 
            G_CALLBACK(on_enough_data_cb), this);

        // get the property defaults
        m_pSourceElement->GetAttribute("do-timestamp", &m_doTimestamp);
        m_pSourceElement->GetAttribute("format", &m_streamFormat);
        m_pSourceElement->GetAttribute("block", &m_blockEnabled);
        m_pSourceElement->GetAttribute("max-bytes", &m_maxBytes);

        // TODO support GST 1.20 properties
        // m_pSourceElement->GetAttribute("max-buffers", &m_maxBuffers);
        // m_pSourceElement->GetAttribute("max-time", &m_maxTime);
        // m_pSourceElement->GetAttribute("leaky-type", &m_leakyType);
        
        if (!m_cudaDeviceProp.integrated)
        {
            m_pBufferOutVidConv->SetAttribute("nvbuf-memory-type", 
                DSL_NVBUF_MEM_TYPE_UNIFIED);
        }

        LOG_INFO("");
        LOG_INFO("Initial property values for AppSourceBintr '" << name << "'");
        LOG_INFO("  is-live           : " << m_isLive);
        LOG_INFO("  do-timestamp      : " << m_doTimestamp);
        LOG_INFO("  stream-format     : " << m_streamFormat);
        LOG_INFO("  block-enabled     : " << m_blockEnabled);
        LOG_INFO("  max-bytes         : " << m_maxBytes);
        LOG_INFO("  media             : " << m_mediaType);
        LOG_INFO("  buffer-in-format  : " << m_bufferInFormat);
        LOG_INFO("  buffer-out-format : " << m_bufferOutFormat);
        LOG_INFO("  width             : " << m_width);
        LOG_INFO("  height            : " << m_height);
        LOG_INFO("  fps-n             : " << m_fpsN);
        LOG_INFO("  fps-d             : " << m_fpsD);

        // TODO support GST 1.20 properties
        // LOG_INFO("max-buffers = " << m_maxBuffers);
        // LOG_INFO("max-time    = " << m_maxTime);
        // LOG_INFO("leaky-type  = " << m_leakyType);

        // add all elementrs as childer to this Bintr
        AddChild(m_pSourceElement);

        g_mutex_init(&m_dataHandlerMutex);
    }

    AppSourceBintr::~AppSourceBintr()
    {
        LOG_FUNC();
        
        g_mutex_clear(&m_dataHandlerMutex);
    }
    
    bool AppSourceBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("AppSourceBintr '" << GetName() 
                << "' is already in a linked state");
            return false;
        }
        if (!m_pSourceElement->LinkToSink(m_pBufferOutVidConv) or
            !m_pBufferOutVidConv->LinkToSink(m_pBufferOutCapsFilter))
        {
            return false;
        }
        
        m_isLinked = true;
        
        return true;
    }

    void AppSourceBintr::UnlinkAll()
    {
        LOG_FUNC();

        if (!m_isLinked)
        {
            LOG_ERROR("AppSourceBintr '" << GetName() 
                << "' is not in a linked state");
            return;
        }
        m_pSourceElement->UnlinkFromSink();
        m_pBufferOutVidConv->UnlinkFromSink();
        m_isLinked = false;
    }

    bool AppSourceBintr::AddDataHandlers(
        dsl_source_app_need_data_handler_cb needDataHandler, 
        dsl_source_app_enough_data_handler_cb enoughDataHandler, 
        void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_dataHandlerMutex);

        if (m_needDataHandler)
        {
            LOG_ERROR("AppSourceBintr '" << GetName() 
                << "' already has data-handler callbacks");
            return false;
        }
        m_needDataHandler = needDataHandler;
        m_enoughDataHandler = enoughDataHandler;
        m_clientData = clientData;
        return true;
    }
        
    bool AppSourceBintr::RemoveDataHandlers()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_dataHandlerMutex);

        if (!m_needDataHandler)
        {
            LOG_ERROR("AppSourceBintr '" << GetName() 
                << "' does not have data-handler callbacks to remove");
            return false;
        }
        m_needDataHandler = NULL;
        m_enoughDataHandler = NULL;
        m_clientData = NULL;
        return true;
    }
    
    bool AppSourceBintr::PushBuffer(void* buffer)
    {
        // Do not log function entry/exit for performance
        
        if (!m_isLinked)
        {
            LOG_ERROR("AppSourceBintr '" << GetName() 
                << "' is not in a linked state");
            return false;
        }
        
        // Push the buffer to the App Source element.
        
        GstFlowReturn retVal = gst_app_src_push_buffer(
            (GstAppSrc*)m_pSourceElement->GetGObject(), (GstBuffer*)buffer);
        if (retVal != GST_FLOW_OK)
        {
            LOG_ERROR("AppSourceBintr '" << GetName() 
                << "' returned " << retVal << " on push-buffer");
            return false;
        }
            
        return true;
    }

    bool AppSourceBintr::PushSample(void* sample)
    {
        // Do not log function entry/exit for performance
        
        if (!m_isLinked)
        {
            LOG_ERROR("AppSourceBintr '" << GetName() 
                << "' is not in a linked state");
            return false;
        }
        
        // Push the sample to the App Source element.
        
        GstFlowReturn retVal = gst_app_src_push_sample(
            (GstAppSrc*)m_pSourceElement->GetGObject(), (GstSample*)sample);
        if (retVal != GST_FLOW_OK)
        {
            LOG_ERROR("AppSourceBintr '" << GetName() 
                << "' returned " << retVal << " on push-sample");
            return false;
        }
            
        return true;
    }

    bool AppSourceBintr::Eos()
    {
        LOG_FUNC();

        if (!m_isLinked)
        {
            LOG_ERROR("AppSourceBintr '" << GetName() 
                << "' is not in a linked state");
            return false;
        }
        GstFlowReturn retVal = gst_app_src_end_of_stream(
            (GstAppSrc*)m_pSourceElement->GetGObject());
        if (retVal != GST_FLOW_OK)
        {
            LOG_ERROR("AppSourceBintr '" << GetName() 
                << "' returned " << retVal << " on end-of-stream");
            return false;
        }
            
        return true;
    }

    void AppSourceBintr::HandleNeedData(uint length)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_dataHandlerMutex);

        if (m_needDataHandler)
        {
            try
            {
                // call the client handler with the length hint.
                m_needDataHandler(length, m_clientData);
            }
            catch(...)
            {
                LOG_ERROR("AppSourceBintr '" << GetName() 
                    << "' threw exception calling client handler function \
                        for 'need-data'");
            }
        }
    }
    
    void AppSourceBintr::HandleEnoughData()
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_dataHandlerMutex);

        if (m_enoughDataHandler)
        {
            try
            {
                // call the client handler with the buffer and process.
                m_enoughDataHandler(m_clientData);
            }
            catch(...)
            {
                LOG_ERROR("AppSourceBintr '" << GetName() 
                    << "' threw exception calling client handler function \
                        for 'enough-data'");
            }
        }
    }

    boolean AppSourceBintr::GetDoTimestamp()
    {
        LOG_FUNC();
        
        return m_doTimestamp;
    }

    bool AppSourceBintr::SetDoTimestamp(boolean doTimestamp)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't set block-enabled for SourceBintr '" 

                << GetName() << "' as it's currently in a linked state");
            return false;
        }

        m_doTimestamp = doTimestamp;
        m_pSourceElement->SetAttribute("do-timestamp", m_doTimestamp);
        return true;
    }


    boolean AppSourceBintr::GetBlockEnabled()
    {
        LOG_FUNC();
        
        return m_blockEnabled;
    }
    
    bool AppSourceBintr::SetBlockEnabled(boolean enabled)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't set block-enabled for AppSourceBintr '" 
                << GetName() << "' as it's currently in a linked state");
            return false;
        }

        m_blockEnabled = enabled;
        m_pSourceElement->SetAttribute("block", m_blockEnabled);
        return true;
    }
    
    uint AppSourceBintr::GetStreamFormat()
    {
        LOG_FUNC();
        
        return m_streamFormat;
    }
    
    bool AppSourceBintr::SetStreamFormat(uint streamFormat)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't set stream-format for AppSourceBintr '" 
                << GetName() << "' as it's currently in a linked state");
            return false;
        }

        m_streamFormat = streamFormat;
        m_pSourceElement->SetAttribute("format", m_streamFormat);
        return true;
    }
    
    uint64_t AppSourceBintr::GetCurrentLevelBytes()
    {
        // do not log function entry/exit for performance reasons
        
        uint64_t currentLevel(0);
        
        m_pSourceElement->GetAttribute("current-level-bytes", 
            &currentLevel);

        return currentLevel;
    }
    
    uint64_t AppSourceBintr::GetMaxLevelBytes()
    {
        LOG_FUNC();

        m_pSourceElement->GetAttribute("max-bytes", 
            &m_maxBytes);

        return m_maxBytes;
    }
    
    bool AppSourceBintr::SetMaxLevelBytes(uint64_t level)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't set max-level for AppSourceBintr '" 
                << GetName() << "' as it's currently in a linked state");
            return false;
        }
        m_maxBytes = level;
        m_pSourceElement->SetAttribute("max-bytes", m_maxBytes);

        return true;
    }
    
//    uint AppSourceBintr::GetLeakyType()
//    {
//        LOG_FUNC();
//        
//        return m_leakyType;
//    }
//    
//    bool AppSourceBintr::SetLeakyType(uint leakyType)
//    {
//        LOG_FUNC();
//
//        if (m_isLinked)
//        {
//            LOG_ERROR("Can't set leaky-type for AppSourceBintr '" 
//                << GetName() << "' as it's currently in a linked state");
//            return false;
//        }
//
//        m_leakyType = leakyType;
//        m_pSourceElement->SetAttribute("leaky-type", m_leakyType);
//        return true;
//    }


    static void on_need_data_cb(GstElement source, uint length,
        gpointer pAppSrcBintr)
    {
        static_cast<AppSourceBintr*>(pAppSrcBintr)->
            HandleNeedData(length);
    }
        
    static void on_enough_data_cb(GstElement source, 
        gpointer pAppSrcBintr)
    {
        static_cast<AppSourceBintr*>(pAppSrcBintr)->
            HandleEnoughData();
    }
        
    //*********************************************************************************
    // Initilize the unique id list for all CsiSourceBintrs 
    std::list<uint> CsiSourceBintr::s_uniqueSensorIds;

    CsiSourceBintr::CsiSourceBintr(const char* name, 
        guint width, guint height, guint fpsN, guint fpsD)
        : SourceBintr(name)
        , m_sensorId(0)
    {
        LOG_FUNC();

        // Media type is fixed to "video/x-raw"
        std::wstring L_mediaType(DSL_MEDIA_TYPE_VIDEO_XRAW);
        m_mediaType.assign(L_mediaType.begin(), L_mediaType.end());

        // Set the buffer-out-format to the default video format
        std::wstring L_bufferOutFormat(DSL_VIDEO_FORMAT_DEFAULT);
        m_bufferOutFormat.assign(L_bufferOutFormat.begin(), 
            L_bufferOutFormat.end());

        m_width = width;
        m_height = height;
        m_fpsN = fpsN;
        m_fpsD = fpsD;

        // Find the first available unique sensor-id
        while(std::find(s_uniqueSensorIds.begin(), s_uniqueSensorIds.end(), 
            m_sensorId) != s_uniqueSensorIds.end())
        {
            m_sensorId++;
        }
        s_uniqueSensorIds.push_back(m_sensorId);
        
        m_pSourceElement = DSL_ELEMENT_NEW("nvarguscamerasrc", name);
        m_pSourceCapsFilter = DSL_ELEMENT_EXT_NEW("capsfilter", name, "1");

        m_pSourceElement->SetAttribute("sensor-id", m_sensorId);
        m_pSourceElement->SetAttribute("bufapi-version", TRUE);

        // Set the full capabilities (format, dimensions, and framerate)
        // Note: nvarguscamerasrc supports NV12 and P010_10LE formats only.
        if (!set_full_caps(m_pSourceCapsFilter, m_mediaType.c_str(), "NV12",
            m_width, m_height, m_fpsN, m_fpsD, true))
        {
            throw;
        }

        // Get property defaults that aren't specifically set
        m_pSourceElement->GetAttribute("do-timestamp", &m_doTimestamp);

//        // ---- Video Converter Setup
        
        LOG_INFO("");
        LOG_INFO("Initial property values for CsiSourceBintr '" << name << "'");
        LOG_INFO("  is-live           : " << m_isLive);
        LOG_INFO("  do-timestamp      : " << m_doTimestamp);
        LOG_INFO("  sensor-id         : " << m_sensorId);
        LOG_INFO("  bufapi-version    : " << TRUE);
        LOG_INFO("  media             : " << m_mediaType << "(memory:NVMM)");
        LOG_INFO("  buffer-out-format : " << m_bufferOutFormat.c_str());
        LOG_INFO("  width             : " << m_width);
        LOG_INFO("  height            : " << m_height);
        LOG_INFO("  framerate         : " << m_fpsN << "/" << m_fpsD);

        AddChild(m_pSourceElement);
        AddChild(m_pSourceCapsFilter);
    }

    CsiSourceBintr::~CsiSourceBintr()
    {
        LOG_FUNC();
        
        s_uniqueSensorIds.remove(m_sensorId);
    }
    
    bool CsiSourceBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("CsiSourceBintr '" << GetName() << "' is already in a linked state");
            return false;
        }
        if (!m_pSourceElement->LinkToSink(m_pSourceCapsFilter) or
            !m_pSourceCapsFilter->LinkToSink(m_pBufferOutVidConv) or
            !m_pBufferOutVidConv->LinkToSink(m_pBufferOutCapsFilter))
        {
            return false;
        }
        m_isLinked = true;
        
        return true;
    }

    void CsiSourceBintr::UnlinkAll()
    {
        LOG_FUNC();

        if (!m_isLinked)
        {
            LOG_ERROR("CsiSourceBintr '" << GetName() << "' is not in a linked state");
            return;
        }
        m_pSourceElement->UnlinkFromSink();
        m_pSourceCapsFilter->UnlinkFromSink();
        m_pBufferOutVidConv->UnlinkFromSink();
        
        m_isLinked = false;
    }
    
    uint CsiSourceBintr::GetSensorId()
    {
        LOG_FUNC();

        return m_sensorId;
    }

    bool CsiSourceBintr::SetSensorId(uint sensorId)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't set sensor-id for CsiSourceBintr '" << GetName() 
                << "' as it is currently in a linked state");
            return false;
        }
        if (m_sensorId == sensorId)
        {
            LOG_WARN("sensor-id for CsiSourceBintr '" << GetName()
                << "' is already set to " << sensorId);
        }
        // Ensure that the sensor-id is unique.
        if(std::find(s_uniqueSensorIds.begin(), s_uniqueSensorIds.end(), 
            sensorId) != s_uniqueSensorIds.end())
        {
            LOG_ERROR("Can't set sensor-id = " << sensorId 
                << " for CsiSourceBintr '" << GetName() 
                << "'. The id is not unqiue");
            return false;
        }

        // remove the old sensor-id from the uiniue id list before updating
        s_uniqueSensorIds.remove(m_sensorId);

        m_sensorId = sensorId;
        s_uniqueSensorIds.push_back(m_sensorId);
        m_pSourceElement->SetAttribute("sensor-id", m_sensorId);
        
        return true;
    }

    //*********************************************************************************
    // Initilize the unique device id list for all UsbSourceBintrs 
    std::list<uint> UsbSourceBintr::s_uniqueDeviceIds;
    std::list<std::string> UsbSourceBintr::s_deviceLocations;

    UsbSourceBintr::UsbSourceBintr(const char* name, 
        guint width, guint height, guint fpsN, guint fpsD)
        : SourceBintr(name)
        , m_deviceId(0)
    {
        LOG_FUNC();

        // Media type is fixed to "video/x-raw"
        std::wstring L_mediaType(DSL_MEDIA_TYPE_VIDEO_XRAW);
        m_mediaType.assign(L_mediaType.begin(), L_mediaType.end());

        // Set the buffer-out-format to the default video format
        std::wstring L_bufferOutFormat(DSL_VIDEO_FORMAT_DEFAULT);
        m_bufferOutFormat.assign(L_bufferOutFormat.begin(), 
            L_bufferOutFormat.end());

        // Update the frame dimensions and framerate
        m_width = width;
        m_height = height;
        m_fpsN = fpsN;
        m_fpsD = fpsD;
        
        m_pSourceElement = DSL_ELEMENT_NEW("v4l2src", name);

        // Find the first available unique device-id
        while(std::find(s_uniqueDeviceIds.begin(), s_uniqueDeviceIds.end(), 
            m_deviceId) != s_uniqueDeviceIds.end())
        {
            m_deviceId++;
        }
        s_uniqueDeviceIds.push_back(m_deviceId);
        
        // create the device-location by adding the device-id as suffex to /dev/video
        m_deviceLocation = "/dev/video" + std::to_string(m_deviceId);
        s_deviceLocations.push_back(m_deviceLocation);
        
        LOG_INFO("Setting device-location = '" << m_deviceLocation 
            << "' for UsbSourceBintr '" << name << "'");

        m_pSourceElement->SetAttribute("device", m_deviceLocation.c_str());

        // Get property defaults that aren't specifically set
        m_pSourceElement->GetAttribute("do-timestamp", &m_doTimestamp);

        if (!m_cudaDeviceProp.integrated)
        {
            m_pdGpuVidConv = DSL_ELEMENT_EXT_NEW("nvvideoconvert", name, "1");
            AddChild(m_pdGpuVidConv);
        }
        
        LOG_INFO("");
        LOG_INFO("Initial property values for UsbSourceBintr '" << name << "'");
        LOG_INFO("  is-live           : " << m_isLive);
        LOG_INFO("  do-timestamp      : " << m_doTimestamp);
        LOG_INFO("  device            : " << m_deviceLocation.c_str());
        LOG_INFO("  media             : " << m_mediaType << "(memory:NVMM)");
        LOG_INFO("  buffer-out-format : " << m_bufferOutFormat.c_str());
        LOG_INFO("  width             : " << m_width);
        LOG_INFO("  height            : " << m_height);
        LOG_INFO("  framerate         : " << m_fpsN << "/" << m_fpsD);

        AddChild(m_pSourceElement);
    }

    UsbSourceBintr::~UsbSourceBintr()
    {
        LOG_FUNC();
        
        // remove from lists so values can be reused by next
        // new USB Source
        s_uniqueDeviceIds.remove(m_deviceId);
        s_deviceLocations.remove(m_deviceLocation);
    }

    bool UsbSourceBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("UsbSourceBintr '" << GetName() << "' is already in a linked state");
            return false;
        }
        
        // x86_64
        if (!m_cudaDeviceProp.integrated)
        {
            if (!m_pSourceElement->LinkToSink(m_pdGpuVidConv) or 
                !m_pdGpuVidConv->LinkToSink(m_pBufferOutVidConv) or
                !m_pBufferOutVidConv->LinkToSink(m_pBufferOutCapsFilter))
            {
                return false;
            }
        }
        else // aarch_64
        {
            if (!m_pSourceElement->LinkToSink(m_pBufferOutVidConv) or 
                !m_pBufferOutVidConv->LinkToSink(m_pBufferOutCapsFilter))
            {
                return false;
            }
        }
        m_isLinked = true;
        
        return true;
    }

    void UsbSourceBintr::UnlinkAll()
    {
        LOG_FUNC();

        if (!m_isLinked)
        {
            LOG_ERROR("UsbSourceBintr '" << GetName() << "' is not in a linked state");
            return;
        }
        
        // x86_64
        m_pSourceElement->UnlinkFromSink();
        if (!m_cudaDeviceProp.integrated)
        {
            m_pdGpuVidConv->UnlinkFromSink();
        }
        m_pBufferOutVidConv->UnlinkFromSink();
        m_isLinked = false;
    }

    const char* UsbSourceBintr::GetDeviceLocation()
    {
        LOG_FUNC();

        return m_deviceLocation.c_str();
    }
    
    bool UsbSourceBintr::SetDeviceLocation(const char* deviceLocation)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't set device-location for UsbSourceBintr '" << GetName() 
                << "' as it is currently in a linked state");
            return false;
        }
        
        // Ensure that the device-location is unique.
        std::string newLocation(deviceLocation);
        
        if (newLocation.find("/dev/video") == std::string::npos)
        {
            LOG_ERROR("Can't set device-location = '" << deviceLocation 
                << "' for UsbSourceBintr '" << GetName() 
                << "'. The string is invalid");
            return false;
        }
        uint newDeviceId(0);
        try
        {
            newDeviceId = std::stoi(newLocation.substr(10, 
                newLocation.size()-10));
        }
        catch(...)
        {
            LOG_ERROR("Can't set device-location = '" << deviceLocation 
                << "' for UsbSourceBintr '" << GetName() 
                << "'. The string is invalid");
            return false;
        }
        
        if(std::find(s_uniqueDeviceIds.begin(), s_uniqueDeviceIds.end(), 
            newDeviceId) != s_uniqueDeviceIds.end())
        {
            LOG_ERROR("Can't set device-location = '" << deviceLocation 
                << "' for UsbSourceBintr '" << GetName() 
                << "'. The location string is not unqiue");
            return false;
        }
        // remove the old device-id and location before updating
        s_uniqueDeviceIds.remove(m_deviceId);
        s_deviceLocations.remove(m_deviceLocation);

        m_deviceId = newDeviceId;
        m_deviceLocation = deviceLocation;
        s_uniqueDeviceIds.push_back(m_deviceId);
        s_deviceLocations.push_back(m_deviceLocation);
        
        m_pSourceElement->SetAttribute("device", deviceLocation);
        return true;
    }

    //*********************************************************************************

    UriSourceBintr::UriSourceBintr(const char* name, const char* uri, bool isLive,
        uint skipFrames, uint dropFrameInterval)
        : ResourceSourceBintr(name, uri)
        , m_numExtraSurfaces(DSL_DEFAULT_NUM_EXTRA_SURFACES)
        , m_skipFrames(skipFrames)
        , m_dropFrameInterval(dropFrameInterval)
        , m_accumulatedBase(0)
        , m_prevAccumulatedBase(0)
        , m_pDecoderStaticSinkpad(NULL)
        , m_bufferProbeId(0)
        , m_repeatEnabled(false)
    {
        LOG_FUNC();
        
        m_isLive = isLive;
        
        // Initialize the mutex regardless of IsLive or not
        g_mutex_init(&m_repeatEnabledMutex);

        m_pSourceElement = DSL_ELEMENT_NEW("uridecodebin", name);
        
        if (!SetUri(uri))
        {   
            throw;
        }

        // New Elementrs for this Source
        m_pSourceQueue = DSL_ELEMENT_EXT_NEW("queue", name, "src");

        // Connect UIR Source Setup Callbacks
        g_signal_connect(m_pSourceElement->GetGObject(), "pad-added", 
            G_CALLBACK(UriSourceElementOnPadAddedCB), this);
        g_signal_connect(m_pSourceElement->GetGObject(), "child-added", 
            G_CALLBACK(OnChildAddedCB), this);
        g_object_set_data(G_OBJECT(m_pSourceElement->GetGObject()), "source", this);

        g_signal_connect(m_pSourceElement->GetGObject(), "source-setup",
            G_CALLBACK(OnSourceSetupCB), this);

        LOG_INFO("");
        LOG_INFO("Initial property values for UriSourceBintr '" << name << "'");
        LOG_INFO("  uri                 : " << m_uri);
        LOG_INFO("  is-live             : " << m_isLive);
        LOG_INFO("  skip-frames         : " << m_skipFrames);
        LOG_INFO("  drop-frame-interval : " << m_dropFrameInterval);

        // Add all new Elementrs as Children to the SourceBintr
        AddChild(m_pSourceElement);
        AddChild(m_pSourceQueue);
        
        // Source Ghost Pad for Source Queue
        m_pSourceQueue->AddGhostPadToParent("src");

        std::string padProbeName = GetName() + "-src-pad-probe";
        m_pSrcPadProbe = DSL_PAD_BUFFER_PROBE_NEW(padProbeName.c_str(), 
            "src", m_pSourceQueue);
    }

    UriSourceBintr::~UriSourceBintr()
    {
        LOG_FUNC();

        g_mutex_clear(&m_repeatEnabledMutex);
    }

    bool UriSourceBintr::SetUri(const char* uri)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set Uri for UriSourceBintr '" << GetName() 
                << "' as it's currently Linked");
            return false;
        }
        // if it's a file source, 
        std::string newUri(uri);
        
        if ((newUri.find("http") == std::string::npos))
        {
            // Setup the absolute File URI and query dimensions
            if (!SetFileUri(uri))
            {
                LOG_ERROR("URI Source'" << uri << "' Not found");
                return false;
            }
        }        
        LOG_INFO("URI Path for File Source '" << GetName() << "' = " << m_uri);
        
        if (m_uri.size())
        {
            m_pSourceElement->SetAttribute("uri", m_uri.c_str());
        }
        
        return true;
    }

    bool UriSourceBintr::SetFileUri(const char* uri)
    {
        LOG_FUNC();

        std::string testUri(uri);
        if (testUri.empty())
        {
            LOG_INFO("File Path for SourceBintr '" << GetName() 
                << "' is empty. Source is in a non playable state");
            return true;
        }

        std::ifstream streamUriFile(uri);
        if (!streamUriFile.good())
        {
            LOG_ERROR("File Source '" << uri << "' Not found");
            return false;
        }
        // File source, not live - setup full path
        char absolutePath[PATH_MAX+1];
        m_uri.assign(realpath(uri, absolutePath));
        m_uri.insert(0, "file:");

        LOG_INFO("File Path = " << m_uri);
        
        // use openCV to open the file and read the Frame width and height properties.
        cv::VideoCapture vidCap;
        vidCap.open(uri, cv::CAP_ANY);

        if (!vidCap.isOpened())
        {
            LOG_ERROR("Failed to open File '" << uri 
                << "' for VideoRenderPlayerBintr '" << GetName() << "'");
            return false;
        }
        m_width = vidCap.get(cv::CAP_PROP_FRAME_WIDTH);
        m_height = vidCap.get(cv::CAP_PROP_FRAME_HEIGHT);
        
        // Note: the m_fpsN and m_fpsD can be calculated from cv.CAP_PROP_FPS
        // if needed prior to playing the file.
        return true;
    }

    bool UriSourceBintr::LinkAll()
    {
        LOG_FUNC();

        if (IsLinked())
        {
            LOG_ERROR("UriSourceBintr '" << GetName() << "' is already in a linked state");
            return false;
        }


        m_isLinked = true;

        return true;
    }

    void UriSourceBintr::UnlinkAll()
    {
        LOG_FUNC();
    
        if (!m_isLinked)
        {
            LOG_ERROR("UriSourceBintr '" << GetName() << "' is not in a linked state");
            return;
        }

        if (HasDewarperBintr())
        {
        }
        else
        {
        }
         
        m_isLinked = false;
    }
    
    void UriSourceBintr::HandleSourceElementOnPadAdded(GstElement* pBin, GstPad* pPad)
    {
        LOG_FUNC();

        // The "pad-added" callback will be called twice for each URI source,
        // once each for the decoded Audio and Video streams. Since we only 
        // want to link to the Video source pad, we need to know which of the
        // two streams this call is for.
        GstCaps* pCaps = gst_pad_query_caps(pPad, NULL);
        GstStructure* structure = gst_caps_get_structure(pCaps, 0);
        std::string name = gst_structure_get_name(structure);
        
        LOG_INFO("Caps structs name " << name);
        if (name.find("video") != std::string::npos)
        {
            m_pGstStaticSinkPad = gst_element_get_static_pad(m_pSourceQueue->GetGstElement(), "sink");
            if (!m_pGstStaticSinkPad)
            {
                LOG_ERROR("Failed to get Static Source Pad for Streaming Source '" 
                    << GetName() << "'");
            }
            
            if (gst_pad_link(pPad, m_pGstStaticSinkPad) != GST_PAD_LINK_OK) 
            {
                LOG_ERROR("Failed to link decodebin to source Tee");
                throw;
            }
            
            // Update the cap memebers for this URI Source Bintr
            gst_structure_get_uint(structure, "width", &m_width);
            gst_structure_get_uint(structure, "height", &m_height);
            gst_structure_get_fraction(structure, "framerate", (gint*)&m_fpsN, (gint*)&m_fpsD);
            
            LOG_INFO("Video decode linked for URI source '" << GetName() << "'");
        }
    }

    void UriSourceBintr::HandleOnChildAdded(GstChildProxy* pChildProxy, GObject* pObject,
        gchar* name)
    {
        LOG_FUNC();
        
        std::string strName = name;

        LOG_INFO("Child object with name '" << strName << "' added");
        
        if (strName.find("decodebin") != std::string::npos)
        {
            g_signal_connect(G_OBJECT(pObject), "child-added",
                G_CALLBACK(OnChildAddedCB), this);
        }

        else if ((strName.find("omx") != std::string::npos))
        {
            if (m_skipFrames)
            {
                g_object_set(pObject, "skip-frames", m_skipFrames, NULL);
            }
            g_object_set(pObject, "disable-dvfs", TRUE, NULL);
        }

        else if (strName.find("nvjpegdec") != std::string::npos)
        {
            g_object_set(pObject, "DeepStream", TRUE, NULL);
        }

        else if ((strName.find("nvv4l2decoder") != std::string::npos))
        {
            LOG_INFO("setting properties for child '" << strName << "'");
            
            if (m_skipFrames)
            {
                g_object_set(pObject, "skip-frames", m_skipFrames, NULL);
            }
            // aarch64 only
            if (m_cudaDeviceProp.integrated)
            {
                g_object_set(pObject, "enable-max-performance", TRUE, NULL);
            }
            g_object_set(pObject, "drop-frame-interval", m_dropFrameInterval, NULL);
            g_object_set(pObject, "num-extra-surfaces", m_numExtraSurfaces, NULL);

            // if the source is from file, then setup Stream buffer probe function
            // to handle the stream restart/loop on GST_EVENT_EOS.
            if (!m_isLive and m_repeatEnabled)
            {
                GstPadProbeType mask = (GstPadProbeType) 
                    (GST_PAD_PROBE_TYPE_EVENT_BOTH |
                    GST_PAD_PROBE_TYPE_EVENT_FLUSH | 
                    GST_PAD_PROBE_TYPE_BUFFER);
                    
                m_pDecoderStaticSinkpad = 
                    gst_element_get_static_pad(GST_ELEMENT(pObject), "sink");
                
                m_bufferProbeId = gst_pad_add_probe(m_pDecoderStaticSinkpad, 
                    mask, StreamBufferRestartProbCB, this, NULL);
            }
        }
    }
    
    GstPadProbeReturn UriSourceBintr::HandleStreamBufferRestart(GstPad* pPad, 
        GstPadProbeInfo* pInfo)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_repeatEnabledMutex);
        
        GstEvent* event = GST_EVENT(pInfo->data);

        if (pInfo->type & GST_PAD_PROBE_TYPE_BUFFER)
        {
            GST_BUFFER_PTS(GST_BUFFER(pInfo->data)) += m_prevAccumulatedBase;
        }
        
        if (pInfo->type & GST_PAD_PROBE_TYPE_EVENT_BOTH)
        {
            if (GST_EVENT_TYPE(event) == GST_EVENT_EOS)
            {
                g_timeout_add(1, StreamBufferSeekCB, this);
            }
            if (GST_EVENT_TYPE(event) == GST_EVENT_SEGMENT)
            {
                GstSegment* segment;

                gst_event_parse_segment(event, (const GstSegment**)&segment);
                segment->base = m_accumulatedBase;
                m_prevAccumulatedBase = m_accumulatedBase;
                m_accumulatedBase += segment->stop;
            }
            switch (GST_EVENT_TYPE (event))
            {
            case GST_EVENT_EOS:
            // QOS events from downstream sink elements cause decoder to drop
            // frames after looping the file since the timestamps reset to 0.
            // We should drop the QOS events since we have custom logic for
            // looping individual sources.
            case GST_EVENT_QOS:
            case GST_EVENT_SEGMENT:
            case GST_EVENT_FLUSH_START:
            case GST_EVENT_FLUSH_STOP:
                return GST_PAD_PROBE_DROP;
            default:
                break;
            }
        }
        return GST_PAD_PROBE_OK;
    }

    void UriSourceBintr::HandleOnSourceSetup(GstElement* pObject, GstElement* arg0)
    {
        if (g_object_class_find_property(G_OBJECT_GET_CLASS(arg0), "latency")) 
        {
            g_object_set(G_OBJECT(arg0), "latency", "cb_sourcesetup set %d latency\n", NULL);
        }
    }
    
    gboolean UriSourceBintr::HandleStreamBufferSeek()
    {
        SetState(GST_STATE_PAUSED, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND);
        
        gboolean retval = gst_element_seek(GetGstElement(), 1.0, GST_FORMAT_TIME,
            (GstSeekFlags)(GST_SEEK_FLAG_KEY_UNIT | GST_SEEK_FLAG_FLUSH),
            GST_SEEK_TYPE_SET, 0, GST_SEEK_TYPE_NONE, GST_CLOCK_TIME_NONE);

        if (!retval)
        {
            LOG_WARN("Failure to seek");
        }

        SetState(GST_STATE_PLAYING, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND);
        return false;
    }

    
    bool UriSourceBintr::AddDewarperBintr(DSL_BASE_PTR pDewarperBintr)
    {
        LOG_FUNC();
        
        if (m_pDewarperBintr)
        {
            LOG_ERROR("Source '" << GetName() << "' allready has a Dewarper");
            return false;
        }
        m_pDewarperBintr = std::dynamic_pointer_cast<DewarperBintr>(pDewarperBintr);
        AddChild(pDewarperBintr);
        return true;
    }

    bool UriSourceBintr::RemoveDewarperBintr()
    {
        LOG_FUNC();

        if (!m_pDewarperBintr)
        {
            LOG_ERROR("Source '" << GetName() << "' does not have a Dewarper");
            return false;
        }
        RemoveChild(m_pDewarperBintr);
        m_pDewarperBintr = nullptr;
        return true;
    }
    
    bool UriSourceBintr::HasDewarperBintr()
    {
        LOG_FUNC();
        
        return (m_pDewarperBintr != nullptr);
    }
    
    void UriSourceBintr::DisableEosConsumer()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_repeatEnabledMutex);
        
        if (m_pDecoderStaticSinkpad)
        {
            if (m_bufferProbeId)
            {
                gst_pad_remove_probe(m_pDecoderStaticSinkpad, m_bufferProbeId);
            }
            gst_object_unref(m_pDecoderStaticSinkpad);
        }
    }
    
    //*********************************************************************************

    FileSourceBintr::FileSourceBintr(const char* name, 
        const char* uri, bool repeatEnabled)
        : UriSourceBintr(name, uri, false, false, 0)
    {
        LOG_FUNC();
        
        // override the default
        m_repeatEnabled = repeatEnabled;
    }
    
    FileSourceBintr::~FileSourceBintr()
    {
        LOG_FUNC();
    }

    bool FileSourceBintr::SetUri(const char* uri)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set File Path for FileSourceBintr '" << GetName() 
                << "' as it's currently linked");
            return false;
        }
        
        if (!SetFileUri(uri))
        {
            return false;
        }
        if (m_uri.size())
        {
            m_pSourceElement->SetAttribute("uri", m_uri.c_str());
        }
        return true;
    }
    
    bool FileSourceBintr::GetRepeatEnabled()
    {
        LOG_FUNC();
        
        return m_repeatEnabled;
    }

    bool FileSourceBintr::SetRepeatEnabled(bool enabled)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Cannot set Repeat Enabled for Source '" << GetName() 
                << "' as it is currently Linked");
            return false;
        }
        
        m_repeatEnabled = enabled;
        return true;
    }

    //*********************************************************************************

    ImageSourceBintr::ImageSourceBintr(const char* name, const char* uri, uint type)
        : ResourceSourceBintr(name, uri)
        , m_mjpeg(FALSE)
    {
        LOG_FUNC();
        
        // override the default source attributes
        m_isLive = False;

        // Media type is fixed to "video/x-raw"
        std::wstring L_mediaType(DSL_MEDIA_TYPE_VIDEO_XRAW);
        m_mediaType.assign(L_mediaType.begin(), L_mediaType.end());

        // Set the buffer-out-format to the default video format
        std::wstring L_bufferOutFormat(DSL_VIDEO_FORMAT_DEFAULT);
        m_bufferOutFormat.assign(L_bufferOutFormat.begin(), 
            L_bufferOutFormat.end());

        // Other components are created conditionaly by file type. 
        if (m_uri.find("jpeg") != std::string::npos or
            m_uri.find("jpg") != std::string::npos)
        {
            LOG_INFO("Setting file format to JPG for ImageSourceBintr '" 
                << GetName() << "'");
            m_format = DSL_IMAGE_FORMAT_JPG;
            m_ext = DSL_IMAGE_EXT_JPG;
            m_pParser = DSL_ELEMENT_NEW("jpegparse", name);
            m_pDecoder = DSL_ELEMENT_NEW("nvv4l2decoder", name); 

            // ---- Video Converter Setup
            
            m_pVidConv = DSL_ELEMENT_NEW("nvvideoconvert", name);
            
            if (!m_cudaDeviceProp.integrated)
            {
                m_pVidConv->SetAttribute("nvbuf-memory-type", 
                    DSL_NVBUF_MEM_TYPE_UNIFIED);
            }
            
            // ---- Caps Filter Setup

            m_pVidConvCapsFilter = DSL_ELEMENT_NEW("capsfilter", name);

            // Set the buffer-out-format to the default
            if (!set_format_caps(m_pVidConvCapsFilter, m_mediaType.c_str(), 
                m_bufferOutFormat.c_str(), true))
            {
                throw;
            }
            
            AddChild(m_pParser);
            AddChild(m_pDecoder);
            AddChild(m_pVidConv);
            AddChild(m_pVidConvCapsFilter);

            m_pVidConvCapsFilter->AddGhostPadToParent("src");
            
            std::string padProbeName = GetName() + "-src-pad-probe";
            m_pSrcPadProbe = DSL_PAD_BUFFER_PROBE_NEW(padProbeName.c_str(), 
                "src", m_pVidConvCapsFilter);

            // If it's an MJPG file or Multi JPG files
            if (m_uri.find("mjpeg") != std::string::npos or
                m_uri.find("mjpg") != std::string::npos or
                type == DSL_IMAGE_TYPE_MULTI)
            {
                LOG_INFO("Setting decoder 'mjpeg' attribute for ImageSourceBintr '" 
                    << GetName() << "'");
                m_mjpeg = TRUE;
                m_pDecoder->SetAttribute("mjpeg", m_mjpeg);
            }
            
        }
        else if (m_uri.find(".png") != std::string::npos)
        {
            LOG_ERROR("Unsuported file type (.png ) '" << m_uri 
                << "' for new Image Source '" << name << "'");
            throw;
        }
        else
        {
            LOG_ERROR("Invalid file type = '" << m_uri 
                << "' for new Image Source '" << name << "'");
            throw;
        }
    }
    
    ImageSourceBintr::~ImageSourceBintr()
    {
        LOG_FUNC();
    }

    //*********************************************************************************

    SingleImageSourceBintr::SingleImageSourceBintr(const char* name, const char* uri)
        : ImageSourceBintr(name, uri, DSL_IMAGE_TYPE_SINGLE)
    {
        LOG_FUNC();
        
        m_pSourceElement = DSL_ELEMENT_NEW("filesrc", name);
        
        if (!SetUri(uri))
        {
            throw;
        }
        AddChild(m_pSourceElement);

        LOG_INFO("");
        LOG_INFO("Initial property values for SingleImageSourceBintr '" << name << "'");
        LOG_INFO("  Elements");
        LOG_INFO("    Source          : " << m_pSourceElement->GetFactoryName());
        LOG_INFO("    Parser          : " << m_pParser->GetFactoryName());
        LOG_INFO("    Decoder         : " << m_pDecoder->GetFactoryName());
        LOG_INFO("  location          : " << uri);
        LOG_INFO("  is-live           : " << m_isLive);
        LOG_INFO("  media in          : " << "image/jpeg");
        LOG_INFO("  media out         : " << m_mediaType << "(memory:NVMM)");
        LOG_INFO("  buffer-out-format : " << m_bufferOutFormat.c_str());
        LOG_INFO("  framerate         : " << m_fpsN << "/" << m_fpsD);
        LOG_INFO("  mjpeg             : " << m_mjpeg);
    }
    
    SingleImageSourceBintr::~SingleImageSourceBintr()
    {
        LOG_FUNC();
    }

    bool SingleImageSourceBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("SingleImageSourceBintr '" << GetName() 
                << "' is already in a linked state");
            return false;
        }
        if (!IsLinkable())
        {
            LOG_ERROR("Unable to Link SingleImageSourceBintr '" << GetName() 
                << "' as its uri has not been set");
            return false;
        }
        if (!m_pSourceElement->LinkToSink(m_pParser) or
            !m_pParser->LinkToSink(m_pDecoder) or
            !m_pDecoder->LinkToSink(m_pVidConv) or
            !m_pVidConv->LinkToSink(m_pVidConvCapsFilter))
        {
            LOG_ERROR("SingleImageSourceBintr '" << GetName() 
                << "' failed to LinkAll");
            return false;
        }
        m_isLinked = true;
        
        return true;
    }

    void SingleImageSourceBintr::UnlinkAll()
    {
        LOG_FUNC();

        if (!m_isLinked)
        {
            LOG_ERROR("SingleImageSourceBintr '" << GetName() 
                << "' is not in a linked state");
            return;
        }
        if (!m_pSourceElement->UnlinkFromSink() or
            !m_pParser->UnlinkFromSink() or
            !m_pDecoder->UnlinkFromSink() or
            !m_pVidConv->UnlinkFromSink())
        {
            LOG_ERROR("SingleImageSourceBintr '" << GetName() 
                << "' failed to UnlinkAll");
            return;
        }    
        m_isLinked = false;
    }

    bool SingleImageSourceBintr::SetUri(const char* uri)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set File Path for ImageFrameSourceBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        
        std::string pathString(uri);
        if (pathString.empty())
        {
            LOG_INFO("File Path for ImageFrameSourceBintr '" << GetName() 
                << "' is empty. Source is in a non playable state");
            return true;
        }
        
        std::ifstream streamUriFile(uri);
        if (!streamUriFile.good())
        {
            LOG_ERROR("Image Source'" << uri << "' Not found");
            return false;
        }
        // File source, not live - setup full path
        char absolutePath[PATH_MAX+1];
        m_uri.assign(realpath(uri, absolutePath));

        // Use OpenCV to determine the new image dimensions
        cv::Mat image = imread(m_uri, cv::IMREAD_COLOR);
        cv::Size imageSize = image.size();
        m_width = imageSize.width;
        m_height = imageSize.height;

        // Set the filepath for the File Source Elementr
        m_pSourceElement->SetAttribute("location", m_uri.c_str());

        return true;
            
    }

    //*********************************************************************************

    MultiImageSourceBintr::MultiImageSourceBintr(const char* name, 
        const char* uri, uint fpsN, uint fpsD)
        : ImageSourceBintr(name, uri, DSL_IMAGE_TYPE_MULTI)
        , m_loopEnabled(false)
        , m_startIndex(0)
        , m_stopIndex(-1)
    {
        LOG_FUNC();
        
        // override the default source attributes
        m_fpsN = fpsN;
        m_fpsD = fpsD;

        m_pSourceElement = DSL_ELEMENT_NEW("multifilesrc", name);

        GstCaps * pCaps = gst_caps_new_simple("image/jpeg", "framerate", 
            GST_TYPE_FRACTION, m_fpsN, m_fpsD, NULL);
        if (!pCaps)
        {
            LOG_ERROR("Failed to create new Simple Capabilities for '" 
                << name << "'");
            throw;  
        }

        m_pSourceElement->SetAttribute("caps", pCaps);
        m_pSourceElement->SetAttribute("loop", m_loopEnabled);
        m_pSourceElement->SetAttribute("start-index", m_startIndex);
        m_pSourceElement->SetAttribute("stop-index", m_stopIndex);
        
        gst_caps_unref(pCaps);        

        LOG_INFO("");
        LOG_INFO("Initial property values for MultiImageSourceBintr '" << name << "'");
        LOG_INFO("  Elements");
        LOG_INFO("    Source          : " << m_pSourceElement->GetFactoryName());
        LOG_INFO("    Parser          : " << m_pParser->GetFactoryName());
        LOG_INFO("    Decoder         : " << m_pDecoder->GetFactoryName());
        LOG_INFO("  location          : " << m_pParser->GetFactoryName());
        LOG_INFO("  is-live           : " << m_isLive);
        LOG_INFO("  media in          : " << "image/jpeg");
        LOG_INFO("  media out         : " << m_mediaType << "(memory:NVMM)");
        LOG_INFO("  buffer-out-format : " << m_bufferOutFormat.c_str());
        LOG_INFO("  framerate         : " << m_fpsN << "/" << m_fpsD);
        LOG_INFO("  loop              : " << m_loopEnabled);
        LOG_INFO("  start-index       : " << m_startIndex);
        LOG_INFO("  stop-index        : " << m_stopIndex);
        
        AddChild(m_pSourceElement);

        if (!SetUri(uri))
        {
            throw;
        }
    }
    
    MultiImageSourceBintr::~MultiImageSourceBintr()
    {
        LOG_FUNC();
    }

    bool MultiImageSourceBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("MultiImageSourceBintr '" << GetName() 
                << "' is already in a linked state");
            return false;
        }
        if (!IsLinkable())
        {
            LOG_ERROR("Unable to Link MultiImageSourceBintr '" << GetName() 
                << "' as its uri has not been set");
            return false;
        }
        if (!m_pSourceElement->LinkToSink(m_pParser) or
            !m_pParser->LinkToSink(m_pDecoder) or
            !m_pDecoder->LinkToSink(m_pVidConv) or
            !m_pVidConv->LinkToSink(m_pVidConvCapsFilter))
        {
            LOG_ERROR("MultiImageSourceBintr '" << GetName() 
                << "' failed to LinkAll");
            return false;
        }
        m_isLinked = true;
        
        return true;
    }

    void MultiImageSourceBintr::UnlinkAll()
    {
        LOG_FUNC();

        if (!m_isLinked)
        {
            LOG_ERROR("MultiImageSourceBintr '" << GetName() 
                << "' is not in a linked state");
            return;
        }
        
        if (!m_pSourceElement->UnlinkFromSink() or
            !m_pParser->UnlinkFromSink() or
            !m_pDecoder->UnlinkFromSink() or
            !m_pVidConv->UnlinkFromSink())
        {
            LOG_ERROR("MultiImageSourceBintr '" << GetName() 
                << "' failed to UnlinkAll");
            return;
        }    
        else
        {
            // TODO
        }
        m_isLinked = false;
    }

    bool MultiImageSourceBintr::SetUri(const char* uri)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set File Path for MultiImageSourceBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        
        std::string pathString(uri);
        if (pathString.empty())
        {
            LOG_INFO("File Path for MultiImageSourceBintr '" << GetName() 
                << "' is empty. Source is in a non playable state");
            return true;
        }
        
        m_uri.assign(uri);
        // Set the filepath for the File Source Elementr
        m_pSourceElement->SetAttribute("location", m_uri.c_str());

        return true;
            
    }

    bool MultiImageSourceBintr::GetLoopEnabled()
    {
        LOG_FUNC();
        
        return m_loopEnabled;
    }
    
    bool MultiImageSourceBintr::SetLoopEnabled(bool loopEnabled)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set loop-enabled for MultiImageSourceBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_loopEnabled = loopEnabled;
        m_pSourceElement->SetAttribute("loop", m_loopEnabled);
        return true;
    }

    void MultiImageSourceBintr::GetIndices(int* startIndex, int* stopIndex)
    {
        LOG_FUNC();
        
        *startIndex = m_startIndex;
        *stopIndex = m_stopIndex;
    }
    
    bool MultiImageSourceBintr::SetIndices(int startIndex, int stopIndex)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set indicies for MultiImageSourceBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_startIndex = startIndex;
        m_stopIndex = stopIndex;
        m_pSourceElement->SetAttribute("start-index", m_startIndex);
        m_pSourceElement->SetAttribute("stop-index", m_stopIndex);
        return true;
    }
        
    //*********************************************************************************

    ImageStreamSourceBintr::ImageStreamSourceBintr(const char* name, 
        const char* uri, bool isLive, uint fpsN, uint fpsD, uint timeout)
        : ResourceSourceBintr(name, uri)
        , m_timeout(timeout)
        , m_timeoutTimerId(0)
    {
        LOG_FUNC();
        
        // Media type is fixed to "video/x-raw"
        std::wstring L_mediaType(DSL_MEDIA_TYPE_VIDEO_XRAW);
        m_mediaType.assign(L_mediaType.begin(), L_mediaType.end());

        // Set the buffer-out-format to the default video format
        std::wstring L_bufferOutFormat(DSL_VIDEO_FORMAT_DEFAULT);
        m_bufferOutFormat.assign(L_bufferOutFormat.begin(), 
            L_bufferOutFormat.end());

        // override default values
        m_isLive = isLive;
        m_fpsN = fpsN;
        m_fpsD = fpsD;

        m_pSourceElement = DSL_ELEMENT_NEW("videotestsrc", name);
        m_pSourceCapsFilter = DSL_ELEMENT_EXT_NEW("capsfilter", name, "source");
        m_pImageOverlay = DSL_ELEMENT_NEW("gdkpixbufoverlay", name); 

        m_pSourceElement->SetAttribute("pattern", 2); // 2 = black
        
        // ---- Video Converter Setup

        m_pVidConv = DSL_ELEMENT_NEW("nvvideoconvert", name);

        m_pVidConv->SetAttribute("gpu-id", m_gpuId);
        m_pVidConv->SetAttribute("nvbuf-memory-type", m_nvbufMemType);

        // ---- Caps Filter Setup

        m_pVidConvCapsFilter = DSL_ELEMENT_EXT_NEW("capsfilter", name, "sink");

        // Set the buffer-out-format to the default
        if (!set_format_caps(m_pVidConvCapsFilter, m_mediaType.c_str(), 
            m_bufferOutFormat.c_str(), true))
        {
            throw;
        }

        LOG_INFO("");
        LOG_INFO("Initial property values for ImageStreamSourceBintr '" << name << "'");
        LOG_INFO("  Elements");
        LOG_INFO("    Source          : " << m_pSourceElement->GetFactoryName());
        LOG_INFO("    Overlay         : " << m_pImageOverlay->GetFactoryName());
        LOG_INFO("  location          : " << uri);
        LOG_INFO("  is-live           : " << m_isLive);
        LOG_INFO("  media             : " << m_mediaType << "(memory:NVMM)");
        LOG_INFO("  buffer-out-format : " << m_bufferOutFormat.c_str());
        LOG_INFO("  framerate         : " << m_fpsN << "/" << m_fpsD);

        // Add all new Elementrs as Children to the SourceBintr
        AddChild(m_pSourceElement);
        AddChild(m_pSourceCapsFilter);
        AddChild(m_pImageOverlay);
        AddChild(m_pVidConv);
        AddChild(m_pVidConvCapsFilter);
        
        // Source Ghost Pad for ImageStreamSourceBintr
        m_pVidConvCapsFilter->AddGhostPadToParent("src");

        std::string padProbeName = GetName() + "-src-pad-probe";
        m_pSrcPadProbe = DSL_PAD_BUFFER_PROBE_NEW(padProbeName.c_str(), 
            "src", m_pVidConvCapsFilter);

        g_mutex_init(&m_timeoutTimerMutex);

        if(uri and !SetUri(uri))
        {
            throw;
        }
    }
    
    ImageStreamSourceBintr::~ImageStreamSourceBintr()
    {
        LOG_FUNC();
        
        g_mutex_clear(&m_timeoutTimerMutex);
    }

    bool ImageStreamSourceBintr::SetUri(const char* uri)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set File Path for ImageStreamSourceBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }
        std::string pathString(uri);
        if (pathString.empty())
        {
            LOG_INFO("File Path for ImageStreamSourceBintr '" << GetName() 
                << "' is empty. Source is in a non playable state");
            return true;
        }
            
        std::ifstream streamUriFile(uri);
        if (!streamUriFile.good())
        {
            LOG_ERROR("Image Source'" << uri << "' Not found");
            return false;
        }
        // File source, not live - setup full path
        char absolutePath[PATH_MAX+1];
        m_uri.assign(realpath(uri, absolutePath));

        // Use OpenCV to determine the new image dimensions
        cv::Mat image = imread(m_uri, cv::IMREAD_COLOR);
        cv::Size imageSize = image.size();
        m_width = imageSize.width;
        m_height = imageSize.height;

        // Set the full capabilities (format and framerate)
        if (!set_full_caps(m_pSourceCapsFilter, m_mediaType.c_str(), 
            m_bufferOutFormat.c_str(), m_width, m_height, m_fpsN, m_fpsD, false))
        {
            return false;
        }
        // Set the filepath for the Image Overlay Elementr
        m_pImageOverlay->SetAttribute("location", m_uri.c_str());
        
        return true;
    }
    
    bool ImageStreamSourceBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("ImageStreamSourceBintr '" << GetName() << "' is already in a linked state");
            return false;
        }
        if (!m_pSourceElement->LinkToSink(m_pSourceCapsFilter) or
            !m_pSourceCapsFilter->LinkToSink(m_pImageOverlay) or
            !m_pImageOverlay->LinkToSink(m_pVidConv) or
            !m_pVidConv->LinkToSink(m_pVidConvCapsFilter))
        {
            LOG_ERROR("ImageStreamSourceBintr '" << GetName() << "' failed to LinkAll");
            return false;
        }
        m_isLinked = true;
        
        if (m_timeout)
        {
            m_timeoutTimerId = g_timeout_add(m_timeout*1000, 
                ImageSourceDisplayTimeoutHandler, this);
        }
        
        return true;
    }

    void ImageStreamSourceBintr::UnlinkAll()
    {
        LOG_FUNC();

        if (!m_isLinked)
        {
            LOG_ERROR("ImageStreamSourceBintr '" << GetName() << "' is not in a linked state");
            return;
        }
        if (m_timeoutTimerId)
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_timeoutTimerMutex);
            g_source_remove(m_timeoutTimerId);
            m_timeoutTimerId = 0;
        }
        
        if (!m_pSourceElement->UnlinkFromSink() or
            !m_pSourceCapsFilter->UnlinkFromSink() or
            !m_pImageOverlay->UnlinkFromSink() or
            !m_pVidConv->UnlinkFromSink())
        {
            LOG_ERROR("ImageStreamSourceBintr '" << GetName() << "' failed to UnlinkAll");
            return;
        }    
        m_isLinked = false;
    }
    
    int ImageStreamSourceBintr::HandleDisplayTimeout()
    {
        LOG_FUNC();
        
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_timeoutTimerMutex);

        // Send the EOS event to end the Image display
        SendEos();
        m_timeoutTimerId = 0;
        
        // Single shot - so don't restart
        return 0;
    }

    uint ImageStreamSourceBintr::GetTimeout()
    {
        LOG_FUNC();
        
        return m_timeout;
    }

    bool ImageStreamSourceBintr::SetTimeout(uint timeout)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Cannot set Timeout for Image Source '" << GetName() 
                << "' as it is currently Linked");
            return false;
        }
        
        m_timeout = timeout;
        return true;
    }

    //*********************************************************************************

    InterpipeSourceBintr::InterpipeSourceBintr(const char* name, 
        const char* listenTo, bool isLive, bool acceptEos, bool acceptEvents)
        : SourceBintr(name)
        , m_listenTo(listenTo)
        , m_acceptEos(acceptEos)
        , m_acceptEvents(acceptEvents)
    {
        LOG_FUNC();
        
        // we need to append the factory name to match the Inter-Pipe
        // sinks element name. 
        m_listenToFullName = m_listenTo + "-interpipesink";
        
        // override the default settings.
        m_isLive = isLive;
        
        m_pSourceElement = DSL_ELEMENT_NEW("interpipesrc", name);
        
        m_pSourceElement->SetAttribute("is-live", m_isLive);
        m_pSourceElement->SetAttribute("listen-to", m_listenToFullName.c_str());
        m_pSourceElement->SetAttribute("accept-eos-event", m_acceptEos);
        m_pSourceElement->SetAttribute("accept-events", m_acceptEvents);
        m_pSourceElement->SetAttribute("allow-renegotiation", TRUE);

        LOG_INFO("");
        LOG_INFO("Initial property values for InterpipeSourceBintr '" << name << "'");
        LOG_INFO("  is-live             : " << m_isLive);
        LOG_INFO("  listen-to           : " << m_listenTo);
        LOG_INFO("  accept-eos-event    : " << m_acceptEos);
        LOG_INFO("  accept-events       : " << m_acceptEvents);
        LOG_INFO("  allow-renegotiation : " << TRUE);

        // Add the new Elementr as a Child to the SourceBintr
        AddChild(m_pSourceElement);
        
        m_pSourceElement->AddGhostPadToParent("src");
        
        std::string padProbeName = GetName() + "-src-pad-probe";
        m_pSrcPadProbe = DSL_PAD_BUFFER_PROBE_NEW(padProbeName.c_str(), 
            "src", m_pSourceElement);
}
    
    InterpipeSourceBintr::~InterpipeSourceBintr()
    {
        LOG_FUNC();
    }

    const char* InterpipeSourceBintr::GetListenTo()
    {
        LOG_FUNC();
        
        return m_listenTo.c_str();
    }
    
    void InterpipeSourceBintr::SetListenTo(const char* listenTo)
    {
        m_listenTo = listenTo;
        m_listenToFullName = m_listenTo + "-interpipesink";
        
        m_pSourceElement->SetAttribute("listen-to", m_listenToFullName.c_str());
    }
    
    bool InterpipeSourceBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("InterpipeSourceBintr '" << GetName() 
                << "' is already in a linked state");
            return false;
        }
        // Single element nothing to link
        m_isLinked = true;
        return true;
    }

    void InterpipeSourceBintr::UnlinkAll()
    {
        LOG_FUNC();

        if (!m_isLinked)
        {
            LOG_ERROR("InterpipeSourceBintr '" << GetName() 
                << "' is not in a linked state");
            return;
        }
        // Single element nothing to link
        m_isLinked = false;
    }
    
    void InterpipeSourceBintr::GetAcceptSettings(bool* acceptEos, 
        bool* acceptEvents)
    {
        LOG_FUNC();
        
        *acceptEos = m_acceptEos;
        *acceptEvents = m_acceptEvents;
    }

    bool InterpipeSourceBintr::SetAcceptSettings(bool acceptEos, 
        bool acceptEvents)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set Accept setting for InterpipeSourceBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_acceptEos = acceptEos;
        m_acceptEvents = acceptEvents;
        
        m_pSourceElement->SetAttribute("accept-eos-event", m_acceptEos);
        m_pSourceElement->SetAttribute("accept-events", m_acceptEvents);
        
        return true;
    }
    
    //*********************************************************************************
    
    RtspSourceBintr::RtspSourceBintr(const char* name, const char* uri, 
        uint protocol, uint skipFrames, uint dropFrameInterval, 
        uint latency, uint timeout)
        : ResourceSourceBintr(name, uri)
        , m_skipFrames(skipFrames)
        , m_dropFrameInterval(dropFrameInterval)
        , m_rtpProtocols(protocol)
        , m_latency(latency)
        , m_bufferTimeout(timeout)
        , m_streamManagerTimerId(0)
        , m_reconnectionManagerTimerId(0)
        , m_connectionData{0}
        , m_reconnectionFailed(false)
        , m_reconnectionSleep(0)
        , m_reconnectionStartTime{0}
        , m_currentState(GST_STATE_NULL)
        , m_previousState(GST_STATE_NULL)
        , m_listenerNotifierTimerId(0)
    {
        m_isLive = true;

        // New RTSP Specific Elementrs for this Source
        m_pSourceElement = DSL_ELEMENT_NEW("rtspsrc", name);
        
        // Pre-decode tee is only used if there is a TapBintr
        m_pPreDecodeTee = DSL_ELEMENT_NEW("tee", name);
        m_pPreDecodeQueue = DSL_ELEMENT_EXT_NEW("queue", name, "decodebin");
        m_pSourceQueue = DSL_ELEMENT_EXT_NEW("queue", name, "src");

        // Configure the source to generate NTP sync values
        configure_source_for_ntp_sync(m_pSourceElement->GetGstElement());
        m_pSourceElement->SetAttribute("location", m_uri.c_str());

        m_pSourceElement->SetAttribute("latency", m_latency);
        m_pSourceElement->SetAttribute("drop-on-latency", true);
        m_pSourceElement->SetAttribute("protocols", m_rtpProtocols);

        g_signal_connect (m_pSourceElement->GetGObject(), "select-stream",
            G_CALLBACK(RtspSourceSelectStreamCB), this);

        // Connect RTSP Source Setup Callbacks
        g_signal_connect(m_pSourceElement->GetGObject(), "pad-added", 
            G_CALLBACK(RtspSourceElementOnPadAddedCB), this);

        LOG_INFO("");
        LOG_INFO("Initial property values for RtspSourceBintr '" << name << "'");
        LOG_INFO("  uri                 : " << m_uri);
        LOG_INFO("  is-live             : " << m_isLive);
        LOG_INFO("  skip-frames         : " << m_skipFrames);
        LOG_INFO("  drop-frame-interval : " << m_dropFrameInterval);

        AddChild(m_pSourceElement);
        AddChild(m_pPreDecodeTee);
        AddChild(m_pPreDecodeQueue);
        AddChild(m_pSourceQueue);

        // Source Ghost Pad for Source Queue as src pad to connect to streammuxer
        m_pSourceQueue->AddGhostPadToParent("src");
        
        // New timestamp PPH to stamp the time of the last buffer 
        // - used to monitor the RTSP connection
        std::string handlerName = GetName() + "-timestamp-pph";
        m_TimestampPph = DSL_PPH_TIMESTAMP_NEW(handlerName.c_str());
        
        std::string padProbeName = GetName() + "-src-pad-probe";
        m_pSrcPadProbe = DSL_PAD_BUFFER_PROBE_NEW(padProbeName.c_str(), 
            "src", m_pSourceQueue);
        m_pSrcPadProbe->AddPadProbeHandler(m_TimestampPph);
        
        g_mutex_init(&m_streamManagerMutex);
        g_mutex_init(&m_reconnectionManagerMutex);
        g_mutex_init(&m_stateChangeMutex);
        
        // Set the default connection param values
        m_connectionData.sleep = DSL_RTSP_RECONNECTION_SLEEP_S;
        m_connectionData.timeout = DSL_RTSP_RECONNECTION_TIMEOUT_S;
    }

    RtspSourceBintr::~RtspSourceBintr()
    {
        LOG_FUNC();
        
        if (m_reconnectionManagerTimerId)
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_reconnectionManagerMutex);
            g_source_remove(m_reconnectionManagerTimerId);
        }

        // Note: don't need t worry about stopping the one-shot m_listenerNotifierTimerId
        
        m_pSrcPadProbe->RemovePadProbeHandler(m_TimestampPph);
        
        g_mutex_clear(&m_streamManagerMutex);
        g_mutex_clear(&m_reconnectionManagerMutex);
        g_mutex_clear(&m_stateChangeMutex);
    }
    
    bool RtspSourceBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("RtspSourceBintr '" << GetName() 
                << "' is already in a linked state");
            return false;
        }

        // Note: this is a workaround for an NVIDIA bug. We need to test the 
        // stream beforewe try and link any pads. Otherwise, unlinking a failed 
        // stream connection from the Streammuxer will result in a deadlock. 
        // Try to open the URL with open CV first.
        cv::VideoCapture capture(m_uri.c_str());

        if (!capture.isOpened())
        {
            LOG_ERROR("RtspSourceBintr '" << GetName() 
                << "' failed to open stream for URI = "
                << m_uri.c_str());
            return false;
        }

        // All elements are linked in the select-stream callback (HandleSelectStream),
        // except for the rtspsrc element which is linked in the pad-added callback.
        m_isLinked = true;
        return true;
    }

    void RtspSourceBintr::UnlinkAll()
    {
        LOG_FUNC();

        if (!m_isLinked)
        {
            LOG_ERROR("RtspSourceBintr '" << GetName() 
                << "' is not in a linked state");
            return;
        }
        
        if (m_streamManagerTimerId)
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_streamManagerMutex);
            
            g_source_remove(m_streamManagerTimerId);
            m_streamManagerTimerId = 0;
            LOG_INFO("Stream management disabled for RTSP Source '" 
                << GetName() << "'");
        }
        if (m_reconnectionManagerTimerId)
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_reconnectionManagerMutex);

            g_source_remove(m_reconnectionManagerTimerId);
            m_reconnectionManagerTimerId = 0;
            LOG_INFO("Reconnection management disabled for RTSP Source '" 
                << GetName() << "'");
        }
        
        m_pPreDecodeQueue->UnlinkFromSink();
        if (HasTapBintr())
        {
            m_pPreDecodeQueue->UnlinkFromSourceTee();
            m_pTapBintr->UnlinkAll();
            m_pTapBintr->UnlinkFromSourceTee();
        }
        m_pParser->UnlinkFromSink();
        m_pDepay->UnlinkFromSink();

        // will be recreated in the select-stream callback on next play
        m_pParser = nullptr;
        m_pDepay = nullptr;
        m_pDecoder = nullptr;

        for (auto const& imap: m_pGstRequestedSourcePads)
        {
            gst_element_release_request_pad(m_pPreDecodeTee->GetGstElement(), 
                imap.second);
            gst_object_unref(imap.second);
        }
        
        m_isLinked = false;
    }

    bool RtspSourceBintr::SetUri(const char* uri)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set Uri for RtspSourceBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }
        std::string newUri(uri);
        if (newUri.find("rtsp") == std::string::npos)
        {
            LOG_ERROR("Invalid URI '" << uri << "' for RTSP Source '" << GetName() << "'");
            return false;
        }        
        m_pSourceElement->SetAttribute("location", m_uri.c_str());
        
        return true;
    }
    
    uint RtspSourceBintr::GetBufferTimeout()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_streamManagerMutex);
        
        return m_bufferTimeout;
    }
    
    void RtspSourceBintr::SetBufferTimeout(uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_streamManagerMutex);
        
        if (m_bufferTimeout == timeout)
        {
            LOG_WARN("Buffer timeout for RTSP Source '" << GetName() 
                << "' is already set to " << timeout);
            return;
        }

        // If we're all ready in a linked state, 
        if (IsLinked()) 
        {
            // If stream management is currently running, shut it down regardless
            if (m_streamManagerTimerId)
            {
                // shutdown the current session
                g_source_remove(m_streamManagerTimerId);
                m_streamManagerTimerId = 0;
                LOG_INFO("Stream management disabled for RTSP Source '" << GetName() << "'");
            }
            // If we have a new timeout value, we can renable
            if (timeout)
            {
                // Start up stream mangement
                m_streamManagerTimerId = g_timeout_add(timeout, 
                    RtspReconnectionMangerHandler, this);
                LOG_INFO("Stream management enabled for RTSP Source '" 
                    << GetName() << "' with timeout = " << timeout);
            }
            // Else, the client is disabling stream mangagement. Shut down the 
            // reconnection cycle if running. 
            else if (m_reconnectionManagerTimerId)
            {
                LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_reconnectionManagerMutex);
                // shutdown the current reconnection cycle
                g_source_remove(m_reconnectionManagerTimerId);
                m_reconnectionManagerTimerId = 0;
                LOG_INFO("Reconnection management disabled for RTSP Source '" << GetName() << "'");
            }
        }
        m_bufferTimeout = timeout;
    }

    void RtspSourceBintr::GetReconnectionParams(uint* sleep, uint* timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_reconnectionManagerMutex);
        
        *sleep = m_connectionData.sleep;
        *timeout = m_connectionData.timeout;
    }
    
    bool RtspSourceBintr::SetReconnectionParams(uint sleep, uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_reconnectionManagerMutex);
        
        if (!sleep or !timeout)
        {
            LOG_INFO("Invalid reconnection params for RTSP Source '" << GetName() << "'");
            return false;
        }

        m_connectionData.sleep = sleep;
        m_connectionData.timeout = timeout;
        return true;
    }

    void RtspSourceBintr::GetConnectionData(dsl_rtsp_connection_data* data)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_streamManagerMutex);

        *data = m_connectionData;
    }
    
    void RtspSourceBintr::_setConnectionData(dsl_rtsp_connection_data data)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_streamManagerMutex);
        
        m_connectionData = data;
    }
    
    void RtspSourceBintr::ClearConnectionStats()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_streamManagerMutex);

        m_connectionData.first_connected = 0;
        m_connectionData.last_connected = 0;
        m_connectionData.last_disconnected = 0;
        m_connectionData.count = 0;
        m_connectionData.retries = 0;
    }

    bool RtspSourceBintr::AddStateChangeListener(dsl_state_change_listener_cb listener, 
        void* userdata)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_streamManagerMutex);
        
        if (m_stateChangeListeners.find(listener) != m_stateChangeListeners.end())
        {   
            LOG_ERROR("RTSP Source state-change-listener is not unique");
            return false;
        }
        m_stateChangeListeners[listener] = userdata;
        
        return true;
    }

    bool RtspSourceBintr::RemoveStateChangeListener(dsl_state_change_listener_cb listener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_streamManagerMutex);
        
        if (m_stateChangeListeners.find(listener) == m_stateChangeListeners.end())
        {   
            LOG_ERROR("RTSP Source state-change-listener");
            return false;
        }
        m_stateChangeListeners.erase(listener);
        
        return true;
    }
    
    bool RtspSourceBintr::AddTapBintr(DSL_BASE_PTR pTapBintr)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can not add Tap to Source '" << GetName() 
                << "' as it's in a Linked state");
            return false;
        }
        if (m_pTapBintr)
        {
            LOG_ERROR("Source '" << GetName() << "' allready has a Tap");
            return false;
        }
        m_pTapBintr = std::dynamic_pointer_cast<TapBintr>(pTapBintr);
        AddChild(pTapBintr);
        return true;
    }

    bool RtspSourceBintr::RemoveTapBintr()
    {
        LOG_FUNC();

        if (!m_pTapBintr)
        {
            LOG_ERROR("Source '" << GetName() << "' does not have a Tap");
            return false;
        }
        if (m_isLinked)
        {
            LOG_ERROR("Can not remove Tap from Source '" << GetName() 
                << "' as it's in a Linked state");
            return false;
        }
        RemoveChild(m_pTapBintr);
        m_pTapBintr = nullptr;
        return true;
    }
    
    bool RtspSourceBintr::HasTapBintr()
    {
        LOG_FUNC();
        
        return (m_pTapBintr != nullptr);
    }

    bool RtspSourceBintr::HandleSelectStream(GstElement *pBin, uint num, GstCaps *caps)
    {
        GstStructure *structure = gst_caps_get_structure(caps, 0);
        std::string media = gst_structure_get_string (structure, "media");
        std::string encoding = gst_structure_get_string (structure, "encoding-name");

        LOG_INFO("Media = '" << media << "' for RtspSourceBitnr '" 
            << GetName() << "'");
        LOG_INFO("Encoding = '" << encoding << "' for RtspSourceBitnr '" 
            << GetName() << "'");

        if (m_pParser == nullptr)
        {
            if (media.find("video") == std::string::npos)
            {
                LOG_WARN("Unsupported media = '" << media 
                    << "' for RtspSourceBitnr '" << GetName() << "'");
                return false;
            }
            if (encoding.find("H26") != std::string::npos)
            {
                if (encoding.find("H264") != std::string::npos)
                {
                    m_pDepay = DSL_ELEMENT_NEW("rtph264depay", GetCStrName());
                    m_pParser = DSL_ELEMENT_NEW("h264parse", GetCStrName());
                }
                else if (encoding.find("H265") != std::string::npos)
                {
                    m_pDepay = DSL_ELEMENT_NEW("rtph265depay", GetCStrName());
                    m_pParser = DSL_ELEMENT_NEW("h265parse", GetCStrName());
                }
                else
                {
                    LOG_ERROR("Unsupported encoding = '" << encoding 
                        << "' for RtspSourceBitnr '" << GetName() << "'");
                    return false;
                }
            }
            else if (encoding.find("JPEG") != std::string::npos)
            {
                m_pDepay = DSL_ELEMENT_NEW("rtpjpegdepay", GetCStrName());
                m_pParser = DSL_ELEMENT_NEW("jpegparse", GetCStrName());
            }
            else
            {
                LOG_ERROR("Unsupported encoding = '" << encoding 
                    << "' for RtspSourceBitnr '" << GetName() << "'");
                return false;
            }

            m_pDecoder = DSL_ELEMENT_NEW("nvv4l2decoder", GetCStrName());
            
            // aarch64 only
            if (m_cudaDeviceProp.integrated)
            {
                m_pDecoder->SetAttribute("enable-max-performance", TRUE);
            }
            m_pDecoder->SetAttribute("drop-frame-interval", m_dropFrameInterval);
            m_pDecoder->SetAttribute("num-extra-surfaces", m_numExtraSurfaces);
            
            LOG_INFO("");
            LOG_INFO("Updated property values for RtspSourceBintr '" << GetName() << "'");
            LOG_INFO("  Media      : " << media);
            LOG_INFO("  Encoding   : " << encoding);
            LOG_INFO("  Elements");
            LOG_INFO("    Depay    : " << m_pDepay->GetFactoryName());
            LOG_INFO("    Parser   : " << m_pParser->GetFactoryName());
            LOG_INFO("    Decoder  : " << m_pDecoder->GetFactoryName());

            // The format specific depay, parser, and decoder bins have been selected, 
            // so we can add them as children to this RtspSourceBintr now.
            AddChild(m_pDepay);
            AddChild(m_pParser);
            AddChild(m_pDecoder);

            if (!m_pPreDecodeQueue->LinkToSink(m_pDecoder) or
                !m_pDecoder->LinkToSink(m_pSourceQueue))
            {
                return false;
            }

            // If we're tapping off of the pre-decode source stream, then link to the pre-decode Tee
            // The Pre-decode Queue will already be linked downstream as the first branch on the Tee
            if (HasTapBintr())
            {
                if (!m_pTapBintr->LinkAll() or 
                    !m_pTapBintr->LinkToSourceTee(m_pPreDecodeTee) or
                    !m_pPreDecodeQueue->LinkToSourceTee(m_pPreDecodeTee, "src_%u") or
                    !m_pDepay->LinkToSink(m_pParser) or 
                    !m_pParser->LinkToSink(m_pPreDecodeTee))
                {
                    return false;
                }
            }
            // otherwise, there is no Tee and we link to the Pre-decode Queue directly
            else
            {
                if (!m_pDepay->LinkToSink(m_pParser) or 
                    !m_pParser->LinkToSink(m_pPreDecodeQueue))
                {
                    return false;
                }            
            }
            if (!gst_element_sync_state_with_parent(m_pDepay->GetGstElement()) or
                !gst_element_sync_state_with_parent(m_pParser->GetGstElement()) or
                !gst_element_sync_state_with_parent(m_pDecoder->GetGstElement()))
            {
                LOG_ERROR("Failed to sync Parser/Decoder states with Parent for RtspSourceBitnr '" 
                    << GetName() << "'");
                return false;
            }
            // Start the Stream mangement timer, only if timeout is enable and not currently running
            if (m_bufferTimeout and !m_streamManagerTimerId)
            {
                m_streamManagerTimerId = g_timeout_add(m_bufferTimeout, 
                    RtspStreamManagerHandler, this);
                LOG_INFO("Starting stream management for RTSP Source '" << GetName() << "'");
            }

            SetCurrentState(GST_STATE_READY);
        }
        return true;
    }
        
    void RtspSourceBintr::HandleSourceElementOnPadAdded(GstElement* pBin, GstPad* pPad)
    {
        LOG_FUNC();

        GstCaps* pCaps = gst_pad_query_caps(pPad, NULL);
        GstStructure* structure = gst_caps_get_structure(pCaps, 0);
        std::string name = gst_structure_get_name(structure);
        std::string media = gst_structure_get_string (structure, "media");
        std::string encoding = gst_structure_get_string (structure, "encoding-name");

        LOG_INFO("Caps structs name " << name);
        LOG_INFO("Media = '" << media << "' for RtspSourceBitnr '" << GetName() << "'");
        
        if (name.find("x-rtp") != std::string::npos and 
            media.find("video")!= std::string::npos)
        {
            // get the Depays static sink pad so we can link the rtspsrc elementr
            // to the depay elementr.
            GstPad* pDepayStaicSinkPad = gst_element_get_static_pad(
                m_pDepay->GetGstElement(), "sink");
            if (!pDepayStaicSinkPad)
            {
                LOG_ERROR("Failed to get Static Source Pad for Streaming Source '" 
                    << GetName() << "'");
                throw;
            }
            
            // Link the rtcpsrc element's added src pad to the sink pad of the Depay
            if (gst_pad_link(pPad, pDepayStaicSinkPad) != GST_PAD_LINK_OK) 
            {
                LOG_ERROR("Failed to link source to de-payload");
                throw;
            }
            
            LOG_INFO("rtspsrc element linked for RtspSourceBintr '" << GetName() << "'");

            // Update the cap memebers for this RtspSourceBintr
            gst_structure_get_uint(structure, "width", &m_width);
            gst_structure_get_uint(structure, "height", &m_height);
            gst_structure_get_fraction(structure, "framerate", (gint*)&m_fpsN, (gint*)&m_fpsD);
            
            LOG_INFO("Frame width = " << m_width << ", height = " << m_height);
            LOG_INFO("FPS numerator = " << m_fpsN << ", denominator = " << m_fpsD);
        }
    }
    
    void RtspSourceBintr::HandleDecodeElementOnPadAdded(GstElement* pBin, GstPad* pPad)
    {
        LOG_FUNC();

        GstCaps* pCaps = gst_pad_query_caps(pPad, NULL);
        GstStructure* structure = gst_caps_get_structure(pCaps, 0);
        std::string name = gst_structure_get_name(structure);
        
        LOG_INFO("Caps structs name " << name);
        if (name.find("video") != std::string::npos)
        {
            GstPad* pQueueStaticSinkPad = 
                gst_element_get_static_pad(m_pSourceQueue->GetGstElement(), "sink");
            if (!pQueueStaticSinkPad)
            {
                LOG_ERROR("Failed to get Static Source Pad for RTSP Source '" 
                    << GetName() << "'");
            }
            
            // Link the decode element's src pad with the source queue's sink pad
            if (gst_pad_link(pPad, pQueueStaticSinkPad) != GST_PAD_LINK_OK) 
            {
                LOG_ERROR("Failed to link decodebin to pipeline");
                throw;
            }
            
            // Start the Stream mangement timer, only if timeout is enable and not currently running
            if (m_bufferTimeout and !m_streamManagerTimerId)
            {
                m_streamManagerTimerId = g_timeout_add(m_bufferTimeout, 
                    RtspStreamManagerHandler, this);
                LOG_INFO("Starting stream management for RTSP Source '" << GetName() << "'");
            }

            SetCurrentState(GST_STATE_READY);

            LOG_INFO("Decode element linked for RtspSourceBintr '" << GetName() << "'");
        }
    }

    int RtspSourceBintr::StreamManager()
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_streamManagerMutex);

        // if currently in a reset cycle then let the ResetStream handler continue to handle
        if (m_connectionData.is_in_reconnect)
        {
            return true;
        }

        struct timeval currentTime;
        gettimeofday(&currentTime, NULL);

        GstState currentState;
        uint stateResult = GetState(currentState, 0);
        SetCurrentState(currentState);
        
        // Get the last buffer time. This timer callback should not be called until after the timer 
        // is started on successful linkup - therefore the lastBufferTime should be non-zero
        struct timeval lastBufferTime;
        m_TimestampPph->GetTime(lastBufferTime);
        if (lastBufferTime.tv_sec == 0)
        {
            LOG_DEBUG("Waiting for first buffer before checking for timeout for source '" 
                << GetName() << "'");
            return true;
        }

        double timeSinceLastBufferMs = 1000.0*(currentTime.tv_sec - lastBufferTime.tv_sec) + 
            (currentTime.tv_usec - lastBufferTime.tv_usec) / 1000.0;

        if (timeSinceLastBufferMs < m_bufferTimeout*1000)
        {
            // Timeout has not been exceeded, so return true to sleep again
            return true;
        }
        LOG_INFO("Buffer timeout of " << m_bufferTimeout << " seconds exceeded for source '" 
            << GetName() << "'");
            
        if (HasTapBintr())
        {
            m_pTapBintr->HandleEos();
        }
        
        // Call the Reconnection Managter directly to start the reconnection cycle,
        if (!ReconnectionManager())
        {
            LOG_INFO("Unable to start re-connection manager for '" << GetName() << "'");
            return false;
        }
            
        LOG_INFO("Starting Re-connection Manager for source '" << GetName() << "'");
        m_reconnectionManagerTimerId = g_timeout_add(1000, RtspReconnectionMangerHandler, this);

        return true;
    }
    
    int RtspSourceBintr::ReconnectionManager()
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_reconnectionManagerMutex);
        do
        {
            timeval currentTime;
            gettimeofday(&currentTime, NULL);
            
            uint stateResult(0);
            GstState currentState;
            
            if (!m_connectionData.is_in_reconnect or m_reconnectionFailed or 
                (currentTime.tv_sec - m_reconnectionStartTime.tv_sec) > m_connectionData.timeout)
            {
                // set the reset-state,
                if (!m_connectionData.is_in_reconnect)
                {
                    m_connectionData.is_connected = false;
                    m_connectionData.retries = 0;
                    m_connectionData.is_in_reconnect = true;
                }
                // if the previous attempt failed
                else if (m_reconnectionFailed == true)
                {
                    m_reconnectionSleep-=1;
                    if (m_reconnectionSleep)
                    {
                        LOG_INFO("Sleeping after failed connection");
                        return true;
                    }
                    m_reconnectionFailed = false;    
                }
                m_connectionData.retries++;

                LOG_INFO("Resetting RTSP Source '" << GetName() 
                    << "' with retry count = " << m_connectionData.retries);
                
                m_reconnectionStartTime = currentTime;

                if (SetState(GST_STATE_NULL, 0) != GST_STATE_CHANGE_SUCCESS)
                {
                    LOG_ERROR("Failed to set RTSP Source '" << GetName() << "' to GST_STATE_NULL");
                    return false;
                }
                // update the internal state variable to notify all client listeners 
                SetCurrentState(GST_STATE_NULL);
                return true;
            }
            else
            {   
                // Waiting for the Source to reconnect, check the state again
                stateResult = GetState(currentState, GST_SECOND);
            }
                
            // update the internal state variable to notify all client listeners 
            SetCurrentState(currentState);
            switch (stateResult) 
            {
                case GST_STATE_CHANGE_NO_PREROLL:
                    LOG_INFO("RTSP Source '" << GetName() 
                        << "' returned GST_STATE_CHANGE_NO_PREROLL");
                    // fall through ... do not break
                case GST_STATE_CHANGE_SUCCESS:
                    if (currentState == GST_STATE_NULL)
                    {
                        // synchronize the source's state with the Pipleine's
                        SyncStateWithParent(currentState, 1);
                        return true;
                    }
                    if (currentState == GST_STATE_PLAYING)
                    {
                        LOG_INFO("Re-connection complete for RTSP Source'" << GetName() << "'");
                        m_connectionData.is_in_reconnect = false;

                        // update the current buffer timestamp to the current reset time
                        m_TimestampPph->SetTime(currentTime);
                        m_reconnectionManagerTimerId = 0;
                        return false;
                    }
                    
                    // If state change completed succesfully, but not yet playing, set explicitely.
                    SetState(GST_STATE_PLAYING, 
                        DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND);
                    break;
                    
                case GST_STATE_CHANGE_ASYNC:
                    LOG_INFO("State change will complete asynchronously for RTSP Source '" 
                        << GetName() << "'");
                    break;

                case GST_STATE_CHANGE_FAILURE:
                    LOG_ERROR("FAILURE occured when trying to sync state for RTSP Source '" 
                        << GetName() << "'");
                    m_reconnectionFailed = true;
                    m_reconnectionSleep = m_connectionData.sleep;
                    return true;

                default:
                    LOG_ERROR("Unknown 'state change result' when trying to sync state for RTSP Source '" 
                        << GetName() << "'");
                    return true;
            }
        }while(true);
    }
    
    GstState RtspSourceBintr::GetCurrentState()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_stateChangeMutex);
        
        LOG_INFO("Returning state " 
            << gst_element_state_get_name((GstState)m_currentState) << 
            " for RtspSourceBintr '" << GetName() << "'");

        return m_currentState;
    }

    void RtspSourceBintr::SetCurrentState(GstState newState)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_stateChangeMutex);

        if (newState != m_currentState)
        {
            LOG_INFO("Changing state from " << 
                gst_element_state_get_name((GstState)m_currentState) << 
                " to " << gst_element_state_get_name((GstState)newState) 
                << " for RtspSourceBintr '" << GetName() << "'");
            
            m_previousState = m_currentState;
            m_currentState = newState;

            struct timeval currentTime;
            gettimeofday(&currentTime, NULL);
            
            if ((m_previousState == GST_STATE_PLAYING) and (m_currentState == GST_STATE_NULL))
            {
                m_connectionData.is_connected = false;
                m_connectionData.last_disconnected = currentTime.tv_sec;
            }
            if (m_currentState == GST_STATE_PLAYING)
            {
                m_connectionData.is_connected = true;
                
                // if first time is empty, this is the first since Pipeline play or stats clear.
                if(!m_connectionData.first_connected)
                {
                    m_connectionData.first_connected = currentTime.tv_sec;
                }
                m_connectionData.last_connected = currentTime.tv_sec;
                m_connectionData.count++;
            }                    
            
            if (m_stateChangeListeners.size())
            {
                std::shared_ptr<DslStateChange> pStateChange = 
                    std::shared_ptr<DslStateChange>(new DslStateChange(m_previousState, m_currentState));
                    
                m_stateChanges.push(pStateChange);
                
                // start the asynchronous notification timer if not currently running
                if (!m_listenerNotifierTimerId)
                {
                    m_listenerNotifierTimerId = g_timeout_add(1, RtspListenerNotificationHandler, this);
                }
            }
        }
    }
    
    int RtspSourceBintr::NotifyClientListeners()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_stateChangeMutex);
        
        while (m_stateChanges.size())
        {
            std::shared_ptr<DslStateChange> pStateChange = m_stateChanges.front();
            m_stateChanges.pop();
            
            // iterate through the map of state-change-listeners calling each
            for(auto const& imap: m_stateChangeListeners)
            {
                try
                {
                    imap.first((uint)pStateChange->m_previousState, 
                        (uint)pStateChange->m_newState, imap.second);
                }
                catch(...)
                {
                    LOG_ERROR("RTSP Source '" << GetName() 
                        << "' threw exception calling Client State-Change-Lister");
                }
            }
            
        }
        // clear the timer id and return false to self remove
        m_listenerNotifierTimerId = 0;
        return false;
    }
    
    // --------------------------------------------------------------------------------------

    static int ImageSourceDisplayTimeoutHandler(gpointer pSource)
    {
        return static_cast<ImageStreamSourceBintr*>(pSource)->
            HandleDisplayTimeout();
    }
    
    static void UriSourceElementOnPadAddedCB(GstElement* pBin, GstPad* pPad, gpointer pSource)
    {
        static_cast<UriSourceBintr*>(pSource)->HandleSourceElementOnPadAdded(pBin, pPad);
    }
    
    static boolean RtspSourceSelectStreamCB(GstElement *pBin, uint num, GstCaps *caps,
        gpointer pSource)
    {
        return static_cast<RtspSourceBintr*>(pSource)->HandleSelectStream(pBin, num, caps);
    }
        
    static void RtspSourceElementOnPadAddedCB(GstElement* pBin, GstPad* pPad, gpointer pSource)
    {
        static_cast<RtspSourceBintr*>(pSource)->HandleSourceElementOnPadAdded(pBin, pPad);
    }
    
    static void RtspDecodeElementOnPadAddedCB(GstElement* pBin, GstPad* pPad, gpointer pSource)
    {
        static_cast<RtspSourceBintr*>(pSource)->HandleDecodeElementOnPadAdded(pBin, pPad);
    }
    
    static void OnChildAddedCB(GstChildProxy* pChildProxy, GObject* pObject,
        gchar* name, gpointer pSource)
    {
        static_cast<UriSourceBintr*>(pSource)->HandleOnChildAdded(pChildProxy, pObject, name);
    }
    
    static void OnSourceSetupCB(GstElement* pObject, GstElement* arg0, 
        gpointer pSource)
    {
        static_cast<UriSourceBintr*>(pSource)->HandleOnSourceSetup(pObject, arg0);
    }
    
    static GstPadProbeReturn StreamBufferRestartProbCB(GstPad* pPad, 
        GstPadProbeInfo* pInfo, gpointer pSource)
    {
        return static_cast<UriSourceBintr*>(pSource)->
            HandleStreamBufferRestart(pPad, pInfo);
    }

    static gboolean StreamBufferSeekCB(gpointer pSource)
    {
        return static_cast<UriSourceBintr*>(pSource)->HandleStreamBufferSeek();
    }

    static int RtspStreamManagerHandler(gpointer pSource)
    {
        return static_cast<RtspSourceBintr*>(pSource)->
            StreamManager();
    }

    static int RtspReconnectionMangerHandler(gpointer pSource)
    {
        return static_cast<RtspSourceBintr*>(pSource)->
            ReconnectionManager();
    }

    static int RtspListenerNotificationHandler(gpointer pSource)
    {
        return static_cast<RtspSourceBintr*>(pSource)->
            NotifyClientListeners();
    }
    
} // SDL namespace
