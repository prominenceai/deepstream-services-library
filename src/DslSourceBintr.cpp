/*
/*
The MIT License

Copyright (c) 2019-2024, Prominence AI, Inc.

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
#include "DslCaps.h"
#include "DslServices.h"
#include "DslSourceBintr.h"
#include "DslPipelineBintr.h"
#include "DslSurfaceTransform.h"
#include <nvdsgstutils.h>
#include <gst/app/gstappsrc.h>

#if (BUILD_WITH_FFMPEG == true) || (BUILD_WITH_OPENCV == true)
#include "DslAvFile.h"
#endif

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
    
    SourceBintr::SourceBintr(const char* name)
        : QBintr(name)
        , m_cudaDeviceProp{0}
        , m_isLive(true)
        , m_fpsN(0)
        , m_fpsD(0)
    {
        LOG_FUNC();

        // Get the Device properties
        cudaGetDeviceProperties(&m_cudaDeviceProp, m_gpuId);

    }
    
    SourceBintr::~SourceBintr()
    {
        LOG_FUNC();

        if (m_isLinked)
        {    
            UnlinkAll();
        }
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
    
    //--------------------------------------------------------------------------------
    
    VideoSourceBintr::VideoSourceBintr(const char* name)
        : SourceBintr(name)
        , m_width(0)
        , m_height(0)
        , m_bufferOutWidth(0)
        , m_bufferOutHeight(0)
        , m_bufferOutFpsN(0)
        , m_bufferOutFpsD(0)
        , m_bufferOutOrientation(DSL_VIDEO_ORIENTATION_NONE)
    {
        LOG_FUNC();

        // Media type is fixed to "video/x-raw"
        std::wstring L_mediaType(DSL_MEDIA_STRING_VIDEO_XRAW);
        m_videoMediaString.assign(L_mediaType.begin(), L_mediaType.end());

        m_mediaType = DSL_MEDIA_TYPE_VIDEO_ONLY;

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

        m_pBufferOutCapsFilter = DSL_ELEMENT_EXT_NEW("capsfilter", 
            name, "vidconv");
        
        // Update the caps with the media, format, and memory:NVMM feature
        if (!SetBufferOutFormat(m_bufferOutFormat.c_str()))
        {
            throw;
        }

        // add both elementrs as children to this Bintr
        AddChild(m_pBufferOutVidConv);
        AddChild(m_pBufferOutCapsFilter);

        // IMPORTANT! Caps Filter is ghost-pad by default - will be changed if 
        // duplicate Source is added.
        m_pBufferOutCapsFilter->AddGhostPadToParent("src");

        // Add the Buffer and DS Event Probes to the caps-filter - src-pad only.
        AddSrcPadProbes(m_pBufferOutCapsFilter->GetGstElement());
    }
    
    VideoSourceBintr::~VideoSourceBintr()
    {
        LOG_FUNC();
    }
    
    bool VideoSourceBintr::LinkToCommon(DSL_NODETR_PTR pSrcNodetr)
    {
        LOG_FUNC();

        // If linking as a Nodetr (element), get the static sink-pad for 
        // and then call the LinkToCommon(GstPad*) function below.
        // IMPORTANT! This Nodetr must be unlinked in the UnlinkCommon()
        GstPad* pStaticSrcPad = gst_element_get_static_pad(
                pSrcNodetr->GetGstElement(), "src");
        if (!pStaticSrcPad)
        {
            LOG_ERROR("Failed to get static src pad for VideoSourceBintr '" 
                << GetName() << "'");
            return false;
        }
        bool retval = LinkToCommon(pStaticSrcPad);

        gst_object_unref(pStaticSrcPad);

        return retval;
    }

    bool VideoSourceBintr::LinkToCommon(GstPad* pSrcPad)
    {
        LOG_FUNC();

        // If the VideoSource has a dewarper, add it as the first common element
        // and link it to the Source's pre video converter queue.  
        if (HasDewarperBintr())
        {
            if (!m_pDewarperBintr->LinkAll())
            {
                LOG_ERROR("Failed to Link Dewarper for VideoSourceBintr '" 
                    << GetName() << "'");
                return false;
            }
            m_linkedCommonElements.push_back(m_pDewarperBintr);
            m_pDewarperBintr->LinkToSink(m_pQueue);
        }

        // Add the queue as first or next element to the vector of common elements.
        m_linkedCommonElements.push_back(m_pQueue);

        // We now have the first element (dewarper or queue) so we can link it
        // with the Source specific pSrcPad passed into this function.
        GstPad* pStaticSinkPad = gst_element_get_static_pad(
                m_linkedCommonElements.front()->GetGstElement(), "sink"); 
        if (!pStaticSinkPad)
        {
            LOG_ERROR("Failed to get static sink pad from first common element '"
                << m_linkedCommonElements.front()->GetName() 
                << "' for VideoSourceBintr '" << GetName() << "'");
            return false;
        }
        if (gst_pad_link(pSrcPad, pStaticSinkPad) != GST_PAD_LINK_OK) 
        {
            LOG_ERROR("Failed to link src pad to sink pad from first common element '"
                << m_linkedCommonElements.front()->GetName() 
                << "' for VideoSourceBintr '" << GetName() << "'");
            return false;
        }

        // Link and add the Video Converter next
        if (!m_linkedCommonElements.back()->LinkToSink(m_pBufferOutVidConv))
        {
            return false;
        }
        m_linkedCommonElements.push_back(m_pBufferOutVidConv);
        
        // If the viderate was created, add it as the next common element
        // and link it to the Source's Video Converter.
        if (m_pBufferOutVidRate)
        {
            if (!m_linkedCommonElements.back()->LinkToSink(m_pBufferOutVidRate))
            {
                return false;
            }
            m_linkedCommonElements.push_back(m_pBufferOutVidRate);
        }            

        // Link to the caps-filter element.
        if (!m_linkedCommonElements.back()->LinkToSink(m_pBufferOutCapsFilter))
        {
            return false;
        }

        if (m_pDuplicateSourceTee)
        {
            // Only add the Caps Filter to the list of linked components
            // if it's not the last component... i.e. there's duplicates.
            m_linkedCommonElements.push_back(m_pBufferOutCapsFilter);
            
            if (!m_linkedCommonElements.back()->LinkToSink(m_pDuplicateSourceTee))
            {
                return false;
            }
            // Link all Duplicate Sources to the Duplicate Source Tee.
            if (!linkAllDuplicates())
            {
                return false;
            }
            // Link the extra Queue back to the Duplicate Source Tee.
            if (!m_pDuplicateSourceTeeQueue->LinkToSourceTee(
                    m_pDuplicateSourceTee, "src_%u"))
            {
                return false;
            }
        }
            
        return true;
    }
    
    void VideoSourceBintr::UnlinkCommon()
    {
        LOG_FUNC();

        // Get a reference to the sink pad of the first common element
        GstPad* pStaticSinkPad = gst_element_get_static_pad(
                m_linkedCommonElements.front()->GetGstElement(), "sink");
        if (!pStaticSinkPad)
        {
            LOG_ERROR("Failed to get static sink pad for Elementr '" 
                << m_linkedCommonElements.front()->GetName() << "'");
            return;
        }
        // Check to see if it is currently linked.  This will be true for all
        // sources that call LinkToCommon(DSL_NODETR_PTR) and false for all 
        // sources that call LinkToCommon(GstPad*) from pad_added callbacks.
        if (gst_pad_is_linked(pStaticSinkPad))
        {                
            GstPad* pSrcPad = gst_pad_get_peer(pStaticSinkPad);
            if (!pSrcPad)
            {
                LOG_ERROR("Failed to get peer src pad for Elementr '" 
                    << m_linkedCommonElements.front()->GetName() << "'");
                return;
            }
            LOG_INFO("Unlinking common front Elementr '" 
                << m_linkedCommonElements.front()->GetName() 
                << "' from its peer src pad");

            if (!gst_pad_unlink(pSrcPad, pStaticSinkPad))
            {
                LOG_ERROR("Failed to unlink src pad for Elementr '" 
                    << m_linkedCommonElements.front()->GetName() << "'");
                return;
            }
            gst_object_unref(pSrcPad);
        }
        gst_object_unref(pStaticSinkPad);

        // iterate through the list of linked Elements, unlinking each
        for (auto const& ivec: m_linkedCommonElements)
        {
            ivec->UnlinkFromSink();
        }
        m_linkedCommonElements.clear();

        // If we're duplicating this source's stream.
        if (m_pDuplicateSourceTee)
        {
            m_pDuplicateSourceTeeQueue->UnlinkFromSourceTee();
            unlinkAllDuplicates();
        }
    }

    bool VideoSourceBintr::linkAllDuplicates()
    {
        LOG_FUNC();
        
        uint index(1);
        for (const auto& imap: m_duplicateSources)
        {
            // For each duplicate source, we need to request a new source pad
            // from the Duplicate-Source Tee element.
            GstPad* pRequestedSrcPad = gst_element_get_request_pad(
                m_pDuplicateSourceTee->GetGstElement(), "src_%u");
            if (!pRequestedSrcPad)
            {
                LOG_ERROR("Failed to get a requested source pad from Tee '" 
                    << m_pDuplicateSourceTee->GetName() <<"'");
                return false;
            }
            // We must save the Requested source pad so we can release it 
            // in the helper function UnlinkAllDuplicates below
            m_requestedDuplicateSrcPads.push_back(pRequestedSrcPad);
            
            LOG_INFO("New request pad = " << std::hex << pRequestedSrcPad
                << " allocated from Tee '" 
                << m_pDuplicateSourceTee->GetName() << "'");
            
            // Next, we must elevate the requested pad so that it can be linked
            // to the Sink pad of the DuplicateSourceBintr. We do this be creating
            // a ghost pad from the requested pad, then active it and add it to  
            // Tee's parent, i.e. this VideoSourceBintr's gst-bin.
        
            // start by creating a new, unique name for the new ghost pad.
            std::string padName = "src_" + std::to_string(index);

            GstPad* pGhostPad = gst_ghost_pad_new(padName.c_str(), 
                pRequestedSrcPad);
            if (!pGhostPad)
            {
                LOG_ERROR("Failed to create a ghost pad for requested source pad = "
                    << std::hex << pRequestedSrcPad);
                return false;
            }
            gst_pad_set_active(pGhostPad, TRUE);
                
            if (!gst_element_add_pad(GetGstElement(), pGhostPad))
            {
                LOG_ERROR("Failed to add new ghost pad '" << padName 
                    << "' to Original Source'" << GetName() << "'");
                return false;
            }
            LOG_INFO("New ghost pad = " << std::hex << pGhostPad
                << " allocated for request pad = " 
                << std::hex << pRequestedSrcPad 
                << "' added to Original Source '" << GetName() << "'");
            
            // We can now get the newly added/elevated source pad by name
            GstPad* pStaticSrcPad = gst_element_get_static_pad(
                GetGstElement(), padName.c_str()); 
                
            // Along with the elevated static sink pad for the DuplicateSourceBintr    
            GstPad* pStaticSinkPad = gst_element_get_static_pad(
                imap.second->GetGstElement(), "sink");
            
            // and link them together... with a new stream now splitting
            // off from the original.
            if (gst_pad_link(pStaticSrcPad, pStaticSinkPad) != GST_PAD_LINK_OK)
            {
                LOG_ERROR("Original Source '" << GetName() 
                    << "' failed to link to Duplicate Source '"
                    << imap.second->GetName() << "'");
                return false;
            }
            LOG_INFO("Original Source '" << GetName() 
                << "' linked to Duplicate Source '" << imap.second->GetName()
                << "' successfully");
            
            // Need to unreference the pointers to the static source and sink pads.
            gst_object_unref(pStaticSrcPad);
            gst_object_unref(pStaticSinkPad);
            
            index++;
        }
        return true;
    }

    bool VideoSourceBintr::unlinkAllDuplicates()
    {
        LOG_FUNC();
        
        uint index(1);
        for (const auto& imap: m_duplicateSources)
        {
            // for each duplicate source, get the elevated static pad for
            // the added ghost pad for duplicate-source-tee element
            std::string padName = "src_" + std::to_string(index);
            
            GstPad* pStaticSrcPad = gst_element_get_static_pad(
                GetGstElement(), padName.c_str()); 
            if (!pStaticSrcPad)
            {
                LOG_ERROR("Original Source '" << GetName() 
                    << "' failed to get static source pad");
                return false;
            }
            
            // get the static sink for the Duplicate Source so we can unlink it
            GstPad* pStaticSinkPad = gst_element_get_static_pad(
                imap.second->GetGstElement(), "sink");
            if (!pStaticSinkPad)
            {
                LOG_ERROR("Duplicate Source '" << imap.second->GetName() 
                    << "' failed to get static sink pad");
                return false;
            }
            
            // unlink the Original Source From the Duplicate Source
            if (gst_pad_is_linked(pStaticSinkPad) and
                (!gst_pad_unlink(pStaticSrcPad, pStaticSinkPad)))
            {
                LOG_ERROR("Original Source '" << GetName() 
                    << "' failed to unlink from Duplicate Source '"
                    << imap.second->GetName() << "'");
                return false;
            }
            LOG_INFO("Original Source '" << GetName() 
                << "' unlinked from Duplicate Source '" << imap.second->GetName()
                << "' successfully");
            
            // Need to remove the elevated ghost pad this Origian Sources's gst-bin. 
            if (!gst_element_remove_pad(GetGstElement(), pStaticSrcPad))
            {
                LOG_ERROR("Failed to remove pad '" << padName 
                    << "' from Original Source'" << GetName() << "'");
                return false;
            }
            LOG_INFO("Elevated static pad '" << padName 
                << "' removed from Original Source'" << GetName() << "'");

            // unreference the static pad pointers. 
            gst_object_unref(pStaticSrcPad);
            gst_object_unref(pStaticSinkPad);

            // Finally, we need to release and unref the requested pad (the one 
            // that was ghosted) back to the duplicate-souce-tee
            GstPad* pRequestedSrcPad = m_requestedDuplicateSrcPads[index-1];
            
            LOG_INFO("Releasing and unreferencing requested source pad = " 
                << pRequestedSrcPad << " for Tee '" 
                << m_pDuplicateSourceTee->GetName() << "'");

            gst_element_release_request_pad(m_pDuplicateSourceTee->GetGstElement(), 
                pRequestedSrcPad);
            gst_object_unref(pRequestedSrcPad);
            
            index++;
        }
        // Clear out the vector of requested (now unreferenced) source pads
        m_requestedDuplicateSrcPads.clear();
        return true;
    }

    void VideoSourceBintr::GetDimensions(uint* width, uint* height)
    {
        LOG_FUNC();
        
        *width = m_width;
        *height = m_height;
    }

    bool VideoSourceBintr::SetBufferOutFormat(const char* format)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't set buffer-out-format for VideoSourceBintr '" << GetName() 
                << "' as it is currently in a linked state");
            return false;
        }

        m_bufferOutFormat = format;
        
        return updateVidConvCaps();
    }
    
    void VideoSourceBintr::GetBufferOutDimensions(uint* width, uint* height)
    {
        LOG_FUNC();
        
        *width = m_bufferOutWidth;
        *height = m_bufferOutHeight;
    }
    
    bool VideoSourceBintr::SetBufferOutDimensions(uint width, uint height)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't set buffer-out-dimensions for VideoSourceBintr '" 
                << GetName() << "' as it is currently in a linked state");
            return false;
        }
        m_bufferOutWidth = width;
        m_bufferOutHeight = height;
        
        return updateVidConvCaps();
    }
    
    void VideoSourceBintr::GetBufferOutFrameRate(uint* fpsN, uint* fpsD)
    {
        LOG_FUNC();
        
        *fpsN = m_bufferOutFpsN;
        *fpsD = m_bufferOutFpsD;
    }
    
    bool VideoSourceBintr::SetBufferOutFrameRate(uint fpsN, uint fpsD)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't set buffer-out-frame-rate for VideoSourceBintr '" 
                << GetName() << "' as it is currently in a linked state");
            return false;
        }
        m_bufferOutFpsN = fpsN;
        m_bufferOutFpsD = fpsD;
        
        // if we're scaling the output frame-rate and there is no viderate element.
        if (fpsN and fpsD and !m_pBufferOutVidRate)
        {
            // time to create the viderate now
            m_pBufferOutVidRate = DSL_ELEMENT_NEW("videorate", GetCStrName());
            
            AddChild(m_pBufferOutVidRate);
        }
        // if we're not scalling and the viderate element has already been created.
        else if ((!fpsN or !fpsD) and m_pBufferOutVidRate)
        {
            RemoveChild(m_pBufferOutVidRate);

            // delete the viderate element now
            m_pBufferOutVidRate = nullptr;
        }
        // Update the output-buffer's caps filter now
        return updateVidConvCaps();
    }
    
    void tokenize(std::string const &str, const char delim,
                std::vector<std::string> &out)
    {
        size_t start;
        size_t end = 0;
     
        while ((start = str.find_first_not_of(delim, end)) != std::string::npos)
        {
            end = str.find(delim, start);
            out.push_back(str.substr(start, end - start));
        }
    }
    
    void VideoSourceBintr::GetBufferOutCropRectangle(uint cropAt, 
        uint* left, uint* top, uint* width, uint* height)
    {
        LOG_FUNC();
        
        const char* cropCString;

        if (cropAt == DSL_VIDEO_CROP_AT_SRC)
        {
            m_pBufferOutVidConv->GetAttribute("src-crop", &cropCString);
        }
        else
        {
            m_pBufferOutVidConv->GetAttribute("dest-crop", &cropCString);
        }
        std::string cropString(cropCString);

        const char delim = ':';
        std::vector<std::string> tokens;
        tokenize(cropString, delim, tokens);

        if (tokens.size() != 4)
        {
            LOG_ERROR("Invalid crop string recieved for VideoSourceBintr '"
                << GetName() << "'");
            return;
        }
        *left = std::stoul(tokens[0]);
        *top = std::stoul(tokens[1]);
        *width = std::stoul(tokens[2]);
        *height = std::stoul(tokens[3]);
    }
    
    bool VideoSourceBintr::SetBufferOutCropRectangle(uint cropAt, 
        uint left, uint top, uint width, uint height)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR(
                "Unable to set buffer-out crop settings for VideoSourceBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        
        std::string cropSettings( 
            std::to_string(left) + ":" +
            std::to_string(top) + ":" +
            std::to_string(width) + ":" +
            std::to_string(height));
        
        if (cropAt == DSL_VIDEO_CROP_AT_SRC)
        {
            m_pBufferOutVidConv->SetAttribute("src-crop", cropSettings.c_str());
        }
        else
        {
            m_pBufferOutVidConv->SetAttribute("dest-crop", cropSettings.c_str());
        }

        return true;
    }

    uint VideoSourceBintr::GetBufferOutOrientation()
    {
        LOG_FUNC();
        
        return m_bufferOutOrientation;
    }
    
    bool VideoSourceBintr::SetBufferOutOrientation(uint orientation)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR(
                "Unable to set buffer-out-orientation for VideoSourceBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_bufferOutOrientation = orientation;
        m_pBufferOutVidConv->SetAttribute("flip-method", m_bufferOutOrientation);

        return true;
    }

    bool VideoSourceBintr::SetGpuId(uint gpuId)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Unable to set GPU ID for VideoSourceBintr '" << GetName() 
                << "' as it's currently linked");
            return false;
        }

        m_gpuId = gpuId;
        m_pBufferOutVidConv->SetAttribute("gpu-id", m_gpuId);
        
        LOG_INFO("VideoSourceBintr '" << GetName() 
            << "' - new GPU ID = " << m_gpuId );
        
        return true;
    }

    bool VideoSourceBintr::SetNvbufMemType(uint nvbufMemType)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR(
                "Unable to set NVIDIA buffer memory type for VideoSourceBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_nvbufMemType = nvbufMemType;
        m_pBufferOutVidConv->SetAttribute("nvbuf-memory-type", m_nvbufMemType);

        return true;
    }

    
    bool VideoSourceBintr::updateVidConvCaps()
    {
        LOG_FUNC();

        DslCaps Caps(m_videoMediaString.c_str(), m_bufferOutFormat.c_str(),
            m_bufferOutWidth, m_bufferOutHeight, 
            m_bufferOutFpsN, m_bufferOutFpsD, true);

        // Set the Caps for the Buffer output
        m_pBufferOutCapsFilter->SetAttribute("caps", &Caps);
        
        return true;
    }

    
    bool VideoSourceBintr::AddDewarperBintr(DSL_BASE_PTR pDewarperBintr)
    {
        LOG_FUNC();
        
        if (m_pDewarperBintr)
        {
            LOG_ERROR("VideoSourceBintr '" << GetName() 
                << "' allready has a Dewarper");
            return false;
        }
        m_pDewarperBintr = std::dynamic_pointer_cast<DewarperBintr>(pDewarperBintr);
        AddChild(pDewarperBintr);
        
        // Need to fix output of the video converter to RGBA for the Dewarper.
        return SetBufferOutFormat("RGBA");
    }

    bool VideoSourceBintr::RemoveDewarperBintr()
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
    
    bool VideoSourceBintr::HasDewarperBintr()
    {
        LOG_FUNC();
        
        return (m_pDewarperBintr != nullptr);
    }
    
    bool VideoSourceBintr::AddDuplicateSource(
        DSL_SOURCE_PTR pDuplicateSource)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR(
                "Unable to add DuplicateSourceBintr '"
                << pDuplicateSource->GetName() << "' to VideoSourceBintr '"
                << GetName() << "' as it's currently linked");
            return false;
        }
        // ensure uniqueness 
        if (m_duplicateSources.find(pDuplicateSource->GetName()) 
            != m_duplicateSources.end())
        {   
            LOG_ERROR("DuplicateSourceBintr '" << pDuplicateSource->GetName()
                << "' has been previously added to VideoSourceBintr '"
                << GetName() << "' and cannot be added again");
            return false;
        }
        // if this is the first DuplicateSourceBintr to be added, then we need
        // to create the required Tee and Queue elements to support duplicates.
        if (!m_duplicateSources.size())
        {
            m_pDuplicateSourceTee = DSL_ELEMENT_EXT_NEW("tee", 
                GetCStrName(), "duplicate");
            m_pDuplicateSourceTeeQueue = DSL_ELEMENT_EXT_NEW("queue", 
                GetCStrName(), "duplicate");
                
            AddChild(m_pDuplicateSourceTee);
            AddChild(m_pDuplicateSourceTeeQueue);

            m_pBufferOutCapsFilter->RemoveGhostPadFromParent("src");
            m_pDuplicateSourceTeeQueue->AddGhostPadToParent("src");
        }
        // add the duplicate to the map of duplicates for this VideoSourceBintr
        m_duplicateSources[pDuplicateSource->GetName()] = pDuplicateSource;
        return true;
    }
    
    bool VideoSourceBintr::RemoveDuplicateSource(
        DSL_SOURCE_PTR pDuplicateSource)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR(
                "Unable to remove DuplicateSourceBintr '"
                << pDuplicateSource->GetName() << "' from VideoSourceBintr '"
                << GetName() << "' as it's currently linked");
            return false;
        }
        // ensure exists
        if (m_duplicateSources.find(pDuplicateSource->GetName()) 
            == m_duplicateSources.end())
        {   
            LOG_ERROR("DuplicateSourceBintr '" << pDuplicateSource->GetName()
                << "' has not been previously added to VideoSourceBintr '"
                << GetName() << "' and cannot removed");
            return false;
        }
        
        // remove the duplicate from the map of duplicates for this VideoSourceBintr
        m_duplicateSources.erase(pDuplicateSource->GetName());
        
        // if this was the last DuplicateSourceBintr to be removed, then we need
        // to delete the Tee and Queue elements used to support duplicates, and
        // set the Caps Filter back as source ghost pad.
        if (!m_duplicateSources.size())
        {
            m_pDuplicateSourceTeeQueue->RemoveGhostPadFromParent("src");
            m_pBufferOutCapsFilter->AddGhostPadToParent("src");
            RemoveChild(m_pDuplicateSourceTee);
            RemoveChild(m_pDuplicateSourceTeeQueue);
            m_pDuplicateSourceTee = nullptr;
            m_pDuplicateSourceTeeQueue = nullptr;
        }
        
        return true;
    }
    
    //*********************************************************************************
    DuplicateSourceBintr::DuplicateSourceBintr(const char* name, 
            const char* original, bool isLive)
        : SourceBintr(name) 
        , m_original(original)
    {
        LOG_FUNC();
        
        m_isLive = isLive;
        
        LOG_INFO("");
        LOG_INFO("Initial property values for DuplicateSourceBintr '" << name << "'");
        LOG_INFO("  original-source   : " << m_original);

        // Source queue is both "sink" and "src" ghost-pad for the DuplicateSourceBintr
        m_pQueue->AddGhostPadToParent("src");
        m_pQueue->AddGhostPadToParent("sink");
    }

    DuplicateSourceBintr::~DuplicateSourceBintr()
    {
        LOG_FUNC();
    }
    
    bool DuplicateSourceBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("DuplicateSourceBintr '" << GetName() 
                << "' is already in a linked state");
            return false;
        }
        // Single queue element, nothing to link
        m_isLinked = true;
        
        return true;
    }

    void DuplicateSourceBintr::UnlinkAll()
    {
        LOG_FUNC();

        if (!m_isLinked)
        {
            LOG_ERROR("DuplicateSourceBintr '" << GetName() 
                << "' is not in a linked state");
            return;
        }
        // Single queue element, nothing to unlink
        m_isLinked = false;
    }

    const char* DuplicateSourceBintr::GetOriginal()
    {
        LOG_FUNC();
        
        return m_original.c_str();
    }
    
    void DuplicateSourceBintr::SetOriginal(const char* original)
    {
        LOG_FUNC();
        
        m_original = original;
    }

    //*********************************************************************************
    AppSourceBintr::AppSourceBintr(const char* name, bool isLive, 
            const char* bufferInFormat, uint width, uint height, uint fpsN, uint fpsD)
        : VideoSourceBintr(name) 
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
        if (!set_full_caps(m_pSourceElement, m_videoMediaString.c_str(), 
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
        
//        if (!m_cudaDeviceProp.integrated)
//        {
//            m_pBufferOutVidConv->SetAttribute("nvbuf-memory-type", 
//                DSL_NVBUF_MEM_TYPE_CUDA_UNIFIED);
//        }

        LOG_INFO("");
        LOG_INFO("Initial property values for AppSourceBintr '" << name << "'");
        LOG_INFO("  buffer-in-format  : " << m_bufferInFormat);
        LOG_INFO("  is-live           : " << m_isLive);
        LOG_INFO("  do-timestamp      : " << m_doTimestamp);
        LOG_INFO("  stream-format     : " << m_streamFormat);
        LOG_INFO("  block-enabled     : " << m_blockEnabled);
        LOG_INFO("  max-bytes         : " << m_maxBytes);
        LOG_INFO("  width             : " << m_width);
        LOG_INFO("  height            : " << m_height);
        LOG_INFO("  fps-n             : " << m_fpsN);
        LOG_INFO("  fps-d             : " << m_fpsD);
        LOG_INFO("  media-out         : " << m_videoMediaString << "(memory:NVMM)");
        LOG_INFO("  buffer-out        : ");
        LOG_INFO("    format          : " << m_bufferOutFormat);
        LOG_INFO("    width           : " << m_bufferOutWidth);
        LOG_INFO("    height          : " << m_bufferOutHeight);
        LOG_INFO("    fps-n           : " << m_bufferOutFpsN);
        LOG_INFO("    fps-d           : " << m_bufferOutFpsD);
        LOG_INFO("    crop-pre-conv   : 0:0:0:0" );
        LOG_INFO("    crop-post-conv  : 0:0:0:0" );
        LOG_INFO("    orientation     : " << m_bufferOutOrientation);
        LOG_INFO("  queue             : " );
        LOG_INFO("    leaky           : " << m_leaky);
        LOG_INFO("    max-size        : ");
        LOG_INFO("      buffers       : " << m_maxSizeBuffers);
        LOG_INFO("      bytes         : " << m_maxSizeBytes);
        LOG_INFO("      time          : " << m_maxSizeTime);
        LOG_INFO("    min-threshold   : ");
        LOG_INFO("      buffers       : " << m_minThresholdBuffers);
        LOG_INFO("      bytes         : " << m_minThresholdBytes);
        LOG_INFO("      time          : " << m_minThresholdTime);

        // TODO support GST 1.20 properties
        // LOG_INFO("max-buffers = " << m_maxBuffers);
        // LOG_INFO("max-time    = " << m_maxTime);
        // LOG_INFO("leaky-type  = " << m_leakyType);

        // add all elementrs as childer to this Bintr
        AddChild(m_pSourceElement);
    }

    AppSourceBintr::~AppSourceBintr()
    {
        LOG_FUNC();
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
        
        if (!LinkToCommon(m_pSourceElement))
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
        UnlinkCommon();
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

    bool AppSourceBintr::SetDimensions(uint width, uint height)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't set dimensions for AppSourceBintr '" 

                << GetName() << "' as it's currently in a linked state");
            return false;
        }
        
        m_width = width;
        m_height = height;

        // Set the full capabilities (format, dimensions, and framerate)
        // NVIDIA plugin = false... this is a GStreamer plugin
        if (!set_full_caps(m_pSourceElement, m_videoMediaString.c_str(), 
            m_bufferInFormat.c_str(), m_width, m_height, m_fpsN, m_fpsD, false))
        {
            return false;
        }
        return true;
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
            LOG_ERROR("Can't set block-enabled for AppSourceBintr '" 

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


    static void on_need_data_cb(GstElement* source, uint length,
        gpointer pAppSrcBintr)
    {
        static_cast<AppSourceBintr*>(pAppSrcBintr)->
            HandleNeedData(length);
    }
        
    static void on_enough_data_cb(GstElement* source, 
        gpointer pAppSrcBintr)
    {
        static_cast<AppSourceBintr*>(pAppSrcBintr)->
            HandleEnoughData();
    }

    //*********************************************************************************
    CustomSourceBintr::CustomSourceBintr(const char* name, bool isLive)
        : VideoSourceBintr(name) 
        , m_nextElementIndex(0)
    {
        LOG_FUNC();
        
        m_isLive = isLive;
        
        LOG_INFO("");
        LOG_INFO("Initial property values for CustomSourceBintr '" << name << "'");
        LOG_INFO("  is-live           : " << m_isLive);
        LOG_INFO("  width             : " << m_width);
        LOG_INFO("  height            : " << m_height);
        LOG_INFO("  fps-n             : " << m_fpsN);
        LOG_INFO("  fps-d             : " << m_fpsD);
        LOG_INFO("  buffer-out        : ");
        LOG_INFO("    format          : " << m_bufferOutFormat);
        LOG_INFO("    width           : " << m_bufferOutWidth);
        LOG_INFO("    height          : " << m_bufferOutHeight);
        LOG_INFO("    fps-n           : " << m_bufferOutFpsN);
        LOG_INFO("    fps-d           : " << m_bufferOutFpsD);
        LOG_INFO("    crop-pre-conv   : 0:0:0:0" );
        LOG_INFO("    crop-post-conv  : 0:0:0:0" );
        LOG_INFO("    orientation     : " << m_bufferOutOrientation);
        LOG_INFO("  queue             : " );
        LOG_INFO("    leaky           : " << m_leaky);
        LOG_INFO("    max-size        : ");
        LOG_INFO("      buffers       : " << m_maxSizeBuffers);
        LOG_INFO("      bytes         : " << m_maxSizeBytes);
        LOG_INFO("      time          : " << m_maxSizeTime);
        LOG_INFO("    min-threshold   : ");
        LOG_INFO("      buffers       : " << m_minThresholdBuffers);
        LOG_INFO("      bytes         : " << m_minThresholdBytes);
        LOG_INFO("      time          : " << m_minThresholdTime);

    }

    CustomSourceBintr::~CustomSourceBintr()
    {
        LOG_FUNC();
    }
    
    bool CustomSourceBintr::AddChild(DSL_ELEMENT_PTR pChild)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't add child '" << pChild->GetName() 
                << "' to CustomSourceBintr '" << m_name 
                << "' as it is currently linked");
            return false;
        }
        if (IsChild(pChild))
        {
            LOG_ERROR("GstElementr '" << pChild->GetName() 
                << "' is already a child of CustomSourceBintr '" 
                << GetName() << "'");
            return false;
        }
 
        // increment next index, assign to the Element.
        pChild->SetIndex(++m_nextElementIndex);

        // Add the shared pointer to the CustomSourceBintr to the indexed map 
        // and as a child.
        m_elementrsIndexed[m_nextElementIndex] = pChild;
        return GstNodetr::AddChild(pChild);
    }
    
    bool CustomSourceBintr::RemoveChild(DSL_ELEMENT_PTR pChild)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't remove child '" << pChild->GetName() 
                << "' from CustomSourceBintr '" << m_name 
                << "' as it is currently linked");
            return false;
        }
        if (!IsChild(pChild))
        {
            LOG_ERROR("GstElementr '" << pChild->GetName() 
                << "' is not a child of CustomSourceBintr '" 
                << GetName() << "'");
            return false;
        }
        
        // Remove the shared pointer to the CustomSourceBintr from the indexed map  
        // and as a child.
        m_elementrsIndexed.erase(pChild->GetIndex());
        return GstNodetr::RemoveChild(pChild);
    }
    
    bool CustomSourceBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("CustomSourceBintr '" << m_name 
                << "' is already linked");
            return false;
        }
        if (!m_elementrsIndexed.size()) 
        {
            LOG_ERROR("CustomSourceBintr '" << m_name 
                << "' has no Elements to link");
            return false;
        }
        for (auto const &imap: m_elementrsIndexed)
        {
            // Link the Elementr to the last/previous Elementr in the vector 
            // of linked Elementrs 
            if (m_elementrsLinked.size() and 
                !m_elementrsLinked.back()->LinkToSink(imap.second))
            {
                return false;
            }
            // Add Elementr to the end of the linked Elementrs vector.
            m_elementrsLinked.push_back(imap.second);

            LOG_INFO("CustomSourceBintr '" << GetName() 
                << "' Linked up child Elementr '" << 
                imap.second->GetName() << "' successfully");                    
        }
        // Link the back element to the common VideoSource buffer out elements
        LinkToCommon(m_elementrsLinked.back());

        m_isLinked = true;
        
        return true;
    }
    
    void CustomSourceBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("CustomSourceBintr '" << m_name 
                << "' is not linked");
            return;
        }
        if (!m_elementrsLinked.size()) 
        {
            LOG_ERROR("CustomSourceBintr '" << m_name 
                << "' has no Elements to unlink");
            return;
        }
        // Unlink the common elements
        UnlinkCommon();
        
        // iterate through the list of Linked Components, unlinking each
        for (auto const& ivector: m_elementrsLinked)
        {
            // all but the tail element will be Linked to Sink
            if (ivector->IsLinkedToSink())
            {
                ivector->UnlinkFromSink();
            }
        }
        m_elementrsLinked.clear();

        m_isLinked = false;
    }

    //*********************************************************************************
    // Initilize the unique id list for all CsiSourceBintrs 
    std::list<uint> CsiSourceBintr::s_uniqueSensorIds;

    CsiSourceBintr::CsiSourceBintr(const char* name, 
        guint width, guint height, guint fpsN, guint fpsD)
        : VideoSourceBintr(name)
        , m_sensorId(0)
    {
        LOG_FUNC();

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
        
        
        // DS 6.2 ONLY - removed in DS 6.3 AND 6.4
        if (NVDS_VERSION_MAJOR < 7 && NVDS_VERSION_MINOR < 3)
        {
            m_pSourceElement->SetAttribute("bufapi-version", TRUE);
        }
        // Set the full capabilities (format, dimensions, and framerate)
        // Note: nvarguscamerasrc supports NV12 and P010_10LE formats only.
        if (!set_full_caps(m_pSourceCapsFilter, m_videoMediaString.c_str(), "NV12",
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
        LOG_INFO("  width             : " << m_width);
        LOG_INFO("  height            : " << m_height);
        LOG_INFO("  fps-n             : " << m_fpsN);
        LOG_INFO("  fps-d             : " << m_fpsD);
        LOG_INFO("  media-out         : " << m_videoMediaString << "(memory:NVMM)");
        LOG_INFO("  buffer-out        : ");
        LOG_INFO("    format          : " << m_bufferOutFormat);
        LOG_INFO("    width           : " << m_bufferOutWidth);
        LOG_INFO("    height          : " << m_bufferOutHeight);
        LOG_INFO("    fps-n           : " << m_bufferOutFpsN);
        LOG_INFO("    fps-d           : " << m_bufferOutFpsD);
        LOG_INFO("    crop-pre-conv   : 0:0:0:0" );
        LOG_INFO("    crop-post-conv  : 0:0:0:0" );
        LOG_INFO("    orientation     : " << m_bufferOutOrientation);
        LOG_INFO("  queue             : " );
        LOG_INFO("    leaky           : " << m_leaky);
        LOG_INFO("    max-size        : ");
        LOG_INFO("      buffers       : " << m_maxSizeBuffers);
        LOG_INFO("      bytes         : " << m_maxSizeBytes);
        LOG_INFO("      time          : " << m_maxSizeTime);
        LOG_INFO("    min-threshold   : ");
        LOG_INFO("      buffers       : " << m_minThresholdBuffers);
        LOG_INFO("      bytes         : " << m_minThresholdBytes);
        LOG_INFO("      time          : " << m_minThresholdTime);

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
            !LinkToCommon(m_pSourceCapsFilter))
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
        UnlinkCommon();
        
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

    V4l2SourceBintr::V4l2SourceBintr(const char* name, const char* deviceLocation)
        : VideoSourceBintr(name)
        , m_deviceLocation(deviceLocation)
    {
        LOG_FUNC();

        // Set the buffer-out-format to the default video format
        std::wstring L_bufferOutFormat(DSL_VIDEO_FORMAT_DEFAULT);
        m_bufferOutFormat.assign(L_bufferOutFormat.begin(), 
            L_bufferOutFormat.end());

        m_pSourceElement = DSL_ELEMENT_NEW("v4l2src", name);

        m_pSourceElement->GetAttribute("device-fd", &m_deviceFd);
        m_pSourceElement->GetAttribute("flags", &m_deviceFlags);
        m_pSourceElement->GetAttribute("brightness", &m_brightness);
        m_pSourceElement->GetAttribute("contrast", &m_contrast);
        m_pSourceElement->GetAttribute("hue", &m_hue);
        
        m_pSourceCapsFilter = DSL_ELEMENT_EXT_NEW("capsfilter", name, "1");

        m_pSourceElement->SetAttribute("device", m_deviceLocation.c_str());

        // Get property defaults that aren't specifically set
        m_pSourceElement->GetAttribute("do-timestamp", &m_doTimestamp);
        
        // Set the capabilities - do not set the format. 
        // Dimensions and framerate are set conditionally (non zero).
        DslCaps Caps(m_videoMediaString.c_str(), NULL, 
            m_width, m_height, m_fpsN, m_fpsD, false);

        m_pSourceCapsFilter->SetAttribute("caps", &Caps);

        if (!m_cudaDeviceProp.integrated)
        {
            m_pdGpuVidConv = DSL_ELEMENT_EXT_NEW("videoconvert", name, "1");
            AddChild(m_pdGpuVidConv);
        }
        
        LOG_INFO("");
        LOG_INFO("Initial property values for V4l2SourceBintr '" << name << "'");
        LOG_INFO("  is-live           : " << m_isLive);
        LOG_INFO("  device            : " << m_deviceLocation.c_str());
        LOG_INFO("  device-name       : " << m_deviceName);
        LOG_INFO("  device-fd         : " << m_deviceFd);
        LOG_INFO("  flags             : " << int_to_hex(m_deviceFlags));
        LOG_INFO("  brightness        : " << m_brightness);
        LOG_INFO("  contrast          : " << m_contrast);
        LOG_INFO("  hue               : " << m_hue);
        LOG_INFO("  width             : " << m_width);
        LOG_INFO("  height            : " << m_height);
        LOG_INFO("  fps-n             : " << m_fpsN);
        LOG_INFO("  fps-d             : " << m_fpsD);
        LOG_INFO("  do-timestamp      : " << m_doTimestamp);
        LOG_INFO("  media-out         : " << m_videoMediaString << "(memory:NVMM)");
        LOG_INFO("  buffer-out        : ");
        LOG_INFO("    format          : " << m_bufferOutFormat);
        LOG_INFO("    width           : " << m_bufferOutWidth);
        LOG_INFO("    height          : " << m_bufferOutHeight);
        LOG_INFO("    fps-n           : " << m_bufferOutFpsN);
        LOG_INFO("    fps-d           : " << m_bufferOutFpsD);
        LOG_INFO("    crop-pre-conv   : 0:0:0:0" );
        LOG_INFO("    crop-post-conv  : 0:0:0:0" );
        LOG_INFO("    orientation     : " << m_bufferOutOrientation);
        LOG_INFO("  queue             : " );
        LOG_INFO("    leaky           : " << m_leaky);
        LOG_INFO("    max-size        : ");
        LOG_INFO("      buffers       : " << m_maxSizeBuffers);
        LOG_INFO("      bytes         : " << m_maxSizeBytes);
        LOG_INFO("      time          : " << m_maxSizeTime);
        LOG_INFO("    min-threshold   : ");
        LOG_INFO("      buffers       : " << m_minThresholdBuffers);
        LOG_INFO("      bytes         : " << m_minThresholdBytes);
        LOG_INFO("      time          : " << m_minThresholdTime);

        AddChild(m_pSourceElement);
        AddChild(m_pSourceCapsFilter);
    }

    V4l2SourceBintr::~V4l2SourceBintr()
    {
        LOG_FUNC();
    }

    bool V4l2SourceBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("V4l2SourceBintr '" << GetName() << "' is already in a linked state");
            return false;
        }
        
        // x86_64
        if (!m_cudaDeviceProp.integrated)
        {
            if (!m_pSourceElement->LinkToSink(m_pSourceCapsFilter) or
                !m_pSourceCapsFilter->LinkToSink(m_pdGpuVidConv) or 
                !LinkToCommon(m_pdGpuVidConv))
            {
                return false;
            }
        }
        else // aarch_64
        {
            if (!m_pSourceElement->LinkToSink(m_pSourceCapsFilter) or
                !LinkToCommon(m_pSourceCapsFilter))
            {
                return false;
            }
        }
        m_isLinked = true;
        
        return true;
    }

    void V4l2SourceBintr::UnlinkAll()
    {
        LOG_FUNC();

        if (!m_isLinked)
        {
            LOG_ERROR("V4l2SourceBintr '" << GetName() << "' is not in a linked state");
            return;
        }
        
        m_pSourceElement->UnlinkFromSink();

        if (!m_cudaDeviceProp.integrated)
        {
            m_pSourceCapsFilter->UnlinkFromSink();
        }
        UnlinkCommon();
        m_isLinked = false;
    }

    const char* V4l2SourceBintr::GetDeviceLocation()
    {
        LOG_FUNC();

        return m_deviceLocation.c_str();
    }
    
    bool V4l2SourceBintr::SetDeviceLocation(const char* deviceLocation)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't set device-location for V4l2SourceBintr '" << GetName() 
                << "' as it is currently in a linked state");
            return false;
        }
        
        m_deviceLocation = deviceLocation;
        
        m_pSourceElement->SetAttribute("device", m_deviceLocation.c_str());
        return true;
    }

    bool V4l2SourceBintr::SetDimensions(uint width, uint height)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't set dimensions for V4l2SourceBintr '" 
                << GetName() << "' as it is currently in a linked state");
            return false;
        }
        m_width = width;
        m_height = height;

        // Set the capabilities - do not set the format. 
        // Dimensions and framerate are set conditionally (non zero).
        DslCaps Caps(m_videoMediaString.c_str(), NULL, 
            m_width, m_height, m_fpsN, m_fpsD, false);

        m_pSourceCapsFilter->SetAttribute("caps", &Caps);
        
        return true;
    }
    
    bool V4l2SourceBintr::SetFrameRate(uint fpsN, uint fpsD)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't set frame-rate for V4l2SourceBintr '" 
                << GetName() << "' as it is currently in a linked state");
            return false;
        }
        m_fpsN = fpsN;
        m_fpsD = fpsD;
        
        // Set the capabilities - do not set the format. 
        // Dimensions and framerate are set conditionally (non zero).
        DslCaps Caps(m_videoMediaString.c_str(), NULL, 
            m_width, m_height, m_fpsN, m_fpsD, false);

        m_pSourceCapsFilter->SetAttribute("caps", &Caps);
        
        return true;
    }
    
    const char* V4l2SourceBintr::GetDeviceName()
    {
        LOG_FUNC();
        
        // default to no device-name
        m_deviceName = "";

        const char* deviceName(NULL);
        m_pSourceElement->GetAttribute("device-name", &deviceName);
        
        // Update if set
        if (deviceName)
        {
            m_deviceName = deviceName;
        }
            
        return m_deviceName.c_str();
    }
    
    int V4l2SourceBintr::GetDeviceFd()
    {
        LOG_FUNC();

        m_pSourceElement->GetAttribute("device-fd", &m_deviceFd);
        return m_deviceFd;
    }
    
    uint V4l2SourceBintr::GetDeviceFlags()
    {
        LOG_FUNC();

        m_pSourceElement->GetAttribute("flags", &m_deviceFlags);
        return m_deviceFlags;
    }
    
    void V4l2SourceBintr::GetPictureSettings(int* brightness, 
        int* contrast, int* hue)
    {
        LOG_FUNC();
        
        m_pSourceElement->GetAttribute("brightness", &m_brightness);
        m_pSourceElement->GetAttribute("contrast", &m_contrast);
        m_pSourceElement->GetAttribute("hue", &m_hue);

        *brightness = m_brightness;
        *contrast = m_contrast;
        *hue = m_hue;
    }

    bool V4l2SourceBintr::SetPictureSettings(int brightness, 
        int contrast, int hue)
    {
        LOG_FUNC();

        if (m_brightness != brightness)
        {
            m_brightness = brightness;
            m_pSourceElement->SetAttribute("brightness", m_brightness);
            LOG_INFO("V4l2SourceBintr '" << GetName() 
                << "' set brightness level to " << m_brightness);
        }
        if (m_contrast != contrast)
        {
            m_contrast = contrast;
            m_pSourceElement->SetAttribute("contrast", m_contrast);            
            LOG_INFO("V4l2SourceBintr '" << GetName() 
                << "' set contrast level to " << m_contrast);
        }
        if (m_hue != hue)
        {
            m_hue = hue;
            m_pSourceElement->SetAttribute("hue", m_hue);
            LOG_INFO("V4l2SourceBintr '" << GetName() 
                << "' set hue level to " << m_hue);
        }
        return true;
    }


    //*********************************************************************************

    UriSourceBintr::UriSourceBintr(const char* name, const char* uri, bool isLive,
        uint skipFrames, uint dropFrameInterval)
        : ResourceSourceBintr(name, uri)
        , m_isFullyLinked(false)
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
        
        m_pSourceElement = DSL_ELEMENT_NEW("uridecodebin", name);
        
        if (!SetUri(uri))
        {   
            throw;
        }

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
        LOG_INFO("  width               : " << m_width);
        LOG_INFO("  height              : " << m_height);
        LOG_INFO("  fps-n               : " << m_fpsN);
        LOG_INFO("  fps-d               : " << m_fpsD);
        LOG_INFO("  media-out           : " << m_videoMediaString << "(memory:NVMM)");
        LOG_INFO("  buffer-out          : ");
        LOG_INFO("    format            : " << m_bufferOutFormat);
        LOG_INFO("    width             : " << m_bufferOutWidth);
        LOG_INFO("    height            : " << m_bufferOutHeight);
        LOG_INFO("    fps-n             : " << m_bufferOutFpsN);
        LOG_INFO("    fps-d             : " << m_bufferOutFpsD);
        LOG_INFO("    crop-pre-conv     : 0:0:0:0" );
        LOG_INFO("    crop-post-conv    : 0:0:0:0" );
        LOG_INFO("    orientation       : " << m_bufferOutOrientation);
        LOG_INFO("  queue             : " );
        LOG_INFO("    leaky           : " << m_leaky);
        LOG_INFO("    max-size        : ");
        LOG_INFO("      buffers       : " << m_maxSizeBuffers);
        LOG_INFO("      bytes         : " << m_maxSizeBytes);
        LOG_INFO("      time          : " << m_maxSizeTime);
        LOG_INFO("    min-threshold   : ");
        LOG_INFO("      buffers       : " << m_minThresholdBuffers);
        LOG_INFO("      bytes         : " << m_minThresholdBytes);
        LOG_INFO("      time          : " << m_minThresholdTime);

        // Add all new Elementrs as Children to the SourceBintr
        AddChild(m_pSourceElement);
    }

    UriSourceBintr::~UriSourceBintr()
    {
        LOG_FUNC();
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
            LOG_INFO("File Path for UriSourceBintr '" << GetName() 
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
        
        // Try to open the file and read the frame-rate and dimensions.

#if (BUILD_WITH_FFMPEG == true) || (BUILD_WITH_OPENCV == true)
        try
        {
            AvInputFile avFile(uri);
            m_fpsN = avFile.fpsN;
            m_fpsD = avFile.fpsD;
            m_width = avFile.videoWidth; 
            m_height = avFile.videoHeight;
        }
        catch(...)
        {
            return false;
        }
#else
        LOG_WARN(
            "Unable to determine video frame-rate and dimensions for URI Source = '"
            << GetName() << "' Extended AV File Services are disabled in the Makefile");
#endif        

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

        if (m_isFullyLinked)
        {
            UnlinkCommon();
        }
        m_isFullyLinked = false;
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
            LinkToCommon(pPad);
            m_isFullyLinked = true;
            
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
                // DS 6.2 ONLY - removed in DS 6.3 AND 6.4
                if (NVDS_VERSION_MAJOR < 7 && NVDS_VERSION_MINOR < 3)
                {
                    g_object_set(pObject, "bufapi-version", TRUE, NULL);
                }
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
                    
                // Note - m_pDecoderStaticSinkpad unreferenced in DisableEosConsumer
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

            AddChild(m_pParser);
            AddChild(m_pDecoder);

            // If it's an MJPG file or Multi JPG files
            if (m_uri.find("mjpeg") != std::string::npos or
                m_uri.find("mjpg") != std::string::npos or
                m_uri.find("mp4") != std::string::npos or
                type == DSL_IMAGE_TYPE_MULTI)
            {
                // aarch64 (Jetson) only
                if (m_cudaDeviceProp.integrated)
                {
                    LOG_INFO("Setting decoder 'mjpeg' attribute for ImageSourceBintr '" 
                        << GetName() << "'");
                    m_mjpeg = TRUE;
                    m_pDecoder->SetAttribute("mjpeg", m_mjpeg);
                }
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

        LOG_INFO("");
        LOG_INFO("Initial property values for SingleImageSourceBintr '" << name << "'");
        LOG_INFO("  location          : " << uri);
        LOG_INFO("  is-live           : " << m_isLive);
        LOG_INFO("  media in          : " << "image/jpeg");
        LOG_INFO("  mjpeg             : " << m_mjpeg);
        LOG_INFO("  width             : " << m_width);
        LOG_INFO("  height            : " << m_height);
        LOG_INFO("  media-out         : " << m_videoMediaString << "(memory:NVMM)");
        LOG_INFO("  buffer-out        : ");
        LOG_INFO("    format          : " << m_bufferOutFormat);
        LOG_INFO("    width           : " << m_bufferOutWidth);
        LOG_INFO("    height          : " << m_bufferOutHeight);
        LOG_INFO("    fps-n           : " << m_bufferOutFpsN);
        LOG_INFO("    fps-d           : " << m_bufferOutFpsD);
        LOG_INFO("    crop-pre-conv   : 0:0:0:0" );
        LOG_INFO("    crop-post-conv  : 0:0:0:0" );
        LOG_INFO("    orientation     : " << m_bufferOutOrientation);
        LOG_INFO("  queue             : " );
        LOG_INFO("    leaky           : " << m_leaky);
        LOG_INFO("    max-size        : ");
        LOG_INFO("      buffers       : " << m_maxSizeBuffers);
        LOG_INFO("      bytes         : " << m_maxSizeBytes);
        LOG_INFO("      time          : " << m_maxSizeTime);
        LOG_INFO("    min-threshold   : ");
        LOG_INFO("      buffers       : " << m_minThresholdBuffers);
        LOG_INFO("      bytes         : " << m_minThresholdBytes);
        LOG_INFO("      time          : " << m_minThresholdTime);

        AddChild(m_pSourceElement);
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
            !LinkToCommon(m_pDecoder))
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
        m_pSourceElement->UnlinkFromSink();
        m_pParser->UnlinkFromSink();
        UnlinkCommon();
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

#if (BUILD_WITH_FFMPEG == true) || (BUILD_WITH_OPENCV == true)
        // Try to open the file and read the dimensions.
        try
        {
            AvInputFile avFile(uri);
            m_width = avFile.videoWidth;
            m_height = avFile.videoHeight;
        }
        catch(...)
        {
            return false;
        }
#else
        LOG_WARN(
            "Unable to determine video frame-rate and dimensions for URI Source = '"
            << GetName() << "' Extended AV File Services are disabled in the Makefile");
#endif        

        char absolutePath[PATH_MAX+1];
        m_uri.assign(realpath(uri, absolutePath));

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
        LOG_INFO("  uri               : " << uri);
        LOG_INFO("  is-live           : " << m_isLive);
        LOG_INFO("  media in          : " << "image/jpeg");
        LOG_INFO("  loop              : " << m_loopEnabled);
        LOG_INFO("  start-index       : " << m_startIndex);
        LOG_INFO("  stop-index        : " << m_stopIndex);
        LOG_INFO("  width             : " << m_width);
        LOG_INFO("  height            : " << m_height);
        LOG_INFO("  fps-n             : " << m_fpsN);
        LOG_INFO("  fps-d             : " << m_fpsD);
        LOG_INFO("  media-out         : " << m_videoMediaString << "(memory:NVMM)");
        LOG_INFO("  buffer-out        : ");
        LOG_INFO("    format          : " << m_bufferOutFormat);
        LOG_INFO("    width           : " << m_bufferOutWidth);
        LOG_INFO("    height          : " << m_bufferOutHeight);
        LOG_INFO("    fps-n           : " << m_bufferOutFpsN);
        LOG_INFO("    fps-d           : " << m_bufferOutFpsD);
        LOG_INFO("    crop-pre-conv   : 0:0:0:0" );
        LOG_INFO("    crop-post-conv  : 0:0:0:0" );
        LOG_INFO("    orientation     : " << m_bufferOutOrientation);
        LOG_INFO("  queue             : " );
        LOG_INFO("    leaky           : " << m_leaky);
        LOG_INFO("    max-size        : ");
        LOG_INFO("      buffers       : " << m_maxSizeBuffers);
        LOG_INFO("      bytes         : " << m_maxSizeBytes);
        LOG_INFO("      time          : " << m_maxSizeTime);
        LOG_INFO("    min-threshold   : ");
        LOG_INFO("      buffers       : " << m_minThresholdBuffers);
        LOG_INFO("      bytes         : " << m_minThresholdBytes);
        LOG_INFO("      time          : " << m_minThresholdTime);
        
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
            !LinkToCommon(m_pDecoder))
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
        
        m_pSourceElement->UnlinkFromSink();
        m_pParser->UnlinkFromSink();
        UnlinkCommon();
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

        m_pSourceElement->SetAttribute("is-live", m_isLive); 
        m_pSourceElement->SetAttribute("pattern", 2); // 2 = black
        
        if(uri and !SetUri(uri))
        {
            throw;
        }

        LOG_INFO("");
        LOG_INFO("Initial property values for ImageStreamSourceBintr '" << name << "'");
        LOG_INFO("  uri               : " << uri);
        LOG_INFO("  is-live           : " << m_isLive);
        LOG_INFO("  width             : " << m_width);
        LOG_INFO("  height            : " << m_height);
        LOG_INFO("  fps-n             : " << m_fpsN);
        LOG_INFO("  fps-d             : " << m_fpsD);
        LOG_INFO("  media-out         : " << m_videoMediaString << "(memory:NVMM)");
        LOG_INFO("  buffer-out        : ");
        LOG_INFO("    format          : " << m_bufferOutFormat);
        LOG_INFO("    width           : " << m_bufferOutWidth);
        LOG_INFO("    height          : " << m_bufferOutHeight);
        LOG_INFO("    fps-n           : " << m_bufferOutFpsN);
        LOG_INFO("    fps-d           : " << m_bufferOutFpsD);
        LOG_INFO("    crop-pre-conv   : 0:0:0:0" );
        LOG_INFO("    crop-post-conv  : 0:0:0:0" );
        LOG_INFO("    orientation     : " << m_bufferOutOrientation);
        LOG_INFO("  queue             : " );
        LOG_INFO("    leaky           : " << m_leaky);
        LOG_INFO("    max-size        : ");
        LOG_INFO("      buffers       : " << m_maxSizeBuffers);
        LOG_INFO("      bytes         : " << m_maxSizeBytes);
        LOG_INFO("      time          : " << m_maxSizeTime);
        LOG_INFO("    min-threshold   : ");
        LOG_INFO("      buffers       : " << m_minThresholdBuffers);
        LOG_INFO("      bytes         : " << m_minThresholdBytes);
        LOG_INFO("      time          : " << m_minThresholdTime);

        // Add all new Elementrs as Children to the SourceBintr
        AddChild(m_pSourceElement);
        AddChild(m_pSourceCapsFilter);
        AddChild(m_pImageOverlay);

    }
    
    ImageStreamSourceBintr::~ImageStreamSourceBintr()
    {
        LOG_FUNC();
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
        
#if (BUILD_WITH_FFMPEG == true) || (BUILD_WITH_OPENCV == true)
        // Try to open the file and read the dimensions.
        try
        {
            AvInputFile avFile(uri);
            m_width = avFile.videoWidth;
            m_height = avFile.videoHeight;
        }
        catch(...)
        {
            return false;
        }
#else
        LOG_WARN(
            "Unable to determine video frame-rate and dimensions for URI Source = '"
            << GetName() << "' Extended AV File Services are disabled in the Makefile");
#endif        
        
        // Set the full capabilities (format and framerate)
        if (!set_full_caps(m_pSourceCapsFilter, m_videoMediaString.c_str(), 
            m_bufferOutFormat.c_str(), m_width, m_height, m_fpsN, m_fpsD, false))
        {
            return false;
        }

        // Setup the full path
        char absolutePath[PATH_MAX+1];
        m_uri.assign(realpath(uri, absolutePath));

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
            !LinkToCommon(m_pImageOverlay))
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
        
        m_pSourceElement->UnlinkFromSink();
        m_pSourceCapsFilter->UnlinkFromSink();
        UnlinkCommon();
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
        : VideoSourceBintr(name)
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
        LOG_INFO("  width               : " << m_width);
        LOG_INFO("  height              : " << m_height);
        LOG_INFO("  fps-n               : " << m_fpsN);
        LOG_INFO("  fps-d               : " << m_fpsD);
        LOG_INFO("  media-out           : " << m_videoMediaString << "(memory:NVMM)");
        LOG_INFO("  buffer-out          : ");
        LOG_INFO("    format            : " << m_bufferOutFormat);
        LOG_INFO("    width             : " << m_bufferOutWidth);
        LOG_INFO("    height            : " << m_bufferOutHeight);
        LOG_INFO("    fps-n             : " << m_bufferOutFpsN);
        LOG_INFO("    fps-d             : " << m_bufferOutFpsD);
        LOG_INFO("    crop-pre-conv     : 0:0:0:0" );
        LOG_INFO("    crop-post-conv    : 0:0:0:0" );
        LOG_INFO("    orientation       : " << m_bufferOutOrientation);
        LOG_INFO("  queue               : " );
        LOG_INFO("    leaky             : " << m_leaky);
        LOG_INFO("    max-size          : ");
        LOG_INFO("      buffers         : " << m_maxSizeBuffers);
        LOG_INFO("      bytes           : " << m_maxSizeBytes);
        LOG_INFO("      time            : " << m_maxSizeTime);
        LOG_INFO("    min-threshold     : ");
        LOG_INFO("      buffers         : " << m_minThresholdBuffers);
        LOG_INFO("      bytes           : " << m_minThresholdBytes);
        LOG_INFO("      time            : " << m_minThresholdTime);

        // Add the new Elementr as a Child to the SourceBintr
        AddChild(m_pSourceElement);
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

        if (!LinkToCommon(m_pSourceElement))
        {
            LOG_ERROR("InterpipeSourceBintr '" << GetName() << "' failed to LinkAll");
            return false;
        }
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
        UnlinkCommon();
        
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
        , m_isFullyLinked(false)
        , m_skipFrames(skipFrames)
        , m_dropFrameInterval(dropFrameInterval)
        , m_numExtraSurfaces(DSL_DEFAULT_NUM_EXTRA_SURFACES)
        , m_rtpProtocols(protocol)
        , m_latency(latency)
        , m_firstConnectTime(0)
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
        // ---------------------------------------------------------------------------
        // The RTSP Source is linked in one of two ways depending on whether
        // A tap-bintr has been added or not.
        //
        // With tap-bintr, the parser is added between the depay and the recordbin
        // as discussed here: https://forums.developer.nvidia.com/t/questions-re-differences-between-rtsp-source-in-deepstream-source-bin-c-deepstream-test-sr-app-c/245307/6
        // The decoder will do its own parsing in this case.
        //
        //                           |->queue->decoder->[common-elements]->
        //        rtcpsrc->depay->tee
        //                           |->parser->tap-bintr
        //
        // Without tap-bintr, we add capsfilter->parser to main stream. This is to 
        // support RTSP Sources that are forwarded trough streaming services. These
        // sources will NOT be able to connect with a tap-bintr as above.
        //
        //        rtcpsrc->depay->capsfilter->parser->decoder->[common-elements]->
        //
        // ---------------------------------------------------------------------------
        
        // update the is-live variable (initiated as false)
        m_isLive = true;

        // New RTSP Specific Elementrs for this Source
        m_pSourceElement = DSL_ELEMENT_NEW("rtspsrc", name);
        
        m_pDepayCapsfilter = DSL_ELEMENT_EXT_NEW("capsfilter", name, "depay");
        
        // Pre-decode tee is only used if there is a TapBintr
        m_pPreDecodeTee = DSL_ELEMENT_NEW("tee", name);
        m_pPreDecodeQueue = DSL_ELEMENT_EXT_NEW("queue", name, "decodebin");
        m_pPreParserQueue = DSL_ELEMENT_EXT_NEW("queue", name, "parser");

        // Get the default properties
        m_pSourceElement->GetAttribute("tls-validation-flags", 
            &m_tlsValidationFlags);
        m_pSourceElement->GetAttribute("udp-buffer-size", &m_udpBufferSize);
        m_pSourceElement->GetAttribute("drop-on-latency", &m_dropOnLatency);
        
        // Configure the source to generate NTP sync values
        configure_source_for_ntp_sync(m_pSourceElement->GetGstElement());
        m_pSourceElement->SetAttribute("location", m_uri.c_str());

        m_pSourceElement->SetAttribute("latency", m_latency);
        m_pSourceElement->SetAttribute("protocols", m_rtpProtocols);

        // Connect RTSP Source Setup Callbacks
        g_signal_connect(m_pSourceElement->GetGObject(), "select-stream",
            G_CALLBACK(RtspSourceSelectStreamCB), this);

        g_signal_connect(m_pSourceElement->GetGObject(), "pad-added", 
            G_CALLBACK(RtspSourceElementOnPadAddedCB), this);

        // Same decoder for H.264, H.265, and JPEG
        m_pDecoder = DSL_ELEMENT_NEW("nvv4l2decoder", GetCStrName());
        
        if (m_skipFrames)
        {
            m_pDecoder->SetAttribute("skip-frames", m_skipFrames);
        }
        // aarch64 (Jetson) only
        if (m_cudaDeviceProp.integrated)
        {
            // For integrated GPUs only:
            // DS 6.2 requires bufapi-version to be enabled. 
            // This feature was deprecated in DS 6.3 and later.            
            if (NVDS_VERSION_MAJOR < 7 && NVDS_VERSION_MINOR < 3)
            {
                m_pDecoder->SetAttribute("bufapi-version", TRUE);
            }
            m_pDecoder->SetAttribute("enable-max-performance", TRUE);
        }
        m_pDecoder->SetAttribute("drop-frame-interval", m_dropFrameInterval);
        m_pDecoder->SetAttribute("num-extra-surfaces", m_numExtraSurfaces);

        LOG_INFO("");
        LOG_INFO("Initial property values for RtspSourceBintr '" << name << "'");
        LOG_INFO("  uri                  : " << m_uri);
        LOG_INFO("  is-live              : " << m_isLive);
        LOG_INFO("  skip-frames          : " << m_skipFrames);
        LOG_INFO("  latency              : " << m_latency);
        LOG_INFO("  drop-on-latency      : " << m_dropOnLatency);
        LOG_INFO("  drop-frame-interval  : " << m_dropFrameInterval);
        LOG_INFO("  tls-validation-flags : " << std::hex << m_tlsValidationFlags);
        LOG_INFO("  udp-buffer-size      : " << m_udpBufferSize);
        LOG_INFO("  width                : " << m_width);
        LOG_INFO("  height               : " << m_height);
        LOG_INFO("  fps-n                : " << m_fpsN);
        LOG_INFO("  fps-d                : " << m_fpsD);
        LOG_INFO("  media-out            : " << m_videoMediaString << "(memory:NVMM)");
        LOG_INFO("  buffer-out           : ");
        LOG_INFO("    format             : " << m_bufferOutFormat);
        LOG_INFO("    width              : " << m_bufferOutWidth);
        LOG_INFO("    height             : " << m_bufferOutHeight);
        LOG_INFO("    fps-n              : " << m_bufferOutFpsN);
        LOG_INFO("    fps-d              : " << m_bufferOutFpsD);
        LOG_INFO("    crop-pre-conv      : 0:0:0:0" );
        LOG_INFO("    crop-post-conv     : 0:0:0:0" );
        LOG_INFO("    orientation        : " << m_bufferOutOrientation);
        LOG_INFO("  queue                : " );
        LOG_INFO("    leaky              : " << m_leaky);
        LOG_INFO("    max-size           : ");
        LOG_INFO("      buffers          : " << m_maxSizeBuffers);
        LOG_INFO("      bytes            : " << m_maxSizeBytes);
        LOG_INFO("      time             : " << m_maxSizeTime);
        LOG_INFO("    min-threshold      : ");
        LOG_INFO("      buffers          : " << m_minThresholdBuffers);
        LOG_INFO("      bytes            : " << m_minThresholdBytes);
        LOG_INFO("      time             : " << m_minThresholdTime);

        AddChild(m_pSourceElement);
        AddChild(m_pDepayCapsfilter);
        AddChild(m_pPreDecodeTee);
        AddChild(m_pPreDecodeQueue);
        AddChild(m_pPreParserQueue);
        AddChild(m_pDecoder);

        // New timestamp PPH to stamp the time of the last buffer 
        // - used to monitor the RTSP connection
        std::string handlerName = GetName() + "-timestamp-pph";
        m_TimestampPph = DSL_PPH_TIMESTAMP_NEW(handlerName.c_str());
        
        m_pSrcPadBufferProbe->AddPadProbeHandler(m_TimestampPph);
        
        // Set the default connection param values
        m_connectionData.sleep = DSL_RTSP_CONNECTION_SLEEP_S;
        m_connectionData.timeout = DSL_RTSP_CONNECTION_TIMEOUT_S;
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
        
        m_pSrcPadBufferProbe->RemovePadProbeHandler(m_TimestampPph);
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

        // Note: all elements are linked in the select-stream and pad-added callbacks.

        // Start the Stream mangement timer, only if timeout is enable and 
        if (m_bufferTimeout)
        {
            // reset the first connect counter in case the pipeline is relinking 
            // and playing after a previous play and stop.
            m_firstConnectTime = 0;
            
            m_streamManagerTimerId = g_timeout_add(
                DSL_RTSP_TEST_FOR_BUFFER_TIMEOUT_PERIOD_MS, 
                RtspStreamManagerHandler, this);
            LOG_INFO("Starting stream management for RTSP Source '" 
                << GetName() << "'");
        }
        
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
        
        if (m_isFullyLinked)
        {
            if (HasTapBintr())
            {
                m_pPreDecodeQueue->UnlinkFromSourceTee();
                m_pPreParserQueue->UnlinkFromSourceTee();
                m_pPreParserQueue->UnlinkFromSink();
                m_pTapBintr->UnlinkAll();
            }
            else
            {
                m_pDepayCapsfilter->UnlinkFromSink();
            }
            m_pDepay->UnlinkFromSink();
            m_pParser->UnlinkFromSink();
            UnlinkCommon();
        }
        m_isLinked = false;
        m_isFullyLinked = false;
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

    void RtspSourceBintr::GetConnectionParams(uint* sleep, uint* timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_reconnectionManagerMutex);
        
        *sleep = m_connectionData.sleep;
        *timeout = m_connectionData.timeout;
    }
    
    bool RtspSourceBintr::SetConnectionParams(uint sleep, uint timeout)
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
    
    uint RtspSourceBintr::GetLatency()
    {
        LOG_FUNC();

        return m_latency;
    }

    bool RtspSourceBintr::SetLatency(uint latency)
    {
        LOG_FUNC();

        if (IsLinked())
        {
            LOG_ERROR("Unable to set latency for RtspSourceBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_latency = latency;
        m_pSourceElement->SetAttribute("latency", m_latency);
    
        return true;
    }
    
    boolean RtspSourceBintr::GetDropOnLatencyEnabled()
    {
        LOG_FUNC();

        return m_dropOnLatency;
    }

    bool RtspSourceBintr::SetDropOnLatencyEnabled(boolean dropOnLatency)
    {
        LOG_FUNC();

        if (IsLinked())
        {
            LOG_ERROR("Unable to set drop-on-latency for RtspSourceBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_dropOnLatency = dropOnLatency;
        m_pSourceElement->SetAttribute("drop-on-latency", m_dropOnLatency);
    
        return true;
    }
    
    guint RtspSourceBintr::GetTlsValidationFlags()
    {
        LOG_FUNC();

        return m_tlsValidationFlags;
    }
    
    bool RtspSourceBintr::SetTlsValidationFlags(uint flags)
    {
        LOG_FUNC();

        if (IsLinked())
        {
            LOG_ERROR("Unable to set tls-validation-flags for RtspSourceBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_tlsValidationFlags = flags;
        m_pSourceElement->SetAttribute("tls-validation-flags", 
            m_tlsValidationFlags);
    
        return true;
    }

    guint RtspSourceBintr::GetUdpBufferSize()
    {
        LOG_FUNC();

        return m_udpBufferSize;
    }
    
    bool RtspSourceBintr::SetUdpBufferSize(uint size)
    {
        LOG_FUNC();

        if (IsLinked())
        {
            LOG_ERROR("Unable to set udp-buffer-size for RtspSourceBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_udpBufferSize = size;
        m_pSourceElement->SetAttribute("udp-buffer-size", 
            m_udpBufferSize);
    
        return true;
    }

    bool RtspSourceBintr::AddStateChangeListener(
        dsl_state_change_listener_cb listener, void* userdata)
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

    bool RtspSourceBintr::HandleSelectStream(GstElement *pBin, 
        uint num, GstCaps *caps)
    {
        GstStructure *structure = gst_caps_get_structure(caps, 0);
        std::string media = gst_structure_get_string (structure, "media");
        std::string encoding = gst_structure_get_string (structure, "encoding-name");

        LOG_INFO("Media = '" << media << "' for RtspSourceBitnr '" 
            << GetName() << "'");
        LOG_INFO("Encoding = '" << encoding << "' for RtspSourceBitnr '" 
            << GetName() << "'");

        // Note we create a parser even if there is no TapBintr just to simplify
        // the logic. The parser is only used/linked if there is TapBintr
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
                GstCaps* pCaps;
                if (encoding.find("H264") != std::string::npos)
                {
                    m_pDepay = DSL_ELEMENT_NEW("rtph264depay", GetCStrName());
                    m_pParser = DSL_ELEMENT_NEW("h264parse", GetCStrName());
                    pCaps = gst_caps_from_string("video/x-h264");
                }
                else if (encoding.find("H265") != std::string::npos)
                {
                    m_pDepay = DSL_ELEMENT_NEW("rtph265depay", GetCStrName());
                    m_pParser = DSL_ELEMENT_NEW("h265parse", GetCStrName());
                    pCaps = gst_caps_from_string("video/x-h265");
                } 
                else
                {
                    LOG_ERROR("Unsupported encoding = '" << encoding 
                        << "' for RtspSourceBitnr '" << GetName() << "'");
                    return false;
                }
                m_pDepayCapsfilter->SetAttribute("caps", pCaps);
                gst_caps_unref(pCaps);  
            }
            else if (encoding.find("JPEG") != std::string::npos)
            {
                m_pDepay = DSL_ELEMENT_NEW("rtpjpegdepay", GetCStrName());
                m_pParser = DSL_ELEMENT_NEW("jpegparse", GetCStrName());

                // aarch64 (Jetson) only
                if (m_cudaDeviceProp.integrated)
                {
                    LOG_INFO("Setting decoder 'mjpeg' attribute for RtspSourceBintr '" 
                        << GetName() << "'");
                    m_pDecoder->SetAttribute("mjpeg", TRUE);
                }
            }
            else
            {
                LOG_ERROR("Unsupported encoding = '" << encoding 
                    << "' for RtspSourceBitnr '" << GetName() << "'");
                return false;
            }

            // The format specific depay, parser, and decoder bins have been selected, 
            // so we can add them as children to this RtspSourceBintr now.
            AddChild(m_pDepay);
            AddChild(m_pParser);

            // If we're tapping off of the pre-decode source stream
            if (HasTapBintr())
            {
                if (!m_pPreDecodeQueue->LinkToSink(m_pDecoder) or
                    !m_pPreDecodeQueue->LinkToSourceTee(m_pPreDecodeTee, "src_%u") or
                    !m_pDepay->LinkToSink(m_pPreDecodeTee) or 
                    !m_pPreParserQueue->LinkToSourceTee(m_pPreDecodeTee, "src_%u") or
                    !m_pPreParserQueue->LinkToSink(m_pParser) or
                    !m_pTapBintr->LinkAll() or 
                    !m_pParser->LinkToSink(m_pTapBintr) or
                    !gst_element_sync_state_with_parent(m_pTapBintr->GetGstElement()))
                {
                    LOG_ERROR("Failed to link and sync states with Parent for RtspSourceBitnr '" 
                        << GetName() << "'");
                    return false;
                }
            }
            // Otherwise, we include the capsfilter and parser before decoder.
            // This seems to be required for certain streaming services. These
            // streams will fail if trying to connect with a TapBintr above.
            // Needs review further... to see if there is a way to support both.
            // See also: https://forums.developer.nvidia.com/t/questions-re-differences-between-rtsp-source-in-deepstream-source-bin-c-deepstream-test-sr-app-c/245307/5
            else
            {
                if (!m_pDepay->LinkToSink(m_pDepayCapsfilter) or
                    !m_pDepayCapsfilter->LinkToSink(m_pParser) or
                    !m_pParser->LinkToSink(m_pDecoder))
                {
                    LOG_ERROR("Failed to link elements for RtspSourceBitnr '" 
                        << GetName() << "'");
                    return false;
                }            
            }
            
            if (!LinkToCommon(m_pDecoder))
            {
                LOG_ERROR(
                    "Failed to link decoder with common elements for RtspSourceBitnr '" 
                    << GetName() << "'");
                return false;
            }
            
            if (!gst_element_sync_state_with_parent(m_pDepay->GetGstElement()) or
                !gst_element_sync_state_with_parent(m_pParser->GetGstElement()))
            {
                LOG_ERROR(
                    "Failed to sync Depay/Parser states with Parent for RtspSourceBitnr '" 
                    << GetName() << "'");
                return false;
            }
        }
        return true;
    }
        
    void RtspSourceBintr::HandleSourceElementOnPadAdded(GstElement* pBin, 
        GstPad* pPad)
    {
        LOG_FUNC();

        GstCaps* pCaps = gst_pad_query_caps(pPad, NULL);
        GstStructure* structure = gst_caps_get_structure(pCaps, 0);
        std::string name = gst_structure_get_name(structure);
        std::string media = gst_structure_get_string (structure, "media");
        std::string encoding = gst_structure_get_string (structure, "encoding-name");

        LOG_INFO("Caps structs name " << name);
        LOG_INFO("Media = '" << media << "' for RtspSourceBitnr '" 
            << GetName() << "'");
        
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
                return;
            }
            
            // Link the rtcpsrc element's added src pad to the sink pad of the Depay
            if (gst_pad_link(pPad, pDepayStaicSinkPad) != GST_PAD_LINK_OK) 
            {
                LOG_ERROR("Failed to link source to de-payload");
                return;
            }
            gst_object_unref(pDepayStaicSinkPad);
            
            LOG_INFO("rtspsrc element linked for RtspSourceBintr '" 
                << GetName() << "'");

            // finally fully linked -- ok to unlink all elements from this point
            m_isFullyLinked = true;

            // Update the cap memebers for this RtspSourceBintr
            gst_structure_get_uint(structure, "width", &m_width);
            gst_structure_get_uint(structure, "height", &m_height);
            gst_structure_get_fraction(structure, "framerate", (gint*)&m_fpsN, 
                (gint*)&m_fpsD);
            
            LOG_INFO("Frame width = " << m_width << ", height = " << m_height);
            LOG_INFO("FPS numerator = " << m_fpsN << ", denominator = " << m_fpsD);
        }
    }
    
    int RtspSourceBintr::StreamManager()
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_streamManagerMutex);

        // if currently in a reset cycle then let the ResetStream 
        // handler continue to handle
        if (m_connectionData.is_in_reconnect)
        {
            return true;
        }

        struct timeval currentTime;
        gettimeofday(&currentTime, NULL);

        GstState currentState;
        uint stateResult = GetState(currentState, 0);
        SetCurrentState(currentState);
        
        // Get the last buffer-time so we can determine if connection is nominal
        struct timeval lastBufferTime;
        m_TimestampPph->GetTime(lastBufferTime);
        
        // If we still haven't received our first buffer... we're waiting for the
        // the first connection attemp to complete
        if (lastBufferTime.tv_sec == 0)
        {
            // Increment the total time we've been waiting for connection.
            m_firstConnectTime += DSL_RTSP_TEST_FOR_BUFFER_TIMEOUT_PERIOD_MS;
            
            // If we haven't exceeded our first connection wait time.
            if (m_firstConnectTime < (m_connectionData.timeout*1000))
            {
                LOG_DEBUG("RtspSourceBintr '" << GetName() 
                    << "' is waiting for first connection" );
                return true;
            }

            LOG_ERROR("First connection timeout for RtspSourceBintr '" 
                << GetName() << "' " );
            m_firstConnectTime = 0;
        }
        else
        {
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
                    LOG_INFO("Sleeping after failed connection");
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
