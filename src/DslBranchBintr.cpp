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
#include "DslBranchBintr.h"

#include <gst/gst.h>

namespace DSL
{
    BranchBintr::BranchBintr(const char* name, bool pipeline)
        : Bintr(name, pipeline) 
    {
        LOG_FUNC();
    }

    bool BranchBintr::AddPreprocBintr(DSL_BASE_PTR pPreprocBintr)
    {
        LOG_FUNC();
        
        if (m_pPreprocBintr)
        {
            LOG_ERROR("Branch '" << GetName() << "' has an exisiting PreprocBintr '" 
                << m_pPreprocBintr->GetName());
            return false;
        }
        m_pPreprocBintr = std::dynamic_pointer_cast<PreprocBintr>(pPreprocBintr);
        
        return AddChild(pPreprocBintr);
    }

    bool BranchBintr::RemovePreprocBintr(DSL_BASE_PTR pPreprocBintr)
    {
        LOG_FUNC();
        
        if (!m_pPreprocBintr)
        {
            LOG_ERROR("Branch '" << GetName() << "' has no PreprocBintr to remove'");
            return false;
        }
        if (m_pPreprocBintr != pPreprocBintr)
        {
            LOG_ERROR("Branch '" << GetName() << "' does not own PreprocBintr' " 
                << pPreprocBintr->GetName() << "'");
            return false;
        }
        if (IsLinked())
        {
            LOG_ERROR("PreprocBintr cannot be removed from Branch '" << GetName() 
                << "' as it is currently linked");
            return false;
        }
        m_pPreprocBintr = nullptr;
        
        LOG_INFO("Removing PreprocBintr '"<< pPreprocBintr->GetName() 
            << "' from Branch '" << GetName() << "'");
        return RemoveChild(pPreprocBintr);
    }

    bool BranchBintr::AddPrimaryInferBintr(DSL_BASE_PTR pPrimaryInferBintr)
    {
        LOG_FUNC();
        
        if (m_pPrimaryInferBintr)
        {
            LOG_ERROR("Branch '" << GetName() << "' has an exisiting PrimaryInferBintr '" 
                << m_pPrimaryInferBintr->GetName());
            return false;
        }
        m_pPrimaryInferBintr = std::dynamic_pointer_cast<PrimaryInferBintr>(pPrimaryInferBintr);
        
        return AddChild(pPrimaryInferBintr);
    }

    bool BranchBintr::RemovePrimaryInferBintr(DSL_BASE_PTR pPrimaryInferBintr)
    {
        LOG_FUNC();
        
        if (!m_pPrimaryInferBintr)
        {
            LOG_ERROR("Branch '" << GetName() << "' has no Primary Infer to remove'");
            return false;
        }
        if (m_pPrimaryInferBintr != pPrimaryInferBintr)
        {
            LOG_ERROR("Branch '" << GetName() << "' does not own Primary Infer' " 
                << pPrimaryInferBintr->GetName() << "'");
            return false;
        }
        if (IsLinked())
        {
            LOG_ERROR("Primary Infer cannot be removed from Branch '" << GetName() 
                << "' as it is currently linked");
            return false;
        }
        m_pPrimaryInferBintr = nullptr;
        
        LOG_INFO("Removing Primary Infer '"<< pPrimaryInferBintr->GetName() 
            << "' from Branch '" << GetName() << "'");
        return RemoveChild(pPrimaryInferBintr);
    }

    bool BranchBintr::AddSegVisualBintr(DSL_BASE_PTR pSegVisualBintr)
    {
        LOG_FUNC();

        if (m_pSegVisualBintr)
        {
            LOG_ERROR("Branch '" << GetName() 
                << "' already has a Segmentation Visualizer");
            return false;
        }
        m_pSegVisualBintr = std::dynamic_pointer_cast<SegVisualBintr>(pSegVisualBintr);
        
        return AddChild(pSegVisualBintr);
    }

    bool BranchBintr::AddTrackerBintr(DSL_BASE_PTR pTrackerBintr)
    {
        LOG_FUNC();

        if (m_pTrackerBintr)
        {
            LOG_ERROR("Branch '" << GetName() << "' already has a Tracker");
            return false;
        }
        m_pTrackerBintr = std::dynamic_pointer_cast<TrackerBintr>(pTrackerBintr);
        
        return AddChild(pTrackerBintr);
    }

    bool BranchBintr::RemoveTrackerBintr(DSL_BASE_PTR pTrackerBintr)
    {
        LOG_FUNC();
        
        if (!m_pTrackerBintr)
        {
            LOG_ERROR("Branch '" << GetName() << "' has no Tracker to remove'");
            return false;
        }
        if (m_pTrackerBintr != pTrackerBintr)
        {
            LOG_ERROR("Branch '" << GetName() << "' does not own Tracker' " 
                << m_pTrackerBintr->GetName() << "'");
            return false;
        }
        if (IsLinked())
        {
            LOG_ERROR("Tracker cannot be removed from Branch '" << GetName() 
                << "' as it is currently linked'");
            return false;
        }
        m_pTrackerBintr = nullptr;
        
        LOG_INFO("Removing Tracker '"<< pTrackerBintr->GetName() 
            << "' from Branch '" << GetName() << "'");
        return RemoveChild(pTrackerBintr);
    }

    bool BranchBintr::AddSecondaryInferBintr(DSL_BASE_PTR pSecondaryInferBintr)
    {
        LOG_FUNC();
        
        // Create the optional Secondary GIEs bintr 
        if (!m_pSecondaryInfersBintr)
        {
            m_pSecondaryInfersBintr = DSL_PIPELINE_SINFERS_NEW("secondary-infer-bin");
            AddChild(m_pSecondaryInfersBintr);
        }
        return m_pSecondaryInfersBintr->
            AddChild(std::dynamic_pointer_cast<SecondaryInferBintr>(pSecondaryInferBintr));
    }

    bool BranchBintr::AddOfvBintr(DSL_BASE_PTR pOfvBintr)
    {
        LOG_FUNC();

        if (m_pOfvBintr)
        {
            LOG_ERROR("Branch '" << GetName() << "' already has an Optical Flow Visualizer ");
            return false;
        }
        m_pOfvBintr = std::dynamic_pointer_cast<OfvBintr>(pOfvBintr);
        
        return AddChild(m_pOfvBintr);
    }

    bool BranchBintr::AddDemuxerBintr(DSL_BASE_PTR pDemuxerBintr)
    {
        LOG_FUNC();

        if (m_pDemuxerBintr)
        {
            LOG_ERROR("Branch '" << GetName() << "' already has a Demuxer");
            return false;
        }
        if (m_pSplitterBintr)
        {
            LOG_ERROR("Branch '" << GetName() << "' already has a Splitter - can't add Demuxer");
            return false;
        }
        if (m_pTilerBintr)
        {
            LOG_ERROR("Branch '" << GetName() << "' already has a Tiler - can't add Demuxer");
            return false;
        }
        m_pDemuxerBintr = std::dynamic_pointer_cast<DemuxerBintr>(pDemuxerBintr);
        
        return AddChild(pDemuxerBintr);
    }

    bool BranchBintr::AddSplitterBintr(DSL_BASE_PTR pSplitterBintr)
    {
        LOG_FUNC();

        if (m_pSplitterBintr)
        {
            LOG_ERROR("Branch '" << GetName() << "' already has a Splitter");
            return false;
        }
        if (m_pDemuxerBintr)
        {
            LOG_ERROR("Branch '" << GetName() << "' already has a Dumxer- can't add Splitter");
            return false;
        }
        m_pSplitterBintr = std::dynamic_pointer_cast<SplitterBintr>(pSplitterBintr);
        
        return AddChild(pSplitterBintr);
    }

    bool BranchBintr::AddTilerBintr(DSL_BASE_PTR pTilerBintr)
    {
        LOG_FUNC();

        if (m_pTilerBintr)
        {
            LOG_ERROR("Branch '" << GetName() << "' already has a Tiler");
            return false;
        }
        if (m_pDemuxerBintr)
        {
            LOG_ERROR("Branch '" << GetName() << "' already has a Demuxer - can't add Tiler");
            return false;
        }
        m_pTilerBintr = std::dynamic_pointer_cast<TilerBintr>(pTilerBintr);
        
        return AddChild(pTilerBintr);
    }

    bool BranchBintr::AddOsdBintr(DSL_BASE_PTR pOsdBintr)
    {
        LOG_FUNC();
        
        if (m_pOsdBintr)
        {
            LOG_ERROR("Branch '" << GetName() << "' has an exisiting OSD '" 
                << m_pOsdBintr->GetName());
            return false;
        }
        if (m_pDemuxerBintr)
        {
            LOG_ERROR("Branch '" << GetName() << "' already has a Demuxer - can't add OSD");
            return false;
        }
        m_pOsdBintr = std::dynamic_pointer_cast<OsdBintr>(pOsdBintr);
        
        return AddChild(pOsdBintr);
    }

    bool BranchBintr::RemoveOsdBintr(DSL_BASE_PTR pOsdBintr)
    {
        LOG_FUNC();
        
        if (!m_pOsdBintr)
        {
            LOG_ERROR("Branch '" << GetName() << "' has no OSD to remove'");
            return false;
        }
        if (m_pOsdBintr != pOsdBintr)
        {
            LOG_ERROR("Branch '" << GetName() << "' does not own OSD' " 
                << m_pOsdBintr->GetName() << "'");
            return false;
        }
        m_pOsdBintr = nullptr;
        
        LOG_INFO("Removing OSD '"<< pOsdBintr->GetName() 
            << "' from Branch '" << GetName() << "'");
        return RemoveChild(pOsdBintr);
    }

    bool BranchBintr::AddSinkBintr(DSL_BASE_PTR pSinkBintr)
    {
        LOG_FUNC();
        
        if (m_pDemuxerBintr)
        {
            LOG_ERROR("Branch '" << GetName() << "' already has a Demuxer - can't add Sink after a Demuxer");
            return false;
        }
        if (m_pSplitterBintr)
        {
            LOG_ERROR("Branch '" << GetName() << "' already has a Tee - can't add Sink after a Tee");
            return false;
        }
        // Create the shared Sinks bintr if it doesn't exist
        if (!m_pMultiSinksBintr)
        {
            m_pMultiSinksBintr = DSL_MULTI_SINKS_NEW("sinks-bin");
            AddChild(m_pMultiSinksBintr);
        }
        return m_pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pSinkBintr));
    }

    bool BranchBintr::IsSinkBintrChild(DSL_BASE_PTR pSinkBintr)
    {
        LOG_FUNC();

        if (!m_pMultiSinksBintr)
        {
            LOG_INFO("Branch '" << GetName() << "' has no Sinks");
            return false;
        }
        return (m_pMultiSinksBintr->IsChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr)));
    }

    bool BranchBintr::RemoveSinkBintr(DSL_BASE_PTR pSinkBintr)
    {
        LOG_FUNC();

        if (!m_pMultiSinksBintr)
        {
            LOG_INFO("Branch '" << GetName() << "' has no Sinks");
            return false;
        }

        // Must cast to SourceBintr first so that correct Instance of RemoveChild is called
        return m_pMultiSinksBintr->RemoveChild(std::dynamic_pointer_cast<Bintr>(pSinkBintr));
    }
    
    bool BranchBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_INFO("Components for Branch '" << GetName() << "' are already assembled");
            return false;
        }
        if (!m_pDemuxerBintr and !m_pSplitterBintr and !m_pMultiSinksBintr)
        {
            LOG_ERROR("Pipline '" << GetName() 
                << "' has no Demuxer, Splitter or Sink - and is unable to link");
            return false;
        }
        if (m_pPreprocBintr and !m_pPrimaryInferBintr)
        {
            LOG_ERROR("Pipline '" << GetName() 
                << "' has a PreprocBintr and no PrimaryInferBintr - and is unable to link");
            return false;
        }
        if (m_pTrackerBintr and !m_pPrimaryInferBintr)
        {
            LOG_ERROR("Pipline '" << GetName() 
                << "' has a Tracker and no PrimaryInferBintr - and is unable to link");
            return false;
        }
        if (m_pSegVisualBintr and !m_pPrimaryInferBintr)
        {
            LOG_ERROR("Pipline '" << GetName() 
                << "' has a Segmentation Visualizer with no PrimaryInferBintr - and is unable to link");
            return false;
        }
        if (m_pSecondaryInfersBintr and !m_pPrimaryInferBintr)
        {
            LOG_ERROR("Pipline '" << GetName() 
                << "' has a SecondayInferbintr with no PrimaryInferBintr - and is unable to link");
            return false;
        }
        
        if (m_pPreprocBintr)
        {
            // Set the SecondarInferBintrs batch size to the current stream muxer batch size, 
            // then LinkAll PrimaryInfer Elementrs and add as the next component in the Branch
            m_pPreprocBintr->SetBatchSize(m_batchSize);
            if (!m_pPreprocBintr->LinkAll() or
                (m_linkedComponents.size() and 
                !m_linkedComponents.back()->LinkToSink(m_pPreprocBintr)))
            {
                return false;
            }
            m_linkedComponents.push_back(m_pPreprocBintr);
            LOG_INFO("Branch '" << GetName() << "' Linked up PreprocBintr '" << 
                m_pPreprocBintr->GetName() << "' successfully");
        }
        
        if (m_pPrimaryInferBintr)
        {
            // Set the SecondarInferBintrs batch size to the current stream muxer batch size, 
            // then LinkAll PrimaryInfer Elementrs and add as the next component in the Branch
            m_pPrimaryInferBintr->SetBatchSize(m_batchSize);
            if (!m_pPrimaryInferBintr->LinkAll() or
                (m_linkedComponents.size() and 
                !m_linkedComponents.back()->LinkToSink(m_pPrimaryInferBintr)))
            {
                return false;
            }
            m_linkedComponents.push_back(m_pPrimaryInferBintr);
            LOG_INFO("Branch '" << GetName() << "' Linked up PrimaryInferBintr '" << 
                m_pPrimaryInferBintr->GetName() << "' successfully");
        }
        
        if (m_pTrackerBintr)
        {
            // LinkAll Tracker Elementrs and add as the next component in the Branch
            m_pTrackerBintr->SetBatchSize(m_batchSize);
            if (!m_pTrackerBintr->LinkAll() or
                (m_linkedComponents.size() and 
                !m_linkedComponents.back()->LinkToSink(m_pTrackerBintr)))
            {
                return false;
            }
            m_linkedComponents.push_back(m_pTrackerBintr);
            LOG_INFO("Branch '" << GetName() << "' Linked up Tracker '" << 
                m_pTrackerBintr->GetName() << "' successfully");
        }
        
        if (m_pSecondaryInfersBintr)
        {
            // Set the SecondaryInferBintr' Primary Infer Name, and set batch sizes
            m_pSecondaryInfersBintr->SetInferOnId(m_pPrimaryInferBintr->GetUniqueId());
            m_pSecondaryInfersBintr->SetBatchSize(m_batchSize);
            
            // LinkAll SecondaryGie Elementrs and add the Bintr as next component in the Branch
            if (!m_pSecondaryInfersBintr->LinkAll() or
                (m_linkedComponents.size() and 
                !m_linkedComponents.back()->LinkToSink(m_pSecondaryInfersBintr)))
            {
                return false;
            }
            m_linkedComponents.push_back(m_pSecondaryInfersBintr);
            LOG_INFO("Branch '" << GetName() << "' Linked up all Secondary GIEs '" << 
                m_pSecondaryInfersBintr->GetName() << "' successfully");
        }

        if (m_pSegVisualBintr)
        {
            // LinkAll Segmentation Visualizer Elementrs and add as the next component in the Branch
            m_pSegVisualBintr->SetBatchSize(m_batchSize);
            if (!m_pSegVisualBintr->LinkAll() or
                (m_linkedComponents.size() and 
                !m_linkedComponents.back()->LinkToSink(m_pSegVisualBintr)))
            {
                return false;
            }
            m_linkedComponents.push_back(m_pSegVisualBintr);
            LOG_INFO("Branch '" << GetName() << "' Linked up Segmentation Visualizer '" << 
                m_pSegVisualBintr->GetName() << "' successfully");
        }

        if (m_pOfvBintr)
        {
            // LinkAll Optical Flow Elementrs and add as the next component in the Branch
            m_pOfvBintr->SetBatchSize(m_batchSize);
            if (!m_pOfvBintr->LinkAll() or
                (m_linkedComponents.size() and 
                !m_linkedComponents.back()->LinkToSink(m_pOfvBintr)))
            {
                return false;
            }
            m_linkedComponents.push_back(m_pOfvBintr);
            LOG_INFO("Branch '" << GetName() << "' Linked up Optical Flow Detector '" << 
                m_pOfvBintr->GetName() << "' successfully");
        }

        // mutually exclusive with Demuxer
        if (m_pTilerBintr)
        {
            // Link All Tiler Elementrs and add as the next component in the Branch
            m_pTilerBintr->SetBatchSize(m_batchSize);
            if (!m_pTilerBintr->LinkAll() or
                (m_linkedComponents.size() and 
                !m_linkedComponents.back()->LinkToSink(m_pTilerBintr)))
            {
                return false;
            }
            m_linkedComponents.push_back(m_pTilerBintr);
            LOG_INFO("Branch '" << GetName() << "' Linked up Tiler '" << 
                m_pTilerBintr->GetName() << "' successfully");
        }

        // mutually exclusive with Demuxer
        if (m_pOsdBintr)
        {
            // LinkAll Osd Elementrs and add as next component in the Branch
            m_pOsdBintr->SetBatchSize(m_batchSize);
            if (!m_pOsdBintr->LinkAll() or
                (m_linkedComponents.size() and 
                !m_linkedComponents.back()->LinkToSink(m_pOsdBintr)))
            {
                return false;
            }
            m_linkedComponents.push_back(m_pOsdBintr);
            LOG_INFO("Branch '" << GetName() << "' Linked up OSD '" << 
                m_pOsdBintr->GetName() << "' successfully");
        }

        // mutually exclusive with TilerBintr, Pipeline-OsdBintr, and Pieline-MultiSinksBintr
        if (m_pDemuxerBintr)
        {
            // Link All Demuxer Elementrs and add as the next ** AND LAST ** 
            // component in the Pipeline
            m_pDemuxerBintr->SetBatchSize(m_batchSize);
            if (!m_pDemuxerBintr->LinkAll() or
                (m_linkedComponents.size() and 
                !m_linkedComponents.back()->LinkToSink(m_pDemuxerBintr)))
            {
                return false;
            }
            m_linkedComponents.push_back(m_pDemuxerBintr);
            LOG_INFO("Branch '" << GetName() << "' Linked up Stream Demuxer '" << 
                m_pDemuxerBintr->GetName() << "' successfully");
        }

        // mutually exclusive with TilerBintr, Pipeline-OsdBintr, 
        // and Pieline-MultiSinksBintr
        if (m_pSplitterBintr)
        {
            // Link All Demuxer Elementrs and add as the next ** AND LAST ** 
            // component in the Pipeline
            m_pSplitterBintr->SetBatchSize(m_batchSize);
            if (!m_pSplitterBintr->LinkAll() or
                (m_linkedComponents.size() and 
                !m_linkedComponents.back()->LinkToSink(m_pSplitterBintr)))
            {
                return false;
            }
            m_linkedComponents.push_back(m_pSplitterBintr);
            LOG_INFO("Branch '" << GetName() << "' Linked up Stream Splitter'" << 
                m_pSplitterBintr->GetName() << "' successfully");
        }

        // mutually exclusive with Demuxer
        if (m_pMultiSinksBintr)
        {
            // Link all Sinks and their elementrs and add as finale (tail) 
            //components in the Branch
            m_pMultiSinksBintr->SetBatchSize(m_batchSize);
            if (!m_pMultiSinksBintr->LinkAll() or
                (m_linkedComponents.size() and 
                !m_linkedComponents.back()->LinkToSink(m_pMultiSinksBintr)))
            {
                return false;
            }
            m_linkedComponents.push_back(m_pMultiSinksBintr);
            LOG_INFO("Branch '" << GetName() << "' Linked up all Sinks '" << 
                m_pMultiSinksBintr->GetName() << "' successfully");
        }
        
        m_isLinked = true;
        return true;
    }
    
    void BranchBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            return;
        }
        // iterate through the list of Linked Components, unlinking each
        for (auto const& ivector: m_linkedComponents)
        {
            // all but the tail m_pMultiSinksBintr will be Linked to Sink
            if (ivector->IsLinkedToSink())
            {
                ivector->UnlinkFromSink();
            }
            ivector->UnlinkAll();
        }
        m_linkedComponents.clear();

        m_isLinked = false;
    }
    
    bool BranchBintr::LinkToSourceTee(DSL_NODETR_PTR pTee, const char* srcPadName)
    {
        LOG_FUNC();
        
        if (!m_linkedComponents.size())
        {
            LOG_ERROR("Unable to link empty Bramch '" << GetName() <<"'");
            return false;
        }

        GstPad* pComponentStaticSinkPad = gst_element_get_static_pad(m_linkedComponents.front()->GetGstElement(), "sink");
        if (!pComponentStaticSinkPad)
        {
            LOG_ERROR("Failed to get static Sink Pad for Branch Bintr '" << GetName() <<"'");
            return false;
        }        
        // Add a sink ghost pad to BranchBintr, using the firt componet's 
        if (!gst_element_add_pad(GetGstElement(), 
            gst_ghost_pad_new("sink", pComponentStaticSinkPad)))
        {
            gst_object_unref(pComponentStaticSinkPad);
            LOG_ERROR("Failed to add Sink Ghost Pad for BranchBintr'" << GetName() << "'");
            return false;
        }
        gst_object_unref(pComponentStaticSinkPad);
        
        return Bintr::LinkToSourceTee(pTee, srcPadName);
    }

} // DSL
