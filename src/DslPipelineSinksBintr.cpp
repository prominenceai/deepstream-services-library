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

#include "Dsl.h"
#include "DslSinkBintr.h"
#include "DslPipelineSinksBintr.h"

namespace DSL
{

    PipelineSinksBintr::PipelineSinksBintr(const char* name)
        : Bintr(name)
        , m_pQueue(NULL)
        , m_pTee(NULL)
    {
        LOG_FUNC();

        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "sink_bin_queue");
        m_pTee = DSL_ELEMENT_NEW(NVDS_ELEM_TEE, "sink_bin_tee");
        
        AddChild(m_pQueue);
        AddChild(m_pTee);

        m_pQueue->AddGhostPadToParent("sink");
        
    }
    
    PipelineSinksBintr::~PipelineSinksBintr()
    {
        LOG_FUNC();
    }
     
    DSL_NODETR_PTR PipelineSinksBintr::AddChild(DSL_NODETR_PTR pChildBintr)
    {
        LOG_FUNC();
        
        if (IsChild(pChildBintr))
        {
            LOG_ERROR("' " << pChildBintr->m_name << "' is already a child of '" << m_name << "'");
            throw;
        }
        
        if (!gst_bin_add(GST_BIN(m_pGstObj), GST_ELEMENT(pChildBintr->m_pGstObj)))
        {
            LOG_ERROR("Failed to add " << pChildBintr->m_name << " to " << m_name);
            throw;
        }
        
        // call the base function to complete the add
        return Bintr::AddChild(pChildBintr);
    }
    
    void PipelineSinksBintr::RemoveChild(DSL_NODETR_PTR pChildBintr)
    {
        LOG_FUNC();

        if (!IsChild(pChildBintr))
        {
            LOG_ERROR("' " << pChildBintr->m_name << "' is NOT a child of '" << m_name << "'");
            throw;
        }
        // unlink the sink from the Tee
        std::dynamic_pointer_cast<SinkBintr>(pChildBintr)->
            m_pStaticSinkPadtr->Unlink();
        
        // call the base function to complete the remove
        Bintr::RemoveChild(pChildBintr);
    }

    void PipelineSinksBintr::RemoveAllChildren()
    {
        LOG_FUNC();
        
        // Removed sinks will be reset to not-in-use
        for (auto &imap: m_pChildren)
        {
            // unlink each sink from the Tee
            std::dynamic_pointer_cast<SinkBintr>(imap.second)->
                m_pStaticSinkPadtr->Unlink();
            
            // call the base function to complete the remove
            Bintr::RemoveChild(imap.second);
        }
    }

    bool PipelineSinksBintr::LinkAll()
    {
        LOG_FUNC();
        
        for (auto const& imap: m_pChildren)
        {
            GstPadTemplate* padtemplate = 
                gst_element_class_get_pad_template(GST_ELEMENT_GET_CLASS(m_pTee->m_pGstObj), "src_%u");
            if (!padtemplate)
            {
                LOG_ERROR("Failed to get Pad Template for '" << m_name << "'");
                return false;
            }
            
            std::shared_ptr<RequestPadtr> pSourcePadtr = 
                std::shared_ptr<RequestPadtr>(new RequestPadtr("src", m_pTee, padtemplate));
            
            pSourcePadtr->LinkTo(std::dynamic_pointer_cast<SinkBintr>(imap.second)->m_pStaticSinkPadtr);
        }
        return true;
    }

    void PipelineSinksBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        for (auto const& imap: m_pChildren)
        {
            // unlink from the Tee Element
            std::dynamic_pointer_cast<SinkBintr>(imap.second)->
                m_pStaticSinkPadtr->Unlink();
        }
    }
}