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
#include "DslOdeHandlerBintr.h"
#include "DslBranchBintr.h"

namespace DSL
{

    OdeHandlerBintr::OdeHandlerBintr(const char* name)
        : Bintr(name)
        , m_isEnabled(true)
    {
        LOG_FUNC();

        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "reporter-queue");
        
        Bintr::AddChild(m_pQueue);

        m_pQueue->AddGhostPadToParent("sink");
        m_pQueue->AddGhostPadToParent("src");

        // New src pad probe for event processing and reporting
        m_pSrcPadProbe = DSL_PAD_PROBE_NEW("reporter-src-pad-probe", "src", m_pQueue);
        
        if (!AddBatchMetaHandler(DSL_PAD_SRC, PadBufferHandler, this))
        {
            LOG_ERROR("OdeHandlerBintr '" << m_name << "' failed to add probe buffer handler on create");
            throw;
        }
    }

    OdeHandlerBintr::~OdeHandlerBintr()
    {
        LOG_FUNC();

        if (m_isLinked)
        {    
            UnlinkAll();
        }
        RemoveAllChildren();
    }

    bool OdeHandlerBintr::AddToParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' reporter to the Parent Pipeline 
        return std::dynamic_pointer_cast<BranchBintr>(pParentBintr)->
            AddOdeHandlerBintr(shared_from_this());
    }
    
    bool OdeHandlerBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("OdeHandlerBintr '" << m_name << "' is already linked");
            return false;
        }

        // single element, noting to link
        m_isLinked = true;
        
        return true;
    }
    
    void OdeHandlerBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("OdeHandlerBintr '" << m_name << "' is not linked");
            return;
        }
        // single element, nothing to unlink
        m_isLinked = false;
    }
    
    bool OdeHandlerBintr::AddChild(DSL_BASE_PTR pChild)
    {
        LOG_FUNC();
        
        return Base::AddChild(pChild);
    }

    bool OdeHandlerBintr::RemoveChild(DSL_BASE_PTR pChild)
    {
        LOG_FUNC();
        
        return Base::RemoveChild(pChild);
    }

    bool OdeHandlerBintr::GetEnabled()
    {
        LOG_FUNC();
        
        return m_isEnabled;
    }
    
    bool OdeHandlerBintr::SetEnabled(bool enabled)
    {
        LOG_FUNC();

        if (m_isEnabled == enabled)
        {
            LOG_ERROR("Can't set Reporting Enabled to the same value of " 
                << enabled << " for OdeHandlerBintr '" << GetName() << "' ");
            return false;
        }
        m_isEnabled = enabled;
        
        if (enabled)
        {
            LOG_INFO("Enabling the OdeHandlerBintr '" << GetName() << "'");
            
            return AddBatchMetaHandler(DSL_PAD_SRC, PadBufferHandler, this);
        }
        LOG_INFO("Disabling the OdeHandlerBintr '" << GetName() << "'");
        
        return RemoveBatchMetaHandler(DSL_PAD_SRC, PadBufferHandler);
    }
    
    bool OdeHandlerBintr::HandlePadBuffer(GstBuffer* pBuffer)
    {
        NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(pBuffer);
        
        for (NvDsMetaList* l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
        {
            NvDsFrameMeta* pFrameMeta = (NvDsFrameMeta *) (l_frame->data);
            if (pFrameMeta != NULL and pFrameMeta->bInferDone)
            {
                for (NvDsMetaList* pMeta = pFrameMeta->obj_meta_list; pMeta != NULL; pMeta = pMeta->next)
                {
                    NvDsObjectMeta* pObjectMeta = (NvDsObjectMeta *) (pMeta->data);
                    if (pObjectMeta != NULL)
                    {
                        for (const auto &imap: m_pChildren)
                        {
                            DSL_ODE_TYPE_PTR pOdeType = std::dynamic_pointer_cast<OdeType>(imap.second);
                            pOdeType->CheckForOccurrence(pFrameMeta, pObjectMeta);
                        }
                    }
                }
            }
        }
        return true;
    }
    
    static boolean PadBufferHandler(void* pBuffer, void* user_data)
    {
        return static_cast<OdeHandlerBintr*>(user_data)->
            HandlePadBuffer((GstBuffer*)pBuffer);
    }
    
}