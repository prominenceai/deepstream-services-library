/*
The MIT License

Copyright (c) 2024, Prominence AI, Inc.

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
#include "DslGstBintr.h"
#include "DslBranchBintr.h"

namespace DSL
{

    GstBintr::GstBintr(const char* name)
        : Bintr(name)
        , m_nextElementIndex(0)
    {
        LOG_FUNC();

        LOG_INFO("");
        LOG_INFO("Initial property values for GstBintr '" << name << "'");
        LOG_INFO("  none - applicable       : " );

   }

    GstBintr::~GstBintr()
    {
        LOG_FUNC();

        if (m_isLinked)
        {    
            UnlinkAll();
        }
    }

    bool GstBintr::AddToParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' GstBintr to the Parent Pipeline 
        return std::dynamic_pointer_cast<BranchBintr>(pParentBintr)->
            AddGstBintr(shared_from_this());
        return true;
    }
    
    bool GstBintr::AddChild(DSL_ELEMENT_PTR pChild)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't add child '" << pChild->GetName() 
                << "' to GstBintr '" << m_name << "' as it is currently linked");
            return false;
        }
        if (IsChild(pChild))
        {
            LOG_ERROR("GstElementr '" << pChild->GetName() 
                << "' is already a child of GstBintr '" << GetName() << "'");
            return false;
        }
 
        // increment next index, assign to the Element.
        pChild->SetIndex(++m_nextElementIndex);

        // Add the shared pointer to the GstBintr to the indexed map and as a child.
        m_elementrsIndexed[m_nextElementIndex] = pChild;
        return GstNodetr::AddChild(pChild);
    }
    
    bool GstBintr::RemoveChild(DSL_ELEMENT_PTR pChild)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't remove child '" << pChild->GetName() 
                << "' from GstBintr '" << m_name << "' as it is currently linked");
            return false;
        }
        if (!IsChild(pChild))
        {
            LOG_ERROR("GstElementr '" << pChild->GetName() 
                << "' is not a child of GstBintr '" << GetName() << "'");
            return false;
        }
        
        // Remove the shared pointer to the GstBintr from the indexed map and 
        // as a child.
        m_elementrsIndexed.erase(pChild->GetIndex());
        return GstNodetr::RemoveChild(pChild);
    }
    
    bool GstBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("GstBintr '" << m_name << "' is already linked");
            return false;
        }
        if (!m_elementrsIndexed.size()) 
        {
            LOG_ERROR("GstBintr '" << m_name << "' has no Elements to link");
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

            LOG_INFO("GstBintr '" << GetName() << "' Linked up child Elementrs as '" << 
                imap.second->GetName() << "' successfully");                    
        }
        // Setup the ghost pads for the first and last Elementrs, which would
        // be the same in the case of one element.
        if (gst_element_get_static_pad(m_elementrsLinked.back().get()->GetGstElement(), "src"))
        {
            m_elementrsLinked.back()->AddGhostPadToParent("src");
        }
        if (gst_element_get_static_pad(m_elementrsLinked.front().get()->GetGstElement(), "sink"))
        {
            m_elementrsLinked.front()->AddGhostPadToParent("sink");
        }       
 
        m_isLinked = true;
        
        return true;
    }
    
    void GstBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("GstBintr '" << m_name << "' is not linked");
            return;
        }
        if (!m_elementrsLinked.size()) 
        {
            LOG_ERROR("GstBintr '" << m_name << "' has no Elements to unlink");
            return;
        }
        // Remove the ghost pads for the first and last element, which would
        // be the same in the case of one element.
        if (gst_element_get_static_pad(m_elementrsLinked.back().get()->GetGstElement(), "src"))
        {
            m_elementrsLinked.back()->RemoveGhostPadFromParent("src");
        }
        if (gst_element_get_static_pad(m_elementrsLinked.front().get()->GetGstElement(), "sink"))
        {
            m_elementrsLinked.front()->RemoveGhostPadFromParent("sink");
        }       
        
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
}