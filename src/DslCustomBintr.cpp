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
#include "DslCustomBintr.h"
#include "DslBranchBintr.h"

namespace DSL
{

    CustomBintr::CustomBintr(const char* name)
        : QBintr(name)
        , m_nextElementIndex(0)
    {
        LOG_FUNC();

        LOG_INFO("");
        LOG_INFO("Initial property values for CustomBintr '" << name << "'");
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

        // Bintr Queue as first element and Sink ghost pad for the CustomBintr
        m_pQueue->AddGhostPadToParent("sink");
   }

    CustomBintr::~CustomBintr()
    {
        LOG_FUNC();

        if (m_isLinked)
        {    
            UnlinkAll();
        }
    }

    bool CustomBintr::AddToParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' CustomBintr to the Parent Pipeline/Branch 
        return std::dynamic_pointer_cast<BranchBintr>(pParentBintr)->
            AddCustomBintr(shared_from_this());
        return true;
    }
    
    bool CustomBintr::RemoveFromParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // remove 'this' CustomBintr from the Parent Branch
        return std::dynamic_pointer_cast<BranchBintr>(pParentBintr)->
            RemoveCustomBintr(shared_from_this());
    }

    bool CustomBintr::AddChild(DSL_ELEMENT_PTR pChild)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't add child '" << pChild->GetName() 
                << "' to CustomBintr '" << m_name << "' as it is currently linked");
            return false;
        }
        if (IsChild(pChild))
        {
            LOG_ERROR("GstElementr '" << pChild->GetName() 
                << "' is already a child of CustomBintr '" << GetName() << "'");
            return false;
        }
 
        // increment next index, assign to the Element.
        pChild->SetIndex(++m_nextElementIndex);

        // Add the shared pointer to the CustomBintr to the indexed map and as a child.
        m_elementrsIndexed[m_nextElementIndex] = pChild;
        return GstNodetr::AddChild(pChild);
    }
    
    bool CustomBintr::RemoveChild(DSL_ELEMENT_PTR pChild)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't remove child '" << pChild->GetName() 
                << "' from CustomBintr '" << m_name << "' as it is currently linked");
            return false;
        }
        if (!IsChild(pChild))
        {
            LOG_ERROR("GstElementr '" << pChild->GetName() 
                << "' is not a child of CustomBintr '" << GetName() << "'");
            return false;
        }
        
        // Remove the shared pointer to the CustomBintr from the indexed map and 
        // as a child.
        m_elementrsIndexed.erase(pChild->GetIndex());
        return GstNodetr::RemoveChild(pChild);
    }
    
    bool CustomBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("CustomBintr '" << m_name << "' is already linked");
            return false;
        }
        if (!m_elementrsIndexed.size()) 
        {
            LOG_ERROR("CustomBintr '" << m_name << "' has no Elements to link");
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

            LOG_INFO("CustomBintr '" << GetName() << "' Linked up child Elementrs as '" << 
                imap.second->GetName() << "' successfully");                    
        }
        // Setup the ghost pad for the last Element, which would
        // be the same in the case of one element.
        m_elementrsLinked.back()->AddGhostPadToParent("src");

        // Link the input queue to the first custom element
        m_pQueue->LinkToSink(m_elementrsLinked.front());

        m_isLinked = true;
        
        return true;
    }
    
    void CustomBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("CustomBintr '" << m_name << "' is not linked");
            return;
        }
        if (!m_elementrsLinked.size()) 
        {
            LOG_ERROR("CustomBintr '" << m_name << "' has no Elements to unlink");
            return;
        }
        // Remove the sink ghost pad from the last element, which would
        // be the same in the case of one element.
        m_elementrsLinked.back()->RemoveGhostPadFromParent("src");
        
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