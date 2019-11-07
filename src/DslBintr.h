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

#ifndef _DSL_BINTR_H
#define _DSL_BINTR_H

#include "Dsl.h"
#include "DslNodetr.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_BINTR_PTR std::shared_ptr<Bintr>
    #define DSL_BINTR_NEW(name) \
        std::shared_ptr<Bintr>(new Bintr(name))    

    /**
     * @class Bintr
     * @brief Implements a base container class for a GST Bin
     */
    class Bintr : public Nodetr
    {
    public:

        /**
         * @brief named container ctor with new Bin 
         */
        Bintr(const char* name)
            : Nodetr(name)
            , m_gpuId(0)
            , m_nvbufMemoryType(0)
            , m_pSinkPad(NULL)
            , m_pSourcePad(NULL)
        { 
            LOG_FUNC(); 

            m_pGstObj = GST_OBJECT(gst_bin_new((gchar*)name));
            if (!m_pGstObj)
            {
                LOG_ERROR("Failed to create a new GST bin for Bintr '" << name << "'");
                throw;  
            }
        };
        
        /**
         * @brief Bintr dtor to release all GST references
         */
        ~Bintr()
        {
            LOG_FUNC();
        };
        
        /**
         * @brief links this Bintr as source to a given Bintr as sink
         * @param pSinkBintr to link to
         */
        void LinkTo(DSL_NODETR_PTR pSink)
        { 
            LOG_FUNC();
            
            // Call the base class to complete the relationship
            Nodetr::LinkTo(pSink);

            // Link Source Bintr to Sink Bintr as elements 
            if (!gst_element_link(GST_ELEMENT(m_pGstObj), GST_ELEMENT(m_pSink->m_pGstObj)))
            {
                LOG_ERROR("Failed to link " << m_name << " to " << pSink->m_name);
                throw;
            }
        };

        /**
         * @brief unlinks this Bintr from a previously linked-to sink Bintr
         */
        void Unlink()
        { 
            LOG_FUNC();

            if (IsLinked())
            {
                gst_element_unlink(GST_ELEMENT(m_pGstObj), GST_ELEMENT(m_pSink->m_pGstObj));

                // Call the base class to complete the unlink
                Nodetr::Unlink();
            }
        };

        /**
         * @brief adds a child Bintr to this parent Bintr
         * @param pChildBintr to add. Once added, calling InUse()
         *  on the Child Bintr will return true
         * @return a shared pointer to the newly added pChild
         */
        DSL_NODETR_PTR AddChild(DSL_NODETR_PTR pChild)
        {
            LOG_FUNC();
            
            if (!gst_bin_add(GST_BIN(m_pGstObj), GST_ELEMENT(pChild->m_pGstObj)))
            {
                LOG_ERROR("Failed to add " << pChild->m_name << " to " << m_name <<"'");
                throw;
            }
            return Nodetr::AddChild(pChild);
        };
        
        /**
         * @brief removes a child Bintr from this parent Bintr
         * @param pChildBintr to remove. Once removed, calling InUse()
         *  on the Child Bintr will return false
         */
        void RemoveChild(DSL_NODETR_PTR pChild)
        {
            LOG_FUNC();
            
            if (!IsChild(pChild))
            {
                LOG_ERROR("'" << pChild->m_name << "' is not a child of '" << m_name <<"'");
                throw;
            }
                            
            if (!gst_bin_remove(GST_BIN(m_pGstObj), GST_ELEMENT(pChild->m_pGstObj)))
            {
                LOG_ERROR("Failed to remove " << pChild->m_name << " from " << m_name <<"'");
                throw;
            }
            Nodetr::RemoveChild(pChild);
        };

        /**
         * @brief Adds this Bintr as a child to a ParentBinter
         * @param pParentBintr to add to
         */
        virtual void AddToParent(DSL_NODETR_PTR pParent)
        {
            LOG_FUNC();
                
            pParent->AddChild(shared_from_this());
        };        
        
        /**
         * @brief removes this Bintr from the provided pParentBintr
         * @param pParentBintr Bintr to remove from
         */
        virtual void RemoveFromParent(DSL_NODETR_PTR pParentBintr)
        {
            LOG_FUNC();
                
            pParentBintr->RemoveChild(shared_from_this());
        };
        
        virtual void AddGhostPad(const char* name, DSL_NODETR_PTR pElementr)
        {
            LOG_FUNC();
            
            // create a new ghost pad with the static Sink pad retrieved from this Elementr's 
            // pGstObj and adds it to the the Elementr's Parent Bintr's pGstObj.
            if (!gst_element_add_pad(GST_ELEMENT(m_pGstObj), 
                gst_ghost_pad_new(name, gst_element_get_static_pad(GST_ELEMENT(pElementr->m_pGstObj), name))))
            {
                LOG_ERROR("Failed to add Pad '" << name << "' for element'" << m_name << "'");
                throw;
            }
        }

        /**
         * @brief virtual function for derived classes to implement
         * a bintr type specific function to link all children.
         */
        virtual bool LinkAll() = 0;
        
        /**
         * @brief virtual function for derived classes to implement
         * a bintr type specific function to unlink all child elements.
         */
        virtual void UnlinkAll() = 0;
        
        
    public:

        /**
         * @brief
         */
        guint m_gpuId;

        /**
         * @brief
         */
        guint m_nvbufMemoryType;

        GstPad* m_pSinkPad;
            
        GstPad* m_pSourcePad;
    };

} // DSL

#endif // _DSL_BINTR_H