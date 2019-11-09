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

#ifndef _DSL_GSTNODETR_H
#define _DSL_GSTNODETR_H

#include "Dsl.h"
#include "DslNodetr.h"

namespace DSL
{

    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_GSTNODETR_PTR std::shared_ptr<GstNodetr>
    #define DSL_GSTNODETR_NEW(name) \
        std::shared_ptr<GstNodetr>(new GstNodetr(name))    

    /**
     * @class Padtr
     * @brief Add some GST Pad linking and Unlinking to the base Nodetr
     */
    class GstNodetr : public Nodetr
    {
    public:
        
        /**
         * @brief ctor for the GstNodetr base class
         * @param[in] name for the new GstNodetr
         */
        GstNodetr(const char* name)
        : Nodetr(name)
        {
            LOG_FUNC();

            LOG_INFO("New GstNodetr '" << m_name << "' created");
        }

        /**
         * @brief dtor for the GstNodetr base class
         */
        ~GstNodetr()
        {
            LOG_FUNC();
            
            if (IsLinkedToSink())
            {
                UnlinkFromSink();
            }
            if (IsLinkedToSource())
            {
                UnlinkFromSource();
            }
            
            LOG_INFO("Nodetr '" << m_name << "' deleted");
        }
        
        /**
         * @brief Creates a new Ghost Sink pad for this Gst Element
         * and adds it to the parent Gst Bin.
         * @throws a general exception on failure
         */
        void AddGhostPadToParent(const char* name)
        {
            LOG_FUNC();

            // create a new ghost pad with the static Sink pad retrieved from this Elementr's 
            // pGstObj and adds it to the the Elementr's Parent Bintr's pGstObj.
            if (!gst_element_add_pad(GST_ELEMENT(m_pParentGstObj), 
                gst_ghost_pad_new(name, gst_element_get_static_pad(GST_ELEMENT(m_pGstObj), name))))
            {
                LOG_ERROR("Failed to add Pad '" << name << "' for element'" << m_name << "'");
                throw;
            }
        }

        /**
         * @brief links this Elementr as Source to a given Sink Elementr
         * @param pSinkBintr to link to
         */
        void LinkToSink(DSL_NODETR_PTR pSink)
        { 
            LOG_FUNC();
            
            // Call the base class to setup the relationship first
            Nodetr::LinkToSink(pSink);

            // Link Source Element to Sink Element 
            if (!gst_element_link(GST_ELEMENT(m_pGstObj), GST_ELEMENT(m_pSink->m_pGstObj)))
            {
                LOG_ERROR("Failed to link " << m_name << " to " << pSink->m_name);
                throw;
            }
        }

        /**
         * @brief unlinks this Elementr from a previously linked-to Sink Elementr
         */
        void UnlinkFromSink()
        { 
            LOG_FUNC();

            if (IsLinkedToSink())
            {
                gst_element_unlink(GST_ELEMENT(m_pGstObj), GST_ELEMENT(m_pSink->m_pGstObj));

                // Call the base class to complete the unlink
                Nodetr::UnlinkFromSink();
            }
        }

        /**
         * @brief unlinks this Elementr from a previously linked-to Source Element
         */
        void UnlinkFromSource()
        { 
            LOG_FUNC();

            if (IsLinkedToSource())
            {
                gst_element_unlink(GST_ELEMENT(m_pSource->m_pGstObj), GST_ELEMENT(m_pGstObj));

                // Call the base class to complete the unlink
                Nodetr::UnlinkFromSource();
            }
        }
    };

} // DSL namespace    

#endif // _DSL_GSTNODETR_H