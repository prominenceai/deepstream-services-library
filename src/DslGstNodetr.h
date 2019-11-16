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

            LOG_INFO("New GstNodetr '" << GetName() << "' created");
        }

        /**
         * @brief dtor for the GstNodetr base class
         */
        ~GstNodetr()
        {
            LOG_FUNC();

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
            if (!gst_element_add_pad(GST_ELEMENT(GetParentGstObject()), 
                gst_ghost_pad_new(name, gst_element_get_static_pad(GetGstElement(), name))))
            {
                LOG_ERROR("Failed to add Pad '" << name << "' for element'" << GetName() << "'");
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
            if (!gst_element_link(GetGstElement(), m_pSink->GetGstElement()))
            {
                LOG_ERROR("Failed to link " << GetName() << " to " << pSink->GetName());
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
//                gst_element_unlink(GetGstElement(), m_pSink->GetGstElement());
                LOG_WARN("Unlinking Elementr '" << GetName() << "' from Sink '" << m_pSink << "'");

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
                gst_element_unlink(m_pSource->GetGstElement(), GetGstElement());

                // Call the base class to complete the unlink
                Nodetr::UnlinkFromSource();
            }
        }
        
        /**
         * @brief Returns the current State of this GstNodetr
         * @return the current state of the GstNodetr. 
         */
        uint GetState()
        {
            LOG_FUNC();
            
            GstState currentState;
            
            if (gst_element_get_state(GetGstElement(), &currentState, NULL, 1) == GST_STATE_CHANGE_ASYNC)
            {
                return DSL_STATE_IN_TRANSITION;
            }
            
            LOG_INFO("Returning a state of '" << gst_element_state_get_name(currentState) 
                << "' for Nodetr '" << GetName());
            
            return currentState;
        }
        
        /**
         * @brief Returns the current State of this Bintr's Parent
         * @return the current state of the Parenet, GST_STATE_NULL if the
         * GstNodetr is currently an orphen. 
         */
        uint GetParentState()
        {
            LOG_FUNC();
            
            if (!m_pParentGstObj)
            {
                return GST_STATE_NULL;
            }
            GstState currentState;
            
            if (gst_element_get_state(GetParentGstElement(), &currentState, NULL, 1) == GST_STATE_CHANGE_ASYNC)
            {
                return DSL_STATE_IN_TRANSITION;
            }
            
            LOG_INFO("Returning a state of '" << gst_element_state_get_name(currentState) 
                << "' for Nodetr '" << GetName());
            
            return currentState;
        }
    };

} // DSL namespace    

#endif // _DSL_GSTNODETR_H