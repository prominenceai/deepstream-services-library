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

#ifndef _DSL_ELEMENTR_H
#define _DSL_ELEMENTR_H

#include "Dsl.h"
#include "DslNodetr.h"
#include "DslPadtr.h"


namespace DSL
{

    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_ELEMENT_PTR std::shared_ptr<Elementr>
    #define DSL_ELEMENT_NEW(type, name) \
        std::shared_ptr<Elementr>(new Elementr(type, name))   

    /**
     * @class Elementr
     * @brief Implements a container class for a GST Element
     */
    class Elementr : public Nodetr
    {
    public:

        /**
         * @brief ctor for the container class
         */
        Elementr(const char* factoryname, const char* name)
            : Nodetr(name)
        { 
            LOG_FUNC(); 
            
            m_pGstObj = GST_OBJECT(gst_element_factory_make(factoryname, name));
            if (!m_pGstObj)
            {
                LOG_ERROR("Failed to create new Element '" << name << "'");
                throw;  
            }
        };
        
        ~Elementr()
        {
            LOG_FUNC();
        };

        /**
         * @brief Sets a GST Element's attribute, owned by this Elementr to a value of uint
         * @param name name of the attribute to set
         * @param value unsigned integer value to set the attribute
         */
        void SetAttribute(const char* name, uint value)
        {
            LOG_FUNC();
            
            LOG_DEBUG("Setting attribute '" << name << "' to uint value '" << value << "'");
            
            g_object_set(G_OBJECT(m_pGstObj), name, value, NULL);
        }
        
        /**
         * @brief Sets a GST Element's attribute, owned by this Elementr to a 
         * null terminated array of characters (char*)
         * @param name name of the attribute to set
         * @param value char* string value to set the attribute
         */
        void SetAttribute(const char* name, const char* value)
        {
            LOG_FUNC();
            
            LOG_DEBUG("Setting attribute '" << name << "' to char* value '" << value << "'");
            
            g_object_set(G_OBJECT(m_pGstObj), name, value, NULL);
        }
        
        /**
         * @brief Sets a GST Element's attribute, owned by this Elementr to a 
         * value of type GstCaps, created with one of gst_caps_new_* 
         * @param name name of the attribute to set
         * @param value char* string value to set the attribute
         */
        void SetAttribute(const char* name, const GstCaps * value)
        {
            LOG_FUNC();
            
            LOG_DEBUG("Setting attribute '" << name << "' to char* value '" << value << "'");
            
            g_object_set(G_OBJECT(m_pGstObj), name, value, NULL);
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

        void LinkTo(DSL_NODETR_PTR pSink)
        { 
            LOG_FUNC();

            // Call the base class to setup the relationship
            Nodetr::LinkTo(pSink);

            // Link Source Bintr to Sink Bintr as elements 
            if (!gst_element_link(GST_ELEMENT(m_pGstObj), GST_ELEMENT(m_pSink->m_pGstObj)))
            {
                LOG_ERROR("Failed to link " << m_name << " to " << pSink->m_name);
                throw;
            }
        }
        
        void Unlink()
        { 
            LOG_FUNC();

            if (IsLinked())
            {
                gst_element_unlink(GST_ELEMENT(m_pGstObj), GST_ELEMENT(m_pSink->m_pGstObj));

                // Call the base class to unlink the shared pointers
                Nodetr::Unlink();
            }
        }
    
    };
}

#endif // _DSL_ELEMENTR_H   