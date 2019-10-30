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
#include "DslPadtr.h"

namespace DSL
{
    /**
     * @class Elementr
     * @brief Implements a container class for a GST Element
     */
    class Elementr : public std::enable_shared_from_this<Elementr>
    {
    public:

        /**
         * @brief ctor for the container class
         */
        Elementr(const char* factoryname, const char* name, GstElement* parentBin)
            : m_name(name)
            , m_pParentBin(parentBin)
            , m_pElement(NULL)
            , m_pLinkedSourceElementr(nullptr)
            , m_pLinkedSinkElementr(nullptr)
            , m_pSinkPad(NULL)
            , m_pSourcePad(NULL)
        { 
            LOG_FUNC(); 
            
            m_pElement = gst_element_factory_make(factoryname, name);
            if (!m_pElement)
            {
                LOG_ERROR("Failed to create new Element '" << name << "'");
                throw;  
            }

            if (!gst_bin_add(GST_BIN(m_pParentBin), m_pElement))
            {
                LOG_ERROR("Failed to add Elementr" << name << " to parent bin");
                throw;
            }
            
        };
        
        ~Elementr()
        {
            LOG_FUNC();
            
            // Clean up all resources
            if (m_pSinkPad)
            {
                gst_object_unref(m_pSinkPad);
            }

            if (m_pSourcePad)
            {
                gst_object_unref(m_pSourcePad);
            }
            
            if (m_pElement and GST_OBJECT_REFCOUNT_VALUE(m_pElement))
            {
                gst_object_unref(m_pElement);
            }
        };

        
        /**
         * @brief Creates a new Ghost Sink pad for this Gst Element
         * and adds it to the parent Gst Bin.
         * @throws a general exception on failure
         */
        void AddSinkGhostPad()
        {
            LOG_FUNC();
            
            // get Sink pad for this element 
            StaticPadtr SinkPadtr(m_pElement, "sink");

            // create a new ghost pad with the Sink pad and add to this bintr's bin
            if (!gst_element_add_pad(m_pParentBin, gst_ghost_pad_new("sink", SinkPadtr.m_pPad)))
            {
                LOG_ERROR("Failed to add Sink Pad for element'" << m_name);
                throw;
            }
        };
        
        /**
         * @brief Creates a new Ghost Source pad for this Gst element
         * and adds it to the parent Gst Bin.
         * @throws a general exception on failure
         */
        void AddSourceGhostPad()
        {
            LOG_FUNC();
            
            // get Source pad for last child element in the ordered list
            StaticPadtr SourcePadtr(m_pElement, "src");

            // create a new ghost pad with the Source pad and add to this bintr's bin
            if (!gst_element_add_pad(m_pParentBin, gst_ghost_pad_new("src", SourcePadtr.m_pPad)))
            {
                LOG_ERROR("Failed to add Source Pad for '" << m_name);
                throw;
            }
        };

        void LinkTo(std::shared_ptr<Elementr> pSinkElement)
        {
            LOG_FUNC();

            m_pLinkedSinkElementr = pSinkElement;

            pSinkElement->m_pLinkedSourceElementr = 
                std::dynamic_pointer_cast<Elementr>(shared_from_this());
            
            if (!gst_element_link(m_pElement, m_pLinkedSinkElementr->m_pElement))
            {
                LOG_ERROR("Failed to link " << m_name << " to "
                    << pSinkElement->m_name);
                throw;
            }
        };
        
        void Unlink()
        {
            LOG_FUNC();
            
            if (m_pLinkedSinkElementr)
            {
            
                gst_element_unlink(m_pElement, m_pLinkedSinkElementr->m_pElement);
                m_pLinkedSinkElementr->m_pLinkedSourceElementr = nullptr;
                m_pLinkedSinkElementr = nullptr;
            }    
        };
        
    public:

        /**
         @brief unique name for this Bintr
         */
        std::string m_name;

        /**
         * @brief pointer to the contained GST Element for this Elementr
         */
        GstElement* m_pElement;
        
        /**
         * @brief pointer to the Parent bin for this Elementr
         */
        GstElement* m_pParentBin;

        /**
         * @brief Shared pointer to Source Elementr if linked
         */
        std::shared_ptr<Elementr> m_pLinkedSourceElementr;
        
        /**
         * @brief Shared pointer to Sink Elementr if linked
         */
        std::shared_ptr<Elementr> m_pLinkedSinkElementr;

        /**
         @brief
         */
        GstPad *m_pSinkPad;
        
        /**
         @brief
         */
        GstPad *m_pSourcePad; 
    };
}

#endif // _DSL_ELEMENTR_H   