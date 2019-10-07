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

#ifndef _DSL_PADTR_H
#define _DSL_PADTR_H

#include "Dsl.h"

namespace DSL
{
    /**
     * @class StaticPadtr
     * @brief Implements a container class for a Gst Static Pad
     */
    class StaticPadtr
    {
    public:
        
        /**
         * @brief ctor for the StaticPadtr class
         * @param[in] pElement element to retreive the static pad from
         * @param[in] name of the static pad to retrieve
         */
        StaticPadtr(GstElement* pElement, const char* name)
        : m_name(name)
        , m_pPad(NULL)
        {
            LOG_FUNC();
            
            m_pPad = gst_element_get_static_pad(pElement, (gchar*)name);
            if (!m_pPad)
            {
                LOG_ERROR("Failed to get Static Pad for '" << name <<" '");
                throw;
            }
        };

        /**
         * @brief dtor for the StaticPadtr class
         */
        ~StaticPadtr()
        {
            LOG_FUNC();

            if (m_pPad)
            {
                gst_object_unref(m_pPad);
            }
        };
        
        /**
         * @brief named for request pad
         */
        std::string m_name;
        
        /**
         * @brief pointer to the contained static pad
         */
        GstPad* m_pPad;
    };
    
    /**
     * @class StaticPadtr
     * @brief Implements a container class for a Gst Static Pad
     */
    class RequestPadtr
    {
    public:
        
        /**
         * @brief ctor for the RequestPadtr class
         * @param[in] pElement element to retreive the static pad from
         * @param[in] name of the static pad to retrieve
         */
        RequestPadtr(GstElement* pElement, const char* name)
            : m_name(name)
            , m_pPad(NULL)
            , m_pElement(pElement)
        {
            LOG_FUNC();
            
            m_pPad = gst_element_get_request_pad(pElement, (gchar*)name);
            if (!m_pPad)
            {
                LOG_ERROR("Failed to get Request Pad for '" << name <<" '");
                throw;
            }
        };

        /**
         * @brief ctor for the RequestPadtr class
         * @param[in] pElement element to retreive the static pad from
         * @param[in] pPadTemplate
         * @param[in] name of the static pad to retrieve
         */
        RequestPadtr(GstElement* pElement, GstPadTemplate* pPadTemplate, const char* name)
            : m_name(name)
            , m_pPad(NULL)
            , m_pElement(pElement)
        {
            LOG_FUNC();
            
            m_pPad = gst_element_request_pad(pElement, pPadTemplate, NULL, NULL);
            if (!m_pPad)
            {
                LOG_ERROR("Failed to get Pad for '" << name <<" '");
                throw;
            }
        };

        /**
         * @brief dtor for the RequestPadtr class
         */
        ~RequestPadtr()
        {
            LOG_FUNC();

            if (m_pPad)
            {
                gst_element_release_request_pad(m_pElement, m_pPad);
            }
        };

        void LinkTo(StaticPadtr& staticPadtr)
        {
            if (gst_pad_link(m_pPad, staticPadtr.m_pPad) != GST_PAD_LINK_OK)
            {
                LOG_ERROR("Failed to link request pad '" << m_name 
                    << "' to static pad '" << staticPadtr.m_name << "'");
                throw;
            }
        }
        
        /**
         * @brief named for request pad
         */
        std::string m_name;
        
        /**
         * @brief pointer to the contained request pad
         */
        GstPad* m_pPad;
        
        /**
         * @brief pointer to the element the request pad was retreived from
         */
        GstElement* m_pElement;

    };
} // DSL namespace    

#endif // _DSL_PADTR_H