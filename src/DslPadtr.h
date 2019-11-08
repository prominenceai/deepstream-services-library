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
#include "DslNodetr.h"

namespace DSL
{

    #define DSL_PADTR_PTR std::shared_ptr<Padtr>
    #define DSL_PADTR_NEW(name) \
        std::shared_ptr<Padtr>(new Padtr(name))    

    #define DSL_STATIC_PADTR_PTR std::shared_ptr<StaticPadtr>
    #define DSL_STATIC_PADTR_NEW(name, parentElement) \
        std::shared_ptr<StaticPadtr>(new StaticPadtr(name, parentElement))    

    #define DSL_REQUEST_PADTR_PTR std::shared_ptr<RequestPadtr>
    #define DSL_REQUEST_PADTR_NEW(name, parentElement) \
        std::shared_ptr<RequestPadtr>(new RequestPadtr(name, parentElement))

    /**
     * @class Padtr
     * @brief Implements a base container class for Gst Pad types
     */
    class Padtr : public Nodetr
    {
    public:
        
        /**
         * @brief ctor for the Padtr base class
         * @param[in] name of the pad element
         */
        Padtr(const char* name)
        : Nodetr(name)
        {
            LOG_FUNC();
        };

        /**
         * @brief dtor for the Padtr base class
         */
        ~Padtr()
        {
            LOG_FUNC();
        };

        /**
         * @brief returns whether the Bintr object is in-use 
         * @return True if the Bintr has a relationship with another Bintr
         */
        void LinkTo(DSL_NODETR_PTR pSink)
        {
            LOG_FUNC();
            
            if (gst_pad_link(GST_PAD(m_pGstObj), GST_PAD(pSink->m_pGstObj)) != GST_PAD_LINK_OK)
            {
                LOG_ERROR("Failed to link source pad '" << m_name << "' to sink pad '" << pSink->m_name << "'");
                throw;
            }
            
            Nodetr::LinkTo(pSink);
        }
        
        void Unlink()
        {

            LOG_FUNC();

            if (IsLinked())
            {
                gst_pad_unlink(GST_PAD(m_pGstObj), GST_PAD(m_pSink->m_pGstObj));

                // Call the base class to complete the unlink
                Nodetr::Unlink();
            }
        }
        
    };

    /**
     * @class StaticPadtr
     * @brief Implements a container class for a Gst Static Pad
     */
    class StaticPadtr : public Padtr
    {
    public:
        
        /**
         * @brief ctor for the StaticPadtr class
         * @param[in] name of the static pad to retrieve
         */
        StaticPadtr(const char* name, DSL_NODETR_PTR pParent)
        : Padtr(name)
        {
            LOG_FUNC();

            m_pGstObj = GST_OBJECT(gst_element_get_static_pad(GST_ELEMENT(pParent->m_pGstObj), (gchar*)name));
            if (!m_pGstObj)
            {
                LOG_ERROR("Failed to get Static Pad for '" << name <<"'");
                throw;
            }
        }

        /**
         * @brief dtor for the StaticPadtr class
         */
        ~StaticPadtr()
        {
            LOG_FUNC();
        }
        
        guint AddPad(GstPadProbeType mask, GstPadProbeCallback callback, gpointer pData)
        {
            LOG_FUNC();
            
            return gst_pad_add_probe(GST_PAD(m_pGstObj), mask, callback, pData, NULL);
        }
    
    };
    
    /**
     * @class StaticPadtr
     * @brief Implements a container class for a Gst Request Pad
     */
    class RequestPadtr : public Padtr
    {
    public:
        
        /**
         * @brief ctor for the RequestPadtr class
         * @param[in] name of the static pad to retrieve
         * @param[in] pElement element to retreive the static pad from
         */
        RequestPadtr(const char* name, DSL_NODETR_PTR pParent)
            : Padtr(name)
        {
            LOG_FUNC();
            
            m_pGstObj = GST_OBJECT(gst_element_get_request_pad(
                GST_ELEMENT(pParent->m_pGstObj), (gchar*)name));
            if (!m_pGstObj)
            {
                LOG_ERROR("Failed to get Request Pad for '" << name <<" '");
                throw;
            }
        }

        /**
         * @brief ctor for the RequestPadtr class
         * @param[in] name of the static pad to retrieve
         * @param[in] pElement element to retreive the static pad from
         * @param[in] pPadTemplate
         */
        RequestPadtr(const char* name, DSL_NODETR_PTR pParent, GstPadTemplate* pPadTemplate)
            : Padtr(name)
        {
            LOG_FUNC();
            
            m_pGstObj = GST_OBJECT(gst_element_request_pad(
                GST_ELEMENT(pParent->m_pGstObj), pPadTemplate, NULL, NULL));
            if (!m_pGstObj)
            {
                LOG_ERROR("Failed to get Pad for '" << name <<" '");
                throw;
            }
        }

        /**
         * @brief dtor for the RequestPadtr class
         */
        ~RequestPadtr()
        {
            LOG_FUNC();
        }

    };
} // DSL namespace    

#endif // _DSL_PADTR_H