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

#ifndef _DSD_BINTR_H
#define _DSD_BINTR_H

#include "Dsd.h"

#define INIT_MEMORY(m) memset(&m, 0, sizeof(m));
#define LINK_TRUE true
#define LINK_FALSE false

namespace DSD
{
    
    /**
     * @class Bintr
     * @brief 
     */
    class Bintr
    {
    public:

        /**
         * @brief 
         */
        Bintr(const std::string& name)
            : m_gpuId(0)
            , m_nvbufMemoryType(0)
            , m_pSinkPad(NULL)
            , m_pSourcePad(NULL)
            , m_pParentBintr(NULL)
            , m_pSourceBintr(NULL)
            , m_pDestBintr(NULL)
        { 
            LOG_FUNC(); 
            LOG_INFO("New bintr:: " << name);
            
            m_name.assign(name);

            m_pBin = gst_bin_new((gchar*)name.c_str());
            if (!m_pBin)
            {
                LOG_ERROR("Failed to create new bin for component'" << name << "'");
                throw;  
            }
            
            g_mutex_init(&m_bintrMutex);
        };
        
        ~Bintr()
        {
            LOG_FUNC();
            LOG_INFO("Delete bintr:: " << m_name);

            g_mutex_clear(&m_bintrMutex);
        };

        std::string m_configFilePath;

        bool LinkTo(Bintr* pDestBintr)
        { 
            LOG_FUNC();
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_bintrMutex);
            
            m_pDestBintr = pDestBintr;
            pDestBintr->m_pSourceBintr = this;
            
            if (!gst_element_link(m_pBin, pDestBintr->m_pBin))
            {
                LOG_ERROR("Failed to link " << m_name << " to "
                    << pDestBintr->m_name);
                throw;
            }
        };

        void AddChild(Bintr* pChildBintr)
        {
            LOG_FUNC();
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_bintrMutex);
            
            pChildBintr->m_pParentBintr = this;
            m_pChildBintrs.push_back(pChildBintr);
                            
            if (!gst_bin_add(GST_BIN(m_pBin), pChildBintr->m_pBin))
            {
                LOG_ERROR("Failed to add " << pChildBintr->m_name << " to " << m_name);
                throw;
            }
        };
        
        GstElement* MakeElement(const gchar * factoryname, const gchar * name, bool linkToPrev)
        {
            LOG_FUNC();
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_bintrMutex);

            GstElement* pElement = gst_element_factory_make(factoryname, name);
            if (!pElement)
            {
                LOG_ERROR("Failed to create new Element '" << name << "'");
                throw;  
            }

            if (!gst_bin_add(GST_BIN(m_pBin), pElement))
            {
                LOG_ERROR("Failed to add " << name << " to " << m_name);
                throw;
            }
            
            if (!linkToPrev)
            {
                // If not the first element
                if (m_pLinkedChildElements.size())
                {
                    // link the previous to the new element 
                    if (!gst_element_link(m_pLinkedChildElements.back(), pElement))
                    {
                        LOG_ERROR("Failed to link new element" << m_name << " to "
                            << m_name);
                        throw;
                    }
                }
                m_pLinkedChildElements.push_back(pElement);
            }
            return pElement;
        };
        
        void AddGhostPads()
        {
            LOG_FUNC();
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_bintrMutex);
            
            // get Sink pad for first child element in the order list
            m_pSinkPad = gst_element_get_static_pad(m_pLinkedChildElements.back(), "sink");
            if (!m_pSinkPad)
            {
                LOG_ERROR("Failed to add Sink Pad for '" << m_name <<" '");
                throw;
            }
            
            // get Source pad for last child element in the ordered list
            m_pSourcePad = gst_element_get_static_pad(m_pLinkedChildElements.back(), "src");
            if (!m_pSourcePad)
            {
                LOG_ERROR("Failed to Source Pad for '" << m_name <<" '");
                throw;
            }
        };
        
    public:

        /**
         @brief
         */
        std::string m_name;

        /**
         @brief
         */
        GstElement* m_pBin;
        
        /**
         @brief
         */
        std::vector<GstElement*> m_pLinkedChildElements;
        
        /**
         @brief
         */
        guint m_gpuId;

        /**
         @brief
         */
        guint m_nvbufMemoryType;

        /**
         @brief
         */
        GstPad *m_pSinkPad;
        
        /**
         @brief
         */
        GstPad *m_pSourcePad; 
        
        /**
         @brief
         */
        Bintr* m_pParentBintr;
        
        /**
         @brief
         */
        std::vector<Bintr*> m_pChildBintrs;
        
        /**
         @brief
         */
        Bintr* m_pSourceBintr;

        /**
         @brief
         */
        Bintr* m_pDestBintr;
        
        /**
         * @brief mutex to protect bintr reentry
         */
        GMutex m_bintrMutex;
    };


} // DSD

#endif // _DSD_BINTR_H