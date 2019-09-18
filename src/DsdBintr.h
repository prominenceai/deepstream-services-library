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
            : m_pBin(NULL)
            , m_pParentBintr(NULL)
            , m_pSourceBintr(NULL)
            , m_pDestBintr(NULL)
        { 
            LOG_FUNC(); 
            LOG_INFO("New bintr:: " << name);
            
            m_name.assign(name);
            m_configFilePath.assign(DS_CONFIG_DIR);
            
            g_mutex_init(&m_bintrMutex);
        };
        
        ~Bintr()
        {
            LOG_FUNC();
            LOG_INFO("Delete bintr:: " << m_name);

            g_mutex_clear(&m_bintrMutex);
        };

        std::string m_configFilePath;

        void LinkTo(Bintr* pDestBintr)
        { 
            LOG_FUNC();
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_bintrMutex);
            
            m_pDestBintr = pDestBintr;
            pDestBintr->m_pSourceBintr = this;

            LOG_INFO("Source Bin " << m_name 
                << " enabled:: " << (bool)m_pBin);
            LOG_INFO("Distination Bin " << pDestBintr->m_name 
                << " enabled:: " << (bool)m_pBin);
            
            if (m_pBin && pDestBintr->m_pBin)
            {
                if (!gst_element_link(m_pBin, pDestBintr->m_pBin))
                {
                    LOG_ERROR("Failed to link " << m_name << " to "
                        << pDestBintr->m_name);
                    throw;
                }
            }
        };

        void AddChild(Bintr* pChildBintr)
        {
            LOG_FUNC();
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_bintrMutex);
            
//            m_pChildBintr = pChildBintr;
            pChildBintr->m_pParentBintr = this;
            
            if (m_pBin && pChildBintr->m_pBin)
            {
                if (!gst_bin_add(GST_BIN(m_pBin), pChildBintr->m_pBin))
                {
                    LOG_ERROR("Failed to add " << pChildBintr->m_pBin << " to " << m_name);
                    throw;
                }
            }
        }
        
    public:
        std::string m_name;

        GstElement* m_pBin;

        Bintr* m_pParentBintr;
        
        Bintr* m_pSourceBintr;

        Bintr* m_pDestBintr;
        
        /**
         * @brief mutex to protect bintr reentry
         */
        GMutex m_bintrMutex;
    };


    class PrimaryGieBintr : public Bintr
    {
    public: 
    
        PrimaryGieBintr(const std::string& gie, 
            const std::string& model,const std::string& infer, 
            guint batchSize, guint bc1, guint bc2, guint bc3, guint bc4)
            : Bintr(gie)
        {
            LOG_FUNC();
            
            INIT_MEMORY(m_nvdsConfig);
            INIT_MEMORY(m_nvdsBin);
            
            m_nvdsConfig.config_file_path = (gchar*)Bintr::m_configFilePath.c_str();;
            m_nvdsConfig.model_engine_file_path = (gchar*)model.c_str();
            m_nvdsConfig.label_file_path = (gchar*)infer.c_str();
            m_nvdsConfig.is_batch_size_set = (gboolean)batchSize;
            m_nvdsConfig.batch_size = batchSize;

            if (!create_primary_gie_bin(&m_nvdsConfig, &m_nvdsBin))
            {
                LOG_ERROR("Failed to create new Primary GIE bin for '" << gie << "'");
                throw;
            }
        };    

        ~PrimaryGieBintr()
        {
            LOG_FUNC();
        };

        
    private:
    
        NvDsGieConfig m_nvdsConfig;

        NvDsPrimaryGieBin m_nvdsBin;
    };
    

} // DSD

#endif // _DSD_BINTR_H