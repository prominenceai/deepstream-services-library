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
            : m_bin(NULL)
            , m_pParentBintr(NULL)
            , m_pSourceBintr(NULL)
            , m_pDestBintr(NULL)
        { 
            LOG_FUNC(); 
            LOG_INFO("New bintr:: " << name);
            
            m_name.assign(name);
            
            g_mutex_init(&m_bintrMutex);
        };
        
        ~Bintr()
        {
            LOG_FUNC();
            LOG_INFO("Delete bintr:: " << m_name);

            g_mutex_clear(&m_bintrMutex);
        };

        void LinkTo(Bintr* pDestBintr)
        { 
            LOG_FUNC();
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_bintrMutex);
            
            m_pDestBintr = pDestBintr;
            pDestBintr->m_pSourceBintr = this;

            LOG_INFO("Source Bin " << m_name 
                << " enabled:: " << (bool)m_bin);
            LOG_INFO("Distination Bin " << pDestBintr->m_name 
                << " enabled:: " << (bool)m_bin);
            
            if (m_bin && pDestBintr->m_bin)
            {
                if (!gst_element_link(m_bin, pDestBintr->m_bin))
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
            
            if (m_bin && pChildBintr->m_bin)
            {
                if (!gst_bin_add(GST_BIN(m_bin), pChildBintr->m_bin))
                {
                    LOG_ERROR("Failed to add " << pChildBintr->m_bin << " to " << m_name);
                    throw;
                }
            }
        }
        
    private:
        std::string m_name;

        GstElement* m_bin;

        Bintr* m_pParentBintr;
        
        Bintr* m_pSourceBintr;

        Bintr* m_pDestBintr;
        
        /**
         * @brief mutex to protect bintr reentry
         */
        GMutex m_bintrMutex;
    };

    class SourceBintr : public Bintr
    {
    public: 
    
        SourceBintr(const std::string& source, guint type, gboolean live, 
            guint width, guint height, guint fps_n, guint fps_d)
            : Bintr(source)
        {
            LOG_FUNC();
            
            INIT_MEMORY(m_nvdsConfig);
            INIT_MEMORY(m_nvdsBin);
            
            m_nvdsConfig.type = (NvDsSourceType)type;
            m_nvdsConfig.live_source = live;
            m_nvdsConfig.source_width = width;
            m_nvdsConfig.source_height = height;
            m_nvdsConfig.source_fps_n = fps_n;
            m_nvdsConfig.source_fps_d = fps_d;
            m_nvdsConfig.camera_csi_sensor_id = 1;

            if (!create_source_bin(&m_nvdsConfig, &m_nvdsBin))
            {
                LOG_ERROR("Failed to create new Source bin for '" << source << "'");
                throw;
            }
            
        };

        ~SourceBintr()
        {
            LOG_FUNC();
        };
        
    private:
    
        NvDsSourceConfig m_nvdsConfig;

        NvDsSrcBin m_nvdsBin;        
    };

    class StreamMuxBintr : public Bintr
    {
    public: 
    
        StreamMuxBintr(const std::string& streammux, 
            gboolean live, guint batchSize, guint batchTimeout, guint width, guint height)
            : Bintr(streammux)
        {
            LOG_FUNC();
            
            INIT_MEMORY(m_nvdsConfig);
            
            m_nvdsConfig.pipeline_width = width;
            m_nvdsConfig.pipeline_height = height;
            m_nvdsConfig.batch_size = batchSize;
            m_nvdsConfig.batched_push_timeout = batchTimeout;
            m_nvdsConfig.live_source = live;
            m_nvdsConfig.is_parsed = TRUE;

            m_pBin = gst_element_factory_make (NVDS_ELEM_STREAM_MUX, "stream_muxer");
            if (!m_pBin) 
            {            
                LOG_ERROR("Failed to create new Stream Muxer bin for '" << streammux << "'");
                throw;
            };
            if (!set_streammux_properties(&m_nvdsConfig, m_pBin))
            {
                LOG_ERROR("Failed to set Stream Muxer properties for '" << streammux << "'");
                throw;
            }
        };    

        ~StreamMuxBintr()
        {
            LOG_FUNC();
        };

        
    private:
    
        NvDsStreammuxConfig m_nvdsConfig;

        GstElement* m_pBin;        
    };
} // DSD

#endif // _DSD_BINTR_H