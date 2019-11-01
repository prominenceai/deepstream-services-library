
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

#ifndef _DSL_PIPELINE_SOURCES_BINTR_H
#define _DSL_PIPELINE_SOURCES_BINTR_H

#include "Dsl.h"
#include "DslSourceBintr.h"

#define DSL_PIPELINE_SOURCES_PTR std::shared_ptr<PipelineSourcesBintr>
#define DSL_PIPELINE_SOURCES_NEW(name) \
    std::shared_ptr<PipelineSourcesBintr>(new DSL::PipelineSourcesBintr(name))

namespace DSL
{
    class PipelineSourcesBintr : public Bintr
    {
    public: 
    
        PipelineSourcesBintr(const char* name);

        ~PipelineSourcesBintr();
        
        void AddChild(std::shared_ptr<Bintr> pChildBintr);
        
        void RemoveChild(std::shared_ptr<Bintr> pChildBintr);
        
        void RemoveAllChildren();
        
        void AddSourceGhostPad();
        
        /**
         * @brief interates through the list of child source bintrs setting 
         * their Sensor Id's and linking to the StreamMux
         */
        void LinkAll();
        
        /**
         * @brief interates through the list of child source bintrs unlinking
         * them from the StreamMux and reseting their Sensor Id's
         */
        void UnlinkAll();

        void SetStreamMuxPlayType(bool areSourcesLive);        
        
        void SetStreamMuxBatchProperties(uint batchSize, uint batchTimeout);
        
        void SetStreamMuxOutputSize(uint width, uint height);

        /**
         * @brief Returns the number sources currently owned by this Sources Bintr
         * @return 
         */
        uint GetNumSourceInUse()
        {
            LOG_FUNC();
            
            return m_pChildBintrs.size();
        }
        

    private:

        DSL_ELEMENT_PTR m_pStreamMux;
        
        /**
         @brief
         */
        bool m_areSourcesLive;

        /**
         @brief
         */
        gint m_batchSize;

        /**
         @brief
         */
        gint m_batchTimeout;
        /**
         @brief
         */
        gint m_streamMuxWidth;

        /**
         @brief
         */
        gint m_streamMuxHeight;

        /**
         @brief
         */
        gboolean m_enablePadding;
    };

    
}

#endif // _DSL_PIPELINE_SOURCES_BINTR_H
