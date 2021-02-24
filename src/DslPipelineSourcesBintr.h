
/*
The MIT License

Copyright (c) 2019-2021, Prominence AI, Inc.

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
#include "DslApi.h"
#include "DslSourceBintr.h"

namespace DSL
{
    #define DSL_PIPELINE_SOURCES_PTR std::shared_ptr<PipelineSourcesBintr>
    #define DSL_PIPELINE_SOURCES_NEW(name) \
        std::shared_ptr<PipelineSourcesBintr>(new PipelineSourcesBintr(name))

    class PipelineSourcesBintr : public Bintr
    {
    public: 
    
        PipelineSourcesBintr(const char* name);

        ~PipelineSourcesBintr();
        
        /**
         * @brief adds a child SourceBintr to this PipelineSourcesBintr
         * @param pChildSource shared pointer to SourceBintr to add
         * @return true if the SourceBintr was added correctly, false otherwise
         */
        bool AddChild(DSL_SOURCE_PTR pChildSource);
        
        /**
         * @brief removes a child SourceBintr from this PipelineSourcesBintr
         * @param pChildElement a shared pointer to SourceBintr to remove
         * @return true if the SourceBintr was removed correctly, false otherwise
         */
        bool RemoveChild(DSL_SOURCE_PTR pChildSource);

        /**
         * @brief overrides the base method and checks in m_pChildSources only.
         */
        bool IsChild(DSL_SOURCE_PTR pChildSource);

        /**
         * @brief overrides the base Noder method to only return the number of 
         * child SourceBintrs and not the total number of children... 
         * i.e. exclude the nuber of child Elementrs from the count
         * @return the number of Child SourceBintrs held by this PipelineSourcesBintr
         */
        uint GetNumChildren()
        {
            LOG_FUNC();
            
            return m_pChildSources.size();
        }

        /**
         * @brief interates through the list of child source bintrs setting 
         * their Sensor Id's and linking to the StreamMux
         */
        bool LinkAll();
        
        /**
         * @brief interates through the list of child source bintrs unlinking
         * them from the StreamMux and reseting their Sensor Id's
         */
        void UnlinkAll();

        /**
         * @brief Gets the current Streammuxer "play-type-is-live" setting
         * @return true if play-type is live, false otherwise
         */
        bool StreamMuxPlayTypeIsLiveGet();        

        /**
         * @brief Sets the current Streammuxer play type based on the first source added
         * @param isLive set to true if all sources are to be Live, and therefore live only.
         */
        void StreamMuxPlayTypeIsLiveSet(bool isLive);        
        
        /**
         * @brief Gets the current batch settings for the SourcesBintr's Stream Muxer
         * @param[out] batchSize current batchSize, default == the number of source
         * @param[out] batchTimeout current batch timeout
         */
        void GetStreamMuxBatchProperties(uint* batchSize, uint* batchTimeout);

        /**
         * @brief Sets the current batch settings for the SourcesBintr's Stream Muxer
         * @param[in] batchSize new batchSize to set, default == the number of sources
         * @param[in] batchTimeout timeout value to set in ms
         */
        void SetStreamMuxBatchProperties(uint batchSize, uint batchTimeout);

        /**
         * @brief Gets the current dimensions for the SourcesBintr's Stream Muxer
         * @param[out] width width in pixels for the current setting
         * @param[out] height height in pixels for the curren setting
         */
        void GetStreamMuxDimensions(uint* width, uint* height);

        /**
         * @brief Set the dimensions for the SourcesBintr's StreamMuxer
         * @param width width in pixels to set the streamMux Output
         * @param height height in pixels to set the StreamMux output
         */
        void SetStreamMuxDimensions(uint width, uint height);
        
        /**
         * @brief Gets the current setting for the PipelineSourcesBintr's Muxer padding
         * @param enable true if enabled, false otherwise.
         */
        void GetStreamMuxPadding(bool* enabled);

        /**
         * @brief Sets, enables/disables the PipelineSourcesBintr's StreamMuxer padding
         * @param enabled set to true to enable padding
         */
        void SetStreamMuxPadding(bool enabled);

        /**
         * @brief Gets the current setting for the PipelineSourcesBintr's StreamMuxer
         * num-surfaces-per-frame seting
         * @param[out] num current setting for the number of surfaces [1..4].
         */
        void GetStreamMuxNumSurfacesPerFrame(uint* num);

        /**
         * @brief Sets the current setting for the PipelineSourcesBintr's StreamMuxer
         * num-surfaces-per-frame seting
         * @param[in] num new value for the number of surfaces [1..4].
         */
        void SetStreamMuxNumSurfacesPerFrame(uint num);

    private:
        /**
         * @brief adds a child Elementr to this PipelineSourcesBintr
         * @param pChildElement a shared pointer to the Elementr to add
         * @return a shared pointer to the Elementr if added correctly, nullptr otherwise
         */
        bool AddChild(DSL_BASE_PTR pChildElement);
        
        /**
         * @brief removes a child Elementr from this PipelineSourcesBintr
         * @param pChildElement a shared pointer to the Elementr to remove
         */
        bool RemoveChild(DSL_BASE_PTR pChildElement);
        
        /**
         * @brief Pad Probe Event Handler to consume all dowstream EOS events
         * Will be created if and when a RTSP source is added to this PipelineSourcesBintr.
         */
        DSL_PPEH_EOS_CONSUMER_PTR m_pEosConsumer;

    public:

        DSL_ELEMENT_PTR m_pStreamMux;
        
        std::map<std::string, DSL_SOURCE_PTR> m_pChildSources;
        
        /**
         @brief
         */
        bool m_areSourcesLive;

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
        bool m_isPaddingEnabled;
        
        /**
         @brief Number of surfaces-per-frame stream-muxer setting
         */
        int m_numSurfacesPerFrame;
    };

    
}

#endif // _DSL_PIPELINE_SOURCES_BINTR_H
