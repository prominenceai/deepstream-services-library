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

#ifndef _DSL_DEMUXER_BINTR_H
#define _DSL_DEMUXER_BINTR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslElementr.h"
#include "DslBintr.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_DEMUXER_PTR std::shared_ptr<DemuxerBintr>
    #define DSL_DEMUXER_NEW(name) \
        std::shared_ptr<DemuxerBintr>(new DemuxerBintr(name))
        
    class DemuxerBintr : public Bintr
    {
    public: 
    
        /**
         * @brief Ctor for the DemuxerBintr class
         * @param[in] name unique name to give to the Demuxer
         */
        DemuxerBintr(const char* name);

        /**
         * @brief dtor for the DemuxerBintr class
         */
        ~DemuxerBintr();

        /**
         * @brief Adds the DemuxerBintr to a Parent Pipeline Bintr
         * @param[in] pParentBintr Parent Pipeline to add this Bintr to
         */
        bool AddToParent(DSL_NODETR_PTR pParentBintr);

        /**
         * @brief Links all Child Elementrs owned by this Bintr
         * @return true if all links were succesful, false otherwise
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elementrs owned by this Bintr
         * Calling UnlinkAll when in an unlinked state has no effect.
         */
        void UnlinkAll();
        
        /**
         * @brief returns a const shared pointer to the Bintr's Demuxer Elementr
         * @return 
         */
        const DSL_NODETR_PTR GetDemuxerElementr()
        {
            LOG_FUNC();
            
            return m_pDemuxer;
        }

    private:

        /**
         * @brief Demuxer element for all muxed input streams
         */
        DSL_ELEMENT_PTR m_pDemuxer;

    };
    
} // DSL

#endif // _DSL_DEMUXER_BINTR_H
