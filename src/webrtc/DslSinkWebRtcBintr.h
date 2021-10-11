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

#ifndef _DSL_SINK_WEBRTC_BINTR_H
#define _DSL_SINK_WEBRTC_BINTR_H

#include "DslSinkBintr.h"

namespace DSL
{

    #define DSL_WEBRTC_SINK_PTR std::shared_ptr<WebRtcSinkBintr>
    #define DSL_WEBRTC_SINK_NEW(name) \
        std::shared_ptr<WebRtcSinkBintr>(new WebRtcSinkBintr(name))

    class WebRtcSinkBintr : public SinkBintr
    {
    public: 
    
        WebRtcSinkBintr(const char* name);

        ~WebRtcSinkBintr();
  
        /**
         * @brief Links all Child Elementrs owned by this Bintr
         * @return true if all links were succesful, false otherwise
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elemntrs owned by this Bintr
         * Calling UnlinkAll when in an unlinked state has no effect.
         */
        void UnlinkAll();
        
        /**
         * @brief sets the current sync and async settings for the SinkBintr
         * @param[in] sync current sync setting, true if set, false otherwise.
         * @param[in] async current async setting, true if set, false otherwise.
         * @return true is successful, false otherwise. 
         */
        bool SetSyncSettings(bool sync, bool async);

    private:

        boolean m_qos;
        
        /**
         * @brief WebRtc Sink element for the Sink Bintr.
         */
        DSL_ELEMENT_PTR m_pWebRtcSink;
    };

}
#endif //_DSL_SINK_BINTR_H
