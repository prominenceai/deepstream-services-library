/*
The MIT License

Copyright (c) 2021, Prominence AI, Inc.

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

#include "Dsl.h"
#include "DslSinkWebRtcBintr.h"

namespace DSL
{
    WebRtcSinkBintr::WebRtcSinkBintr(const char* name)
        : SinkBintr(name, true, false)
        , m_qos(false)
    {
        LOG_FUNC();
        
        m_pWebRtcSink = DSL_ELEMENT_NEW(NVDS_ELEM_SINK_FAKESINK, "sink-bin-webrtc");
        m_pWebRtcSink->SetAttribute("enable-last-sample", false);
        m_pWebRtcSink->SetAttribute("max-lateness", -1);
        m_pWebRtcSink->SetAttribute("sync", m_sync);
        m_pWebRtcSink->SetAttribute("async", m_async);
        m_pWebRtcSink->SetAttribute("qos", m_qos);
        
        AddChild(m_pWebRtcSink);
    }
    
    WebRtcSinkBintr::~WebRtcSinkBintr()
    {
        LOG_FUNC();
    
        if (IsLinked())
        {    
            UnlinkAll();
        }
    }

    bool WebRtcSinkBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("WebRtcSinkBintr '" << GetName() << "' is already linked");
            return false;
        }
        if (!m_pQueue->LinkToSink(m_pWebRtcSink))
        {
            return false;
        }
        m_isLinked = true;
        return true;
    }
    
    void WebRtcSinkBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("WebRtcSinkBintr '" << GetName() << "' is not linked");
            return;
        }
        m_pQueue->UnlinkFromSink();
        m_isLinked = false;
    }

    bool WebRtcSinkBintr::SetSyncSettings(bool sync, bool async)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set Sync/Async Settings for WebRtcSinkBintr '" << GetName() 
                << "' as it's currently linked");
            return false;
        }
        m_sync = sync;
        m_async = async;
        
        m_pWebRtcSink->SetAttribute("sync", m_sync);
        m_pWebRtcSink->SetAttribute("async", m_async);
        
        return true;
    }
    
} // DSL