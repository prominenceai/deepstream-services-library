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
#include "DslPipelineBusSyncMgr.h"
#include "DslServices.h"

namespace DSL
{
    PipelineBusSyncMgr::PipelineBusSyncMgr(const GstObject* pGstPipeline)
    {
        LOG_FUNC();

        GstBus* pGstBus = gst_pipeline_get_bus(GST_PIPELINE(pGstPipeline));

        // install the sync handler for the message bus
        gst_bus_set_sync_handler(pGstBus, bus_sync_handler, this, NULL);        

        gst_object_unref(pGstBus);

        g_mutex_init(&m_busSyncMutex);
    }

    PipelineBusSyncMgr::~PipelineBusSyncMgr()
    {
        LOG_FUNC();
        
        g_mutex_clear(&m_busSyncMutex);
    }
    
    GstBusSyncReply PipelineBusSyncMgr::HandleBusSyncMessage(GstMessage* pMessage)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_busSyncMutex);

        switch (GST_MESSAGE_TYPE(pMessage))
        {
        case GST_MESSAGE_ELEMENT:
        
            if (gst_is_video_overlay_prepare_window_handle_message(pMessage))
            {
                // A Window Sink component is signaling to prepare 
                // the window handle. Call into the Window-Sink registry services
                // to get the Owner of the nveglglessink element

                DSL_WINDOW_SINK_PTR pWindowSink =
                    std::dynamic_pointer_cast<WindowSinkBintr>(
                        DSL::Services::GetServices()->_sinkWindowGet(
                            GST_MESSAGE_SRC(pMessage)));
                        
                // If the sink is found -- should always true.
                if (pWindowSink)
                {
                    pWindowSink->CreateXWindow();
                }
                else
                {
                    LOG_ERROR("Failed to find WindowSinkBintr in registry");
                }
                    
                UNREF_MESSAGE_ON_RETURN(pMessage);
                return GST_BUS_DROP;
            }
            break;
        default:
            break;
        }
        return GST_BUS_PASS;
    }

    static GstBusSyncReply bus_sync_handler(GstBus* bus, 
        GstMessage* pMessage, gpointer pData)
    {
        return static_cast<PipelineBusSyncMgr*>(pData)->
            HandleBusSyncMessage(pMessage);
    }

} // DSL   