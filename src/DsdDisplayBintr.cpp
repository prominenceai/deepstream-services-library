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

#include "Dsd.h"
#include "DsdDisplayBintr.h"

namespace DSD
{

    DisplayBintr::DisplayBintr(const std::string& display, Display* pGstDisplay,
        guint rows, guint columns, guint width, guint height)
        : Bintr(display)
        , m_pGstDisplay(pGstDisplay)
        , m_rows(rows)
        , m_columns(columns)
        , m_width(width)
        , m_height(height)
        , m_gpuId(0)
        , m_enablePadding(FALSE)
        , m_nvbufMemoryType(0)
        , m_pQueue(NULL) 
        , m_pTiler(NULL)
        , m_pSinkGst(NULL)
        , m_pSrcGst(NULL)
        , m_window(0)
    {
        LOG_FUNC();
        
        m_pBin = gst_bin_new("tiled_display_bin");
        if (!m_pBin) {
            LOG_ERROR("Failed to create new bin for Tiled Display '" << display);
            throw;
        }

        m_pQueue = gst_element_factory_make(NVDS_ELEM_QUEUE, "tiled_display_queue");
        if (!m_pQueue) {
            LOG_ERROR("Failed to create new Queue for Tiled Display '" << display);
            throw;
        }

        m_pTiler = gst_element_factory_make (NVDS_ELEM_TILER, "tiled_display_tiler");
        if (!m_pTiler) {
            LOG_ERROR("Failed to create new Tiler for Display '" << display);
            throw;
        }

        g_object_set(G_OBJECT(m_pTiler), "width", m_width, NULL);
        g_object_set(G_OBJECT(m_pTiler), "height", m_height, NULL);
        g_object_set(G_OBJECT(m_pTiler), "rows", m_rows, NULL);
        g_object_set(G_OBJECT(m_pTiler), "columns", m_columns, NULL);
        g_object_set(G_OBJECT(m_pTiler), "gpu-id", m_gpuId, NULL);
        g_object_set(G_OBJECT(m_pTiler), "nvbuf-memory-type", m_nvbufMemoryType, NULL);
        
        gst_bin_add_many(GST_BIN(m_pBin), m_pQueue, m_pTiler, NULL);

        if (!gst_element_link(m_pQueue, m_pTiler))
        {
            LOG_ERROR("Failed to link Queue to Tiler for Display '" << display <<" '");
            throw;
        }

        m_pSinkGst = gst_element_get_static_pad(m_pQueue, "sink");
        if (!m_pSinkGst)
        {
            LOG_ERROR("Failed to Sink Pad for Display '" << display <<" '");
            throw;
        }
        
        m_pSrcGst = gst_element_get_static_pad(m_pTiler, "src");
        if (!m_pSrcGst)
        {
            LOG_ERROR("Failed to Source Pad for Display '" << display <<" '");
            throw;
        }

        gst_element_add_pad(m_pBin, gst_ghost_pad_new("sink", m_pSinkGst));
        gst_element_add_pad(m_pBin, gst_ghost_pad_new("src", m_pSrcGst));
        
        m_window = XCreateSimpleWindow(m_pGstDisplay, 
            RootWindow(m_pGstDisplay, DefaultScreen(m_pGstDisplay)), 
            0, 0, m_width, m_height, 2, 0x00000000, 0x00000000);            

        if (!m_window)
        {
            LOG_ERROR("Failed to create new X Window for Display '" << display <<" '");
            throw;
        }

        XSetWindowAttributes attr = {0};
        
        attr.event_mask = ButtonPress | KeyRelease;
        XChangeWindowAttributes(m_pGstDisplay, m_window, CWEventMask, &attr);

        Atom wmDeleteMessage = XInternAtom(m_pGstDisplay, "WM_DELETE_WINDOW", False);
        if (wmDeleteMessage != None)
        {
            XSetWMProtocols(m_pGstDisplay, m_window, &wmDeleteMessage, 1);
        }
        XMapRaised(m_pGstDisplay, m_window);
        XSync(m_pGstDisplay, 1);       
        
    };    

    DisplayBintr::~DisplayBintr()
    {
        LOG_FUNC();
    };
    
}