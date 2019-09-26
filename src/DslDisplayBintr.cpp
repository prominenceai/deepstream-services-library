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

#include "Dsl.h"
#include "DslDisplayBintr.h"
#include "DslPipeline.h"

namespace DSL
{

    DisplayBintr::DisplayBintr(const char* display, Display* pXDisplay,
        guint rows, guint columns, guint width, guint height)
        : Bintr(display)
        , m_pXDisplay(pXDisplay)
        , m_rows(rows)
        , m_columns(columns)
        , m_width(width)
        , m_height(height)
        , m_enablePadding(FALSE)
        , m_pQueue(NULL) 
        , m_pTiler(NULL)
        , m_window(0)
    {
        LOG_FUNC();

        // Queue and Tiler elements will be linked in the order created.
        m_pQueue = MakeElement(NVDS_ELEM_QUEUE, "tiled_display_queue", LINK_TRUE);
        m_pTiler = MakeElement(NVDS_ELEM_TILER, "tiled_display_tiler", LINK_TRUE);

        g_object_set(G_OBJECT(m_pTiler), "width", m_width, NULL);
        g_object_set(G_OBJECT(m_pTiler), "height", m_height, NULL);
        g_object_set(G_OBJECT(m_pTiler), "rows", m_rows, NULL);
        g_object_set(G_OBJECT(m_pTiler), "columns", m_columns, NULL);
        g_object_set(G_OBJECT(m_pTiler), "gpu-id", m_gpuId, NULL);
        g_object_set(G_OBJECT(m_pTiler), "nvbuf-memory-type", m_nvbufMemoryType, NULL);

        // Add Sink and Source pads for Queue and Tiler
        AddGhostPads();
        
        m_window = XCreateSimpleWindow(m_pXDisplay, 
            RootWindow(m_pXDisplay, DefaultScreen(m_pXDisplay)), 
            0, 0, m_width, m_height, 2, 0x00000000, 0x00000000);            

        if (!m_window)
        {
            LOG_ERROR("Failed to create new X Window for Display '" << display <<" '");
            throw;
        }

        XSetWindowAttributes attr = {0};
        
        attr.event_mask = ButtonPress | KeyRelease;
        XChangeWindowAttributes(m_pXDisplay, m_window, CWEventMask, &attr);

        Atom wmDeleteMessage = XInternAtom(m_pXDisplay, "WM_DELETE_WINDOW", False);
        if (wmDeleteMessage != None)
        {
            XSetWMProtocols(m_pXDisplay, m_window, &wmDeleteMessage, 1);
        }
        XMapRaised(m_pXDisplay, m_window);
    }    

    DisplayBintr::~DisplayBintr()
    {
        LOG_FUNC();
    }
    
    void DisplayBintr::AddToPipeline(std::shared_ptr<Pipeline> pPipeline)
    {
        LOG_FUNC();
        
        // add 'this' display to the Parent Pipeline 
        pPipeline->AddDisplayBintr(shared_from_this());
    }
    
}