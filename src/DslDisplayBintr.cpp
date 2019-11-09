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
#include "DslPipelineBintr.h"

namespace DSL
{

    DisplayBintr::DisplayBintr(const char* name, guint width, guint height)
        : Bintr(name)
        , m_rows(1)
        , m_columns(1)
        , m_width(width)
        , m_height(height)
        , m_enablePadding(FALSE)
    {
        LOG_FUNC();

        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "tiled_display_queue");
        m_pTiler = DSL_ELEMENT_NEW(NVDS_ELEM_TILER, "tiled_display_tiler");
        
        m_pTiler->SetAttribute("gpu-id", m_gpuId);
        m_pTiler->SetAttribute("nvbuf-memory-type", m_nvbufMemoryType);

        AddChild(m_pQueue);
        AddChild(m_pTiler);

        m_pQueue->AddGhostPadToParent("sink");
        m_pTiler->AddGhostPadToParent("src");
    }

    DisplayBintr::~DisplayBintr()
    {
        LOG_FUNC();

        UnlinkAll();
    }
    
    bool DisplayBintr::LinkAll()
    {
        LOG_FUNC();
        
        m_pQueue->LinkToSink(m_pTiler);
        
        return true;
    }
    
    void DisplayBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        m_pQueue->UnlinkFromSink();
    }
    
    void DisplayBintr::AddToParent(DSL_NODETR_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' display to the Parent Pipeline 
        std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
            AddDisplayBintr(shared_from_this());
    }

    void DisplayBintr::SetTiles(uint rows, uint columns)
    {
        LOG_FUNC();

        m_rows = rows;
        m_columns = columns;
    
        m_pTiler->SetAttribute("rows", m_rows);
        m_pTiler->SetAttribute("columns", m_rows);
    }
    
    void DisplayBintr::GetTiles(uint& rows, uint& columns)
    {
        LOG_FUNC();
        
        rows = m_rows;
        columns = m_columns;
    }
    
    void DisplayBintr::SetDimensions(uint width, uint height)
    {
        LOG_FUNC();
        
        m_width = width;
        m_height = height;

        m_pTiler->SetAttribute("width", m_width);
        m_pTiler->SetAttribute("height", m_height);
    }
    
    void DisplayBintr::GetDimensions(uint& width, uint& height)
    {
        LOG_FUNC();
        
        width = m_width;
        height = m_height;
    }
}