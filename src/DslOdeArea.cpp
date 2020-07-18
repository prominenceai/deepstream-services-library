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
#include "DslOdeArea.h"

namespace DSL
{

    OdeArea::OdeArea(const char* name, DSL_RGBA_RECTANGLE_PTR pRectangle, bool display)
        : Base(name)
        , m_pRectangle(pRectangle)
        , m_display(display)
    {
        LOG_FUNC();
        
    }
    
    OdeArea::~OdeArea()
    {
        LOG_FUNC();
    }
        
    void OdeArea::OverlayFrame(GstBuffer* pBuffer, NvDsFrameMeta* pFrameMeta)
    {
        LOG_FUNC();
        
        if (!m_display)
        {
            return;
        }
        
        // If this is the first time seeing a frame for the reported Source Id.
        if (m_frameNumPerSource.find(pFrameMeta->source_id) == m_frameNumPerSource.end())
        {
            // Initial the frame number for the new source
            m_frameNumPerSource[pFrameMeta->source_id] = 0;
        }
        
        // If the last frame number for the reported source is less than the current frame
        if (m_frameNumPerSource[pFrameMeta->source_id] < pFrameMeta->frame_num)
        {
            // Update the frame number so we only add the rectangle once
            m_frameNumPerSource[pFrameMeta->source_id] = pFrameMeta->frame_num;
            
            m_pRectangle->OverlayFrame(pBuffer, pFrameMeta);
            
        }
    }
}