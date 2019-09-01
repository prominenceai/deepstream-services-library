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

#include "Dss.h"
#include "DssPipeline.h"

#include <gst/gst.h>

namespace DSS
{
 
    Pipeline::Pipeline()
        : m_pPipeline(NULL)
    {
        LOG_FUNC();
    }

    Pipeline::~Pipeline()
    {
        LOG_FUNC();
    }
    
    bool Pipeline::Pause()
    {
        LOG_FUNC();
        
        return (gst_element_set_state(m_pPipeline, 
            GST_STATE_PAUSED) != GST_STATE_CHANGE_FAILURE);
    }

    bool Pipeline::Play()
    {
        LOG_FUNC();
        
        return (gst_element_set_state(m_pPipeline, 
            GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE);
    }
    

} // DSS