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
#include "DslSegVisualBintr.h"

namespace DSL
{
    SegVisualElementr::SegVisualElementr(const char* name, 
        uint width, uint height)
        : Elementr("nvsegvisual", name)
        , m_width(width)
        , m_height(height)
        , m_batchSize(0)
        , m_gpuId(0)
    {
        LOG_FUNC();
        
        SetAttribute("width", m_width);
        SetAttribute("height", m_height);
        SetAttribute("gpu-id", m_gpuId);
    }

    SegVisualElementr::~SegVisualElementr()
    {
        LOG_FUNC();
    }

    void SegVisualElementr::GetDimensions(uint* width, uint* height)
    {
        LOG_FUNC();
        
        *width = m_width;
        *height = m_height;
    }

    bool SegVisualElementr::SetDimensions(uint width, uint height)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set dimensions for SegVisualElementr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_width = width;
        m_height = height;

        SetAttribute("width", m_width);
        SetAttribute("height", m_height);
        
        return true;
    }

    uint SegVisualElementr::GetBatchSize()
    {
        LOG_FUNC();
        
        return m_batchSize;
    };
    
    bool SegVisualElementr::SetBatchSize(uint batchSize)
    {
        LOG_FUNC();
        LOG_INFO("Setting batch size to '" << batchSize 
            << "' for SegVisualElementr '" << GetName() << "'");
        
        m_batchSize = batchSize;
        
        SetAttribute("batch-size", m_batchSize);
        return true;
    };

    uint SegVisualElementr::GetGpuId()
    {
        LOG_FUNC();

        LOG_DEBUG("Returning a GPU ID of " << m_gpuId <<"' for Bintr '" << GetName() << "'");
        return m_gpuId;
    }

    bool SegVisualElementr::SetGpuId(uint gpuId)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set GPU ID for Bintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }
        m_gpuId = gpuId;
        SetAttribute("gpu-id", m_batchSize);
        
        return true;
    }
    
}