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

#include "Dsl.h"
#include "DslOdeTrackedObject.h"

namespace DSL
{
    TrackedObject::TrackedObject(uint64_t trackingId, uint64_t frameNumber,
        const NvOSD_RectParams& rectParams)
        : m_trackingId(trackingId)
        , m_frameNumber(frameNumber)
    {
        LOG_FUNC();
        
        timeval creationTime;
        gettimeofday(&creationTime, NULL);
        m_creationTimeMs = creationTime.tv_sec*1000.0 + creationTime.tv_usec/1000.0;
        
        std::shared_ptr pRectParams = std::shared_ptr<NvOSD_RectParams>(new NvOSD_RectParams);
        *pRectParams = rectParams;
        
        m_rectangleTrace.push(pRectParams);
    }
    
    void TrackedObject::PushRectangle(const NvOSD_RectParams& rectParams)
    {
        LOG_FUNC();
        
        std::shared_ptr pRectParams = std::shared_ptr<NvOSD_RectParams>(new NvOSD_RectParams);
        *pRectParams = rectParams;
        
        m_rectangleTrace.push(pRectParams);
    }
}    