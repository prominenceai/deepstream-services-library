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
    TrackedObject::TrackedObject(uint64_t trackingId, uint64_t frameNumber)
        : m_trackingId(trackingId)
        , m_frameNumber(frameNumber)
        , m_maxHistory(0)
    {
        LOG_FUNC();
        
        timeval creationTime;
        gettimeofday(&creationTime, NULL);
        m_creationTimeMs = creationTime.tv_sec*1000.0 + creationTime.tv_usec/1000.0;
    }
    
    TrackedObject::TrackedObject(uint64_t trackingId, uint64_t frameNumber,
        const NvOSD_RectParams* pRectParams, uint maxHistory)
        : m_trackingId(trackingId)
        , m_frameNumber(frameNumber)
        , m_maxHistory(maxHistory)
    {
        LOG_FUNC();
        
        timeval creationTime;
        gettimeofday(&creationTime, NULL);
        m_creationTimeMs = creationTime.tv_sec*1000.0 + creationTime.tv_usec/1000.0;
        
        PushBbox(pRectParams);
    }
    
    void TrackedObject::PushBbox(const NvOSD_RectParams* pRectParams)
    {
        LOG_FUNC();
        
        if (!m_maxHistory)
        {
            LOG_ERROR("PushBbx called with m_maxHistory set to 0");
        }
        
        std::shared_ptr pBboxCoords = 
            std::shared_ptr<NvBbox_Coords>(new NvBbox_Coords);
        memcpy(&(*pBboxCoords), pRectParams, sizeof(NvBbox_Coords));

        if (m_bboxTrace.size() == m_maxHistory)
        {
           m_bboxTrace.pop_front();
        }
        
        m_bboxTrace.push_back(pBboxCoords);
    }
    
    std::shared_ptr<std::vector<dsl_coordinate>> TrackedObject::GetTrace(uint testPoint)
    {
        // Create the trace - i.e. a vector of pre-sized blank coordinates
        std::shared_ptr<std::vector<dsl_coordinate>> pTraceCoordinates =
            std::shared_ptr<std::vector<dsl_coordinate>>(
                new std::vector<dsl_coordinate>(m_bboxTrace.size()));
            
        uint traceIdx(0);
        
        for (const auto & ideque: m_bboxTrace)
        {
            dsl_coordinate traceCoordinate;
            
            switch (testPoint)
            {
            case DSL_BBOX_POINT_CENTER :
                traceCoordinate.x = round(ideque->left + ideque->width/2);
                traceCoordinate.y = round(ideque->top + ideque->height/2);
                break;
            case DSL_BBOX_POINT_NORTH_WEST :
                traceCoordinate.x = round(ideque->left);
                traceCoordinate.y = round(ideque->top);
                break;
            case DSL_BBOX_POINT_NORTH :
                traceCoordinate.x = round(ideque->left + ideque->width/2);
                traceCoordinate.y = round(ideque->top);
                break;
            case DSL_BBOX_POINT_NORTH_EAST :
                traceCoordinate.x = round(ideque->left + ideque->width);
                traceCoordinate.y = round(ideque->top);
                break;
            case DSL_BBOX_POINT_EAST :
                traceCoordinate.x = round(ideque->left + ideque->width);
                traceCoordinate.y = round(ideque->top + ideque->height/2);
                break;
            case DSL_BBOX_POINT_SOUTH_EAST :
                traceCoordinate.x = round(ideque->left + ideque->width);
                traceCoordinate.y = round(ideque->top + ideque->height);
                break;
            case DSL_BBOX_POINT_SOUTH :
                traceCoordinate.x = round(ideque->left + ideque->width/2);
                traceCoordinate.y = round(ideque->top + ideque->height);
                break;
            case DSL_BBOX_POINT_SOUTH_WEST :
                traceCoordinate.x = round(ideque->left);
                traceCoordinate.y = round(ideque->top + ideque->height);
                break;
            case DSL_BBOX_POINT_WEST :
                traceCoordinate.x = round(ideque->left);
                traceCoordinate.y = round(ideque->top + ideque->height/2);
                break;
            default:
                LOG_ERROR("Invalid DSL_BBOX_POINT = '" << testPoint 
                    << "' for Tracked Object ");
                throw;
            }          
            // copy the calculated coordinates into the pre-sized vector 
            // at the current index of the bbox history.
            pTraceCoordinates->at(traceIdx++) = traceCoordinate;

            LOG_INFO("Trace point x=" << traceCoordinate.x 
                << ",y=" << traceCoordinate.x);
        }
        return pTraceCoordinates;
    }
}    