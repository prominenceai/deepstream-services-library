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
        const NvOSD_RectParams* pRectParams, uint maxHistory)
        : m_trackingId(trackingId)
        , m_maxHistory(maxHistory)
        , m_triggered(false)
    {
        // No function log - avoid overhead.
        
        timeval creationTime;
        gettimeofday(&creationTime, NULL);
        m_creationTimeMs = creationTime.tv_sec*1000.0 + creationTime.tv_usec/1000.0;
        
        Update(frameNumber, pRectParams);
    }
    
    void TrackedObject::SetMaxHistory(uint maxHistory)
    {
        LOG_FUNC();
        
        m_maxHistory = maxHistory;
    }
    
    void TrackedObject::Update(uint64_t currentFrameNumber, 
        const NvOSD_RectParams* pRectParams)
    {
        // No function log - avoid overhead.
        
        // update the tracked object's frame number - the filter used for purging.
        m_frameNumber = currentFrameNumber;
        
        // Copy only the rectangle coordinates of the Object's RectParams
        std::shared_ptr pBboxCoords = 
            std::shared_ptr<NvBbox_Coords>(new NvBbox_Coords);
        memcpy(&(*pBboxCoords), pRectParams, sizeof(NvBbox_Coords));

        // If maintaining bbox trace-point history
        if (m_maxHistory)
        {
            // Maintain max queue depth
            while (m_bboxTrace.size() >= m_maxHistory)
            {
               m_bboxTrace.pop_front();
            }
            
            m_bboxTrace.push_back(pBboxCoords);
        }
    }

    double TrackedObject::GetDurationMs()
    {
        timeval currentTime;
        gettimeofday(&currentTime, NULL);
        
        return (currentTime.tv_sec*1000.0 + currentTime.tv_usec/1000.0) -
            m_creationTimeMs;
    }

    
    std::shared_ptr<std::vector<dsl_coordinate>> TrackedObject::GetTrace(
        uint testPoint, uint method)
    {
        // No function log - avoid overhead.
        
        if (method == DSL_OBJECT_TRACE_TEST_METHOD_END_POINTS)
        {
            // Create the trace - i.e. a vector of pre-sized blank coordinates
            std::shared_ptr<std::vector<dsl_coordinate>> pTraceCoordinates =
                std::shared_ptr<std::vector<dsl_coordinate>>(
                    new std::vector<dsl_coordinate>(2));
            
            dsl_coordinate traceCoordinate{0};
            getCoordinate(m_bboxTrace.front(), testPoint, traceCoordinate);
            pTraceCoordinates->at(0) = traceCoordinate;
            
            getCoordinate(m_bboxTrace.back(), testPoint, traceCoordinate);
            pTraceCoordinates->at(1) = traceCoordinate;
            
            return pTraceCoordinates;
        }

        // Create the trace - i.e. a vector of pre-sized blank coordinates
        std::shared_ptr<std::vector<dsl_coordinate>> pTraceCoordinates =
            std::shared_ptr<std::vector<dsl_coordinate>>(
                new std::vector<dsl_coordinate>(m_bboxTrace.size()));
            
        uint traceIdx(0);
        dsl_coordinate traceCoordinate{0};
        
        for (const auto& ideque: m_bboxTrace)
        {
            getCoordinate(ideque, testPoint, traceCoordinate);
            // copy the calculated coordinates into the pre-sized vector 
            // at the current index of the bbox history.
            pTraceCoordinates->at(traceIdx++) = traceCoordinate;

            LOG_DEBUG("Trace point x=" << traceCoordinate.x 
                << ",y=" << traceCoordinate.y);
        }
        
        return pTraceCoordinates;
    }

    void TrackedObject::getCoordinate(std::shared_ptr<NvBbox_Coords> pBbox, 
        uint testPoint, dsl_coordinate& traceCoordinate)
    {
        switch (testPoint)
        {
        case DSL_BBOX_POINT_CENTER :
            traceCoordinate.x = round(pBbox->left + pBbox->width/2);
            traceCoordinate.y = round(pBbox->top + pBbox->height/2);
            break;
        case DSL_BBOX_POINT_NORTH_WEST :
            traceCoordinate.x = round(pBbox->left);
            traceCoordinate.y = round(pBbox->top);
            break;
        case DSL_BBOX_POINT_NORTH :
            traceCoordinate.x = round(pBbox->left + pBbox->width/2);
            traceCoordinate.y = round(pBbox->top);
            break;
        case DSL_BBOX_POINT_NORTH_EAST :
            traceCoordinate.x = round(pBbox->left + pBbox->width);
            traceCoordinate.y = round(pBbox->top);
            break;
        case DSL_BBOX_POINT_EAST :
            traceCoordinate.x = round(pBbox->left + pBbox->width);
            traceCoordinate.y = round(pBbox->top + pBbox->height/2);
            break;
        case DSL_BBOX_POINT_SOUTH_EAST :
            traceCoordinate.x = round(pBbox->left + pBbox->width);
            traceCoordinate.y = round(pBbox->top + pBbox->height);
            break;
        case DSL_BBOX_POINT_SOUTH :
            traceCoordinate.x = round(pBbox->left + pBbox->width/2);
            traceCoordinate.y = round(pBbox->top + pBbox->height);
            break;
        case DSL_BBOX_POINT_SOUTH_WEST :
            traceCoordinate.x = round(pBbox->left);
            traceCoordinate.y = round(pBbox->top + pBbox->height);
            break;
        case DSL_BBOX_POINT_WEST :
            traceCoordinate.x = round(pBbox->left);
            traceCoordinate.y = round(pBbox->top + pBbox->height/2);
            break;
        default:
            LOG_ERROR("Invalid DSL_BBOX_POINT = '" << testPoint 
                << "' for Tracked Object ");
            throw;
        }          
    }
    
    //********************************************************************************
    
    TrackedObjects::TrackedObjects(uint maxHistory)
        : m_maxHistory(maxHistory)
    {
        LOG_FUNC();
    }
    
    bool TrackedObjects::IsTracked(uint sourceId, uint64_t trackingId)
    {
        // No function log - avoid overhead.

        // If the sourceId does not exist, then not tracked.
        if (m_trackedObjectsPerSource.find(sourceId) == 
            m_trackedObjectsPerSource.end())
        {
            return false;
        }
        std::shared_ptr<TrackedObjectsT> pTrackedObjects = 
            m_trackedObjectsPerSource[sourceId];
            
        // else, if this is the first occurrence of a specific object for this source
        if (pTrackedObjects->find(trackingId) == pTrackedObjects->end())
        {
            return false;
        }
        
        // else, the object is currently being tracked.
        return true;
    }
    
    std::shared_ptr<TrackedObject> TrackedObjects::GetObject(
        uint sourceId, uint64_t trackingId)
    {
        // No function log - avoid overhead.

        // If the sourceId does not exist, then not tracked.
        if (m_trackedObjectsPerSource.find(sourceId) == 
            m_trackedObjectsPerSource.end())
        {
            return nullptr;
        }
        std::shared_ptr<TrackedObjectsT> pTrackedObjects = 
            m_trackedObjectsPerSource[sourceId];
            
        // else, if this is the first occurrence of a specific object for this source
        if (pTrackedObjects->find(trackingId) == pTrackedObjects->end())
        {
            return nullptr;
        }
        
        // else, the object is currently being tracked.
        return pTrackedObjects->at(trackingId);
    }
    
    bool TrackedObjects::Track(NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // No function log - avoid overhead.

        // if this is the first occurrence of any object for this source
        if (m_trackedObjectsPerSource.find(pFrameMeta->source_id) == 
            m_trackedObjectsPerSource.end())
        {
            LOG_DEBUG("First object detected with id = " << pObjectMeta->object_id 
                << " for source = " << pFrameMeta->source_id);
            
            // create a new tracked object for this tracking Id and source
            std::shared_ptr<TrackedObject> pTrackedObject = 
                std::shared_ptr<TrackedObject>(new TrackedObject(
                    pObjectMeta->object_id, pFrameMeta->frame_num, 
                    &pObjectMeta->rect_params, m_maxHistory));
                
            // create a map of tracked objects for this source    
            std::shared_ptr<TrackedObjectsT> pTrackedObjects = 
                std::shared_ptr<TrackedObjectsT>(new TrackedObjectsT());
                
            // insert the new tracked object into the new map    
            pTrackedObjects->insert(std::pair<uint64_t, 
                std::shared_ptr<TrackedObject>>(pObjectMeta->object_id, 
                    pTrackedObject));
                
            // add the map of tracked objects for this source to the map of 
            // all tracked objects.
            m_trackedObjectsPerSource[pFrameMeta->source_id] = pTrackedObjects;
            
            return true;
        }
        std::shared_ptr<TrackedObjectsT> pTrackedObjects = 
            m_trackedObjectsPerSource[pFrameMeta->source_id];
            
        // else, if this is the first occurrence of a specific object for this source
        if (pTrackedObjects->find(pObjectMeta->object_id) == pTrackedObjects->end())
        {
            LOG_DEBUG("New object detected with id = " << pObjectMeta->object_id 
                << " for source = " << pFrameMeta->source_id);
            
            // create a new tracked object for this tracking Id and source
            std::shared_ptr<TrackedObject> pTrackedObject = 
                std::shared_ptr<TrackedObject>(new TrackedObject(
                    pObjectMeta->object_id, pFrameMeta->frame_num,
                    &pObjectMeta->rect_params, m_maxHistory));

            // insert the new tracked object into the new map    
            pTrackedObjects->insert(std::pair<uint64_t, 
                std::shared_ptr<TrackedObject>>(pObjectMeta->object_id, 
                    pTrackedObject));
            
            return true;
        }
        
        LOG_ERROR("Object with id = " << pObjectMeta->object_id 
            << " for source = " << pFrameMeta->source_id 
            << " is already being tracked");
        return false;
    }

    void TrackedObjects::DeleteObject(uint sourceId, uint64_t trackingId)
    {
        // If the sourceId does not exist, then not tracked.
        if (m_trackedObjectsPerSource.find(sourceId) == 
            m_trackedObjectsPerSource.end())
        {
            LOG_ERROR("Source = " << sourceId 
                << " is not being tracked");
            return;
        }
        std::shared_ptr<TrackedObjectsT> pTrackedObjects = 
            m_trackedObjectsPerSource[sourceId];
            
        // else, if this is the first occurrence of a specific object for this source
        if (pTrackedObjects->find(trackingId) == pTrackedObjects->end())
        {
            LOG_ERROR("Object with id = " << trackingId 
                << " for source = " << sourceId 
                << " is not being tracked");
            return;
        }
        
        LOG_WARN("size = " << pTrackedObjects->size());
        // else, the object is currently being tracked.
        pTrackedObjects->erase(trackingId);
        LOG_WARN("size = " << pTrackedObjects->size());
    }    

    void TrackedObjects::Purge(uint64_t currentFrameNumber)
    {
        // No function log - avoid overhead.

        for (const auto &trackedObjects: m_trackedObjectsPerSource)
        {
            std::shared_ptr<TrackedObjectsT> pTrackedObjects = 
                trackedObjects.second;

            auto trackedObject = pTrackedObjects->cbegin();
            while (trackedObject != pTrackedObjects->cend())
            {
                if (trackedObject->second->GetFrameNumber() != currentFrameNumber)
                {
                    LOG_DEBUG("Purging tracked object with id = " 
                        << trackedObject->first << " for source = " 
                        << trackedObjects.first);
                        
                    // use the return value to update the iterator, as erase invalidates it
                    trackedObject = pTrackedObjects->erase(trackedObject);
                }
                else {
                    trackedObject++;
                }            
            }
        }
    }
    
    void TrackedObjects::Clear()
    {
        m_trackedObjectsPerSource.clear();
    }
    
    double TrackedObjects::GetCreationTime(NvDsFrameMeta* pFrameMeta, 
        NvDsObjectMeta* pObjectMeta)
    {
        // No function log - avoid overhead.
        
        if (!IsTracked(pFrameMeta->source_id, pObjectMeta->object_id))
        {
            LOG_ERROR("Object with id = " << pObjectMeta->object_id 
                << " for source = " << pFrameMeta->source_id 
                << " is NOT being tracked");
            return 0;
        }
        std::shared_ptr<TrackedObjectsT> pTrackedObjects = 
            m_trackedObjectsPerSource[pFrameMeta->source_id];
            
        return pTrackedObjects->at(pObjectMeta->object_id)->GetDurationMs();
    }

    void TrackedObjects::SetMaxHistory(uint maxHistory)
    {
        LOG_FUNC();
        
        for (const auto &trackedObjects: m_trackedObjectsPerSource)
        {
            std::shared_ptr<TrackedObjectsT> pTrackedObjects = 
                trackedObjects.second;
                
            for (const auto &trackedObject: *pTrackedObjects)
            {
                trackedObject.second->SetMaxHistory(maxHistory);
            }
        }
    }
    
}    