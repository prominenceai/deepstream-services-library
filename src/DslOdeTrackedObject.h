/*
The MIT License

Copyright (c) 2022, Prominence AI, Inc.

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

#ifndef _DSL_ODE_TRACKED_OBJECT_H
#define _DSL_ODE_TRACKED_OBJECT_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslOdeBase.h"
#include "DslDisplayTypes.h"

namespace DSL
{
    /**
     * @class TrackedObject
     * @file DslOdeTrackedObject.h
     * @brief Implements a Tracked Object with a history of bbox coordinates.
     */
    class TrackedObject
    {
    public:

        /**
         * @brief Ctor for the TrackedObject class
         * @param[in] unique trackingId for the tracked object
         * @param[in] frameNumber the object was first detected
         * @param[in] pCoordinates bounding box coordinates from the object's meta 
         * when first detected
         * @param[in] pColor shared pointer to an RGBA Color Type to
         * set a unique color for the tracked object. 
         * @param[in] maxHistory maximum number of bbox coordinates to track
         */
        TrackedObject(uint64_t trackingId, uint64_t frameNumber,
            const NvBbox_Coords* pCoordinates, DSL_RGBA_COLOR_PTR pColor, 
            uint maxHistory);
            
        /**
         * @brief Sets the max history for this tracked object
         * @param maxHistory new max history setting.
         */
        void SetMaxHistory(uint maxHistory);
        
        /**
         * @brief function to update the tracked-object's last frame number and 
         * push a new set of positional bbox coordinates on to the tracked 
         * object's m_bboxTrace queue.
         * @param[in] currentFrameNumber new frame number to save
         * @param[in] pCoordinates new bounding box coordinates to push.
         */
        void Update(uint64_t currentFrameNumber, const NvBbox_Coords* pCoordinates);
        
        /**
         * @brief calculates the duration of time the object has been tracked.
         * @return the duration in units of ms
         */
        double GetDurationMs();
        
        /**
         * @brief Gets the current size of the bounding box trace.
         * @return current size of the bbox trace.
         */
        size_t BboxTraceSize(){return m_pBboxTrace->size();};
        
        /**
         * @brief Gets the coordinates for a specific test-point for the 
         * first bounding box in the TrackedObject's history.
         * @param[in] testPoint to generate the coordinates with
         * @return first coordinates 
         */
        dsl_coordinate GetFirstCoordinate(uint testPoint);
        
        /**
         * @brief Gets the coordinates for a specific test-point for the 
         * last bounding box in the TrackedObject's history.
         * @param[in] testPoint to generate the coordinates with
         * @return last coordinates
         */
        dsl_coordinate GetLastCoordinate(uint testPoint);
        
        /**
         * @brief Returns a vector of coordinates defining the TrackedObject's
         * trace for a specfic test-point on the object's bounding box
         * @param[in] testPoint test-point to generate the trace with.
         * @param[in] method one of the DSL_OBJECT_TRACE_TEST_METHOD_* constants
         * @param[in] lineWidth the width value to assign to the line.
         * @return shared pointer to a vector of coordinates.
         */
        DSL_RGBA_MULTI_LINE_PTR GetTrace(uint testPoint, uint method, 
            uint lineWidth);
            
        /**
         * @brief used to query if the tracked object has a previous Trace
         * from a previous line cross event.
         */
        bool HasPreviousTrace(){return m_pPrevBboxTrace != nullptr;};

        /**
         * @brief Returns a vector of coordinates defining the TrackedObject's
         * previous trace for a specfic test-point on the object's bounding box.
         * @param[in] testPoint test-point to generate the trace with.
         * @param[in] method one of the DSL_OBJECT_TRACE_TEST_METHOD_* constants
         * @param[in] lineWidth the width value to assign to the line.
         * @return shared pointer to a vector of coordinates.
         */
        DSL_RGBA_MULTI_LINE_PTR GetPreviousTrace(uint testPoint, uint method, 
            uint lineWidth);
            
        /**
         * @brief Handles an ODE Occurrence for this tracked object. The current
         * m_pBboxTrace is moved to the m_pPreviousBboxTrace and a new/empty
         * m_pBboxTrace is created.
         */
        void HandleOccurrence();

        /**
         * @brief unique tracking id for the tracked object.
         */
        uint64_t trackingId;
        
        /**
         * @brief frame number for the tracked object, updated on detection 
         * within a new frame.
         */
        uint64_t frameNumber;
        
        /**
         * @brief total number of consecutive tracked frames
         */
        uint frameCount;
        
        /**
         * @brief number of tracked frames prior to event
         */
        uint preEventFrameCount;

        /**
         * @brief number of tracked events
         */
        uint onEventFrameCount;
    
    private:

        /**
         * @brief Get an x,y coordinate from a Bbox based on this Trigger's
         * client specified test-point
         * @param[in] pBbox to optain the coordinate from
         * @param[in] testPoint one of the DSL_BBOX_POINT_* constants
         * @param[out] traceCoordinate x,y coordinate value.
         */
        void getCoordinate(std::shared_ptr<NvBbox_Coords> pBbox, 
            uint testPoint, dsl_coordinate& traceCoordinate);
        
        /**
         * @brief time of creation for this Tracked Object, used to test 
         * for object persistence.
         */
        double m_creationTimeMs;
        
        /**
         * @brief maximum number of bbox coordinates to maintain/trace.
         */
        uint m_maxHistory;
        
        /**
         * @brief a max sized queue of Rectangle Params.
         */
        std::shared_ptr<std::deque<std::shared_ptr<NvBbox_Coords>>> m_pBboxTrace;
        
        /**
         * @brief a max sized queue of Rectangle Params.
         */
        std::shared_ptr<std::deque<std::shared_ptr<NvBbox_Coords>>> m_pPrevBboxTrace;
        
        /**
         * @brief used to identify the tracked object with an RGBA color.
         */
        DSL_RGBA_COLOR_PTR m_pColor;
        
    };
    
    //*******************************************************************************

    /**
     * @class TrackedObjects
     * @file DslOdeTrackedObject.h
     * @brief Manages a map of tracked objects.
     */
    class TrackedObjects
    {
    public:
        /**
         * @brief Ctor TrackedObjects class
         * @param[in] maxHistory maximum number of bbox coordinates to track
         */
        TrackedObjects(uint maxHistory);
        
        /**
         * @brief determines if an object is currently tracked for a given source.
         * @param[in] sourceId source to filter on
         * @param[in] trackingId unique tracking id of the object to query
         * @return true if the object is currenty tracked, false otherwise.
         */ 
        bool IsTracked(uint sourceId, uint64_t trackingId);
        
        /**
         * @brief adds an untracked object to the collection of tracked objects
         * per source.
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - 
         * the frame that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @param[in] pColor RGBA color to assign to the new Tracked object.
         * @return a shared pointer to the newly tracked object, null_ptr otherwise.
         */
        std::shared_ptr<TrackedObject> Track(NvDsFrameMeta* pFrameMeta, 
            NvDsObjectMeta* pObjectMeta, DSL_RGBA_COLOR_PTR pColor);
        
        /**
         * @brief Gets the Tracked Object for a specified source and tranking Id
         * @param[in] sourceId source to filter on
         * @param[in] trackingId unique tracking id of the object to query
         * @return a shared pointer if found, null_ptr otherwise.
         */
        std::shared_ptr<TrackedObject> GetObject(uint sourceId, 
            uint64_t trackingId);
        
        /**
         * @brief Deletes a Tracked Object by source Id and tracking Id
         * @param[in] sourceId source to filter on
         * @param[in] trackingId unique tracking id of the object to query
         */
        void DeleteObject(uint sourceId, uint64_t trackingId);
        
        /**
         * @brief Purges all tracked objects that are not in the current frame
         * @param currentFrameNumber current frame number to use as a purge filter.
         */
        void Purge(uint64_t currentFrameNumber);
        
        /**
         * @brief Clears/deletes all tracked objects
         */
        void Clear();
        
        /**
         * @brief Query to determine if the container is empty
         * @return true if empty of tracked objects, false otherwise
         */
        bool IsEmpty(){return m_trackedObjectsPerSource.empty();};
        
        /**
         * @brief gets the time of tracked object creation
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - 
         * the frame that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta
         * @return Creation date if found, 0 otherwise.
         */
        double GetCreationTime(NvDsFrameMeta* pFrameMeta, 
            NvDsObjectMeta* pObjectMeta);
        
        /**
         * @brief Sets the max history for all objects tracked
         * @param maxHistory new max history setting.
         */
        void SetMaxHistory(uint maxHistory);
        
    private:
    
        /**
         * @brief maximum number of bbox coordinates to maintain/trace
         */
        uint m_maxHistory;
        
        /**
         * @brief map of tracked objects - Key = unique Tracking Id
         */
        typedef std::map <uint64_t, std::shared_ptr<TrackedObject>> TrackedObjectsT;

        /**
         * @brief map of tracked objects per source - Key = source Id
         */
        std::map <uint, std::shared_ptr<TrackedObjectsT>> m_trackedObjectsPerSource;
    };    
}

#endif // _DSL_ODE_TRACKED_OBJECT_H
