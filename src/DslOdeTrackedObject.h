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

namespace DSL
{
    struct TrackedObject
    {

        /**
         * @brief Ctor 1 of 2 for the TrackedObject struct
         * @param[in] unique trackingId for the tracked object
         * @param frameNumber the object was first detected
         */
        TrackedObject(uint64_t trackingId, uint64_t frameNumber);
        
        /**
         * @brief Ctor 2 of 2 for the TrackedObject struct
         * @param[in] unique trackingId for the tracked object
         * @param frameNumber the object was first detected
         * @param rectParams rectParms from the object's meta when first detected
         * @param maxHistory maximum number of bbox coordinates to track
         */
        TrackedObject(uint64_t trackingId, uint64_t frameNumber,
            const NvOSD_RectParams* rectParams, uint maxHistory);
        
        /**
         * @brief function to push a new set of positional bbox coordinates
         * on to the tracked object's m_bboxTrace queue
         * @param rectParams new rectangle params to push
         */
        void PushBbox(const NvOSD_RectParams* rectParams);
        
        /**
         * @brief returns a vector of coordinates defining the TrackedObject's
         * trace for a specfic test-point on the object's bounding box
         * @param testPoint test-point to generate the trace with.
         * @return shared pointer to a vector of coordinates.
         */
        std::shared_ptr<std::vector<dsl_coordinate>> 
            GetTrace(uint testPoint);
        
        /**
         * @brief unique tracking id for the tracked object.
         */
        uint64_t m_trackingId;
        
        /**
         * @brief frame number for the tracked object, updated on detection within a new frame
         */
        uint64_t m_frameNumber;
        
        /**
         * @brief time of creation for this Tracked Object, used to test for object persistence
         */
        double m_creationTimeMs;
        
        /**
         * @brief maximum number of bbox coordinates to maintain/trace
         */
        uint m_maxHistory;
        
        /**
         * @brief a fixed size queue of Rectangle Params 
         */
        std::deque<std::shared_ptr<NvBbox_Coords>> m_bboxTrace;
    };

    /**
     * @brief map of tracked objects - unique Tracking Id as key
     */
    typedef std::map <uint64_t, std::shared_ptr<TrackedObject>> TrackedObjects;

}

#endif // _DSL_ODE_TRACKED_OBJECT_H
