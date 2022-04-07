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
    template <typename T, int MaxLen, typename Container=std::deque<T>>
    class FixedQueue : public std::queue<T, Container> {
    public:
        void push(const T& value) {
            if (this->size() == MaxLen) {
               this->c.pop_front();
            }
            std::queue<T, Container>::push(value);
        }
    };
    

    struct TrackedObject
    {

        /**
         * @brief Ctor for the TrackedObject struct
         * @param[in] unique trackingId for the tracked object
         * @param frameNumber the object was first detected
         * @param rectParams rectParms from the object's meta when first detected
         */
        TrackedObject(uint64_t trackingId, uint64_t frameNumber,
            const NvOSD_RectParams& rectParams);
        
        /**
         * @brief function to push a new set of positional rectParms
         * on to the tracked object's m_rectangleTrace queue
         * @param rectParams new rectangle params to push
         */
        void PushRectangle(const NvOSD_RectParams& rectParams);
        
        /**
         * @brief unique id for the tracked object
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
         * @brief a fixed size queue of Rectangle Params 
         */
        FixedQueue<std::shared_ptr<NvOSD_RectParams>, 10> m_rectangleTrace;
    };

    /**
     * @brief map of tracked objects - unique Tracking Id as key
     */
    typedef std::map <uint64_t, std::shared_ptr<TrackedObject>> TrackedObjects;

}

#endif // _DSL_ODE_TRACKED_OBJECT_H
