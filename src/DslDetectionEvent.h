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

#ifndef _DSL_DETECTION_EVENT_H
#define _DSL_DETECTION_EVENT_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslEvent.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_EVENT_DETECTION_PTR std::shared_ptr<DetectionEvent>

    #define DSL_EVENT_FIRST_OCCURRENCE_PTR std::shared_ptr<FirstOccurrenceEvent>
    #define DSL_EVENT_FIRST_OCCURRENCE_NEW(name, classId) \
        std::shared_ptr<FirstOccurrenceEvent>(new FirstOccurrenceEvent(name, classId))
        
    class DetectionEvent : public EventBase
    {
    public: 
    
        DetectionEvent(const char* name, uint classId);

        ~DetectionEvent();
        
        /**
         * @brief Gets the ClassId filter used for Object detection 
         * @return 
         */
        uint GetClassId();
        
        /**
         * @brief Sets the ClassId filter for Object detection 
         * @return 
         */
        void SetClassId(uint classId);
        
        /**
         * @brief Gets the current Minimum rectangle width and height to trigger the event
         * a value of 0 means no minimum
         * @param[out] minWidth current minimum width value in use
         * @param[out] minHeight current minimum height value in use
         */
        void GetMinDimensions(uint* minWidth, uint* minHeight);

        /**
         * @brief Sets new Minimum rectangle width and height to trigger the event
         * a value of 0 means no minimum
         * @param[in] minWidth current minimum width value in use
         * @param[in] minHeight current minimum height value in use
         */
        void SetMinDimensions(uint minWidth, uint minHeight);
        
        /**
         * @brief Gets the current Minimum frame count to trigger the event (n of d frames)
         * @param[out] minFrameCountN frame count numeratior  
         * @param[out] minFrameCountD frame count denominator
         */
        void GetMinFrameCount(uint* minFrameCountN, uint* minFrameCountD);
        
        /**
         * @brief Sets the current Minimum frame count to trigger an event (n of d frames)
         * @param[out] minFrameCountN frame count numeratior  
         * @param[out] minFrameCountD frame count denominator
         */
        void SetMinFrameCount(uint minFrameCountN, uint minFrameCountD);

    private:

        /**
         * @brief GIE Class Id filter for this event
         */
        uint m_classId;
        
        /**
         * @brief Minimum rectangle width to trigger event
         */
        uint m_minWidth;

        /**
         * @brief Minimum rectangle height to trigger event
         */
        uint m_minHeight;

        /**
         * @brief Minimum frame count numerator to trigger event
         */
        uint m_minFrameCountN;

        /**
         * @brief Minimum frame count denominator to trigger event
         */
        uint m_minFrameCountD;

    };
    
    class FirstOccurrenceEvent : public DetectionEvent
    {
    public:
    
        FirstOccurrenceEvent(const char* name, uint classId);
        
        ~FirstOccurrenceEvent();
        
    private:
    
    };

    class FirstAbsenceEvent : public DetectionEvent
    {
    public:
    
        FirstAbsenceEvent(const char* name, uint classId);
        
        ~FirstAbsenceEvent();
        
    private:
    
    };
}


#endif // _DSL_EVENT_H
