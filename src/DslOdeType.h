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

#ifndef _DSL_ODE_TYPE_H
#define _DSL_ODE_TYPE_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslBase.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_ODE_TYPE_PTR std::shared_ptr<OdeType>

    #define DSL_ODE_FIRST_OCCURRENCE_PTR std::shared_ptr<FirstOccurrenceEvent>
    #define DSL_ODE_FIRST_OCCURRENCE_NEW(name, classId) \
        std::shared_ptr<FirstOccurrenceEvent>(new FirstOccurrenceEvent(name, classId))
        
    #define DSL_ODE_FIRST_ABSENCE_PTR std::shared_ptr<FirstAbsenceEvent>
    #define DSL_ODE_FIRST_ABSENCE_NEW(name, classId) \
        std::shared_ptr<FirstAbsenceEvent>(new FirstAbsenceEvent(name, classId))
        
    #define DSL_ODE_EVERY_OCCURRENCE_PTR std::shared_ptr<EveryOccurrenceEvent>
    #define DSL_ODE_EVERY_OCCURRENCE_NEW(name, classId) \
        std::shared_ptr<EveryOccurrenceEvent>(new EveryOccurrenceEvent(name, classId))
        
    #define DSL_ODE_EVERY_ABSENCE_PTR std::shared_ptr<EveryAbsenceEvent>
    #define DSL_ODE_EVERY_ABSENCE_NEW(name, classId) \
        std::shared_ptr<EveryAbsenceEvent>(new EveryAbsenceEvent(name, classId))
        
    class OdeType : public Base
    {
    public: 
    
        OdeType(const char* name, uint eventType, uint classId, uint64_t limit);

        ~OdeType();

        /**
         * @brief total count of all events
         */
        static uint64_t s_eventCount;
        
        /**
         * @brief Function to check a given Object Meta data structure for the occurence of an event
         * and to invoke all Event Actions owned by the event
         * @param[in] pFrameMeta pointer to the containing NvDsFrameMeta data
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        virtual bool CheckForOccurrence(GstBuffer* pBuffer, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta) = 0;
        
        /**
         * @brief Function called to process all Occurrence/Absence data for the current frame
         * @param[in] pFrameMeta pointer to NvDsFrameMeta data for post processing
         * @return true if Occurrence on post process, false otherwise
         */
        virtual bool PostProcessFrame(GstBuffer* pBuffer,
            NvDsFrameMeta* pFrameMeta){return false;};
        
        /**
         * @brief Gets the ClassId filter used for Object detection 
         * @return the current ClassId filter value
         */
        uint GetClassId();
        
        /**
         * @brief Sets the ClassId filter for Object detection 
         * @param[in] classId new filter value to use
         */
        void SetClassId(uint classId);
        
        /**
         * @brief Get the SourceId filter used for Object detection
         * A value of 0 indicates no filter.
         * @return the current SourceId filter value
         */
        uint GetSourceId();
        
        /**
         * @brief sets the SourceId filter for Object detection
         * @param[in] sourceId new filter value to use
         */
        void SetSourceId(uint sourceId);
        
        /**
         * @brief Gets the Minimuum Inference Confidence to trigger the event
         * @return the current Minimum Confidence value in use [0..1.0]
         */
        float GetMinConfidence();
        
        /**
         * @brief Sets the Minumum Inference Confidence to trigger the event
         * @param minConfidence new Minumum Confidence value to use
         */
        void SetMinConfidence(float minConfidence);
        
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

    protected:
    
        /**
         * @brief Common function to check if an Object's meta data meets the min criteria for ODE 
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to test for min criteria
         * @return true if Min Criteria is met, false otherwise
         */
        bool CheckForMinCriteria(NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
    
        /**
         * @brief Mutex to ensure mutual exlusion for propery get/sets
         */
        GMutex m_propertyMutex;

    
    public:
    
        // access made public for performace reasons

        /**
         * @brief Wide string name used for C/Python API
         */
        std::wstring m_wName;

        /**
         * @brief Unique DSL_ODE_TYPE_... identifer defined in dsl.h
         */
        uint m_eventType;
    
        /**
         * @brief trigger count, incremented on every event occurrence
         */
        uint64_t m_triggered;    
    
        /**
         * @brief trigger limit, once reached, actions will no longer be invoked
         */
        uint64_t m_limit;

        /**
         * @brief number of occurrences for the current frame, 
         * reset on exit of PostProcessFrame
         */
        uint m_occurrences; 

        /**
         * @brief GIE Class Id filter for this event
         */
        uint m_classId;
        
        /**
         * @brief unique source stream Id filter for this event
         * 0 indicates filter is disabled
         */
        uint m_sourceId;
        
        /**
         * Mininum inference confidence to trigger event [0.0..1.0]
         */
        float m_minConfidence;
        
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
    
    class FirstOccurrenceEvent : public OdeType
    {
    public:
    
        FirstOccurrenceEvent(const char* name, uint classId);
        
        ~FirstOccurrenceEvent();

        /**
         * @brief Function to check a given Object Meta data structure for a First Occurence event
         * and to invoke all Event Actions owned by the event
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        
    private:
    
    };

    class EveryOccurrenceEvent : public OdeType
    {
    public:
    
        EveryOccurrenceEvent(const char* name, uint classId);
        
        ~EveryOccurrenceEvent();

        /**
         * @brief Function to check a given Object Meta data structure for an Every Occurence event
         * and to invoke all Event Actions owned by the event
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
    };

    class FirstAbsenceEvent : public OdeType
    {
    public:
    
        FirstAbsenceEvent(const char* name, uint classId);
        
        ~FirstAbsenceEvent();

        /**
         * @brief Function to check a given Object Meta data structure for Object occurrence
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame for an Absence Event 
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return true if an ODE occurred during post processing
         */
        bool PostProcessFrame(GstBuffer* pBuffer, NvDsFrameMeta* pFrameMeta);

    private:
    
    };

    class EveryAbsenceEvent : public OdeType
    {
    public:
    
        EveryAbsenceEvent(const char* name, uint classId);
        
        ~EveryAbsenceEvent();

        /**
         * @brief Function to check a given Object Meta data structure for Object occurrence
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame for an Absence Event 
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return true if an ODE occurred during post processing
         */
        bool PostProcessFrame(GstBuffer* pBuffer, NvDsFrameMeta* pFrameMeta);

    private:
    
    };
    
    
}

#endif // _DSL_ODE_H
