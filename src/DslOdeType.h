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

    #define DSL_ODE_TYPE_OCCURRENCE_PTR std::shared_ptr<OccurrenceOdeType>
    #define DSL_ODE_TYPE_OCCURRENCE_NEW(name, classId, limit) \
        std::shared_ptr<OccurrenceOdeType>(new OccurrenceOdeType(name, classId, limit))
        
    #define DSL_ODE_TYPE_ABSENCE_PTR std::shared_ptr<AbsenceOdeType>
    #define DSL_ODE_TYPE_ABSENCE_NEW(name, classId, limit) \
        std::shared_ptr<AbsenceOdeType>(new AbsenceOdeType(name, classId, limit))
        
    #define DSL_ODE_TYPE_SUMMATION_PTR std::shared_ptr<SummationOdeType>
    #define DSL_ODE_TYPE_SUMMATION_NEW(name, classId, limit) \
        std::shared_ptr<SummationOdeType>(new SummationOdeType(name, classId, limit))
        
    class OdeType : public Base
    {
    public: 
    
        OdeType(const char* name, uint classId, uint limit);

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
         * @brief Function called to pre process the current frame data prior to checking for Occurrences
         * @param[in] pFrameMeta pointer to NvDsFrameMeta data for pre processing
         */
        virtual void PreProcessFrame(GstBuffer* pBuffer,
            NvDsFrameMeta* pFrameMeta);
        
        /**
         * @brief Function called to process all Occurrence/Absence data for the current frame
         * @param[in] pFrameMeta pointer to NvDsFrameMeta data for post processing
         * @return true if Occurrence on post process, false otherwise
         */
        virtual bool PostProcessFrame(GstBuffer* pBuffer,
            NvDsFrameMeta* pFrameMeta){return false;};
        
        /**
         * @brief Gets the current Enabled setting, default = true
         * @return the current Enabled setting
         */
        bool GetEnabled();
        
        /**
         * @brief Sets the Enabled setting for ODE type
         * @param[in] the new value to use
         */
        void SetEnabled(bool enabled);
        
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
         * @brief Gets the current detection area rectange settings, 
         * @param[out] left location of the area's left side on the x-axis in pixels, from the left of the frame
         * @param[out] top location of the area's top on the y-axis in pixels, from the top of the frame
         * @param[out] width width of the area in pixels
         * @param[out] height of the area in pixels
         */
        void GetArea(uint* left, uint* top, uint* width, uint* height);

        /**
         * @brief sets the currenty detection area rectange settings, 
         * @param[in] left location of the area's left side on the x-axis in pixels, from the left of the frame
         * @param[in] top location of the area's top on the y-axis in pixels, from the top of the frame
         * @param[in] width width of the area in pixels
         * @param[in] height of the area in pixels
         */
        void SetArea(uint left, uint top, uint width, uint height);
        
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
        bool checkForMinCriteria(NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
        /**
         * @brief helper function for doesOverlap
         * @param value to check if in range
         * @param min min value of range check
         * @param max max value of range check
         * @return trun if value in range of min-max
         */
        bool valueInRange(int value, int min, int max);
        
        /**
         * @brief Determines if an object's rectangle overlaps with the ODE Type's area
         * @param rectParams object's rectangle to check for overlap
         * @return true if the object's rectangle overlaps, false otherwise
         */
        bool doesOverlap(NvOSD_RectParams rectParams);
    
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
         * @brief enabled flag.
         */
        bool m_enabled;

        /**
         * @brief trigger count, incremented on every event occurrence
         */
        uint64_t m_triggered;    
    
        /**
         * @brief trigger limit, once reached, actions will no longer be invoked
         */
        uint m_limit;

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
         * @brief Rectangle array for object detection 
         */
        NvOSD_RectParams m_areaParams;

        /**
         * @brief Minimum frame count numerator to trigger event
         */
        uint m_minFrameCountN;

        /**
         * @brief Minimum frame count denominator to trigger event
         */
        uint m_minFrameCountD;

    };
    
    class OccurrenceOdeType : public OdeType
    {
    public:
    
        OccurrenceOdeType(const char* name, uint classId, uint limit);
        
        ~OccurrenceOdeType();

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

    class AbsenceOdeType : public OdeType
    {
    public:
    
        AbsenceOdeType(const char* name, uint classId, uint limit);
        
        ~AbsenceOdeType();

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
    
    class SummationOdeType : public OdeType
    {
    public:
    
        SummationOdeType(const char* name, uint classId, uint limit);
        
        ~SummationOdeType();

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
         * @brief Function to post process the frame and generate a Summation Event 
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return true if an ODE occurred during post processing
         */
        bool PostProcessFrame(GstBuffer* pBuffer, NvDsFrameMeta* pFrameMeta);

    private:
    
    };
    
}

#endif // _DSL_ODE_H
