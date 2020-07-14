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
    #define DSL_ODE_TRIGGER_PTR std::shared_ptr<OdeTrigger>
    
    #define DSL_ODE_TRIGGER_ALWAYS_PTR std::shared_ptr<AlwaysOdeTrigger>
    #define DSL_ODE_TRIGGER_ALWAYS_NEW(name, when) \
        std::shared_ptr<AlwaysOdeTrigger>(new AlwaysOdeTrigger(name, when))

    #define DSL_ODE_TRIGGER_ABSENCE_PTR std::shared_ptr<AbsenceOdeTrigger>
    #define DSL_ODE_TRIGGER_ABSENCE_NEW(name, classId, limit) \
        std::shared_ptr<AbsenceOdeTrigger>(new AbsenceOdeTrigger(name, classId, limit))

    #define DSL_ODE_TRIGGER_INTERSECTION_PTR std::shared_ptr<IntersectionOdeTrigger>
    #define DSL_ODE_TRIGGER_INTERSECTION_NEW(name, classId, limit) \
        std::shared_ptr<IntersectionOdeTrigger>(new IntersectionOdeTrigger(name, classId, limit))

    #define DSL_ODE_TRIGGER_OCCURRENCE_PTR std::shared_ptr<OccurrenceOdeTrigger>
    #define DSL_ODE_TRIGGER_OCCURRENCE_NEW(name, classId, limit) \
        std::shared_ptr<OccurrenceOdeTrigger>(new OccurrenceOdeTrigger(name, classId, limit))

    #define DSL_ODE_TRIGGER_SUMMATION_PTR std::shared_ptr<SummationOdeTrigger>
    #define DSL_ODE_TRIGGER_SUMMATION_NEW(name, classId, limit) \
        std::shared_ptr<SummationOdeTrigger>(new SummationOdeTrigger(name, classId, limit))
        
    #define DSL_ODE_TRIGGER_CUSTOM_PTR std::shared_ptr<CustomOdeTrigger>
    #define DSL_ODE_TRIGGER_CUSTOM_NEW(name, classId, limit, clientChecker, clientData) \
        std::shared_ptr<CustomOdeTrigger>(new CustomOdeTrigger(name, classId, limit, clientChecker, clientData))

    #define DSL_ODE_TRIGGER_MINIMUM_PTR std::shared_ptr<MinimumOdeTrigger>
    #define DSL_ODE_TRIGGER_MINIMUM_NEW(name, classId, limit, minimum) \
        std::shared_ptr<MinimumOdeTrigger>(new MinimumOdeTrigger(name, classId, limit, minimum))
        
    #define DSL_ODE_TRIGGER_MAXIMUM_PTR std::shared_ptr<MaximumOdeTrigger>
    #define DSL_ODE_TRIGGER_MAXIMUM_NEW(name, classId, limit, maximum) \
        std::shared_ptr<MaximumOdeTrigger>(new MaximumOdeTrigger(name, classId, limit, maximum))

    #define DSL_ODE_TRIGGER_RANGE_PTR std::shared_ptr<RangeOdeTrigger>
    #define DSL_ODE_TRIGGER_RANGE_NEW(name, classId, limit, lower, upper) \
        std::shared_ptr<RangeOdeTrigger>(new RangeOdeTrigger(name, classId, limit, lower, upper))

    class OdeTrigger : public Base
    {
    public: 
    
        OdeTrigger(const char* name, uint classId, uint limit);

        ~OdeTrigger();

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
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta){return false;};

        /**
         * @brief Function called to pre process the current frame data prior to checking for Occurrences
         * @param[in] pFrameMeta pointer to NvDsFrameMeta data for pre processing
         */
        virtual void PreProcessFrame(GstBuffer* pBuffer,
            NvDsFrameMeta* pFrameMeta);
        
        /**
         * @brief Function called to process all Occurrence/Absence data for the current frame
         * @param[in] pFrameMeta pointer to NvDsFrameMeta data for post processing
         * @return the number of ODE Occurrences triggered on post process
         */
        virtual uint PostProcessFrame(GstBuffer* pBuffer,
            NvDsFrameMeta* pFrameMeta){return m_occurrences;};

        /**
         * @brief Adds an ODE Action as a child to this ODE Type
         * @param[in] pChild pointer to ODE Action to add
         * @return true if successful, false otherwise
         */
        bool AddAction(DSL_BASE_PTR pChild);
        
        /**
         * @brief Removes a child ODE Action from this ODE Type
         * @param[in] pChild pointer to ODE Action to remove
         * @return true if successful, false otherwise
         */
        bool RemoveAction(DSL_BASE_PTR pChild);
        
        /**
         * @brief Removes all child ODE Actions from this ODE Type
         */
        void RemoveAllActions();
        
        /**
         * @brief Adds an ODE Area as a child to this ODE Type
         * @param[in] pChild pointer to ODE Area to add
         * @return true if successful, false otherwise
         */
        bool AddArea(DSL_BASE_PTR pChild);
        
        /**
         * @brief Removes a child ODE Area from this ODE Type
         * @param[in] pChild pointer to ODE Area to remove
         * @return true if successful, false otherwise
         */
        bool RemoveArea(DSL_BASE_PTR pChild);
        
        /**
         * @brief Removes all child ODE Areas from this ODE Type
         */
        void RemoveAllAreas();
        
        
        /**
         * @brief Resets the Trigger
         */
        void Reset();
        
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
        double GetMinConfidence();
        
        /**
         * @brief Sets the Minumum Inference Confidence to trigger the event
         * @param minConfidence new Minumum Confidence value to use
         */
        void SetMinConfidence(double minConfidence);
        
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
         * @brief Gets the current Maximum rectangle width and height to trigger the event
         * a value of 0 means no maximim
         * @param[out] maxWidth current maximim width value in use
         * @param[out] maxHeight current maximim height value in use
         */
        void GetMaxDimensions(uint* maxWidth, uint* maxHeight);

        /**
         * @brief Sets new Maximum rectangle width and height to trigger the event
         * a value of 0 means no maximim
         * @param[in] maxWidth current maximim width value in use
         * @param[in] maxHeight current maximim height value in use
         */
        void SetMaxDimensions(uint maxWidth, uint maxHeight);
        
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

        /**
         * @brief Gets the current "inferrence-done" only setting
         * If enabled, the bInferDone flag must be set to trigger ODE Occurrence
         * @return true if enabled, false otherwise. Default=false
         */
        bool GetInferDoneOnlySetting();
        
        /**
         * @brief Set the current "on-inferrence-frame only" setting
         * @param[in] onInferOnly  if true/enabled, only frames with
         * the bInfrDone
         */
        void SetInferDoneOnlySetting(bool inferDoneOnly);
        
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
         * @brief Determines if two rectangles overlaps 
         * @param[in] a rectangle A for test
         * @param[in] b rectangle A for test
         * @return true if the object's rectangle overlaps, false otherwise
         */
        bool doesOverlap(NvOSD_RectParams a, NvOSD_RectParams b);
        
        /**
         * @brief Map of ODE Areas to use for minimum critera
         */
        std::map <std::string, DSL_BASE_PTR> m_pOdeAreas;
        
        /**
         * @brief Map of child ODE Actions to invoke on ODE occurrence
         */
        std::map <std::string, DSL_BASE_PTR> m_pOdeActions;
    
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
         * Mininum inference confidence to trigger an ODE occurrence [0.0..1.0]
         */
        float m_minConfidence;
        
        /**
         * @brief Minimum rectangle width to trigger an ODE occurrence
         */
        uint m_minWidth;

        /**
         * @brief Minimum rectangle height to trigger an ODE occurrence
         */
        uint m_minHeight;

        /**
         * @brief Maximum rectangle width to trigger an ODE occurrence
         */
        uint m_maxWidth;

        /**
         * @brief Maximum rectangle height to trigger an ODE occurrence
         */
        uint m_maxHeight;

        /**
         * @brief Minimum frame count numerator to trigger an ODE occurrence
         */
        uint m_minFrameCountN;

        /**
         * @brief Minimum frame count denominator to trigger an ODE occurrence
         */
        uint m_minFrameCountD;
        
        /**
         * @brief if set, the Frame meta value "bInferDone" must be set
         * to trigger an occurrence
         */
        bool m_inferDoneOnly;

    };
    
    class AlwaysOdeTrigger : public OdeTrigger
    {
    public:
    
        AlwaysOdeTrigger(const char* name, uint when);
        
        ~AlwaysOdeTrigger();

        /**
         * @brief Function called to pre-process the current frame data
         * This trigger will not look for any occurrences
         * @param[in] pFrameMeta pointer to NvDsFrameMeta data for pre-processing
         */
        void PreProcessFrame(GstBuffer* pBuffer,
            NvDsFrameMeta* pFrameMeta);

        /**
         * @brief Function to post-process the frame for an Absence Event 
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, NvDsFrameMeta* pFrameMeta);
        
    private:
    
        /**
         * When to Trigger, pre-occurrence-checking, Post-occurrence-checking, or both
         */
        uint m_when;
    
    };


    class OccurrenceOdeTrigger : public OdeTrigger
    {
    public:
    
        OccurrenceOdeTrigger(const char* name, uint classId, uint limit);
        
        ~OccurrenceOdeTrigger();

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

    class AbsenceOdeTrigger : public OdeTrigger
    {
    public:
    
        AbsenceOdeTrigger(const char* name, uint classId, uint limit);
        
        ~AbsenceOdeTrigger();

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
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, NvDsFrameMeta* pFrameMeta);

    private:
    
    };
    
    class SummationOdeTrigger : public OdeTrigger
    {
    public:
    
        SummationOdeTrigger(const char* name, uint classId, uint limit);
        
        ~SummationOdeTrigger();

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
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, NvDsFrameMeta* pFrameMeta);

    private:
    
    };
    
    class IntersectionOdeTrigger : public OdeTrigger
    {
    public:
    
        IntersectionOdeTrigger(const char* name, uint classId, uint limit);
        
        ~IntersectionOdeTrigger();

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
         * @brief Function to post process the frame and generate a Intersection Event 
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, NvDsFrameMeta* pFrameMeta);

    private:
    
        /**
         * @brief list of pointers to NvDsObjectMeta data
         * Each object occurrence that matches the min criteria will be added
         * to list to be checked for intersection on PostProcessFrame
         */ 
        std::vector<NvDsObjectMeta*> m_occurrenceMetaList;
    
    };

    class CustomOdeTrigger : public OdeTrigger
    {
    public:
    
        CustomOdeTrigger(const char* name, 
            uint classId, uint limit, dsl_ode_check_for_occurrence_cb clientChecker, void* clientData);
        
        ~CustomOdeTrigger();

        /**
         * @brief Function to check a given Object Meta data structure for an Occurrence that meets the min criteria
         * and to inkoke the client provided "client_checker". If the client returns TRUE, all Event Actions owned 
         * by the trigger will be invoked.
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief client provided Check for Occurrence callback
         */
        dsl_ode_check_for_occurrence_cb m_clientChecker;
        
        /**
         * @brief client data to be returned to the client on callback
         */
        void* m_clientData;
    
    };    

    class MinimumOdeTrigger : public OdeTrigger
    {
    public:
    
        MinimumOdeTrigger(const char* name, uint classId, uint limit, uint minimum);
        
        ~MinimumOdeTrigger();

        /**
         * @brief Function to check a given Object Meta data structure for Object occurrence, 
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a Minimum ODE occurrence if the 
         * number of occurrences is less that the Trigger's Minimum value
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, NvDsFrameMeta* pFrameMeta);

    private:
    
        /**
         * @brief minimum object count before for ODE occurrence
         */
        uint m_minimum;
        
    };

    class MaximumOdeTrigger : public OdeTrigger
    {
    public:
    
        MaximumOdeTrigger(const char* name, uint classId, uint limit, uint maximum);
        
        ~MaximumOdeTrigger();

        /**
         * @brief Function to check a given Object Meta data structure for Object occurrence, 
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a Maximum ODE occurrence if the 
         * number of occurrences is greater that the Trigger's Maximum value
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, NvDsFrameMeta* pFrameMeta);

    private:
    
        /**
         * @brief maximum object count before for ODE occurrence
         */
        uint m_maximum;
    
    };

    class RangeOdeTrigger : public OdeTrigger
    {
    public:
    
        RangeOdeTrigger(const char* name, uint classId, uint limit, uint lower, uint upper);
        
        ~RangeOdeTrigger();

        /**
         * @brief Function to check a given Object Meta data structure for Object occurrence, 
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a Range ODE occurrence if the 
         * number of occurrences is with in range of the Trigger's Upper and Lower values
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, NvDsFrameMeta* pFrameMeta);

    private:
    
        /**
         * @brief Lower range of the object count for ODE occurrence
         */
        uint m_lower;
    
        /**
         * @brief Lower range of the object count for ODE occurrence
         */
        uint m_upper;
    
    };

}

#endif // _DSL_ODE_H
