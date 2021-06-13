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
    #define DSL_ODE_TRIGGER_ALWAYS_NEW(name, source, when) \
        std::shared_ptr<AlwaysOdeTrigger>(new AlwaysOdeTrigger(name, source, when))

    #define DSL_ODE_TRIGGER_ABSENCE_PTR std::shared_ptr<AbsenceOdeTrigger>
    #define DSL_ODE_TRIGGER_ABSENCE_NEW(name, source, classId, limit) \
        std::shared_ptr<AbsenceOdeTrigger>(new AbsenceOdeTrigger(name, source, classId, limit))

    #define DSL_ODE_TRIGGER_INSTANCE_PTR std::shared_ptr<InstanceOdeTrigger>
    #define DSL_ODE_TRIGGER_INSTANCE_NEW(name, source, classId, limit) \
        std::shared_ptr<InstanceOdeTrigger>(new InstanceOdeTrigger(name, source, classId, limit))

    #define DSL_ODE_TRIGGER_OCCURRENCE_PTR std::shared_ptr<OccurrenceOdeTrigger>
    #define DSL_ODE_TRIGGER_OCCURRENCE_NEW(name, source, classId, limit) \
        std::shared_ptr<OccurrenceOdeTrigger>(new OccurrenceOdeTrigger(name, source, classId, limit))

    #define DSL_ODE_TRIGGER_SUMMATION_PTR std::shared_ptr<SummationOdeTrigger>
    #define DSL_ODE_TRIGGER_SUMMATION_NEW(name, source, classId, limit) \
        std::shared_ptr<SummationOdeTrigger>(new SummationOdeTrigger(name, source, classId, limit))
        
    #define DSL_ODE_TRIGGER_CUSTOM_PTR std::shared_ptr<CustomOdeTrigger>
    #define DSL_ODE_TRIGGER_CUSTOM_NEW(name, \
    source, classId, limit, clientChecker, clientPostProcessor, clientData) \
        std::shared_ptr<CustomOdeTrigger>(new CustomOdeTrigger(name, \
            source, classId, limit, clientChecker, clientPostProcessor, clientData))

    #define DSL_ODE_TRIGGER_PERSISTENCE_PTR std::shared_ptr<PersistenceOdeTrigger>
    #define DSL_ODE_TRIGGER_PERSISTENCE_NEW(name, source, classId, limit, minimum, maximum) \
        std::shared_ptr<PersistenceOdeTrigger> \
			(new PersistenceOdeTrigger(name, source, classId, limit, minimum, maximum))

    #define DSL_ODE_TRIGGER_COUNT_PTR std::shared_ptr<CountOdeTrigger>
    #define DSL_ODE_TRIGGER_COUNT_NEW(name, source, classId, limit, minimum, maximum) \
        std::shared_ptr<CountOdeTrigger> \
            (new CountOdeTrigger(name, source, classId, limit, minimum, maximum))

    #define DSL_ODE_TRIGGER_SMALLEST_PTR std::shared_ptr<SmallestOdeTrigger>
    #define DSL_ODE_TRIGGER_SMALLEST_NEW(name, source, classId, limit) \
        std::shared_ptr<SmallestOdeTrigger>(new SmallestOdeTrigger(name, source, classId, limit))

    #define DSL_ODE_TRIGGER_LARGEST_PTR std::shared_ptr<LargestOdeTrigger>
    #define DSL_ODE_TRIGGER_LARGEST_NEW(name, source, classId, limit) \
        std::shared_ptr<LargestOdeTrigger>(new LargestOdeTrigger(name, source, classId, limit))

    #define DSL_ODE_TRIGGER_NEW_LOW_PTR std::shared_ptr<NewLowOdeTrigger>
    #define DSL_ODE_TRIGGER_NEW_LOW_NEW(name, source, classId, limit, preset) \
        std::shared_ptr<NewLowOdeTrigger>(new NewLowOdeTrigger(name, source, classId, limit, preset))

    #define DSL_ODE_TRIGGER_NEW_HIGH_PTR std::shared_ptr<NewHighOdeTrigger>
    #define DSL_ODE_TRIGGER_NEW_HIGH_NEW(name, source, classId, limit, preset) \
        std::shared_ptr<NewHighOdeTrigger>(new NewHighOdeTrigger(name, source, classId, limit, preset))

    // Triggers for ClassA - ClassB Testing

    #define DSL_ODE_TRIGGER_AB_PTR std::shared_ptr<ABOdeTrigger>
    
    #define DSL_ODE_TRIGGER_DISTANCE_PTR std::shared_ptr<DistanceOdeTrigger>
    #define DSL_ODE_TRIGGER_DISTANCE_NEW(name, source, classIdA, classIdB, \
        limit, maximum, minimum, testPoint, testMethod) std::shared_ptr<DistanceOdeTrigger> \
            (new DistanceOdeTrigger(name, source, classIdA, classIdB, limit, \
            maximum, minimum, testPoint, testMethod))

    #define DSL_ODE_TRIGGER_INTERSECTION_PTR std::shared_ptr<IntersectionOdeTrigger>
    #define DSL_ODE_TRIGGER_INTERSECTION_NEW(name, source, classIdA, classIdB, limit) \
        std::shared_ptr<IntersectionOdeTrigger> \
            (new IntersectionOdeTrigger(name, source, classIdA, classIdB, limit))



    class OdeTrigger : public Base
    {
    public: 
    
        OdeTrigger(const char* name, const char* source, uint classId, uint limit);

        ~OdeTrigger();

        /**
         * @brief total count of all events
         */
        static uint64_t s_eventCount;
        
        /**
         * @brief Function to check a given Object Meta data structure for the 
         * occurence of an event and to invoke all Event Actions owned by the event
         * @param[in] pBuffer pointer to the GST Buffer containing all meta
         * @param[in] pBatchMeta aquired from pBuffer containing the Frame and Object meta
         * @param[in] pFrameMeta pointer to the containing NvDsFrameMeta data
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        virtual bool CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta){return false;};

        /**
         * @brief Function called to pre process the current frame data prior to 
         * checking for Occurrences
         * @param[in] pBuffer pointer to the GST Buffer containing all meta
         * @param[in] pBatchMeta aquired from pBuffer containing the Frame meta
         * @param[in] pFrameMeta pointer to NvDsFrameMeta data for pre processing
         */
        virtual void PreProcessFrame(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
            NvDsFrameMeta* pFrameMeta);
        
        /**
         * @brief Function called to process all Occurrence/Absence data for the current frame
         * @param[in] pBuffer pointer to the GST Buffer containing all meta
         * @param[in] pBatchMeta aquired from pBuffer containing the Frame and Object meta
         * @param[in] pFrameMeta pointer to NvDsFrameMeta data for post processing
         * @return the number of ODE Occurrences triggered on post process
         */
        virtual uint PostProcessFrame(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
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
        virtual void Reset();
        
        /**
         * @brief Timer callback function to handle the Reset timer timeout
         * @return false always to destroy the one shot timer.
         */
        int HandleResetTimeout();
        
        /**
         * @brief Gets the current timeout value for the auto-reset timer.
         * @return current timeout value. 0 = disabled (default).
         */
        uint GetResetTimeout();
        
        /**
         * @brief Set the timeout value for the auto-reset timer.
         * @param[in] timeout new timeout value to use. Set to 0 to disable.
         */
        void SetResetTimeout(uint timeout);
        
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
         * @brief Gets the trigger limit for this ODE Trigger 
         * @return the current limit value
         */
        uint GetLimit();
        
        /**
         * @brief Sets the ClassId filter for Object detection 
         * @param[in] limit new trigger limit value to use
         */
        void SetLimit(uint limit);
        
        /**
         * @brief Get the Source filter used for Object detection
         * A value of NULL indicates no filter.
         * @return the current Source filter value
         */
        const char* GetSource();
        
        /**
         * @brief sets the Source filter for Object detection
         * @param[in] source new source name as filter value to use
         */
        void SetSource(const char* source);
        
        /**
         * @brief Note: this service is for testing purposes only. It is
         * used to set the Source Id filter, which is normally queried 
         * and set at runtime by the trigger. 
         * @param id Source Id to use for test scenario
         */
        void _setSourceId(int id);
        
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
        void GetMinDimensions(float* minWidth, float* minHeight);

        /**
         * @brief Sets new Minimum rectangle width and height to trigger the event
         * a value of 0 means no minimum
         * @param[in] minWidth current minimum width value in use
         * @param[in] minHeight current minimum height value in use
         */
        void SetMinDimensions(float minWidth, float minHeight);
        
        /**
         * @brief Gets the current Maximum rectangle width and height to trigger the event
         * a value of 0 means no maximim
         * @param[out] maxWidth current maximim width value in use
         * @param[out] maxHeight current maximim height value in use
         */
        void GetMaxDimensions(float* maxWidth, float* maxHeight);

        /**
         * @brief Sets new Maximum rectangle width and height to trigger the event
         * a value of 0 means no maximim
         * @param[in] maxWidth current maximim width value in use
         * @param[in] maxHeight current maximim height value in use
         */
        void SetMaxDimensions(float maxWidth, float maxHeight);
        
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
        
        /**
         * @brief Gets the current process interval for this Trigger
         * @return the current process interval, default = 0
         */
        uint GetInterval();
        
        /**
         * @brief Sets the process interval for this Trigger
         * @param interval new interval to use in units of frames
         */
        void SetInterval(uint interval);
        
    protected:
    
        /**
         * @brief Common function to check if an Object's meta data meets the 
         * min criteria for ODE 
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame 
         * that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to test for min criteria
         * @return true if Min Criteria is met, false otherwise
         */
        bool CheckForMinCriteria(NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
        /**
         * @brief Common function to check if a Frame's source id meets the criteria for ODE
         * @param sourceId a Frame's Source Id to check against the trigger's source filter if set.
         * @return true if Source Id criteria is met, false otherwise
         */
        bool CheckForSourceId(int sourceId);
        
        /**
         * @brief Increments the Trigger Occurrence counter and checks to see
         * if the count has been exceeded. If so, starts the reset timer if a 
         * timeout value is set/enabled.
         */
        void IncrementAndCheckTriggerCount();
        
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
        
        /**
         * @brief auto-reset timeout in units of seconds
         */
        uint m_resetTimeout;

        /**
         * @brief gnome timer Id for the auto-reset timeout
         */
        uint m_resetTimerId;
        
        /**
         * @brief Mutex for timer reset logic
         */
        GMutex m_resetTimerMutex;
        
        /**
         * @brief process interval, default = 0
         */
        uint m_interval;
        
        /**
         * @brief current number of frames in the current interval
         */
        uint m_intervalCounter;
        
        /**
         * @brief flag to identify frames that should be skipped, if m_skipFrameInterval > 0
         */
         bool m_skipFrame;
         
    public:
    
        // access made public for performance reasons

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
         * @brief unique source name filter for this event
         * NULL indicates filter is disabled
         */
        std::string m_source;
        
        /**
         * @brief unique source id filter for this event
         * -1 indicates not set ... updated on first use.
         */
        int m_sourceId;
        
        /**
         * @brief GIE Class Id filter for this event
         */
        uint m_classId;
        
        /**
         * Mininum inference confidence to trigger an ODE occurrence [0.0..1.0]
         */
        float m_minConfidence;
        
        /**
         * @brief Minimum rectangle width to trigger an ODE occurrence
         */
        float m_minWidth;

        /**
         * @brief Minimum rectangle height to trigger an ODE occurrence
         */
        float m_minHeight;

        /**
         * @brief Maximum rectangle width to trigger an ODE occurrence
         */
        float m_maxWidth;

        /**
         * @brief Maximum rectangle height to trigger an ODE occurrence
         */
        float m_maxHeight;

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
    
    static int TriggerResetTimeoutHandler(gpointer pTrigger);
    
    
    class AlwaysOdeTrigger : public OdeTrigger
    {
    public:
    
        AlwaysOdeTrigger(const char* name, const char* source, uint when);
        
        ~AlwaysOdeTrigger();

        /**
         * @brief Function called to pre-process the current frame data
         * This trigger will not look for any occurrences
         * @param[in] pFrameMeta pointer to NvDsFrameMeta data for pre-processing
         */
        void PreProcessFrame(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
            NvDsFrameMeta* pFrameMeta);

        /**
         * @brief Function to post-process the frame for an Absence Event 
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, NvDsFrameMeta* pFrameMeta);
        
    private:
    
        /**
         * When to Trigger, pre-occurrence-checking, Post-occurrence-checking, or both
         */
        uint m_when;
    
    };


    class OccurrenceOdeTrigger : public OdeTrigger
    {
    public:
    
        OccurrenceOdeTrigger(const char* name, const char* source, uint classId, uint limit);
        
        ~OccurrenceOdeTrigger();

        /**
         * @brief Function to check a given Object Meta data structure for an Every Occurence event
         * and to invoke all Event Actions owned by the event
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
    };

    class AbsenceOdeTrigger : public OdeTrigger
    {
    public:
    
        AbsenceOdeTrigger(const char* name, const char* source, uint classId, uint limit);
        
        ~AbsenceOdeTrigger();

        /**
         * @brief Function to check a given Object Meta data structure for Object occurrence
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame for an Absence Event 
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, NvDsFrameMeta* pFrameMeta);

    private:
    
    };

    class InstanceOdeTrigger : public OdeTrigger
    {
    public:
    
        InstanceOdeTrigger(const char* name, const char* source, uint classId, uint limit);
        
        ~InstanceOdeTrigger();

        /**
         * @brief Function to check a given Object Meta data structure for New Instances of a Class
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
            
    private:
        /**
         * @brief map of last Tracking Ids per unique source_id-class_id combination
         */
        std::map <std::string, uint64_t> m_instances;
    
    };
    
    class SummationOdeTrigger : public OdeTrigger
    {
    public:
    
        SummationOdeTrigger(const char* name, const char* source, uint classId, uint limit);
        
        ~SummationOdeTrigger();

        /**
         * @brief Function to check a given Object Meta data structure for Object occurrence
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame 
         * Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame 
         * that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a Summation Event 
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            NvDsDisplayMeta* pDisplayMeta, NvDsFrameMeta* pFrameMeta);

    private:
    
    };
    
    class CustomOdeTrigger : public OdeTrigger
    {
    public:
    
        CustomOdeTrigger(const char* name, const char* source, 
            uint classId, uint limit, dsl_ode_check_for_occurrence_cb clientChecker, 
            dsl_ode_post_process_frame_cb clientPostProcessor, void* clientData);
        
        ~CustomOdeTrigger();

        /**
         * @brief Function to check a given Object Meta data structure for an 
         * Occurrence that meets the min criteria and to inkoke the client provided 
         * "client_checker". If the client returns TRUE, all Event Actions owned 
         * by the trigger will be invoked.
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame 
         * Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame 
         * that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */

        bool CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        /**
         * @brief Function to call the client provided callback to post process the frame 
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            NvDsDisplayMeta* pDisplayMeta, NvDsFrameMeta* pFrameMeta);
        
    private:
    
        /**
         * @brief client provided Check for Occurrence callback
         */
        dsl_ode_check_for_occurrence_cb m_clientChecker;
        
        /**
         * @brief client provided Post-Process Frame callback
         */
        dsl_ode_post_process_frame_cb m_clientPostProcessor;
        
        /**
         * @brief client data to be returned to the client on callback
         */
        void* m_clientData;
    
    };    

    class MinimumOdeTrigger : public OdeTrigger
    {
    public:
    
        MinimumOdeTrigger(const char* name, const char* source, uint classId, uint limit, uint minimum);
        
        ~MinimumOdeTrigger();

        /**
         * @brief Function to check a given Object Meta data structure for Object occurrence, 
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame 
         * Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame 
         * that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a Minimum ODE occurrence if the 
         * number of occurrences is less that the Trigger's Minimum value
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            NvDsDisplayMeta* pDisplayMeta, NvDsFrameMeta* pFrameMeta);

    private:
    
        /**
         * @brief minimum object count before for ODE occurrence
         */
        uint m_minimum;
        
    };

    class MaximumOdeTrigger : public OdeTrigger
    {
    public:
    
        MaximumOdeTrigger(const char* name, const char* source, uint classId, uint limit, uint maximum);
        
        ~MaximumOdeTrigger();

        /**
         * @brief Function to check a given Object Meta data structure for Object occurrence, 
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame 
         * Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame 
         * that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a Maximum ODE occurrence if the 
         * number of occurrences is greater that the Trigger's Maximum value
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            NvDsDisplayMeta* pDisplayMeta,  NvDsFrameMeta* pFrameMeta);

    private:
    
        /**
         * @brief maximum object count before for ODE occurrence
         */
        uint m_maximum;
		
    };

	struct TrackedObject
	{

		TrackedObject(uint64_t trackingId, uint64_t frameNumber)
			: m_trackingId(trackingId)
			, m_frameNumber(frameNumber)
		{
			timeval creationTime;
			gettimeofday(&creationTime, NULL);
			m_creationTimeMs = creationTime.tv_sec*1000.0 + creationTime.tv_usec/1000.0;
		}
		
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
	};
	
	/**
	 * @brief map of tracked objects - unique Tracking Id as key
	 */
	typedef std::map <uint64_t, std::shared_ptr<TrackedObject>> TrackedObjects;

    class PersistenceOdeTrigger : public OdeTrigger
    {
    public:
    
        PersistenceOdeTrigger(const char* name, 
			const char* source, uint classId, uint limit, uint minimum, uint maximum);
        
        ~PersistenceOdeTrigger();

        /**
         * @brief Gets the current Minimum and Maximum time settings in use. 
         * a value of 0 means no minimum or maximum
         * @param[out] minimim current minimum time setting in use
         * @param[out] maximum current maximum time setting in use
         */
        void GetRange(uint* minimum, uint* maximum);

        /**
         * @brief Sets new Minimum and Maximum time settings to use.
         * a value of 0 means no minimum or maximum
         * @param[in] minimum new minimum time value to use
         * @param[in] maximum new maximum time value to use
         */
        void SetRange(uint minimum, uint maximum);

        /**
         * @brief Function to check a given Object Meta data structure for Object occurrence, 
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame 
         * Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame 
         * that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a Persistence ODE occurrence 
		 * if any unique object has been tracked for a period of time within the Trigger's 
		 * minimum and maximum duration value
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            NvDsDisplayMeta* pDisplayMeta,  NvDsFrameMeta* pFrameMeta);

    private:

        /**
         * @brief minimum duration of object persistence - 0 = no minimum.
         */
        double m_minimumMs;
    
        /**
         * @brief maximum duration of object persistence - 0 = no maximum
         */
        double m_maximumMs;
    
		/**
		 * @brief map of tracked objects per source - Key = source Id
		 */
		std::map <uint, std::shared_ptr<TrackedObjects>> m_trackedObjectsPerSource;
    };

    class CountOdeTrigger : public OdeTrigger
    {
    public:
    
        CountOdeTrigger(const char* name, 
			const char* source, uint classId, uint limit, uint minimum, uint maximum);
        
        ~CountOdeTrigger();

        /**
         * @brief Gets the current Minimum and Maximum count setting in use. 
         * a value of 0 means no minimum or maximum
         * @param[out] minimim current minimum count setting in use
         * @param[out] maximum current maximum count setting in use
         */
        void GetRange(uint* minimum, uint* maximum);

        /**
         * @brief Sets new Minimum and Maximum count settings to use.
         * a value of 0 means no minimum or maximum
         * @param[in] minimum new minimum count value to use
         * @param[in] maximum new maximum count value to use
         */
        void SetRange(uint minimum, uint maximum);

        /**
         * @brief Function to check a given Object Meta data structure for Object occurrence, 
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame 
         * Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame 
         * that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a Count ODE occurrence if the 
         * number of occurrences is with in the Trigger's minimum and maximum settings
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            NvDsDisplayMeta* pDisplayMeta,  NvDsFrameMeta* pFrameMeta);

    private:
    
        /**
         * @brief Minimum object count for ODE occurrence, 0 = no minimum
         */
        uint m_minimum;
    
        /**
         * @brief Maximum object count for ODE occurrence, 0 = no maximum
         */
        uint m_maximum;
    
    };

    class SmallestOdeTrigger : public OdeTrigger
    {
    public:
    
        SmallestOdeTrigger(const char* name, const char* source, uint classId, uint limit);
        
        ~SmallestOdeTrigger();

        /**
         * @brief Function to check a given Object Meta data structure for Object occurrence
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame 
         * Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame 
         * that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a Smallest Object Event 
         * if at least one object is found
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            NvDsDisplayMeta* pDisplayMeta,  NvDsFrameMeta* pFrameMeta);

    private:
    
        /**
         * @brief list of pointers to NvDsObjectMeta data
         * Each object occurrence that matches the min criteria will be added
         * to list to be checked for Smallest object on PostProcessFrame
         */ 
        std::vector<NvDsObjectMeta*> m_occurrenceMetaList;
    
    };

    class LargestOdeTrigger : public OdeTrigger
    {
    public:
    
        LargestOdeTrigger(const char* name, const char* source, uint classId, uint limit);
        
        ~LargestOdeTrigger();

        /**
         * @brief Function to check a given Object Meta data structure for Object occurrence
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame 
         * Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame 
         * that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a Largest Object Event 
         * if at least one object is found
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            NvDsDisplayMeta* pDisplayMeta,  NvDsFrameMeta* pFrameMeta);

    private:
    
        /**
         * @brief list of pointers to NvDsObjectMeta data
         * Each object occurrence that matches the min criteria will be added
         * to list to be checked for Largest object on PostProcessFrame
         */ 
        std::vector<NvDsObjectMeta*> m_occurrenceMetaList;
    
    };
    
    class NewLowOdeTrigger : public OdeTrigger
    {
    public:
    
        NewLowOdeTrigger(const char* name, 
            const char* source, uint classId, uint limit, uint preset);
        
        ~NewLowOdeTrigger();

        /**
         * @brief Resets the Trigger
         */
        virtual void Reset();

        /**
         * @brief Function to check a given Object Meta data structure for Object occurrence
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame 
         * Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame 
         * that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a New Low Count Event 
         * if the current Frame's object count is less than the current low
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            NvDsDisplayMeta* pDisplayMeta, NvDsFrameMeta* pFrameMeta);

    private:
    
        /**
         * @brief initial low value to use on first play and reset.
         */
        uint m_preset;
        
        /**
         * @brief current lowest count value, updated on new low.
         * Set to m_preset on trigger create and reset.
         */
        uint m_currentLow;
    
    };

    class NewHighOdeTrigger : public OdeTrigger
    {
    public:
    
        NewHighOdeTrigger(const char* name, 
            const char* source, uint classId, uint limit, uint preset);
        
        ~NewHighOdeTrigger();

        /**
         * @brief Resets the Trigger
         */
        void Reset();

        /**
         * @brief Function to check a given Object Meta data structure for Object occurrence
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame 
         * Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame 
         * that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a New High Count Event 
         * if the current Frame's object count is greater than the current high
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            NvDsDisplayMeta* pDisplayMeta, NvDsFrameMeta* pFrameMeta);

    private:
    
        /**
         * @brief initial high value to use on first play and reset.
         */
        uint m_preset;
        
        /**
         * @brief current highest count value, updated on new high.
         * Set to m_preset on trigger create and reset.
         */
        uint m_currentHigh;
    
    };

    class ABOdeTrigger : public OdeTrigger
    {
    public:
    
        ABOdeTrigger(const char* name, const char* source, 
            uint classIdA, uint classIdB, uint limit);
        
        ~ABOdeTrigger();

        /**
         * @brief Function to check a given Object Meta data structure for Object occurrence
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame 
         * Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the 
         * frame that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a Distance Event 
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        virtual uint PostProcessFrame(GstBuffer* pBuffer, 
            NvDsDisplayMeta* pDisplayMeta, NvDsFrameMeta* pFrameMeta);

        /**
         * @brief Gets the ClassIdA and ClassIdB filters used for Object detection 
         * @param[out] classA Class Id for Class A 
         * @param[out] classB Class Id for Class B
         * @return the current ClassId filter value
         */
        void GetClassIdAB(uint* classA, uint* classB);
        
        /**
         * @brief Sets the ClassId filter for Object detection 
         * @param[in] classA Class Id for Class A 
         * @param[in] classB Class Id for Class B
         * @param[in] classId new filter value to use
         */
        void SetClassIdAB(uint classIdA, uint classIdB);

    protected:

        /**
         * @brief Function to post process the frame and generate a Distance Event - Class A Only
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        virtual uint PostProcessFrameA(GstBuffer* pBuffer, 
            NvDsDisplayMeta* pDisplayMeta, NvDsFrameMeta* pFrameMeta) = 0;

        /**
         * @brief Function to post process the frame and generate a Distance Event - Class A/B
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        virtual uint PostProcessFrameAB(GstBuffer* pBuffer, 
            NvDsDisplayMeta* pDisplayMeta, NvDsFrameMeta* pFrameMeta) = 0;

        /**
         * @brief list of pointers to NvDsObjectMeta data for Class A
         * Each object occurrence of Class A that matches the criteria will be added
         * to list to be checked for distance on PostProcessFrame
         */ 
        std::vector<NvDsObjectMeta*> m_occurrenceMetaListA;

        /**
         * @brief list of pointers to NvDsObjectMeta data for Class B
         * Each object occurrence of Class B that matches the min criteria will be added
         * to list to be checked for distance on PostProcessFrame
         */ 
        std::vector<NvDsObjectMeta*> m_occurrenceMetaListB;
        
        /**
         * @brief boolean flag to specify if A-A testing or A-B testing
         */
        bool m_classIdAOnly;
    
        /**
         * @brief Class ID to for A objects for A-B distance calculation
         */
        uint m_classIdA;

        /**
         * @brief Class ID to for A objects for A-B distance calculation
         */
        uint m_classIdB;
    };

    class DistanceOdeTrigger : public ABOdeTrigger
    {
    public:
    
        DistanceOdeTrigger(const char* name, const char* source, 
            uint classIdA, uint classIdB, uint limit, uint minimum, uint maximum, 
            uint testPoint, uint testMethod);
        
        ~DistanceOdeTrigger();
        
        /**
         * @brief Gets the current Minimum and Maximum distance setting in use. 
         * a value of 0 means no minimum or maximum
         * @param[out] minimim current minimum distance setting in use
         * @param[out] maximum current maximum distance setting in use
         */
        void GetRange(uint* minimum, uint* maximum);

        /**
         * @brief Sets new Minimum and Maximum distance settings to use.
         * a value of 0 means no minimum or maximum
         * @param[in] minimum new minimum distance value to use
         * @param[in] maximum new maximum distance value to use
         */
        void SetRange(uint minimum, uint maximum);

        /**
         * @brief Gets the current Test Point and Test Methods parameters in use. 
         * @param[out] testPoint current test point value in use
         * @param[out] testMethod current test method value in use
         */
        void GetTestParams(uint* testPoint, uint* testMethod);
        
        /**
         * @brief Sets the current Test Point and Test Methods parameters in use. 
         * @param[in] testPoint new test point value to use
         * @param[in] testMethod new test method value to use
         */
        void SetTestParams(uint testPoint, uint testMethod);
        

    private:

        /**
         * @brief Function to post process the frame and generate a Distance Event - Class A Only
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrameA(GstBuffer* pBuffer, 
            NvDsDisplayMeta* pDisplayMeta, NvDsFrameMeta* pFrameMeta);

        /**
         * @brief Function to post process the frame and generate a Distance Event - Class A/B
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrameAB(GstBuffer* pBuffer, 
            NvDsDisplayMeta* pDisplayMeta, NvDsFrameMeta* pFrameMeta);

    
        /**
         * @brief Calculates the distance between two objects based on the current
         * m_bboxTestPoint setting. Either point-to-point or edge-to-edge
         * @param pObjectMetaA[in] pointer to Object A's meta data with location and dimension
         * @param pObjectMetaB[in] pointer to Object B's meta data with location and dimension
         * @return true if the objects are within minimum or beyond the maximum distance
         * as mesured by the DSL_DISTANCE_METHOD
         */
        bool CheckDistance(NvDsObjectMeta* pObjectMetaA, NvDsObjectMeta* pObjectMetaB);
    
        
        /**
         * @brief minimum distance between objects to trigger ODE occurrence
         */
        uint m_minimum;
        
        /**
         * @brief maximum distance between objects to trigger ODE occurrence
         */
        uint m_maximum;
        
        /**
         * @brief the bounding box point to measure distance
         */
        uint m_testPoint;
        
        /**
         * @brief the method to use to measure distance between objects.
         */
        uint m_testMethod;
    };

    class IntersectionOdeTrigger : public ABOdeTrigger
    {
    public:
    
        IntersectionOdeTrigger(const char* name, 
            const char* source, uint classIdA, uint classIdB, uint limit);
        
        ~IntersectionOdeTrigger();

    private:

        /**
         * @brief Function to post process the frame and generate an Intersection Event - Class A only
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrameA(GstBuffer* pBuffer, 
            NvDsDisplayMeta* pDisplayMeta, NvDsFrameMeta* pFrameMeta);
    
        /**
         * @brief Function to post process the frame and generate an Intersection Event - Class A/B testing
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrameAB(GstBuffer* pBuffer, 
            NvDsDisplayMeta* pDisplayMeta, NvDsFrameMeta* pFrameMeta);
    };

}

#endif // _DSL_ODE_H
