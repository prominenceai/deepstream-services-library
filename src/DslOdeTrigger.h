/*
The MIT License

Copyright (c) 2019-2022, Prominence AI, Inc.

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

#ifndef _DSL_ODE_TRIGGER_H
#define _DSL_ODE_TRIGGER_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslOdeBase.h"
#include "DslOdeTrackedObject.h"
#include "DslDisplayTypes.h"

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
        std::shared_ptr<AbsenceOdeTrigger>(new AbsenceOdeTrigger(name, \
            source, classId, limit))

    #define DSL_ODE_TRIGGER_INSTANCE_PTR std::shared_ptr<InstanceOdeTrigger>
    #define DSL_ODE_TRIGGER_INSTANCE_NEW(name, source, classId, limit) \
        std::shared_ptr<InstanceOdeTrigger>(new InstanceOdeTrigger(name, \
            source, classId, limit))

    #define DSL_ODE_TRIGGER_OCCURRENCE_PTR std::shared_ptr<OccurrenceOdeTrigger>
    #define DSL_ODE_TRIGGER_OCCURRENCE_NEW(name, source, classId, limit) \
        std::shared_ptr<OccurrenceOdeTrigger>(new OccurrenceOdeTrigger(name, \
            source, classId, limit))

    #define DSL_ODE_TRIGGER_SUMMATION_PTR std::shared_ptr<SummationOdeTrigger>
    #define DSL_ODE_TRIGGER_SUMMATION_NEW(name, source, classId, limit) \
        std::shared_ptr<SummationOdeTrigger>(new SummationOdeTrigger(name, \
            source, classId, limit))
        
    #define DSL_ODE_TRIGGER_CUSTOM_PTR std::shared_ptr<CustomOdeTrigger>
    #define DSL_ODE_TRIGGER_CUSTOM_NEW(name, \
    source, classId, limit, clientChecker, clientPostProcessor, clientData) \
        std::shared_ptr<CustomOdeTrigger>(new CustomOdeTrigger(name, \
            source, classId, limit, clientChecker, clientPostProcessor, clientData))

    #define DSL_ODE_TRIGGER_COUNT_PTR std::shared_ptr<CountOdeTrigger>
    #define DSL_ODE_TRIGGER_COUNT_NEW(name, source, classId, limit, minimum, maximum) \
        std::shared_ptr<CountOdeTrigger> (new CountOdeTrigger(name, \
            source, classId, limit, minimum, maximum))

    #define DSL_ODE_TRIGGER_SMALLEST_PTR std::shared_ptr<SmallestOdeTrigger>
    #define DSL_ODE_TRIGGER_SMALLEST_NEW(name, source, classId, limit) \
        std::shared_ptr<SmallestOdeTrigger>(new SmallestOdeTrigger(name, \
            source, classId, limit))

    #define DSL_ODE_TRIGGER_LARGEST_PTR std::shared_ptr<LargestOdeTrigger>
    #define DSL_ODE_TRIGGER_LARGEST_NEW(name, source, classId, limit) \
        std::shared_ptr<LargestOdeTrigger>(new LargestOdeTrigger(name, \
            source, classId, limit))

    #define DSL_ODE_TRIGGER_NEW_LOW_PTR std::shared_ptr<NewLowOdeTrigger>
    #define DSL_ODE_TRIGGER_NEW_LOW_NEW(name, source, classId, limit, preset) \
        std::shared_ptr<NewLowOdeTrigger>(new NewLowOdeTrigger(name, \
            source, classId, limit, preset))

    #define DSL_ODE_TRIGGER_NEW_HIGH_PTR std::shared_ptr<NewHighOdeTrigger>
    #define DSL_ODE_TRIGGER_NEW_HIGH_NEW(name, source, classId, limit, preset) \
        std::shared_ptr<NewHighOdeTrigger>(new NewHighOdeTrigger(name, \
            source, classId, limit, preset))

    // Triggers thta track objects

    #define DSL_ODE_TRACKING_TRIGGER_PTR std::shared_ptr<TrackingOdeTrigger>
    
    #define DSL_ODE_TRIGGER_CROSS_PTR std::shared_ptr<CrossOdeTrigger>
    #define DSL_ODE_TRIGGER_CROSS_NEW(name, \
        source, classId, limit, minTracePoints, maxTracePoints, testMethod, pColor) \
        std::shared_ptr<CrossOdeTrigger>(new CrossOdeTrigger(name, \
            source, classId, limit, minTracePoints, maxTracePoints, testMethod, pColor))

    #define DSL_ODE_TRIGGER_PERSISTENCE_PTR std::shared_ptr<PersistenceOdeTrigger>
    #define DSL_ODE_TRIGGER_PERSISTENCE_NEW(name, \
        source, classId, limit, minimum, maximum) \
        std::shared_ptr<PersistenceOdeTrigger> \
            (new PersistenceOdeTrigger(name, \
                source, classId, limit, minimum, maximum))
                
    #define DSL_ODE_TRIGGER_LATEST_PTR std::shared_ptr<LatestOdeTrigger>
    #define DSL_ODE_TRIGGER_LATEST_NEW(name, source, classId, limit) \
        std::shared_ptr<LatestOdeTrigger>(new LatestOdeTrigger(name, \
            source, classId, limit))

    #define DSL_ODE_TRIGGER_EARLIEST_PTR std::shared_ptr<EarliestOdeTrigger>
    #define DSL_ODE_TRIGGER_EARLIEST_NEW(name, source, classId, limit) \
        std::shared_ptr<EarliestOdeTrigger>(new EarliestOdeTrigger(name, \
            source, classId, limit))


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

    // *****************************************************************************

    /**
     * @class OdeTrigger
     * @brief Implements a super/abstract class for all ODE Triggers
     */
    class OdeTrigger : public OdeBase
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
        virtual bool CheckForOccurrence(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta){return false;};

        /**
         * @brief Function called to pre process the current frame data prior to 
         * checking for Occurrences
         * @param[in] pBuffer pointer to the GST Buffer containing all meta
         * @param[in] pBatchMeta aquired from pBuffer containing the Frame meta
         * @param[in] pFrameMeta pointer to NvDsFrameMeta data for pre processing
         */
        virtual void PreProcessFrame(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta);
        
        /**
         * @brief Function called to process all Occurrence/Absence data for the current frame
         * @param[in] pBuffer pointer to the GST Buffer containing all meta
         * @param[in] pBatchMeta aquired from pBuffer containing the Frame and Object meta
         * @param[in] pFrameMeta pointer to NvDsFrameMeta data for post processing
         * @return the number of ODE Occurrences triggered on post process
         */
        virtual uint PostProcessFrame(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta);

        /**
         * @brief Adds an ODE Action as a child to this OdeTrigger
         * @param[in] pChild pointer to ODE Action to add
         * @return true if successful, false otherwise
         */
        bool AddAction(DSL_BASE_PTR pChild);
        
        /**
         * @brief Removes a child ODE Action from this OdeTrigger
         * @param[in] pChild pointer to ODE Action to remove
         * @return true if successful, false otherwise
         */
        bool RemoveAction(DSL_BASE_PTR pChild);
        
        /**
         * @brief Removes all child ODE Actions from this OdeTrigger
         */
        void RemoveAllActions();
        
        /**
         * @brief Adds an ODE Area as a child to this OdeTrigger
         * @param[in] pChild pointer to ODE Area to add
         * @return true if successful, false otherwise
         */
        bool AddArea(DSL_BASE_PTR pChild);
        
        /**
         * @brief Removes a child ODE Area from this OdeTrigger
         * @param[in] pChild pointer to ODE Area to remove
         * @return true if successful, false otherwise
         */
        bool RemoveArea(DSL_BASE_PTR pChild);
        
        /**
         * @brief Removes all child ODE Areas from this OdeTrigger
         */
        void RemoveAllAreas();

        /**
         * @brief Adds a (one at most) ODE Accumulator as a child to this OdeTrigger.
         * @param[in] pChild pointer to ODE Accumulator to add.
         * @return true if successful, false otherwise
         */
        bool AddAccumulator(DSL_BASE_PTR pAccumulator);
        
        /**
         * @brief Removes the child ODE Accumulator from this OdeTrigger
         * @return true if successful, false otherwise
         */
        bool RemoveAccumulator();
        
        /**
         * @brief Adds a (one at most) ODE HeatMapper as a child to this OdeTrigger.
         * @param[in] pChild pointer to ODE Heat-Mapper to add.
         * @return true if successful, false otherwise
         */
        bool AddHeatMapper(DSL_BASE_PTR pHeatMapper);
        
        /**
         * @brief Removes the child ODE Heat-Mapper from this OdeTrigger
         * @return true if successful, false otherwise
         */
        bool RemoveHeatMapper();
        
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
         * @brief Returns the current state of the Reset Timer.
         * @return ture if the Reset Timer is running, false otherwise.
         */
        bool IsResetTimerRunning();
        
        /**
         * @brief Adds a "limit state-change listener" function to be notified
         * on Trigger LIMIT_REACHED, LIMIT_CHANGED, and COUNTS_RESET.
         * @return ture if the listener function was successfully added, false otherwise.
         */
        bool AddLimitStateChangeListener(
            dsl_ode_trigger_limit_state_change_listener_cb listener, void* clientData);

        /**
         * @brief Removes a "limit event listener" function previously added
         * with a call to AddLimitStateChangeListener.
         * @return true if the listener function was successfully removed, false otherwise.
         */
        bool RemoveLimitStateChangeListener(
            dsl_ode_trigger_limit_state_change_listener_cb listener);
        
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
         * @brief Gets the trigger event limit for this ODE Trigger 
         * @return the current frame limit value
         */
        uint GetEventLimit();
        
        /**
         * @brief Sets the event limit for Object detection.
         * @param[in] limit new trigger frame limit value to use.
         */
        void SetEventLimit(uint limit);
        
        /**
         * @brief Gets the trigger frame limit for this ODE Trigger.
         * @return the current frame limit value.
         */
        uint GetFrameLimit();
        
        /**
         * @brief Sets the frame limit for Object detection.
         * @param[in] limit new trigger frame limit value to use.
         */
        void SetFrameLimit(uint limit);
        
        /**
         * @brief Gets the source filter used for Object detection
         * A value of NULL indicates no filter.
         * @return the current Source filter value
         */
        const char* GetSource();
        
        /**
         * @brief Sets the source filter for Object detection
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
         * @brief Gets the inference component name filter used for Object detection
         * A value of NULL indicates no filter.
         * @return the current inference component name filter value
         */
        const char* GetInfer();
        
        /**
         * @brief sets the inference component name filter for Object detection
         * @param[in] infer new inference component name as filter value to use
         */
        void SetInfer(const char* infer);
        
        /**
         * @brief Note: this service is for testing purposes only. It is
         * used to set the Infer Id filter, which is normally queried 
         * and set at runtime by the trigger. 
         * @param id Infer Id to use for test scenario
         */
        void _setInferId(int id);
        
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
         * @brief Gets the Maximum Inference Confidence to trigger the event
         * @return the current Maximum Confidence value in use [0..1.0]
         */
        float GetMaxConfidence();
        
        /**
         * @brief Sets the Maximum Inference Confidence to trigger the event
         * @param maxConfidence new Maximum Confidence value to use
         */
        void SetMaxConfidence(float maxConfidence);
        
        /**
         * @brief Gets the Minimuum Tracker Confidence to trigger the event
         * @return the current Minimum Confidence value in use [0..1.0]
         */
        float GetMinTrackerConfidence();
        
        /**
         * @brief Sets the Minimum Tracker Confidence to trigger the event
         * @param minConfidence new Minumum Confidence value to use
         */
        void SetMinTrackerConfidence(float minConfidence);
        
        /**
         * @brief Gets the Maximum Tracker Confidence to trigger the event
         * @return the current Maximum Confidence value in use [0..1.0]
         */
        float GetMaxTrackerConfidence();
        
        /**
         * @brief Sets the Maximum Tracker Confidence to trigger the event
         * @param minConfidence new Maximum Confidence value to use
         */
        void SetMaxTrackerConfidence(float maxConfidence);
        
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
         * min criteria for ODE occurrence.
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - 
         * the frame that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to test 
         * for min criteria
         * @return true if Min Criteria is met, false otherwise
         */
        bool CheckForMinCriteria(NvDsFrameMeta* pFrameMeta, 
            NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Common function to check if an Object's bbox fails within
         * one of the Triggers Areas
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to test 
         * for within
         * @return true if the bbox is within one of the trigger's area, false otherwise
         */
        bool CheckForInside(NvDsObjectMeta* pObjectMeta);
        
        /**
         * @brief Common function to check if a Frame's source id meets the 
         * criteria for ODE occurrence.
         * @param sourceId a Frame's Source Id to check against the trigger's 
         * source filter if set.
         * @return true if Source Id criteria is met, false otherwise
         */
        bool CheckForSourceId(int sourceId);
        
        /**
         * @brief Common function to check if an Objects's infer component id 
         * meets the criteria for ODE occurrence.
         * @param inferId an object's inference component Id to check against 
         * the trigger's infer filter if set.
         * @return true if Source Id criteria is met, false otherwise
         */
        bool CheckForInferId(int inferId);
        
        /**
         * @brief Increments the Trigger Occurrence counter and checks to see
         * if the count has been exceeded. If so, starts the reset timer if a 
         * timeout value is set/enabled.
         */
        void IncrementAndCheckTriggerCount();

        /**
         * @brief Index variable to incremment/assign on ODE Area add.
         */
        uint m_nextAreaIndex;
        
        /**
         * @brief Map of child ODE Areas to use for minimum critera
         */
        std::map <std::string, DSL_BASE_PTR> m_pOdeAreas;
        
        /**
         * @brief Map of child ODE Areas indexed by thier add-order for execution
         */
        std::map <uint, DSL_BASE_PTR> m_pOdeAreasIndexed;

        /**
         * @brief Index variable to incremment/assign on ODE Action add.
         */
        uint m_nextActionIndex;

        /**
         * @brief Map of child ODE Actions owned by this trigger
         */
        std::map <std::string, DSL_BASE_PTR> m_pOdeActions;
        
        /**
         * @brief Map of child ODE Actions indexed by their add-order for execution
         */
        std::map <uint, DSL_BASE_PTR> m_pOdeActionsIndexed;
        
        /**
         * @brief optional metric accumulator owned by the ODE Trigger.
         */
        DSL_BASE_PTR m_pAccumulator;
    
        /**
         * @brief optional ODE Heat-Mapper owned by the ODE Trigger.
         */
        DSL_BASE_PTR m_pHeatMapper;
    
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
         * @brief map of all currently registered limit-state-change-listeners
         * callback functions mapped with the user provided data
         */
        std::map<dsl_ode_trigger_limit_state_change_listener_cb, 
            void*>m_limitStateChangeListeners;
        
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
         * @brief trigger count, incremented on every event occurrence
         */
        uint64_t m_triggered;    
    
        /**
         * @brief trigger event limit, once reached, actions will no longer be invoked
         */
        uint m_eventLimit;

        /**
         * @brief number of Frames the trigger has processed.
         */
        uint64_t m_frameCount;
        
        /**
         * @brief trigger frame limit, once reached, actions will no longer be invoked
         */
        uint m_frameLimit;

        /**
         * @brief number of occurrences for the current frame, 
         * reset on exit of PostProcessFrame
         */
        uint m_occurrences; 
        
        /**
         * @brief number of occurrences in the accumlated over all frames, reset on
         * Trigger reset. Only updated if/when the Trigger has an ODE Accumulator. 
         */
        uint m_occurrencesAccumulated;
        

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
         * @brief unique inference component name filter for this event
         * NULL indicates filter is disabled
         */
        std::string m_infer;
        
        /**
         * @brief unique inference component id filter for this event
         * -1 indicates not set ... updated on first use.
         */
        int m_inferId;
        
        /**
         * @brief GIE Class Id filter for this event
         */
        uint m_classId;
        
        /**
         * Mininum inference confidence to trigger an ODE occurrence [0.0..1.0]
         */
        float m_minConfidence;
        
        /**
         * Maximum inference confidence to trigger an ODE occurrence [0.0..1.0]
         */
        float m_maxConfidence;
        
        /**
         * Mininum tracker confidence to trigger an ODE occurrence [0.0..1.0]
         */
        float m_minTrackerConfidence;
        
        /**
         * Maximum tracker confidence to trigger an ODE occurrence [0.0..1.0]
         */
        float m_maxTrackerConfidence;
        
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
         * @brief process interval, default = 0
         */
        uint m_interval;
        
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
        void PreProcessFrame(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta);

        /**
         * @brief Function to post-process the frame for an Absence Event 
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData, NvDsFrameMeta* pFrameMeta);
        
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
        bool CheckForOccurrence(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData,
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
        bool CheckForOccurrence(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame for an Absence Event 
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta);

    private:
    
    };

    class TrackingOdeTrigger : public OdeTrigger
    {
    public:
    
        TrackingOdeTrigger(const char* name, const char* source, uint classId, 
            uint limit, uint maxTracePoints);
        
        ~TrackingOdeTrigger();
        
        /**
         * @brief Overrides the base Reset in order to clear m_trackedObjectsPerSource
         */
        void Reset();

    protected:

        /**
         * @brief map of tracked objects per source - Key = source Id
         */
        std::shared_ptr<TrackedObjects> m_pTrackedObjectsPerSource;
    
    };

    class CrossOdeTrigger : public TrackingOdeTrigger
    {
    public:
    
        CrossOdeTrigger(const char* name, const char* source, uint classId, 
            uint limit, uint minFrameCount, uint maxTracePoints, 
            uint testMethod, DSL_RGBA_COLOR_PTR pColor);
        
        ~CrossOdeTrigger();

        /**
         * @brief Function to check a given Object Meta data structure for to determine if the object has
         * crossed the Trigger's Area - line or line segment of a polygon.
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a Cross Accumulation Event
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData, NvDsFrameMeta* pFrameMeta);

        /**
         * @brief Gets the current max-trace-point setting for this CrossOdeTrigger.
         * @param[out] minFrameCount current min frame-count setting.
         * @param[out] maxTracePoints current max trace-point setting.
         * @param[out] testMethod one of the DSL_OBJECT_TRACE_TEST_METHOD_* constants.
         */
        void GetTestSettings(uint* minFrameCount, 
            uint* maxTracePoints, uint* testMethod);

        /**
         * @brief Sets the max-trace-point setting for this CrossOdeTrigger.
         * @param[in] minFrameCount current min frame-count setting.
         * @param[in] maxTracePoints current max trace-point setting.
         * @param[in] testMethod one of the DSL_OBJECT_TRACE_TEST_METHOD_* constants.
         */
        void SetTestSettings(uint minFrameCount,
            uint maxTracePoints, uint testMethod);
        
        /**
         * @brief Gets the current trace setting for this CrossOdeTrigger.
         * @param[out] enabled true if trace display is enabled, false otherwise.
         * @param[out] color name of the RGBA Color for the trace display.
         * @param[out] lineWidth for the trace display when enabled.
         */
        void GetViewSettings(bool* enabled, const char** color, uint* lineWidth);
        
        /**
         * @brief Gets the current trace setting for this CrossOdeTrigger.
         * @param[in] enabled true if trace display is enabled, false otherwise.
         * @param[in] pColor shared pointer to RGBA Color to use for the trace display.
         * @param[in] lineWidth for the trace display if enabled.
         */
        void SetViewSettings(bool enabled, DSL_RGBA_COLOR_PTR pColor, uint lineWidth);

        /**
         * @brief Overrides the base Reset in order to clear m_occurrencesIn and
         * m_occurrencesOut
         */
        void Reset();
            
    private:

        /**
         * @brief maximum number of trace points to use in cross detection
         */
        uint m_maxTracePoints;

        /**
         * @brief number of occurrences in the "in-direction" for the current frame, 
         * reset on exit of PostProcessFrame
         */
        uint m_occurrencesIn;

        /**
         * @brief number of occurrences in the "out-direction" for the current frame, 
         * reset on exit of PostProcessFrame
         */
        uint m_occurrencesOut;

        /**
         * @brief number of occurrences in the "in-direction" accumlated over 
         * all frames reset on Trigger reset. Only updated if/when the Trigger
         * has an ODE Accumulator. 
         */
        uint m_occurrencesInAccumulated;

        /**
         * @brief number of occurrences in the "out-direction" accumulated over, 
         * all frames reset on Trigger reset. Only updated if/when the Trigger
         * has an ODE Accumulator. 
         */
        uint m_occurrencesOutAccumulated;

        /**
         * @brief minimum number of consective frames required to trigger an event
         * on both sides of the line. 
         */
        uint m_minFrameCount;

        /**
         * @brief method to test object trace line crossing. All-points or end-points.
         */
        uint m_testMethod;
        
        /**
         * @brief true if object trace display is enabled, false otherwise.
         */
        bool m_traceEnabled;
        
        /**
         * @brief shared pointer to RGBA Color to use for the object trace display.
         */
        DSL_RGBA_COLOR_PTR m_pTraceColor;
        
        /**
         * @brief line width for the object trace in units of pixels.
         */
        uint m_traceLineWidth;
    
    };
    
    class InstanceOdeTrigger : public TrackingOdeTrigger
    {
    public:
    
        InstanceOdeTrigger(const char* name, 
            const char* source, uint classId, uint limit);
        
        ~InstanceOdeTrigger();
        
        /**
         * @brief Gets the current instance and suppression count settings for the
         * InstanceOdeTrigger.
         * @param instanceCount[out] the number of consecutive instances to trigger ODE
         * occurrence. Default = 1.
         * @param suppressionCount[out] the number of consecutive instances to suppress
         * ODE occurrence once the instance_count has been reached. Default = 0 (suppress 
         * indefinitely).
         */
        void GetCountSettings(uint* instanceCount, uint* suppressionCount);
        
        /**
         * @brief Sets the instance and suppression count settings for the
         * InstanceOdeTrigger to use.
         * @param instanceCount[in] the number of consecutive instances to trigger ODE
         * occurrence. Default = 1.
         * @param suppressionCount[in] the number of consecutive instances to suppress
         * ODE occurrence once the instance_count has been reached. Default = 0 (suppress 
         * indefinitely).
         */
        void SetCountSettings(uint instanceCount, uint suppressionCount);

        /**
         * @brief Overrides the base Reset in order to clear m_instances
         */
        void Reset();

        /**
         * @brief Function to check a given Object Meta data structure for New Instances of a Class
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
            
        /**
         * @brief Function to post process the frame and purge all tracked objects, 
         * for all sources that are not in the current frame.
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta);
            
    private:
    
        /**
         * @brief map of last Tracking Ids per unique source_id-class_id combination
         */
        std::map <std::string, uint64_t> m_instances;
        
        /**
         * @brief the number of consecutive instances to trigger an ODE occurrence
         * before suppressing.
         */
        uint m_instanceCount;
        
        /**
         * @brief the number of consecutive instances to suppressing ODE occurrence
         * once instanceCount has been reached. Default = 0 = suppress indefinitely
         */
        uint m_suppressionCount;
    
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
        bool CheckForOccurrence(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a Summation Event 
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta);

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

        bool CheckForOccurrence(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        /**
         * @brief Function to call the client provided callback to post process the frame 
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta);
        
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
    
        MinimumOdeTrigger(const char* name, const char* source, 
            uint classId, uint limit, uint minimum);
        
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
        bool CheckForOccurrence(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a Minimum ODE occurrence if the 
         * number of occurrences is less that the Trigger's Minimum value
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta);

    private:
    
        /**
         * @brief minimum object count before for ODE occurrence
         */
        uint m_minimum;
        
    };

    class MaximumOdeTrigger : public OdeTrigger
    {
    public:
    
        MaximumOdeTrigger(const char* name, const char* source, 
            uint classId, uint limit, uint maximum);
        
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
        bool CheckForOccurrence(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a Maximum ODE occurrence if the 
         * number of occurrences is greater that the Trigger's Maximum value
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData,  
            NvDsFrameMeta* pFrameMeta);

    private:
    
        /**
         * @brief maximum object count before for ODE occurrence
         */
        uint m_maximum;
        
    };

    class PersistenceOdeTrigger : public TrackingOdeTrigger
    {
    public:
    
        PersistenceOdeTrigger(const char* name, const char* source, uint classId, 
            uint limit, uint minimum, uint maximum);
        
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
        bool CheckForOccurrence(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData, 
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
            std::vector<NvDsDisplayMeta*>& displayMetaData,  
            NvDsFrameMeta* pFrameMeta);

    private:

        /**
         * @brief minimum duration of object persistence - 0 = no minimum.
         */
        double m_minimumMs;
    
        /**
         * @brief maximum duration of object persistence - 0 = no maximum
         */
        double m_maximumMs;
    };

    class CountOdeTrigger : public OdeTrigger
    {
    public:
    
        CountOdeTrigger(const char* name, 
            const char* source, uint classId, uint limit, 
            uint minimum, uint maximum);
        
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
        bool CheckForOccurrence(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a Count ODE occurrence if the 
         * number of occurrences is with in the Trigger's minimum and maximum settings
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData,  
            NvDsFrameMeta* pFrameMeta);

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
    
        SmallestOdeTrigger(const char* name, const char* source, 
            uint classId, uint limit);
        
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
        bool CheckForOccurrence(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a Smallest Object Event 
         * if at least one object is found
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData,  
            NvDsFrameMeta* pFrameMeta);

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
    
        LargestOdeTrigger(const char* name, const char* source, 
            uint classId, uint limit);
        
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
        bool CheckForOccurrence(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a Largest Object Event 
         * if at least one object is found
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData,  
            NvDsFrameMeta* pFrameMeta);

    private:
    
        /**
         * @brief list of pointers to NvDsObjectMeta data
         * Each object occurrence that matches the min criteria will be added
         * to list to be checked for Largest object on PostProcessFrame
         */ 
        std::vector<NvDsObjectMeta*> m_occurrenceMetaList;
    
    };

    class LatestOdeTrigger : public TrackingOdeTrigger
    {
    public:
    
        LatestOdeTrigger(const char* name, const char* source, 
            uint classId, uint limit);
        
        ~LatestOdeTrigger();

        /**
         * @brief Function to check a given Object Meta data structure for Object occurrence
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame 
         * Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame 
         * that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a Newest Object Event 
         * if at least one object is found
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData,  
            NvDsFrameMeta* pFrameMeta);

    private:
    
        /**
         * @brief pointer to the Latest - least Persistent - object in the current frame
         */
        NvDsObjectMeta* m_pLatestObjectMeta;
        
        /**
         * @brief Tracked time for the m_pLatestObjectMeta
         */
        double m_latestTrackedTimeMs;
    };

    class EarliestOdeTrigger : public TrackingOdeTrigger
    {
    public:
    
        EarliestOdeTrigger(const char* name, const char* source, 
            uint classId, uint limit);
        
        ~EarliestOdeTrigger();

        /**
         * @brief Function to check a given Object Meta data structure for Object occurrence
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame 
         * Meta - that holds the Object Meta
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - the frame 
         * that holds the Object Meta
         * @param[in] pObjectMeta pointer to a NvDsObjectMeta data to check
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a Oldest Object Event 
         * if at least one object is found
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData,  
            NvDsFrameMeta* pFrameMeta);

    private:
    
        /**
         * @brief pointer to the Earliest - most Persistent - object in the current frame
         */
        NvDsObjectMeta* m_pEarliestObjectMeta;
        
        /**
         * @brief Tracked time for the m_pOldestObjectMeta
         */
        double m_earliestTrackedTimeMs;
    };
    
    class NewLowOdeTrigger : public OdeTrigger
    {
    public:
    
        NewLowOdeTrigger(const char* name, 
            const char* source, uint classId, uint limit, uint preset);
        
        ~NewLowOdeTrigger();

        /**
         * @brief Overrides the base Reset to reset the m_currentLow to m_preset
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
        bool CheckForOccurrence(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a New Low Count Event 
         * if the current Frame's object count is less than the current low
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta);

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
         * @brief Overrides the base Reset to reset the m_currentHigh to m_preset
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
        bool CheckForOccurrence(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a New High Count Event 
         * if the current Frame's object count is greater than the current high
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrame(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta);

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
        
        
        /**
         * @brief Accumlative count of new high events accross all frames.
         */
        uint64_t m_occurrencesNewHighAccumulated;
    
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
        bool CheckForOccurrence(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Function to post process the frame and generate a Distance Event 
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        virtual uint PostProcessFrame(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta);

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
            std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta) = 0;

        /**
         * @brief Function to post process the frame and generate a Distance Event - Class A/B
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        virtual uint PostProcessFrameAB(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta) = 0;

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
            std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta);

        /**
         * @brief Function to post process the frame and generate a Distance Event - Class A/B
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrameAB(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta);

    
        /**
         * @brief Calculates the distance between two objects based on the current
         * m_bboxTestPoint setting. Either point-to-point or edge-to-edge
         * @param pObjectMetaA[in] pointer to Object A's meta data with location and dimension
         * @param pObjectMetaB[in] pointer to Object B's meta data with location and dimension
         * @return true if the objects are within minimum or beyond the maximum distance
         * as mesured by the DSL_DISTANCE_METHOD
         */
        bool CheckDistance(NvDsObjectMeta* pObjectMetaA, 
            NvDsObjectMeta* pObjectMetaB);
    
        
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
            std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta);
    
        /**
         * @brief Function to post process the frame and generate an Intersection Event - Class A/B testing
         * @param[in] pBuffer pointer to batched stream buffer - that holds the Frame Meta
         * @param[in] pFrameMeta Frame meta data to post process.
         * @return the number of ODE Occurrences triggered on post process
         */
        uint PostProcessFrameAB(GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta);
    };

}

#endif // _DSL_ODE_TRIGGER_H
