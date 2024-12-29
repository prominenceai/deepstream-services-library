/*
The MIT License

Copyright (c) 2019-2024 Prominence AI, Inc.

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

#ifndef _DSL_DE_TRIGGER_BASE_H
#define _DSL_DE_TRIGGER_BASE_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslDeBase.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_DE_TRIGGER_PTR std::shared_ptr<DeTriggerBase>

    /**
     * @class DeTriggerBase
     * @brief Implements a super/abstract class for all DE and SDE Triggers
     */
    class DeTriggerBase : public DeBase
    {
    public: 
    
        DeTriggerBase(const char* name, const char* source, uint classId, uint limit);

        ~DeTriggerBase();

        /**
         * @brief Adds an DE Action as a child to this DeTriggerBase
         * @param[in] pChild pointer to DE Action to add
         * @return true if successful, false otherwise
         */
        bool AddAction(DSL_BASE_PTR pChild);
        
        /**
         * @brief Removes a child DE Action from this DeTriggerBase
         * @param[in] pChild pointer to DE Action to remove
         * @return true if successful, false otherwise
         */
        bool RemoveAction(DSL_BASE_PTR pChild);
        
        /**
         * @brief Removes all child DE Actions from this DeTriggerBase
         */
        void RemoveAllActions();

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
            dsl_trigger_limit_state_change_listener_cb listener, void* clientData);

        /**
         * @brief Removes a "limit event listener" function previously added
         * with a call to AddLimitStateChangeListener.
         * @return true if the listener function was successfully removed, false otherwise.
         */
        bool RemoveLimitStateChangeListener(
            dsl_trigger_limit_state_change_listener_cb listener);
        
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
         * @brief Gets the trigger event limit for this DE Trigger 
         * @return the current frame limit value
         */
        uint GetEventLimit();
        
        /**
         * @brief Sets the event limit for Object detection.
         * @param[in] limit new trigger frame limit value to use.
         */
        void SetEventLimit(uint limit);
        
        /**
         * @brief Gets the trigger frame limit for this DE Trigger.
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
         * @brief Common function to check if a Frame's source id meets the 
         * criteria for DE occurrence.
         * @param sourceId a Frame's Source Id to check against the trigger's 
         * source filter if set.
         * @return true if Source Id criteria is met, false otherwise
         */
        bool CheckForSourceId(int sourceId);
        
        /**
         * @brief Common function to check if an Objects's infer component id 
         * meets the criteria for DE occurrence.
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
         * If enabled, the bInferDone flag must be set to trigger DE Occurrence
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
         * @brief Index variable to incremment/assign on DE Action add.
         */
        uint m_nextActionIndex;

        /**
         * @brief Map of child DE Actions owned by this trigger
         */
        std::map <std::string, DSL_BASE_PTR> m_pActions;
        
        /**
         * @brief Map of child DE Actions indexed by their add-order for execution
         */
        std::map <uint, DSL_BASE_PTR> m_pActionsIndexed;
        
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
        DslMutex m_resetTimerMutex;

        /**
         * @brief map of all currently registered limit-state-change-listeners
         * callback functions mapped with the user provided data
         */
        std::map<dsl_trigger_limit_state_change_listener_cb, 
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

        /**
         * @brief Wide string name used for C/Python API
         */
        std::wstring m_wName;
        
        /**
         * @brief number of occurrences for the current frame, 
         * reset on exit of PostProcessFrame
         */
        uint m_occurrences; 
        
        /**
         * @brief number of occurrences in the accumlated over all frames, reset on
         * Trigger reset. Only updated if/when the Trigger has an DE Accumulator. 
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
         * Mininum inference confidence to trigger an DE occurrence [0.0..1.0]
         */
        float m_minConfidence;
        
        /**
         * Maximum inference confidence to trigger an DE occurrence [0.0..1.0]
         */
        float m_maxConfidence;
        
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
         * @brief process interval, default = 0
         */
        uint m_interval;
        
        /**
         * @brief Minimum frame count numerator to trigger an DE occurrence
         */
        uint m_minFrameCountN;

        /**
         * @brief Minimum frame count denominator to trigger an DE occurrence
         */
        uint m_minFrameCountD;
        
        /**
         * @brief if set, the Frame meta value "bInferDone" must be set
         * to trigger an occurrence
         */
        bool m_inferDoneOnly;

    };

    int TriggerResetTimeoutHandler(gpointer pTrigger);    

}

#endif // _DSL_DE_TRIGGER_BASE_H