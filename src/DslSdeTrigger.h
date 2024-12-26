/*
The MIT License

Copyright (c) 2024 Prominence AI, Inc.

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

#ifndef _DSL_SDE_TRIGGER_H
#define _DSL_SDE_TRIGGER_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslDeTriggerBase.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_SDE_TRIGGER_PTR std::shared_ptr<SdeTrigger>

    #define DSL_SDE_TRIGGER_OCCURRENCE_PTR std::shared_ptr<OccurrenceSdeTrigger>
    #define DSL_SDE_TRIGGER_OCCURRENCE_NEW(name, source, classId, limit) \
        std::shared_ptr<OccurrenceSdeTrigger>(new OccurrenceSdeTrigger(name, \
            source, classId, limit))

    #define DSL_SDE_TRIGGER_OCCURRENCE_PTR std::shared_ptr<OccurrenceSdeTrigger>
    #define DSL_SDE_TRIGGER_OCCURRENCE_NEW(name, source, classId, limit) \
        std::shared_ptr<OccurrenceSdeTrigger>(new OccurrenceSdeTrigger(name, \
            source, classId, limit))

    // *****************************************************************************

    /**
     * @class SdeTrigger
     * @brief Implements a super/abstract class for all SDE Triggers    
     */
    class SdeTrigger : public DeTriggerBase
    {
    public: 
    
        SdeTrigger(const char* name, const char* source, uint classId, uint limit);

        ~SdeTrigger();

        /**
         * @brief total count of all events
         */
        static uint64_t s_eventCount;
        
        /**
         * @brief Function to check a Audio Frame's Meta data structure for the 
         * occurence of an event and to invoke all Event Actions owned by the Trigger
         * @param[in] pBuffer pointer to the GST Buffer containing all meta
         * @param[in] pFrameMeta pointer to the containing NvDsAudioFrameMeta data
         * @return true if Occurrence, false otherwise
         */
        virtual bool CheckForOccurrence(GstBuffer* pBuffer, 
            NvDsAudioFrameMeta* pFrameMeta){return false;};

    protected:
    
        /**
         * @brief Common function to check if an Audio Frame's meta data meets the 
         * min criteria for SDE occurrence.
         * @param[in] pFrameMeta pointer to the parent NvDsFrameMeta data - 
         * the frame that holds the Object Meta
         * @return true if Min Criteria is met, false otherwise
         */
        bool CheckForMinCriteria(NvDsAudioFrameMeta* pFrameMeta);

    };

    // *****************************************************************************
    
    class OccurrenceSdeTrigger : public SdeTrigger
    {
    public:
    
        OccurrenceSdeTrigger(const char* name, const char* source, uint classId, uint limit);
        
        ~OccurrenceSdeTrigger();

        /**
         * @brief Function to check a Audio Frame's Meta data structure for the 
         * occurence of an event and to invoke all Event Actions owned by the Trigger
         * @param[in] pBuffer pointer to the GST Buffer containing all meta
         * @param[in] pFrameMeta pointer to the containing NvDsAudioFrameMeta data
         * @return true if Occurrence, false otherwise
         */
        bool CheckForOccurrence(GstBuffer* pBuffer, 
            NvDsAudioFrameMeta* pFrameMeta);

    private:
    
    };

}

#endif // _DSL_SDE_TRIGGER_H
