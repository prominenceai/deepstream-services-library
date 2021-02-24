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

#ifndef _DSL_SOURCE_METER_H
#define _DSL_SOURCE_METER_H

#include "Dsl.h"

namespace DSL
{
    #define DSL_SOURCE_METER_PTR std::shared_ptr<SourceMeter>
    #define DSL_SOURCE_METER_NEW(name) \
        std::shared_ptr<SourceMeter>(new SourceMeter(name))

    /**
     * @class SourceMeter
     * @brief Implements a Meter to measure FPS over two seperate epics, one session, the other interval.
     */
    class SourceMeter
    {
    public:
        
        /**
         * @brief ctor for the Source Meter
         * @param sourceId unique Id of the Source being metered.
         */
        SourceMeter(uint sourceId)
            : m_sourceId(sourceId)
            , m_timeStamp{0}
            , m_sessionStartTime{0}
            , m_intervalStartTime{0}
            , m_sessionFrameCount(0)
            , m_intervalFrameCount(0)
            {};

        /**
         * @brief Updates the Timestamp for the Source meter
         */
        void Timestamp()
        {
            gettimeofday(&m_timeStamp, NULL);
            
            // one-time initialization of start times after creation.
            if (!m_sessionStartTime.tv_sec)
            {
                m_sessionStartTime = m_timeStamp;
                m_intervalStartTime = m_timeStamp;
            }
        }
        
        /**
         * @brief Increments both frame counters. must be called on each buffer
         */
        void IncrementFrameCounts()
        {
            m_sessionFrameCount++;
            m_intervalFrameCount++;
        }
        
        /**
         * @brief Resets the Session parameters only
         */
        void SessionReset()
        {
            m_sessionStartTime = m_timeStamp;
            m_sessionFrameCount = 0;
        };
        
        /**
         * @brief Resets the Interval parameters only
         */
        void IntervalReset()
        {
            m_intervalStartTime = m_timeStamp;
            m_intervalFrameCount = 0;
        };
        
        /**
         * @brief Calculates the Average frames-per-second over a full session
         * @return Average Session FPS
         */
        double GetSessionFpsAvg()
        {
            if (!m_sessionFrameCount)
            {
                return 0;
            }
            // convert both secconds and micro-seconds to milli-seconds and add to greate single number
            uint64_t sessionTime =             
                (uint64_t)(m_timeStamp.tv_sec*1000 + m_timeStamp.tv_usec/1000) - 
                (uint64_t)(m_sessionStartTime.tv_sec*1000 + m_sessionStartTime.tv_usec/1000);

            double sessionFpsAvg = (double)m_sessionFrameCount / ((double)sessionTime/1000);        
            
            LOG_INFO("Source '" << m_sourceId << "' session FPS avg = " << sessionFpsAvg);
            return sessionFpsAvg;
        }
        
        /**
         * @brief Calculates the Average frames-per-second over a single interval
         * @return Average interval FPS
         */
        double GetIntervalFpsAvg()
        {
            if (!m_intervalFrameCount)
            {
                return 0;
            }
            uint64_t intervalTime =             
                (uint64_t)(m_timeStamp.tv_sec*1000 + m_timeStamp.tv_usec/1000) - 
                (uint64_t)(m_intervalStartTime.tv_sec*1000 + m_intervalStartTime.tv_usec/1000);

            double intervalFpsAvg = (double)m_intervalFrameCount / ((double)intervalTime/1000);

            LOG_INFO("Source '" << m_sourceId << "' interval FPS avg = " << intervalFpsAvg);
            return intervalFpsAvg;
        }
    
    private:
    
        /**
         * @brief unique source Id for the soure being metered
         */
        int m_sourceId;
        
        /**
         * @brief timestamp updated on each buffer with frame meta for the unique source
         */
        struct timeval m_timeStamp;
        
        /**
         * @brief timestamp for the start of the current session
         */
        struct timeval m_sessionStartTime;

        /**
         * @brief timestamp for the start of the current session
         */
        struct timeval m_intervalStartTime;
        
        /**
         * @brief Frame count since the start of the current session
         */
        uint m_intervalFrameCount;

        /**
         * @brief Frame count since the start of the current interval
         */
        uint m_sessionFrameCount;
    };
}
#endif // _DSL_SOURCE_METER_H
