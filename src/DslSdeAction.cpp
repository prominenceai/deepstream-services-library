
/*
The MIT License

Copyright (c) 2024, Prominence AI, Inc.

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

#include "DslServices.h"
#include "DslSdeTrigger.h"
#include "DslSdeAction.h"

namespace DSL
{
    SdeAction::SdeAction(const char* name)
        : DeBase(name)
    {
        LOG_FUNC();
    }

    SdeAction::~SdeAction()
    {
        LOG_FUNC();
    }
    
    std::string SdeAction::Ntp2Str(uint64_t ntp)
    {
        time_t secs = round(ntp/1000000000);
        time_t usecs = ntp%1000000000;  // gives us fraction of seconds
        usecs *= 1000000; // multiply by 1e6
        usecs >>= 32; // and divide by 2^32
        
        struct tm currentTm;
        localtime_r(&secs, &currentTm);        
        
        char dateTime[65] = {0};
        char dateTimeUsec[85];
        strftime(dateTime, sizeof(dateTime), "%Y-%m-%d %H:%M:%S", &currentTm);
        snprintf(dateTimeUsec, sizeof(dateTimeUsec), "%s.%06ld", dateTime, usecs);

        return std::string(dateTimeUsec);
    }

    // ********************************************************************

    PrintSdeAction::PrintSdeAction(const char* name,
        bool forceFlush)
        : SdeAction(name)
        , m_forceFlush(forceFlush)
        , m_flushThreadFunctionId(0)
    {
        LOG_FUNC();
    }

    PrintSdeAction::~PrintSdeAction()
    {
        LOG_FUNC();

        if (m_flushThreadFunctionId)
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_ostreamMutex);
            g_source_remove(m_flushThreadFunctionId);
        }
    }

    void PrintSdeAction::HandleOccurrence(DSL_BASE_PTR pSdeTrigger, 
        GstBuffer* pBuffer, NvDsAudioFrameMeta* pFrameMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (!m_enabled)
        {
            return;
        }
        DSL_SDE_TRIGGER_PTR pTrigger = 
            std::dynamic_pointer_cast<SdeTrigger>(pSdeTrigger);
        
        std::cout << "Trigger Name        : " << pTrigger->GetName() << "\n";
        std::cout << "  Unique SDE Id     : " << pTrigger->s_eventCount << "\n";
        std::cout << "  NTP Timestamp     : " << Ntp2Str(pFrameMeta->buf_pts) << "\n";
        std::cout << "  Source Data       : ------------------------" << "\n";
        if (pFrameMeta->bInferDone)
        {
            std::cout << "    Inference       : Yes\n";
        }
        else
        {
            std::cout << "    Inference       : No\n";
        }
        std::cout << "    Source Id       : " << int_to_hex(pFrameMeta->source_id) << "\n";
        std::cout << "    Batch Id        : " << pFrameMeta->batch_id << "\n";
        std::cout << "    Pad Index       : " << pFrameMeta->pad_index << "\n";
        std::cout << "    Frame           : " << pFrameMeta->frame_num << "\n";
        std::cout << "    Samples/Frame   : " << pFrameMeta->num_samples_per_frame << "\n";
        std::cout << "    Sample Rate     : " << pFrameMeta->sample_rate << "\n";
        std::cout << "    Channels        : " << pFrameMeta->num_channels << "\n";

        std::cout << "  Sound Data        : ------------------------" << "\n";
        std::cout << "    Class Id        : " << pFrameMeta->class_id << "\n";
        std::cout << "    Class Label     : " << pFrameMeta->class_label << "\n";
        std::cout << "    Confidence      : " << pFrameMeta->confidence << "\n";

        std::cout << "  Criteria          : ------------------------" << "\n";
        std::cout << "    Class Id        : " << pTrigger->m_classId << "\n";
        std::cout << "    Min Infer Conf  : " << pTrigger->m_minConfidence << "\n";

        // If we're force flushing the stream and the flush
        // handler is not currently added to the idle thread
        if (m_forceFlush and !m_flushThreadFunctionId)
        {
            m_flushThreadFunctionId = g_idle_add(PrintActionFlush, this);
        }
    }

    bool PrintSdeAction::Flush()
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_ostreamMutex);
        
        std::cout << std::flush;
        
        // end the thread
        m_flushThreadFunctionId = 0;
        return false;
    }

    static gboolean PrintActionFlush(gpointer pAction)
    {
        return static_cast<PrintSdeAction*>(pAction)->Flush();
    }

}
