
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
        std::cout << "    Sample Rate     : " << pFrameMeta->sample_rate << "\n";
        std::cout << "    Samples/Frame   : " << pFrameMeta->num_samples_per_frame << "\n";
        std::cout << "    Channels        : " << pFrameMeta->num_channels << "\n";

        std::cout << "  Sound Data        : ------------------------" << "\n";
        std::cout << "    Class Id        : " << pFrameMeta->class_id << "\n";
        std::cout << "    Class Label     : " << pFrameMeta->class_label << "\n";
        std::cout << "    Confidence      : " << pFrameMeta->confidence << "\n";

        std::cout << "  Criteria          : ------------------------" << "\n";
        std::cout << "    Source Id       : " << int_to_hex(pTrigger->m_sourceId) << "\n";
        std::cout << "    Class Id        : " << pTrigger->m_classId << "\n";
        std::cout << "    Min Infer Conf  : " << pTrigger->m_minConfidence << "\n";
        std::cout << "    Max Infer Conf  : " << pTrigger->m_maxConfidence << "\n";
        std::cout << "    Interval        : " << pTrigger->m_interval << "\n";

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

// ********************************************************************

    MonitorSdeAction::MonitorSdeAction(const char* name, 
        dsl_sde_monitor_occurrence_cb clientMonitor, void* clientData)
        : SdeAction(name)
        , m_clientMonitor(clientMonitor)
        , m_clientData(clientData)
    {
        LOG_FUNC();
    }

    MonitorSdeAction::~MonitorSdeAction()
    {
        LOG_FUNC();
    }
    
    void MonitorSdeAction::HandleOccurrence(DSL_BASE_PTR pBase, 
        GstBuffer* pBuffer, NvDsAudioFrameMeta* pFrameMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (!m_enabled)
        {
            return;
        }
        try
        {
            DSL_SDE_TRIGGER_PTR pTrigger 
                = std::dynamic_pointer_cast<SdeTrigger>(pBase);
                
            dsl_sde_occurrence_info info{0};
            
            // convert the Trigger Name to wchar string type (client format)
            std::wstring wstrTriggerName(pTrigger->GetName().begin(), 
                pTrigger->GetName().end());
            info.trigger_name = wstrTriggerName.c_str();
            info.unique_sde_id = pTrigger->s_eventCount;
            info.ntp_timestamp = pFrameMeta->ntp_timestamp;
            info.source_info.inference_done = pFrameMeta->bInferDone;
            info.source_info.source_id = pFrameMeta->source_id;
            info.source_info.batch_id = pFrameMeta->batch_id;
            info.source_info.pad_index = pFrameMeta->pad_index;
            info.source_info.frame_num = pFrameMeta->frame_num;
            info.source_info.sample_rate = pFrameMeta->sample_rate;
            info.source_info.num_samples_per_frame = pFrameMeta->num_samples_per_frame;
            info.source_info.num_channels = pFrameMeta->num_channels;
            
            // Automatic varaibles needs to be valid for call to the client callback
            // Create here at higher scope - in case it is used for Object metadata.
            std::wstring wstrLabel;
            std::wstring wstrClassifierLabels;
            
            info.sound_info.class_id = pFrameMeta->class_id;

            std::string strLabel(pFrameMeta->class_label);
            wstrLabel.assign(strLabel.begin(), strLabel.end());
            info.sound_info.label = wstrLabel.c_str();

            info.sound_info.inference_confidence =  pFrameMeta->confidence;

                // look for classifier meta to find labels like licence plate numbers
            // if (pObjectMeta->classifier_meta_list)
            // {
            //     std::ostringstream labelStream;
                
            //     for (NvDsClassifierMetaList* pClassifierMetaList = 
            //             pObjectMeta->classifier_meta_list; pClassifierMetaList; 
            //                 pClassifierMetaList = pClassifierMetaList->next)
            //     {
            //         NvDsClassifierMeta* pClassifierMeta = 
            //             (NvDsClassifierMeta*)(pClassifierMetaList->data);
            //         if (pClassifierMeta != NULL)
            //         {
            //             for (NvDsLabelInfoList* pLabelInfoList = 
            //                     pClassifierMeta->label_info_list; pLabelInfoList; 
            //                         pLabelInfoList = pLabelInfoList->next)
            //             {
            //                 NvDsLabelInfo* pLabelInfo = 
            //                     (NvDsLabelInfo*)(pLabelInfoList->data);
            //                 if(pLabelInfo != NULL)
            //                 {
            //                     if (labelStream.str().size())
            //                     {
            //                         labelStream << " ";
            //                     }
            //                     labelStream << pLabelInfo->result_label;
            //                 }
            //             }
            //         }
            //     }
            //     std::string classifierlabels(labelStream.str());
            //     wstrClassifierLabels.assign(classifierlabels.begin(), 
            //         classifierlabels.end());
            //     info.object_info.classiferLabels = wstrClassifierLabels.c_str();
            // }
            
            // Trigger criteria set for this SDE occurrence.
            info.criteria_info.source_id = pTrigger->m_sourceId;
            info.criteria_info.class_id = pTrigger->m_classId;
            info.criteria_info.min_inference_confidence = pTrigger->m_minConfidence;
            info.criteria_info.max_inference_confidence = pTrigger->m_maxConfidence;
            info.criteria_info.interval = pTrigger->m_interval;
            
            // Call the Client's monitor callback with the info and client-data
            m_clientMonitor(&info, m_clientData);
        }
        catch(...)
        {
            LOG_ERROR("Monitor SDE Action '" << GetName() 
                << "' threw exception calling client callback");
        }
    }

}
