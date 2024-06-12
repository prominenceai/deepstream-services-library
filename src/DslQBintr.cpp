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

#include "DslQBintr.h"

namespace DSL
{
    QBintr::QBintr(const char* name)
        : Bintr(name)
    { 
        LOG_FUNC();

        // Persist the wstring name to pass to the client callbacks on queue
        // overrun/underrun
        m_wstrName.assign(m_name.begin(), m_name.end());

        // Create the Queue element for the QBintr
        m_pQueue = DSL_ELEMENT_NEW("queue", name);

        // Get all default values
        m_pQueue->GetAttribute("leaky", &m_leaky);
        m_pQueue->GetAttribute("max-size-buffers", &m_maxSizeBuffers);
        m_pQueue->GetAttribute("max-size-bytes", &m_maxSizeBytes);
        m_pQueue->GetAttribute("max-size-time", &m_maxSizeTime);
        m_pQueue->GetAttribute("min-threshold-buffers", &m_minThresholdBuffers);
        m_pQueue->GetAttribute("min-threshold-bytes", &m_minThresholdBytes);
        m_pQueue->GetAttribute("min-threshold-time", &m_minThresholdTime);

        // Connect the local static overrrun callback so that queue overruns
        // can be logged as warning messages by default.
        g_signal_connect(m_pQueue->GetGObject(), "overrun",
            G_CALLBACK(QueueOverrunCB), this);

        // Underrun signal occur often (queue level = 0), so we don't 
        // connect the underrun signal until requested by the client. 

        // and the Queue element as a child of this QBintr
        AddChild(m_pQueue);
    }


    QBintr::~QBintr()
    { 
        LOG_FUNC(); 
    }

    uint64_t QBintr::GetQueueCurrentLevel(uint unit)
    {
        LOG_FUNC(); 

        uint64_t currentLevel(0);
        if (unit == DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS)
        {
            uint currentLevelBuffers;
            m_pQueue->GetAttribute("current-level-buffers", &currentLevelBuffers);
            currentLevel = currentLevelBuffers;
        }
        else if (unit == DSL_COMPONENT_QUEUE_UNIT_OF_BYTES)
        {
            uint currentLevelBytes;
            m_pQueue->GetAttribute("current-level-bytes", &currentLevelBytes);
            currentLevel = currentLevelBytes;
        }
        else
        {
            m_pQueue->GetAttribute("current-level-time", &currentLevel);
        }
        return currentLevel;
    }

    void QBintr::PrintQueueCurrentLevel(uint unit)
    {
        LOG_FUNC(); 

        if (unit == DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS)
        {
            uint currentLevelBuffers(0);
            m_pQueue->GetAttribute("current-level-buffers", &currentLevelBuffers);
            std::cout << "'current-level-buffers' = " << currentLevelBuffers 
                << "/" << m_maxSizeBuffers << " for QBintr '" << GetName() 
                << "'" << std::endl;
        }
        else if (unit == DSL_COMPONENT_QUEUE_UNIT_OF_BYTES)
        {
            uint currentLevelBytes(0);
            m_pQueue->GetAttribute("current-level-bytes", &currentLevelBytes);
            std::cout << "'current-level-bytes' = " << currentLevelBytes 
                << "/" << m_maxSizeBytes << " for QBintr '" << GetName() 
                << "'" << std::endl;
        }
        else
        {
            uint64_t currentLevelTime(0);
            m_pQueue->GetAttribute("current-level-time", &currentLevelTime);
            std::cout << "'current-level-time' = " << currentLevelTime 
                << "/" << m_maxSizeTime << " for QBintr '" << GetName() 
                << "'" << std::endl;
        }
    }

    uint QBintr::GetQueueLeaky()
    {
        LOG_FUNC(); 

        m_pQueue->GetAttribute("leaky", &m_leaky);
        return m_leaky;
    }

    bool QBintr::SetQueueLeaky(uint leaky)
    {
        LOG_FUNC(); 

        if (IsLinked())
        {
            LOG_ERROR("Unable to set leaky for QBintr '" << GetName() 
                << "' as it's currently linked");
            return false;
        }
        m_leaky = leaky;
        m_pQueue->SetAttribute("leaky", m_leaky);
        return true;
    }

    uint64_t QBintr::GetQueueMaxSize(uint unit)
    {
        LOG_FUNC(); 

        uint64_t maxSize(0);
        if (unit == DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS)
        {
            m_pQueue->GetAttribute("max-size-buffers", &m_maxSizeBuffers);
            maxSize = m_maxSizeBuffers;
        }
        else if (unit == DSL_COMPONENT_QUEUE_UNIT_OF_BYTES)
        {
            m_pQueue->GetAttribute("max-size-bytes", &m_maxSizeBytes);
            maxSize = m_maxSizeBytes;
        }
        else
        {
            m_pQueue->GetAttribute("max-size-time", &m_maxSizeTime);
            maxSize = m_maxSizeTime;
        }
        return maxSize;
    }

    bool QBintr::SetQueueMaxSize(uint unit, uint64_t maxSize)
    {
        LOG_FUNC(); 

        if (IsLinked())
        {
            LOG_ERROR("Unable to set max-size for QBintr '" << GetName() 
                << "' as it's currently linked");
            return false;
        }

        if (unit == DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS)
        {
            m_maxSizeBuffers = (uint)maxSize;
            m_pQueue->SetAttribute("max-size-buffers", m_maxSizeBuffers);
        }
        else if (unit == DSL_COMPONENT_QUEUE_UNIT_OF_BYTES)
        {
            m_maxSizeBytes = (uint)maxSize;
            m_pQueue->SetAttribute("max-size-bytes", m_maxSizeBytes);
        }
        else
        {
            m_maxSizeTime = maxSize;
            m_pQueue->SetAttribute("max-size-time", m_maxSizeTime);
        }
        return true;
    }

    uint64_t QBintr::GetQueueMinThreshold(uint unit)
    {
        LOG_FUNC(); 

        uint64_t minThreshold(0);
        if (unit == DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS)
        {
            m_pQueue->GetAttribute("min-threshold-buffers", &m_minThresholdBuffers);
            minThreshold = m_minThresholdBuffers;
        }
        else if (unit == DSL_COMPONENT_QUEUE_UNIT_OF_BYTES)
        {
            m_pQueue->GetAttribute("min-threshold-bytes", &m_minThresholdBytes);
            minThreshold = m_minThresholdBytes;
        }
        else
        {
            m_pQueue->GetAttribute("min-threshold-time", &m_minThresholdTime);
            minThreshold = m_minThresholdTime;
        }
        return minThreshold;
    }

    bool QBintr::SetQueueMinThreshold(uint unit, uint64_t minThreshold)
    {
        LOG_FUNC(); 

        if (IsLinked())
        {
            LOG_ERROR("Unable to set min-threshold for QBintr '" << GetName() 
                << "' as it's currently linked");
            return false;
        }

        if (unit == DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS)
        {
            m_minThresholdBuffers = (uint)minThreshold;
            m_pQueue->SetAttribute("min-threshold-buffers", m_minThresholdBuffers);
        }
        else if (unit == DSL_COMPONENT_QUEUE_UNIT_OF_BYTES)
        {
            m_minThresholdBytes = (uint)minThreshold;
            m_pQueue->SetAttribute("min-threshold-bytes", m_minThresholdBytes);
        }
        else
        {
            m_minThresholdTime = minThreshold;
            m_pQueue->SetAttribute("min-threshold-time", m_minThresholdTime);
        }
        return true;
    }

    bool QBintr::AddQueueOverrunListener(dsl_component_queue_overrun_listener_cb listener, 
        void* clientData)
    {
        LOG_FUNC();
        
        if (m_queueOverrunListeners.find(listener) != m_queueOverrunListeners.end())
        {   
            LOG_ERROR("Queue overrun listener is not unique");
            return false;
        }
        m_queueOverrunListeners[listener] = clientData;
        
        return true;
    }

    bool QBintr::RemoveQueueOverrunListener(dsl_component_queue_overrun_listener_cb listener)
    {
        LOG_FUNC();
        
        if (m_queueOverrunListeners.find(listener) == m_queueOverrunListeners.end())
        {   
            LOG_ERROR("Queue overrun listener was not found");
            return false;
        }
        m_queueOverrunListeners.erase(listener);
        
        return true;
    }
        
    bool QBintr::AddQueueUnderrunListener(
            dsl_component_queue_underrun_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        
        if (m_queueUnderrunListeners.find(listener) != m_queueUnderrunListeners.end())
        {   
            LOG_ERROR("Queue underrun listener is not unique");
            return false;
        }

        // connect the callback to the underrun signal on firt listener. 
        if (m_queueUnderrunListeners.empty())
        {
            g_signal_connect(m_pQueue->GetGObject(), "underrun",
                G_CALLBACK(QueueUnderrunCB), this);            
        }
        m_queueUnderrunListeners[listener] = clientData;
        
        return true;
    }

    bool QBintr::RemoveQueueUnderrunListener(
            dsl_component_queue_underrun_listener_cb listener)
    {
        LOG_FUNC();
        
        if (m_queueUnderrunListeners.find(listener) == m_queueUnderrunListeners.end())
        {   
            LOG_ERROR("Queue underrun listener was not found");
            return false;
        }
        m_queueUnderrunListeners.erase(listener);
        
        return true;
    }

    void QBintr::HandleQueueOverrun() 
    {
        LOG_FUNC();

        LOG_WARN("Queue overrun signal received for Component " 
            << GetName() << "'");

        // iterate through the map of queue-overrun-listeners calling each
        for(auto const& imap: m_queueOverrunListeners)
        {
            try
            {
                imap.first(m_wstrName.c_str(), imap.second);
            }
            catch(...)
            {
                LOG_ERROR("Component '" << GetName() 
                    << "' threw exception calling Client queue-overrun-listener");
            }
        }
    }
    
    void QBintr::HandleQueueUnderrun() 
    {
        LOG_FUNC();

        LOG_DEBUG("Queue underrun signal received for Component " 
            << GetName() << "'");
            
        // iterate through the map of queue-underrun-listeners calling each
        for(auto const& imap: m_queueUnderrunListeners)
        {
            try
            {
                imap.first(m_wstrName.c_str(), imap.second);
            }
            catch(...)
            {
                LOG_ERROR("Component '" << GetName() 
                    << "' threw exception calling Client queue-underrun-listener");
            }
        }
    }
    
}
