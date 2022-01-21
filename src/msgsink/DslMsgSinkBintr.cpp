/*
The MIT License

Copyright (c) 2021, Prominence AI, Inc.

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

#include "Dsl.h"
#include "DslApi.h"
#include "DslMsgSinkBintr.h"

namespace DSL
{
    MsgSinkBintr::MsgSinkBintr(const char* name, const char* converterConfigFile, 
        uint payloadType, const char* brokerConfigFile, const char* protoLib, 
        const char* connectionString, const char* topic)
        : SinkBintr(name, true, false) // used for fake sink only
        , m_converterConfigFile(converterConfigFile)
        , m_payloadType(payloadType)
        , m_brokerConfigFile(brokerConfigFile)
        , m_connectionString(connectionString)
        , m_protoLib(protoLib)
        , m_topic(topic)
        , m_qos(false)
{
        LOG_FUNC();
        
        m_pTee = DSL_ELEMENT_NEW(NVDS_ELEM_TEE, "sink-bin-msg-tee");
        m_pMsgConverterQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "sink-bin-msgconverter-queue");
        m_pMsgConverter = DSL_ELEMENT_NEW(NVDS_ELEM_MSG_CONV, "sink-bin-msgconverter");
        m_pMsgBroker = DSL_ELEMENT_NEW(NVDS_ELEM_MSG_BROKER, "sink-bin-msgbroker");
        m_pFakeSinkQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "sink-bin-msg-fake-queue");
        m_pFakeSink = DSL_ELEMENT_NEW(NVDS_ELEM_SINK_FAKESINK, "sink-bin-msg-fake");
        
        m_pMsgConverter->SetAttribute("config", m_converterConfigFile.c_str());
        m_pMsgConverter->SetAttribute("payload-type", m_payloadType);

        m_pMsgBroker->SetAttribute("proto-lib", m_protoLib.c_str());
        m_pMsgBroker->SetAttribute("conn-str", m_connectionString.c_str());
        m_pMsgBroker->SetAttribute("sync", false);

        m_pFakeSink->SetAttribute("enable-last-sample", false);
        m_pFakeSink->SetAttribute("max-lateness", -1);
        m_pFakeSink->SetAttribute("sync", m_sync);
        m_pFakeSink->SetAttribute("async", m_async);
        m_pFakeSink->SetAttribute("qos", m_qos);
        
        if (brokerConfigFile)
        {
            m_pMsgBroker->SetAttribute("config", m_brokerConfigFile.c_str());
        }
        if (m_topic.size())
        {
            m_pMsgBroker->SetAttribute("topic", m_topic.c_str());
        }
        
        AddChild(m_pTee);
        AddChild(m_pMsgConverterQueue);
        AddChild(m_pMsgConverter);
        AddChild(m_pMsgBroker);
        AddChild(m_pFakeSinkQueue);
        AddChild(m_pFakeSink);
    }

    MsgSinkBintr::~MsgSinkBintr()
    {
        LOG_FUNC();
    
        if (IsLinked())
        {    
            UnlinkAll();
        }
    }

    bool MsgSinkBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("MsgSinkBintr '" << GetName() << "' is already linked");
            return false;
        }
        if (!m_pQueue->LinkToSink(m_pTee) or
            !m_pMsgConverterQueue->LinkToSourceTee(m_pTee, "src_%u") or
            !m_pMsgConverterQueue->LinkToSink(m_pMsgConverter) or
            !m_pMsgConverter->LinkToSink(m_pMsgBroker) or
            !m_pFakeSinkQueue->LinkToSourceTee(m_pTee, "src_%u") or
            !m_pFakeSinkQueue->LinkToSink(m_pFakeSink))
        {
            return false;
        }
        m_isLinked = true;
        return true;
    }
    
    void MsgSinkBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("MsgSinkBintr '" << GetName() << "' is not linked");
            return;
        }
        m_pFakeSinkQueue->UnlinkFromSink();
        m_pFakeSinkQueue->UnlinkFromSourceTee();
        m_pMsgConverter->UnlinkFromSink();
        m_pMsgConverterQueue->UnlinkFromSink();
        m_pMsgConverterQueue->UnlinkFromSourceTee();
        m_pQueue->UnlinkFromSink();
        m_isLinked = false;
    }
    
    bool MsgSinkBintr::SetSyncSettings(bool sync, bool async)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set Sync/Async Settings for MsgSinkBintr '" << GetName() 
                << "' as it's currently linked");
            return false;
        }
        m_sync = sync;
        m_async = async;

        m_pMsgBroker->SetAttribute("sink", m_sync);
        return true;
    }

    void MsgSinkBintr::GetConverterSettings(const char** converterConfigFile,
        uint* payloadType)
    {
        LOG_FUNC();
        
        *converterConfigFile = m_converterConfigFile.c_str();
        *payloadType = m_payloadType;
    }

    bool MsgSinkBintr::SetConverterSettings(const char* converterConfigFile,
        uint payloadType)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set Message Conveter Settings for MsgSinkBintr '" << GetName() 
                << "' as it's currently linked");
            return false;
        }
        m_converterConfigFile.assign(converterConfigFile);
        m_payloadType = payloadType;

        m_pMsgConverter->SetAttribute("config", m_converterConfigFile.c_str());
        m_pMsgConverter->SetAttribute("payload-type", m_payloadType);
        return true;
    }

    void MsgSinkBintr::GetBrokerSettings(const char** brokerConfigFile,
        const char** connectionString, const char** topic)
    {
        LOG_FUNC();
        
        *brokerConfigFile = m_brokerConfigFile.c_str();
        *connectionString = m_connectionString.c_str();
        *topic = m_topic.c_str();
    }

    bool MsgSinkBintr::SetBrokerSettings(const char* brokerConfigFile,
        const char* connectionString, const char* topic)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set Message Broker Settings for MsgSinkBintr '" << GetName() 
                << "' as it's currently linked");
            return false;
        }

        m_brokerConfigFile.assign(brokerConfigFile);
        m_connectionString.assign(connectionString);
        m_topic.assign(topic);
        
        m_pMsgBroker->SetAttribute("config", m_brokerConfigFile.c_str());
        m_pMsgBroker->SetAttribute("conn-str", m_connectionString.c_str());
        m_pMsgBroker->SetAttribute("topic", m_topic.c_str());
        return true;
    }

    // -------------------------------------------------------------------------------
    
    AzureMsgSinkBintr::AzureMsgSinkBintr(const char* name, 
        const char* converterConfigFile, uint payloadType, const char* brokerConfigFile, 
        const char* connectionString, const char* topic)
        : MsgSinkBintr(name, converterConfigFile, payloadType, brokerConfigFile,
            NVDS_AZURE_PROTO_LIB, connectionString, topic)
    {
        LOG_FUNC();
    }        
    
} // DSL