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

#ifndef _DSL_SINK_MSG_BINTR_H
#define _DSL_SINK_MSG_BINTR_H

#include "Dsl.h"
#include "DslSinkBintr.h"

namespace DSL
{

    #define DSL_MSG_SINK_PTR std::shared_ptr<MsgSinkBintr>
    
    #define DSL_AZURE_MSG_SINK_PTR std::shared_ptr<AzureMsgSinkBintr>
    #define DSL_AZURE_MSG_SINK_NEW(name, \
            converterConfigFile, payloadType, brokerConfigFile, \
            connectionString, topic) \
        std::shared_ptr<AzureMsgSinkBintr>(new AzureMsgSinkBintr(name, \
            converterConfigFile, payloadType, brokerConfigFile, \
            connectionString, topic))

    /**
     * @class MsgSinkBintr 
     * @brief Implements a Message Sink Bin Container Class (Bintr)
     */
    class MsgSinkBintr : public SinkBintr
    {
    public: 
    
        /**
         * @brief Ctor for the MsgSinkBintr class
         */
        MsgSinkBintr(const char* name, const char* converterConfigFile, 
        uint payloadType, const char* brokerConfigFile, const char* protoLib, 
        const char* connectionString, const char* topic);

        /**
         * @brief Dtor for the MsgSinkBintr class
         */
        ~MsgSinkBintr();
  
        /**
         * @brief Links all Child Elementrs owned by this MsgSinkBintr
         * @return true if all links were successful, false otherwise
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elemntrs owned by this MsgSinkBintr
         * Calling UnlinkAll when in an unlinked state has no effect.
         */
        void UnlinkAll();
        
        /**
         * @brief Gets the current message converter settings for the MsgSinkBintr.
         * @param[out] converterConfigFile absolute file-path to the current
         * message converter config file in use.
         * @param[out] payloadType current payload type setting.
         */
        void GetConverterSettings(const char** converterConfigFile,
            uint* payloadType);
            
        /**
         * @brief Sets the current message converter settings for the MsgSinkBintr.
         * @param[in] converterConfigFile absolute or relate file-path to a new
         * message converter config file to use.
         * @param[in] payloadType new payload type setting to use.
         * @return true if successful, false otherwise.
         */
        bool SetConverterSettings(const char* converterConfigFile,
            uint payloadType);

        /**
         * @brief Gets the current message broker settings for the MsgSinBintr.
         * @param[out] brokerConfigFile absolute file-path to the current message
         * borker config file in use.
         * @param[out] connectionString current connection string in use.
         * @param[out] topic (optional) message topic current in use.
         */
        void GetBrokerSettings(const char** brokerConfigFile,
            const char** connectionString, const char** topic);

        /**
         * @brief Sets the message broker settings for the MsgSinBintr.
         * @param[in] brokerConfigFile absolute or relative file-path to 
         * a new message borker config file to use.
         * @param[in] connectionString new connection string in use.
         * @param[in] topic (optional) new message topic to use.
         * @return true if successful, false otherwise.
         */
        bool SetBrokerSettings(const char* brokerConfigFile,
            const char* connectionString, const char* topic);

        /**
         * @brief sets the current sync and async settings for the SinkBintr
         * @param[in] sync current sync setting, true if set, false otherwise.
         * @param[in] async parameter is unused -- setting has no affect.
         * @return true is successful, false otherwise. 
         */
        bool SetSyncSettings(bool sync, bool async);

    private:

        /**
         * @brief qualitiy of service setting for the fake sink.
         */
        boolean m_qos;
        
        /**
         * @brief absolute path to the message converter config file is use.
         */
        std::string m_converterConfigFile;
        
        /**
         * @brief payload type, one of the DSL_MSG_PAYLOAD_<*> constants 
         */
        uint m_payloadType; 
        
        /**
         * @brief absolute path to the message broker config file in use.
         */
        std::string m_brokerConfigFile; 
        
        /**
         * @brief connection string used as end-point for communication with server.
         */
        std::string m_connectionString;
        
        /**
         * @brief Absolute pathname to the library that contains the protocol adapter.
         */
        std::string m_protoLib; 
        
        /**
         * @brief (optional) message topic name.
         */
        std::string m_topic;
    
        /**
         * @brief Tee element for this MsgSinkBintr 
         */
        DSL_ELEMENT_PTR m_pTee;

        /**
         * @brief Tee Src Queue for the message-converter element for this MsgSinkBintr 
         */
        DSL_ELEMENT_PTR m_pMsgConverterQueue;
        
        /**
         * @brief NVIDIA message-converter element for this MsgSinkBintr 
         */
        DSL_ELEMENT_PTR m_pMsgConverter;

        /**
         * @brief NVIDIA message-broker element for this MsgSinkBintr.
         */
        DSL_ELEMENT_PTR m_pMsgBroker;

        /**
         * @brief Tee Src Queue for the Fake Sink element for this MsgSinkBintr 
         */
        DSL_ELEMENT_PTR m_pFakeSinkQueue;

        /**
         * @brief Fake Sink element for the MsgSinkBintr.
         */
        DSL_ELEMENT_PTR m_pFakeSink;

    };

    /**
     * @class AzureMsgSinkBintr 
     * @brief Implements an Azure Protocol adapted Message Sink Bin 
     * Container Class (Bintr)
     */
    class AzureMsgSinkBintr : public MsgSinkBintr
    {
    public: 
    
        /**
         * @brief Ctor for the AzureMsgSinkBintr class
         */
        AzureMsgSinkBintr(const char* name, const char* converterConfigFile, 
            uint payloadType, const char* brokerConfigFile, 
            const char* connectionString, const char* topic);
    };
    
}
#endif //_DSL_SINK_MSG_BINTR_H
