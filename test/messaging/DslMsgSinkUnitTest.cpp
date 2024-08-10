/*
The MIT License

Copyright (c) 2022-2024, Prominence AI, Inc.

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

#include "catch.hpp"
#include "Dsl.h"
#include "DslSinkBintr.h"

using namespace DSL;

static std::string sinkName("msg-sink");
static std::string conversionConfigFile(
    "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test4/dstest4_msgconv_config.txt");
//static uint payloadType(PAYLOAD_DEEPSTREAM_MINIMAL);
static uint payloadType(0);
static std::string protocolLib("/opt/nvidia/deepstream/deepstream/lib/libnvds_azure_proto.so");
static std::string brokerConfigFile(
    "/opt/nvidia/deepstream/deepstream/sources/libs/azure_protocol_adaptor/device_client/cfg_azure.txt");
static std::string topic("DSL_MESSAGE_TOPIC");

static std::string connectionString(
    "HostName=prominenceai-hub.azure-devices.net;DeviceId=nano1;SharedAccessKey=abcdefghijklmnopqrstuvwzyz");

SCENARIO( "A new MsgSinkBintr is created correctly",  "[MsgSinkBintr]" )
{
    GIVEN( "Attributes for a new MsgSinkBintr" ) 
    {
        WHEN( "The MsgSinkBintr is created " )
        {
            DSL_MESSAGE_SINK_PTR pSinkBintr = 
                DSL_MESSAGE_SINK_NEW(sinkName.c_str(), conversionConfigFile.c_str(), 
                    payloadType, brokerConfigFile.c_str(), protocolLib.c_str(),
                    connectionString.c_str(), topic.c_str());
            
            THEN( "The correct attribute values are returned" )
            {
                boolean retEnabled(false);
                REQUIRE( pSinkBintr->GetSyncEnabled(&retEnabled) == true );;
                REQUIRE( retEnabled == true );;
                
                REQUIRE( pSinkBintr->GetMetaType() == NVDS_EVENT_MSG_META);
                const char* cRetConversionConfigFile;
                uint retPayloadType;
                pSinkBintr->GetConverterSettings(&cRetConversionConfigFile, &retPayloadType);
                
                std::string retConversionConfigFile(cRetConversionConfigFile);
                REQUIRE( retConversionConfigFile == conversionConfigFile );
                REQUIRE( retPayloadType == payloadType );

                const char* cRetBrokerConfigFile;
                const char* cRetProtocolLib;
                const char* cRetConnectionString;
                const char* cRetTopic;
                pSinkBintr->GetBrokerSettings(&cRetBrokerConfigFile,
                    &cRetProtocolLib, &cRetConnectionString, &cRetTopic);

                std::string retBrokerConfigFile(cRetBrokerConfigFile);
                std::string retProtocolLib(cRetProtocolLib);
                std::string retConnectionString(cRetConnectionString);
                std::string retTopic(cRetTopic);
                
                REQUIRE ( retBrokerConfigFile == brokerConfigFile);
                REQUIRE ( retProtocolLib == protocolLib);
                REQUIRE ( retConnectionString == connectionString);
                REQUIRE ( retTopic == topic );
                
                std::string debugDir(pSinkBintr->GetDebugDir());
                REQUIRE ( debugDir == "" );
            }
        }
    }
}

SCENARIO( "A new MsgSinkBintr can LinkAll Child Elementrs", "[MsgSinkBintr]" )
{
    GIVEN( "A new MsgSinkBintr in an Unlinked state" ) 
    {
        std::string sinkName("msg-sink");

            DSL_MESSAGE_SINK_PTR pSinkBintr = 
                DSL_MESSAGE_SINK_NEW(sinkName.c_str(), conversionConfigFile.c_str(), 
                    payloadType, brokerConfigFile.c_str(), protocolLib.c_str(),
                    connectionString.c_str(), topic.c_str());

        REQUIRE( pSinkBintr->IsLinked() == false );

        WHEN( "A new MsgSinkBintr is Linked" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );

            THEN( "The MsgSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == true );
            }
        }
    }
}
