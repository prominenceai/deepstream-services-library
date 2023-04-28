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

#include "catch.hpp"
#include "Dsl.h"
#include "DslApi.h"

SCENARIO( "The Components container is updated correctly on multiple new components", "[component-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring sourceName  = L"csi-source";
        std::wstring windowSinkName = L"window-sink";
        std::wstring tilerName = L"tiler";

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "Several new components are created" ) 
        {

            REQUIRE( dsl_source_csi_new(sourceName.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_sink_window_new(windowSinkName.c_str(), 0, 0, 1280, 720) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_tiler_new(tilerName.c_str(), 1280, 720) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                // TODO complete verification after addition of Iterator API
                REQUIRE( dsl_component_list_size() == 3 );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    
    
SCENARIO( "Multiple new components can Set and Get their GPU ID", "[component-api]" )
{
    GIVEN( "Three new components" ) 
    {
        std::wstring sourceName  = L"csi-source";
        std::wstring windowSinkName = L"window-sink";
        std::wstring tilerName = L"tiler";
        std::wstring pgieName = L"pgie";
        std::wstring osdName = L"osd";
     
        std::wstring infer_config_file(
            L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt");
        uint GPUID0(0);
        uint GPUID1(1);
        uint retGpuId(0);

        uint retNvbufMem(99);

        REQUIRE( dsl_source_csi_new(sourceName.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_infer_gie_primary_new(pgieName.c_str(), infer_config_file.c_str(), NULL, 0) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_osd_new(osdName.c_str(), true, true, true, false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_sink_window_new(windowSinkName.c_str(), 0, 0, 1280, 720) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_tiler_new(tilerName.c_str(), 1280, 720) == DSL_RESULT_SUCCESS );
        
        retNvbufMem = 99;
        retNvbufMem = 99;
        REQUIRE( dsl_component_nvbuf_mem_type_get(pgieName.c_str(), &retNvbufMem) == DSL_RESULT_SUCCESS );
        REQUIRE( retNvbufMem == DSL_NVBUF_MEM_TYPE_DEFAULT);
        retNvbufMem = 99;
        REQUIRE( dsl_component_nvbuf_mem_type_get(tilerName.c_str(), &retNvbufMem) == DSL_RESULT_SUCCESS );
        REQUIRE( retNvbufMem == DSL_NVBUF_MEM_TYPE_DEFAULT);
        retNvbufMem = 99;
        REQUIRE( dsl_component_nvbuf_mem_type_get(osdName.c_str(), &retNvbufMem) == DSL_RESULT_SUCCESS );
        REQUIRE( retNvbufMem == DSL_NVBUF_MEM_TYPE_DEFAULT);

//        // Note:  WindowSink mem type supported on x86_64 Only
//        REQUIRE( dsl_component_nvbuf_mem_type_get(windowSinkName.c_str(), &retNvbufMem) == DSL_RESULT_SUCCESS );
//        REQUIRE( retNvbufMem == DSL_NVBUF_MEM_TYPE_DEFAULT);

        WHEN( "Several new components are called to Set their GPU ID" ) 
        {
            uint newNvbufMemType(DSL_NVBUF_MEM_TYPE_CUDA_UNIFIED);

//            const wchar_t* components[] = {L"csi-source", L"pgie", L"tiler", L"osd", L"window-sink", NULL};
//            REQUIRE( dsl_component_nvbuf_mem_type_set_many(components, newNvbufMemType) == DSL_RESULT_SUCCESS );
            const wchar_t* components[] = {L"csi-source", L"pgie", L"tiler", L"osd", NULL};
            REQUIRE( dsl_component_nvbuf_mem_type_set_many(components, newNvbufMemType) == DSL_RESULT_SUCCESS );

            THEN( "All components return the correct GPU ID of get" ) 
            {
                retNvbufMem = 99;
                REQUIRE( dsl_component_nvbuf_mem_type_get(sourceName.c_str(), &retNvbufMem) == DSL_RESULT_SUCCESS );
                REQUIRE( retNvbufMem == newNvbufMemType );
                retNvbufMem = 99;
                REQUIRE( dsl_component_nvbuf_mem_type_get(pgieName.c_str(), &retNvbufMem) == DSL_RESULT_SUCCESS );
                REQUIRE( retNvbufMem == newNvbufMemType );
                retNvbufMem = 99;
                REQUIRE( dsl_component_nvbuf_mem_type_get(tilerName.c_str(), &retNvbufMem) == DSL_RESULT_SUCCESS );
                REQUIRE( retNvbufMem == newNvbufMemType );
                retNvbufMem = 99;
                REQUIRE( dsl_component_nvbuf_mem_type_get(osdName.c_str(), &retNvbufMem) == DSL_RESULT_SUCCESS );
                REQUIRE( retNvbufMem == newNvbufMemType );
//                REQUIRE( dsl_component_nvbuf_mem_type_get(windowSinkName.c_str(), &retNvbufMem) == DSL_RESULT_SUCCESS );
//                REQUIRE( retNvbufMem == newNvbufMemType );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    
    
SCENARIO( "The Component API checks for NULL input parameters", "[component-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring componentName  = L"test-component";
        uint gpuId(0);
        uint nvbufMemType(DSL_NVBUF_MEM_TYPE_DEFAULT);
        
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_component_delete(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_delete_many(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_gpuid_get(NULL, &gpuId) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_gpuid_set(NULL, gpuId) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_gpuid_set_many(NULL, gpuId) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_nvbuf_mem_type_get(NULL, &nvbufMemType) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_nvbuf_mem_type_set(NULL, nvbufMemType) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_nvbuf_mem_type_set_many(NULL, nvbufMemType) == DSL_RESULT_INVALID_INPUT_PARAM );
                
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

