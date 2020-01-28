/*
The MIT License

Copyright (c) 2019-Present, ROBERT HOWELL

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
        std::wstring overlaySinkName = L"overlay-sink";
        std::wstring tilerName = L"tiler";

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "Several new components are created" ) 
        {

            REQUIRE( dsl_source_csi_new(sourceName.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), 0, 0, 1280, 720) == DSL_RESULT_SUCCESS );
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
        std::wstring overlaySinkName = L"overlay-sink";
        std::wstring tilerName = L"tiler";
        uint GPUID0(0);
        uint GPUID1(1);
        uint retGpuId(0);

        REQUIRE( dsl_source_csi_new(sourceName.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), 0, 0, 1280, 720) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_tiler_new(tilerName.c_str(), 1280, 720) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_component_gpuid_get(sourceName.c_str(), &retGpuId) == DSL_RESULT_SUCCESS );
        REQUIRE( retGpuId == GPUID0);
        REQUIRE( dsl_component_gpuid_get(overlaySinkName.c_str(), &retGpuId) == DSL_RESULT_SUCCESS );
        REQUIRE( retGpuId == GPUID0);
        REQUIRE( dsl_component_gpuid_get(tilerName.c_str(), &retGpuId) == DSL_RESULT_SUCCESS );
        REQUIRE( retGpuId == GPUID0);

        WHEN( "Several new components are called to Set their GPU ID" ) 
        {

            const wchar_t* components[] = {L"csi-source", L"tiler", L"overlay-sink", NULL};
            REQUIRE( dsl_component_gpuid_set_many(components, GPUID1) == DSL_RESULT_SUCCESS );

            THEN( "All components return the correct GPU ID of get" ) 
            {
                REQUIRE( dsl_component_gpuid_get(sourceName.c_str(), &retGpuId) == DSL_RESULT_SUCCESS );
                REQUIRE( retGpuId == GPUID1 );
                REQUIRE( dsl_component_gpuid_get(overlaySinkName.c_str(), &retGpuId) == DSL_RESULT_SUCCESS );
                REQUIRE( retGpuId == GPUID1 );
                REQUIRE( dsl_component_gpuid_get(tilerName.c_str(), &retGpuId) == DSL_RESULT_SUCCESS );
                REQUIRE( retGpuId == GPUID1 );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    
    
