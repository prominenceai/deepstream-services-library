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
        std::wstring tiledDisplayName = L"tiled-display";

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( *(dsl_component_list_all()) == NULL );

        WHEN( "Several new components are created" ) 
        {

            REQUIRE( dsl_source_csi_new(sourceName.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), 0, 0, 1280, 720) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_display_new(tiledDisplayName.c_str(), 1280, 720) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 3 );
                
                const wchar_t** returnedList = dsl_component_list_all();
                               
                REQUIRE( *(returnedList) != NULL );
                std::wstring returnedSourceName = *returnedList;
                std::wcout << returnedSourceName;
//                REQUIRE( returnedSourceName == sourceName );
                
                returnedList++;
                REQUIRE( *(returnedList) != NULL );
                std::wstring returnedOverlaySinkName = *returnedList;
//                REQUIRE( returnedOverlaySinkName == overlaySinkName );

                returnedList++;
                REQUIRE( *(returnedList) != NULL );
                std::wstring returnedTiledDisplayName = *returnedList;
//                REQUIRE( returnedTiledDisplayName == tiledDisplayName );

                returnedList++;
                REQUIRE( *(returnedList) == NULL );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
                REQUIRE( *(dsl_component_list_all()) == NULL );
            }
        }
    }
}    
    
