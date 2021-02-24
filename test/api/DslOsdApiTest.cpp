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
#include "DslApi.h"

SCENARIO( "The Components container is updated correctly on new OSD", "[osd-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring osdName(L"on-screen-display");

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new OSD is created" ) 
        {

            REQUIRE( dsl_osd_new(osdName.c_str(), false, false) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}

SCENARIO( "The Components container is updated correctly on OSD delete", "[osd-api]" )
{
    GIVEN( "A new OSD in memory" ) 
    {
        std::wstring osdName(L"on-screen-display");

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_osd_new(osdName.c_str(), false, false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "The new OSD is created" ) 
        {
            REQUIRE( dsl_component_delete(osdName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size is updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}


SCENARIO( "An OSD in use can't be deleted", "[osd-api]" )
{
    GIVEN( "A new OSD and new pPipeline" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring osdName(L"on-screen-display");

        REQUIRE( dsl_osd_new(osdName.c_str(), false, false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        WHEN( "The OSD is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                osdName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The OSD can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(osdName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An OSD, once removed from a Pipeline, can be deleted", "[osd-api]" )
{
    GIVEN( "A new pPipeline with a child OSD" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring osdName(L"on-screen-display");

        REQUIRE( dsl_osd_new(osdName.c_str(), false, false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            osdName.c_str()) == DSL_RESULT_SUCCESS );
            
        WHEN( "The OSD is from the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(), 
                osdName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The OSD can be deleted" ) 
            {
                REQUIRE( dsl_component_delete(osdName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An OSD in use can't be added to a second Pipeline", "[osd-api]" )
{
    GIVEN( "A new OSD and two new pPipelines" ) 
    {
        std::wstring pipelineName1(L"test-pipeline-1");
        std::wstring pipelineName2(L"test-pipeline-2");
        std::wstring osdName(L"on-screen-display");

        REQUIRE( dsl_osd_new(osdName.c_str(), false, false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName2.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The OSD is added to the first Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName1.c_str(), 
                osdName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The OSD can't be added to the second Pipeline" ) 
            {
                REQUIRE( dsl_pipeline_component_add(pipelineName2.c_str(), 
                    osdName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "An OSD's display-text property can be updated", "[osd-api]" )
{
    GIVEN( "A new OSD in memory" ) 
    {
        std::wstring osdName(L"on-screen-display");
        boolean preEnabled(false), retEnabled(false);

        REQUIRE( dsl_osd_new(osdName.c_str(), preEnabled, false) == DSL_RESULT_SUCCESS );
        dsl_osd_text_enabled_get(osdName.c_str(), &retEnabled);
        REQUIRE( preEnabled == retEnabled);
        
        WHEN( "The OSD's display-text property is Enabled" ) 
        {
            preEnabled = false;
            REQUIRE( dsl_osd_text_enabled_set(osdName.c_str(), preEnabled) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on Get" ) 
            {
                dsl_osd_text_enabled_get(osdName.c_str(), &retEnabled);
                REQUIRE( preEnabled == retEnabled );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "An OSD's Clock Enabled Setting can be updated", "[osd-api]" )
{
    GIVEN( "A new OSD in memory" ) 
    {
        std::wstring osdName(L"on-screen-display");
        boolean preEnabled(false), retEnabled(false);

        REQUIRE( dsl_osd_new(osdName.c_str(), false, preEnabled) == DSL_RESULT_SUCCESS );
        dsl_osd_clock_enabled_get(osdName.c_str(), &retEnabled);
        REQUIRE( preEnabled == retEnabled);
        
        WHEN( "The OSD's Clock Enabled is Set" ) 
        {
            preEnabled = false;
            REQUIRE( dsl_osd_clock_enabled_set(osdName.c_str(), preEnabled) == DSL_RESULT_SUCCESS);
            
            THEN( "The correct value is returned on Get" ) 
            {
                dsl_osd_clock_enabled_get(osdName.c_str(), &retEnabled);
                REQUIRE( preEnabled == retEnabled);

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "An OSD's Clock Offsets can be updated", "[osd-api]" )
{
    GIVEN( "A new OSD in memory" ) 
    {
        std::wstring osdName(L"on-screen-display");
        boolean enabled(true);
        uint preOffsetX(100), preOffsetY(100);
        uint retOffsetX(0), retOffsetY(0);

        REQUIRE( dsl_osd_new(osdName.c_str(), false, enabled) == DSL_RESULT_SUCCESS );
        
        WHEN( "The OSD's Clock Offsets are Set" ) 
        {
            REQUIRE( dsl_osd_clock_offsets_set(osdName.c_str(), preOffsetX, preOffsetY) == DSL_RESULT_SUCCESS);
            
            THEN( "The correct values are returned on Get" ) 
            {
                dsl_osd_clock_offsets_get(osdName.c_str(), &retOffsetX, &retOffsetY);
                REQUIRE( preOffsetX == retOffsetX);
                REQUIRE( preOffsetY == retOffsetY);

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "An OSD's Clock Font can be updated", "[osd-api]" )
{
    GIVEN( "A new OSD in memory" ) 
    {
        std::wstring osdName(L"on-screen-display");
        boolean enabled(true);
        uint preOffsetX(100), preOffsetY(100);
        uint retOffsetX(0), retOffsetY(0);
        std::wstring newFont(L"arial");
        uint newSize(16);

        REQUIRE( dsl_osd_new(osdName.c_str(), false, enabled) == DSL_RESULT_SUCCESS );
        
        WHEN( "The OSD's Clock Font is Set" ) 
        {
            REQUIRE( dsl_osd_clock_font_set(osdName.c_str(), newFont.c_str(), newSize) == DSL_RESULT_SUCCESS);
            
            THEN( "The correct values are returned on Get" ) 
            {
                const wchar_t *retFontPtr;
                uint retSize;
                dsl_osd_clock_font_get(osdName.c_str(), &retFontPtr, &retSize);
                std::wstring retFont(retFontPtr);
                REQUIRE( retFont == newFont );
                REQUIRE( retSize == newSize );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

//SCENARIO( "An OSD's Clock Color can be updated", "[osd-api]" )
//{
//    GIVEN( "A new OSD in memory" ) 
//    {
//        std::wstring osdName(L"on-screen-display");
//        boolean enabled(true);
//        uint preRed(0xFF), preGreen(0xFF), preBlue(0xFF);
//        uint retRed(0x00), retGreen(0x00), retBlue(0x00);
//
//        REQUIRE( dsl_osd_new(osdName.c_str(), enabled) == DSL_RESULT_SUCCESS );
//        
//        WHEN( "The OSD's Clock Color are Set" ) 
//        {
//            REQUIRE( dsl_osd_clock_color_set(osdName.c_str(), preRed, preGreen, preBlue) == DSL_RESULT_SUCCESS );
//            
//            THEN( "The correct values are returned on Get" ) 
//            {
//                dsl_osd_clock_color_get(osdName.c_str(), &retRed, &retGreen, &retBlue);
//                REQUIRE( preRed == retRed);
//                REQUIRE( preGreen == retGreen);
//                REQUIRE( preBlue == retBlue);
//            }
//        }
//        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
//    }
//}


static boolean pad_probe_handler_cb1(void* buffer, void* user_data)
{
}
static boolean pad_probe_handler_cb2(void* buffer, void* user_data)
{
}
    
SCENARIO( "A Sink Pad Probe Handler can be added and removed from a OSD", "[osd-api]" )
{
    GIVEN( "A new OSD and a Custom PPH" ) 
    {
        std::wstring osdName(L"on-screen-display");
        std::wstring customPpmName(L"custom-ppm");

        REQUIRE( dsl_osd_new(osdName.c_str(), false, false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pph_custom_new(customPpmName.c_str(), pad_probe_handler_cb1, NULL) == DSL_RESULT_SUCCESS );

        WHEN( "A Sink Pad Probe Handler is added to the OSD" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE( dsl_osd_pph_remove(osdName.c_str(), customPpmName.c_str(), DSL_PAD_SINK) == DSL_RESULT_OSD_HANDLER_REMOVE_FAILED );

            REQUIRE( dsl_osd_pph_add(osdName.c_str(), customPpmName.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
            
            THEN( "The Padd Probe Handler can then be removed" ) 
            {
                REQUIRE( dsl_osd_pph_remove(osdName.c_str(), customPpmName.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "A Sink Pad Probe Handler is added to the Primary OSD" ) 
        {
            REQUIRE( dsl_osd_pph_add(osdName.c_str(), customPpmName.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
            
            THEN( "Attempting to add the same Sink Pad Probe Handler twice failes" ) 
            {
                REQUIRE( dsl_osd_pph_add(osdName.c_str(), customPpmName.c_str(), DSL_PAD_SINK) == DSL_RESULT_OSD_HANDLER_ADD_FAILED );
                REQUIRE( dsl_osd_pph_remove(osdName.c_str(), customPpmName.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Source Pad Probe Handler can be added and removed froma a OSD", "[osd-api]" )
{
    GIVEN( "A new OSD and a Custom PPH" ) 
    {
        std::wstring osdName(L"on-screen-display");
        std::wstring customPpmName(L"custom-ppm");

        REQUIRE( dsl_osd_new(osdName.c_str(), false, false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pph_custom_new(customPpmName.c_str(), pad_probe_handler_cb1, NULL) == DSL_RESULT_SUCCESS );

        WHEN( "A Sink Pad Probe Handler is added to the OSD" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE( dsl_osd_pph_remove(osdName.c_str(), customPpmName.c_str(), DSL_PAD_SRC) == DSL_RESULT_OSD_HANDLER_REMOVE_FAILED );

            REQUIRE( dsl_osd_pph_add(osdName.c_str(), customPpmName.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
            
            THEN( "The Meta Batch Handler can then be removed" ) 
            {
                REQUIRE( dsl_osd_pph_remove(osdName.c_str(), customPpmName.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "A Sink Pad Probe Handler is added to the Primary OSD" ) 
        {
            REQUIRE( dsl_osd_pph_add(osdName.c_str(), customPpmName.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
            
            THEN( "Attempting to add the same Sink Pad Probe Handler twice failes" ) 
            {
                REQUIRE( dsl_osd_pph_add(osdName.c_str(), customPpmName.c_str(), DSL_PAD_SRC) == DSL_RESULT_OSD_HANDLER_ADD_FAILED );
                REQUIRE( dsl_osd_pph_remove(osdName.c_str(), customPpmName.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

