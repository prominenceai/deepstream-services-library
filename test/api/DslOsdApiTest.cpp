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

            REQUIRE( dsl_osd_new(osdName.c_str(), false) == DSL_RESULT_SUCCESS );

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
        REQUIRE( dsl_osd_new(osdName.c_str(), false) == DSL_RESULT_SUCCESS );
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

        REQUIRE( dsl_osd_new(osdName.c_str(), false) == DSL_RESULT_SUCCESS );
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

        REQUIRE( dsl_osd_new(osdName.c_str(), false) == DSL_RESULT_SUCCESS );
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

        REQUIRE( dsl_osd_new(osdName.c_str(), false) == DSL_RESULT_SUCCESS );
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

SCENARIO( "An OSD's Clock Enabled Setting can be updated", "[osd-api]" )
{
    GIVEN( "A new OSD in memory" ) 
    {
        std::wstring osdName(L"on-screen-display");
        boolean preEnabled(false), retEnabled(false);

        REQUIRE( dsl_osd_new(osdName.c_str(), preEnabled) == DSL_RESULT_SUCCESS );
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

        REQUIRE( dsl_osd_new(osdName.c_str(), enabled) == DSL_RESULT_SUCCESS );
        
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

        REQUIRE( dsl_osd_new(osdName.c_str(), enabled) == DSL_RESULT_SUCCESS );
        
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

SCENARIO( "An OSD's Clock Color can be updated", "[osd-api]" )
{
    GIVEN( "A new OSD in memory" ) 
    {
        std::wstring osdName(L"on-screen-display");
        boolean enabled(true);
        uint preRed(0xFF), preGreen(0xFF), preBlue(0xFF);
        uint retRed(0x00), retGreen(0x00), retBlue(0x00);

        REQUIRE( dsl_osd_new(osdName.c_str(), enabled) == DSL_RESULT_SUCCESS );
        
        WHEN( "The OSD's Clock Color are Set" ) 
        {
            REQUIRE( dsl_osd_clock_color_set(osdName.c_str(), preRed, preGreen, preBlue) == DSL_RESULT_SUCCESS);
            
            THEN( "The correct values are returned on Get" ) 
            {
                dsl_osd_clock_color_get(osdName.c_str(), &retRed, &retGreen, &retBlue);
                REQUIRE( preRed == retRed);
                REQUIRE( preGreen == retGreen);
                REQUIRE( preBlue == retBlue);
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}

static boolean batch_meta_handler_cb1(void* batch_meta, void* user_data)
{
}
static boolean batch_meta_handler_cb2(void* batch_meta, void* user_data)
{
}
    
SCENARIO( "A Sink Pad Batch Meta Handler can be added and removed from a OSD", "[osd-api]" )
{
    GIVEN( "A new pPipeline with a new OSD" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring osdName(L"on-screen-display");

        REQUIRE( dsl_osd_new(osdName.c_str(), false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            osdName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A Sink Pad Batch Meta Handler is added to the OSD" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE( dsl_osd_batch_meta_handler_remove(osdName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1) == DSL_RESULT_OSD_HANDLER_REMOVE_FAILED );

            REQUIRE( dsl_osd_batch_meta_handler_add(osdName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "The Meta Batch Handler can then be removed" ) 
            {
                REQUIRE( dsl_osd_batch_meta_handler_remove(osdName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Source Pad Batch Meta Handler can be added and removed froma a OSD", "[osd-api]" )
{
    GIVEN( "A new pPipeline with a new OSD" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring osdName(L"on-screen-display");

        REQUIRE( dsl_osd_new(osdName.c_str(), false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            osdName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A Source Pad Batch Meta Handler is added to the OSD" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE( dsl_osd_batch_meta_handler_remove(osdName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1) == DSL_RESULT_OSD_HANDLER_REMOVE_FAILED );

            REQUIRE( dsl_osd_batch_meta_handler_add(osdName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "The Meta Batch Handler can then be removed" ) 
            {
                REQUIRE( dsl_osd_batch_meta_handler_remove(osdName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The same Sink Pad Meta Batch Handler can not be added to the OSD twice", "[osd-api]" )
{
    GIVEN( "A new pPipeline with a new OSD" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring osdName(L"on-screen-display");

        REQUIRE( dsl_osd_new(osdName.c_str(), false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            osdName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A Sink Pad Meta Batch Handler is added to the OSD " ) 
        {
            REQUIRE( dsl_osd_batch_meta_handler_add(osdName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "The same Sink Pad Meta Batch Handler can not be added again" ) 
            {
                REQUIRE( dsl_osd_batch_meta_handler_add(osdName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1, NULL)
                    == DSL_RESULT_OSD_HANDLER_ADD_FAILED );
                
                REQUIRE( dsl_osd_batch_meta_handler_remove(osdName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The same Source Pad Meta Batch Handler can not be added to the OSD twice", "[osd-api]" )
{
    GIVEN( "A new pPipeline with a new OSD" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring osdName(L"on-screen-display");

        REQUIRE( dsl_osd_new(osdName.c_str(), false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            osdName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A Source Pad Meta Batch Handler is added to the OSD " ) 
        {
            REQUIRE( dsl_osd_batch_meta_handler_add(osdName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Sink Pad Meta Batch Handler can not be added" ) 
            {
                REQUIRE( dsl_osd_batch_meta_handler_add(osdName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1, NULL)
                    == DSL_RESULT_OSD_HANDLER_ADD_FAILED );
                
                REQUIRE( dsl_osd_batch_meta_handler_remove(osdName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "An invalid On-Screen Display is caught by all Set and Get API calls", "[osd-api]" )
{
    GIVEN( "A new Fake Sink as incorrect Source Type" ) 
    {
        std::wstring fakeSinkName(L"fake-sink");
            
        uint currBitrate(0);
        uint currInterval(0);
    
        uint newBitrate(2500000);
        uint newInterval(10);

        WHEN( "The On-Screen Display Get-Set API called with a Fake sink" )
        {
            
            REQUIRE( dsl_sink_fake_new(fakeSinkName.c_str()) == DSL_RESULT_SUCCESS);

            THEN( "The On-Screen Display APIs fail correctly")
            {
                REQUIRE ( dsl_osd_batch_meta_handler_add(fakeSinkName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1, NULL) == DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE );
                REQUIRE ( dsl_osd_batch_meta_handler_remove(fakeSinkName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1) == DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A OSD can Enable and Disable Kitti output",  "[gie-api]" )
{
    GIVEN( "A new OSD in memory" ) 
    {
        std::wstring osdName(L"on-screen-display");

        REQUIRE( dsl_osd_new(osdName.c_str(), false) == DSL_RESULT_SUCCESS );
        
        WHEN( "The OSD's Kitti output is enabled" )
        {
            REQUIRE( dsl_osd_kitti_output_enabled_set(osdName.c_str(), true, L"./") == DSL_RESULT_SUCCESS );

            THEN( "The Kitti output can then be disabled" )
            {
                REQUIRE( dsl_osd_kitti_output_enabled_set(osdName.c_str(), false, L"") == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

