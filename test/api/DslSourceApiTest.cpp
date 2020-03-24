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

SCENARIO( "The Components container is updated correctly on new source", "[source-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring sourceName(L"csi-source");
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Source is created" ) 
        {
            REQUIRE( dsl_source_csi_new(sourceName.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    
    
SCENARIO( "The Components container is updated correctly on Source Delete", "[source-api]" )
{
    GIVEN( "One Source im memory" ) 
    {
        std::wstring sourceName(L"csi-source");
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);

        REQUIRE( dsl_component_list_size() == 0 );


        REQUIRE( dsl_source_csi_new(sourceName.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        
        WHEN( "The Source is deleted" )
        {
            REQUIRE( dsl_component_delete(sourceName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list and contents are updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Source in use can't be deleted", "[source-api]" )
{
    GIVEN( "A new Source and new pPipeline" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring sourceName(L"csi-source");
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_csi_new(sourceName.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Source is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                sourceName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Source can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(sourceName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Source, once removed from a Pipeline, can be deleted", "[source-api]" )
{
    GIVEN( "A new Pipeline with a Child CSI Source" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring sourceName(L"csi-source");
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_csi_new(sourceName.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            sourceName.c_str()) == DSL_RESULT_SUCCESS );
            
        WHEN( "The Source is removed from the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(),
                sourceName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Source can be deleted successfully" ) 
            {
                REQUIRE( dsl_component_delete(sourceName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new CSI Camera Source returns the correct attribute values", "[source-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring sourceName(L"csi-source");
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Source is created" ) 
        {
            REQUIRE( dsl_source_csi_new(sourceName.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                uint ret_width(0), ret_height(0), ret_fps_n(0), ret_fps_d(0);
                REQUIRE( dsl_source_dimensions_get(sourceName.c_str(), &ret_width, &ret_height) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_source_frame_rate_get(sourceName.c_str(), &ret_fps_n, &ret_fps_d) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_width == width );
                REQUIRE( ret_height == height );
                REQUIRE( ret_fps_n == fps_n );
                REQUIRE( ret_fps_d == fps_d );
                REQUIRE( dsl_source_is_live(sourceName.c_str()) == true );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "A new USB Camera Source returns the correct attribute values", "[source-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring sourceName(L"usb-source");
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new USB Source is created" ) 
        {
            REQUIRE( dsl_source_usb_new(sourceName.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                uint ret_width(0), ret_height(0), ret_fps_n(0), ret_fps_d(0);
                REQUIRE( dsl_source_dimensions_get(sourceName.c_str(), &ret_width, &ret_height) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_source_frame_rate_get(sourceName.c_str(), &ret_fps_n, &ret_fps_d) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_width == width );
                REQUIRE( ret_height == height );
                REQUIRE( ret_fps_n == fps_n );
                REQUIRE( ret_fps_d == fps_d );
                REQUIRE( dsl_source_is_live(sourceName.c_str()) == true );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    


SCENARIO( "A Client is able to update the Source in-use max", "[source-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_source_num_in_use_max_get() == DSL_DEFAULT_SOURCE_IN_USE_MAX );
        REQUIRE( dsl_source_num_in_use_get() == 0 );
        
        WHEN( "The in-use-max is updated by the client" )   
        {
            uint new_max = 128;
            
            REQUIRE( dsl_source_num_in_use_max_set(new_max) == true );
            
            THEN( "The new in-use-max will be returned to the client on get" )
            {
                REQUIRE( dsl_source_num_in_use_max_get() == new_max );
            }
        }
    }
}

SCENARIO( "A Source added to a Pipeline updates the in-use number", "[source-api]" )
{
    GIVEN( "A new Source and new Pipeline" )
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring sourceName(L"csi-source");
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_csi_new(sourceName.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_num_in_use_get() == 0 );

        WHEN( "The Source is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                sourceName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct in-use number is returned to the client" )
            {
                REQUIRE( dsl_source_num_in_use_get() == 1 );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Source removed from a Pipeline updates the in-use number", "[source-api]" )
{
    GIVEN( "A new Pipeline with a Source" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring sourceName(L"csi-source");
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);

        REQUIRE( dsl_component_list_size() == 0 );

        
        REQUIRE( dsl_source_csi_new(sourceName.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            sourceName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_num_in_use_get() == 1 );

        WHEN( "The Source is removed from, the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(),
                sourceName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct in-use number is returned to the client" )
            {
                REQUIRE( dsl_source_num_in_use_get() == 0 );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "Adding multiple Sources to a Pipelines updates the in-use number", "[source-api]" )
{
    std::wstring sourceName1  = L"csi-source1";
    std::wstring pipelineName1  = L"test-pipeline1";
    std::wstring sourceName2  = L"csi-source2";
    std::wstring pipelineName2  = L"test-pipeline2";

    GIVEN( "Two new Sources and two new Pipeline" )
    {
        REQUIRE( dsl_source_csi_new(sourceName1.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_csi_new(sourceName2.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName2.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_num_in_use_get() == 0 );

        WHEN( "Each Sources is added to a different Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName1.c_str(), 
                sourceName1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_component_add(pipelineName2.c_str(), 
                sourceName2.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct in-use number is returned to the client" )
            {
                REQUIRE( dsl_source_num_in_use_get() == 2 );

                REQUIRE( dsl_pipeline_component_remove(pipelineName1.c_str(), 
                    sourceName1.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_component_remove(pipelineName2.c_str(), 
                    sourceName2.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_source_num_in_use_get() == 0 );
            }
        }
    }
}

SCENARIO( "Adding greater than max Sources to all Pipelines fails", "[source-api]" )
{
    std::wstring sourceName1  = L"csi-source1";
    std::wstring pipelineName1  = L"test-pipeline1";
    std::wstring sourceName2  = L"csi-source2";
    std::wstring pipelineName2  = L"test-pipeline2";
    std::wstring sourceName3  = L"csi-source3";
    std::wstring pipelineName3  = L"test-pipeline3";

    GIVEN( "Two new Sources and two new Pipeline" )
    {
        REQUIRE( dsl_source_csi_new(sourceName1.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_csi_new(sourceName2.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName2.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_csi_new(sourceName3.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName3.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_num_in_use_get() == 0 );

        // Reduce the max to less than 3
        REQUIRE( dsl_source_num_in_use_max_set(2) == true );

        WHEN( "The max number of sources are added to Pipelines" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName1.c_str(), 
                sourceName1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_component_add(pipelineName2.c_str(), 
                sourceName2.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "Adding an additional Source to a Pipeline will fail" )
            {
                REQUIRE( dsl_pipeline_component_add(pipelineName3.c_str(), 
                    sourceName3.c_str()) == DSL_RESULT_PIPELINE_SOURCE_MAX_IN_USE_REACHED );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                
                // Set back to default for other tests
                REQUIRE( dsl_source_num_in_use_max_set(DSL_DEFAULT_SOURCE_IN_USE_MAX) == true );
                REQUIRE( dsl_source_num_in_use_get() == 0 );
            }
        }
    }
}


SCENARIO( "A Source not-in-use can not be Paused or Resumed", "[source-api]" )
{
    GIVEN( "A new Source not in use by a Pipeline" ) 
    {
        std::wstring sourceName(L"csi-source");
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Source is not in use by a Pipeline" )
        {
            REQUIRE( dsl_source_csi_new(sourceName.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );

            THEN( "The Source can not be Paused as it's not in use" ) 
            {
                REQUIRE( dsl_source_pause(sourceName.c_str())  == DSL_RESULT_SOURCE_NOT_IN_USE );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "A new Source is not in use by a Pipeline" )
        {
            REQUIRE( dsl_source_csi_new(sourceName.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );
    
            THEN( "The Source can not be Resumed as it's not in use" ) 
            {
                REQUIRE( dsl_source_resume(sourceName.c_str())  == DSL_RESULT_SOURCE_NOT_IN_USE );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    
    
SCENARIO( "A Source in-use but in a null-state can not be Paused or Resumed", "[source-api]" )
{
    GIVEN( "A new Source not in use by a Pipeline" ) 
    {
        std::wstring pipelineName  = L"test-pipeline";

        std::wstring sourceName  = L"csi-source";
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);

        REQUIRE( dsl_component_list_size() == 0 );


        WHEN( "A new Source is in-use by a new Pipeline in a null-state" )
        {
            REQUIRE( dsl_source_csi_new(sourceName.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                sourceName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Source can not be Paused as it's in a null-state" ) 
            {
                REQUIRE( dsl_source_pause(sourceName.c_str())  == DSL_RESULT_SOURCE_NOT_IN_PLAY );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "A new Source is in-use by a new Pipeline in a null-state" )
        {
            REQUIRE( dsl_source_csi_new(sourceName.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                sourceName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Source can not be Resumed as it's not in use" ) 
            {
                REQUIRE( dsl_source_resume(sourceName.c_str())  == DSL_RESULT_SOURCE_NOT_IN_PAUSE );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    
    
SCENARIO( "An invalid Source is caught by all Set and Get API calls", "[source-api]" )
{
    GIVEN( "A new Fake Sink as incorrect Source Type" ) 
    {
        std::wstring fakeSinkName(L"fake-sink");
            
        uint currBitrate(0);
        uint currInterval(0);
    
        uint newBitrate(2500000);
        uint newInterval(10);

        WHEN( "The File Sink Get-Set API called with a Fake sink" )
        {
            
            REQUIRE( dsl_sink_fake_new(fakeSinkName.c_str()) == DSL_RESULT_SUCCESS);

            THEN( "The Source Pause and Resume APIs fail correctly")
            {
                uint width(0), height(0);
                uint fps_n(0), fps_d(0);
                REQUIRE( dsl_source_dimensions_get(fakeSinkName.c_str(), &width, &height) == DSL_RESULT_SOURCE_COMPONENT_IS_NOT_SOURCE);
                REQUIRE( dsl_source_frame_rate_get(fakeSinkName.c_str(), &fps_n, &fps_d) == DSL_RESULT_SOURCE_COMPONENT_IS_NOT_SOURCE);
                REQUIRE( dsl_source_pause(fakeSinkName.c_str()) == DSL_RESULT_SOURCE_COMPONENT_IS_NOT_SOURCE);
                REQUIRE( dsl_source_resume(fakeSinkName.c_str()) == DSL_RESULT_SOURCE_COMPONENT_IS_NOT_SOURCE);
                REQUIRE( dsl_source_is_live(fakeSinkName.c_str()) == DSL_RESULT_SOURCE_COMPONENT_IS_NOT_SOURCE);

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Dewarper can be added to and removed from a Decode Source Component", "[source-api]" )
{
    GIVEN( "A new Source and new Dewarper" )
    {
        std::wstring sourceName = L"uri-source";
        std::wstring uri = L"./test/streams/sample_1080p_h264.mp4";
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0);

        std::wstring dewarperName(L"dewarper");
        std::wstring defConfigFile(L"./test/configs/config_dewarper.txt");

        REQUIRE( dsl_source_uri_new(sourceName.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_dewarper_new(dewarperName.c_str(), defConfigFile.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Dewarper is added to the Source" ) 
        {
            REQUIRE( dsl_source_decode_dewarper_add(sourceName.c_str(), 
                dewarperName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Dewarper can be removed" )
            {
                // A second call must fail
                REQUIRE( dsl_source_decode_dewarper_add(sourceName.c_str(), 
                    dewarperName.c_str()) == DSL_RESULT_SOURCE_DEWARPER_ADD_FAILED );

                REQUIRE( dsl_source_decode_dewarper_remove(sourceName.c_str()) == DSL_RESULT_SUCCESS );

                // A second time must fail
                REQUIRE( dsl_source_decode_dewarper_remove(sourceName.c_str()) == DSL_RESULT_SOURCE_DEWARPER_REMOVE_FAILED );
                    
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "Adding an invalid Dewarper to a Decode Source Component fails", "[source-api]" )
{
    GIVEN( "A new Source and a Fake Sink as invalid Dewarper" )
    {
        std::wstring sourceName = L"uri-source";
        std::wstring uri = L"./test/streams/sample_1080p_h264.mp4";
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0);

        std::wstring fakeSinkName(L"fake-sink");

        REQUIRE( dsl_source_uri_new(sourceName.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        WHEN( "A Fake Sink is used as Dewarper" ) 
        {
            REQUIRE( dsl_sink_fake_new(fakeSinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "Adding the Fake Sink as a Dewarper will fail" )
            {
                REQUIRE( dsl_source_decode_dewarper_add(sourceName.c_str(), 
                    fakeSinkName.c_str()) == DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE );
                    
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}
