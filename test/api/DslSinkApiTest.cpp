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

static uint new_buffer_cb(uint data_type,
    void* buffer, void* client_data)
{
    return DSL_FLOW_OK;
}

SCENARIO( "The Components container is updated correctly on new and delete App Sink", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring sinkName = L"app-sink";

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new App Sink is created" ) 
        {
            REQUIRE( dsl_sink_app_new(sinkName.c_str(), DSL_SINK_APP_DATA_TYPE_BUFFER, 
                new_buffer_cb, NULL) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean sync(false);
                REQUIRE( dsl_sink_sync_enabled_get(sinkName.c_str(), 
                    &sync) == DSL_RESULT_SUCCESS );
                REQUIRE( sync == true );

                // delete and check the component count
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An App Sink fails to create when an invalid data-type is provided", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring sinkName = L"app-sink";

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When an invalid data type is specified" ) 
        {
            uint invalid_data_type(DSL_SINK_APP_DATA_TYPE_BUFFER+1);

            THEN( "The Sink must fail to create" ) 
            {
                REQUIRE( dsl_sink_app_new(sinkName.c_str(), invalid_data_type, 
                    new_buffer_cb, NULL) == DSL_RESULT_SINK_SET_FAILED );
            }
        }
    }
}

SCENARIO( "An App Sink can update it Sync setting correctly", "[sink-api]" )
{
    GIVEN( "A new App Sink component" ) 
    {
        std::wstring sinkName = L"app-sink";

        REQUIRE( dsl_sink_app_new(sinkName.c_str(), DSL_SINK_APP_DATA_TYPE_BUFFER, 
            new_buffer_cb, NULL) == DSL_RESULT_SUCCESS );

        // check the default
        boolean retSync(TRUE);
        REQUIRE( dsl_sink_sync_enabled_get(sinkName.c_str(), 
            &retSync) == DSL_RESULT_SUCCESS );
        REQUIRE( retSync == TRUE );

        WHEN( "The App Sink's sync value is updated" ) 
        {
            boolean newSync(FALSE);
            REQUIRE( dsl_sink_sync_enabled_set(sinkName.c_str(), 
                newSync) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is retruned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                REQUIRE( dsl_sink_sync_enabled_get(sinkName.c_str(), 
                    &retSync) == DSL_RESULT_SUCCESS );
                REQUIRE( retSync == newSync );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "An App Sink can update its data-type setting correctly", "[sink-api]" )
{
    GIVEN( "A new App Sink Component" ) 
    {
        std::wstring sinkName = L"app-sink";
        uint init_data_type(DSL_SINK_APP_DATA_TYPE_BUFFER);
        
        REQUIRE( dsl_sink_app_new(sinkName.c_str(), init_data_type, 
            new_buffer_cb, NULL) == DSL_RESULT_SUCCESS );

        // Check the intial value
        uint ret_data_type;
        REQUIRE( dsl_sink_app_data_type_get(sinkName.c_str(), 
            &ret_data_type) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_data_type == init_data_type );

        WHEN( "The App Sink's data-type is updated" ) 
        {
            uint new_data_type(DSL_SINK_APP_DATA_TYPE_SAMPLE);
            
            REQUIRE( dsl_sink_app_data_type_set(sinkName.c_str(), 
                new_data_type) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_sink_app_data_type_get(sinkName.c_str(), 
                    &ret_data_type) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_data_type == new_data_type );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "An ivalid data-type is provided" ) 
        {
            uint new_data_type(DSL_SINK_APP_DATA_TYPE_BUFFER+1);

            THEN( "The set data-type service must fail" ) 
            {
                REQUIRE( dsl_sink_app_data_type_set(sinkName.c_str(), 
                    new_data_type) == DSL_RESULT_SINK_SET_FAILED);
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    
    
SCENARIO( "The Components container is updated correctly on new Fake Sink", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring sinkName = L"fake-sink";

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Fake Sink is created" ) 
        {
            REQUIRE( dsl_sink_fake_new(sinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean sync(false);
                REQUIRE( dsl_sink_sync_enabled_get(sinkName.c_str(), 
                    &sync) == DSL_RESULT_SUCCESS );
                REQUIRE( sync == true );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "The Components container is updated correctly on Fake Sink delete", "[sink-api]" )
{
    GIVEN( "A Fake Sink Component" ) 
    {
        std::wstring sinkName = L"fake-sink";


        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_fake_new(sinkName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A new Fake Sink is deleted" ) 
        {
            REQUIRE( dsl_component_delete(sinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Fake Sink can update it Sync/Async attributes", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring sinkName = L"fake-sink";

        REQUIRE( dsl_sink_fake_new(sinkName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A the Fake Sink's attributes are updated from the default" ) 
        {
            boolean newSync(false);
            REQUIRE( dsl_sink_sync_enabled_set(sinkName.c_str(), newSync) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retSync(true);
                REQUIRE( dsl_sink_sync_enabled_get(sinkName.c_str(), &retSync) == DSL_RESULT_SUCCESS );
                REQUIRE( retSync == newSync );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "The Components container is updated correctly on new Overlay Sink", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        // Get the Device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        
        if (deviceProp.integrated)
        {
            std::wstring overlaySinkName = L"overlay-sink";
            uint displayId(0);
            uint depth(0);
            uint offsetX(0);
            uint offsetY(0);
            uint sinkW(1280);
            uint sinkH(720);

            REQUIRE( dsl_component_list_size() == 0 );

            WHEN( "A new Overlay Sink is created" ) 
            {

                REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), displayId, depth, 
                    offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

                THEN( "The list size is updated correctly" ) 
                {
                    REQUIRE( dsl_component_list_size() == 1 );
                }
            }
            REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
        }
    }
}    

SCENARIO( "The Components container is updated correctly on Overlay Sink delete", "[sink-api]" )
{
    GIVEN( "An Overlay Sink Component" ) 
    {
        // Get the Device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        
        if (deviceProp.integrated)
        {
            std::wstring overlaySinkName = L"overlay-sink";
            uint displayId(0);
            uint depth(0);
            uint offsetX(0);
            uint offsetY(0);
            uint sinkW(0);
            uint sinkH(0);

            REQUIRE( dsl_component_list_size() == 0 );
            REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), displayId, depth, 
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

            WHEN( "A new Overlay Sink is deleted" ) 
            {
                REQUIRE( dsl_component_delete(overlaySinkName.c_str()) == DSL_RESULT_SUCCESS );
                
                THEN( "The list size updated correctly" )
                {
                    REQUIRE( dsl_component_list_size() == 0 );
                }
            }
        }
    }
}


SCENARIO( "A Overlay Sink can update its Sync/Async attributes", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        // Get the Device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        
        if (deviceProp.integrated)
        {
            std::wstring overlaySinkName = L"overlay-sink";
            uint displayId(0);
            uint depth(0);
            uint offsetX(0);
            uint offsetY(0);
            uint sinkW(0);
            uint sinkH(0);

            REQUIRE( dsl_component_list_size() == 0 );
            REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), displayId, depth, 
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

            WHEN( "A the Window Sink's attributes are updated from the default" ) 
            {
                boolean newSync(false);
                REQUIRE( dsl_sink_sync_enabled_set(overlaySinkName.c_str(),     
                    newSync) == DSL_RESULT_SUCCESS );

                THEN( "The list size is updated correctly" ) 
                {
                    REQUIRE( dsl_component_list_size() == 1 );
                    boolean retSync(true);
                    REQUIRE( dsl_sink_sync_enabled_get(overlaySinkName.c_str(), 
                        &retSync) == DSL_RESULT_SUCCESS );
                    REQUIRE( retSync == newSync );
                    REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                }
            }
        }
    }
}    

SCENARIO( "A Overlay Sink's Offsets can be updated", "[sink-api]" )
{
    GIVEN( "A new Overlay Sink in memory" ) 
    {
        // Get the Device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        
        if (deviceProp.integrated)
        {
            std::wstring sinkName = L"overlay-sink";
            uint displayId(0);
            uint depth(0);
            uint offsetX(0);
            uint offsetY(0);
            uint sinkW(0);
            uint sinkH(0);

            uint preOffsetX(100), preOffsetY(100);
            uint retOffsetX(0), retOffsetY(0);

            REQUIRE( dsl_sink_overlay_new(sinkName.c_str(), displayId, depth, 
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

            WHEN( "The Window Sink's Offsets are Set" ) 
            {
                REQUIRE( dsl_sink_render_offsets_set(sinkName.c_str(), 
                    preOffsetX, preOffsetY) == DSL_RESULT_SUCCESS);
                
                THEN( "The correct values are returned on Get" ) 
                {
                    dsl_sink_render_offsets_get(sinkName.c_str(), &retOffsetX, &retOffsetY);
                    REQUIRE( preOffsetX == retOffsetX);
                    REQUIRE( preOffsetY == retOffsetY);

                    REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                }
            }
        }
    }
}

SCENARIO( "A Overlay Sink's Dimensions can be updated", "[sink-api]" )
{
    GIVEN( "A new Overlay Sink in memory" ) 
    {
        // Get the Device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        
        if (deviceProp.integrated)
        {
            std::wstring sinkName = L"overlay-sink";
            uint displayId(0);
            uint depth(0);
            uint offsetX(0);
            uint offsetY(0);
            uint sinkW(0);
            uint sinkH(0);

            uint preSinkW(1280), preSinkH(720);
            uint retSinkW(0), retSinkH(0);

            REQUIRE( dsl_sink_overlay_new(sinkName.c_str(), displayId, depth, 
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

            WHEN( "The Overlay Sink's Dimensions are Set" ) 
            {
                REQUIRE( dsl_sink_render_dimensions_set(sinkName.c_str(), 
                    preSinkW, preSinkH) == DSL_RESULT_SUCCESS);
                
                THEN( "The correct values are returned on Get" ) 
                {
                    dsl_sink_render_dimensions_get(sinkName.c_str(), &retSinkW, &retSinkH);
                    REQUIRE( preSinkW == retSinkW);
                    REQUIRE( preSinkH == retSinkH);

                    REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                }
            }
        }
    }
}

SCENARIO( "An Overlay Sink can be Reset", "[sink-api]" )
{
    GIVEN( "Attributes for a new Overlay Sink" ) 
    {
        // Get the Device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        
        if (deviceProp.integrated)
        {
            std::wstring overlaySinkName = L"overlay-sink";
            uint displayId(0);
            uint depth(0);
            uint offsetX(0);
            uint offsetY(0);
            uint sinkW(0);
            uint sinkH(0);

            WHEN( "The Overlay Sink is created" ) 
            {
                REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), displayId, depth,
                    offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
                
                THEN( "The Overlay Sink can be reset after creation" ) 
                {
                    REQUIRE( dsl_sink_render_reset(overlaySinkName.c_str()) == DSL_RESULT_SUCCESS );

                    REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                }
            }
        }
    }
}

SCENARIO( "An Overlay Sink in use can't be deleted", "[sink-api]" )
{
    GIVEN( "A new Overlay Sink and new pPipeline" ) 
    {
        // Get the Device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        
        if (deviceProp.integrated)
        {
            std::wstring pipelineName  = L"test-pipeline";
            std::wstring overlaySinkName = L"overlay-sink";
            uint displayId(0);
            uint depth(0);
            uint offsetX(0);
            uint offsetY(0);
            uint sinkW(1280);
            uint sinkH(720);

            REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), displayId, depth,
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_component_list_size() == 1 );
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_list_size() == 1 );

            WHEN( "The Overlay Sink is added to the Pipeline" ) 
            {
                REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                    overlaySinkName.c_str()) == DSL_RESULT_SUCCESS );

                THEN( "The Overlay Sink can't be deleted" ) 
                {
                    REQUIRE( dsl_component_delete(overlaySinkName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );
                    
                    REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                    REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                    REQUIRE( dsl_pipeline_list_size() == 0 );
                    REQUIRE( dsl_component_list_size() == 0 );
                }
            }
        }
    }
}

SCENARIO( "An Overlay Sink, once removed from a Pipeline, can be deleted", "[sink-api]" )
{
    GIVEN( "A new Sink owned by a new pPipeline" ) 
    {
        // Get the Device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        
        if (deviceProp.integrated)
        {
            std::wstring pipelineName  = L"test-pipeline";
            std::wstring overlaySinkName = L"overlay-sink";
            uint displayId(0);
            uint depth(0);
            uint offsetX(0);
            uint offsetY(0);
            uint sinkW(1280);
            uint sinkH(720);

            REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), displayId, depth, 
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
                
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                overlaySinkName.c_str()) == DSL_RESULT_SUCCESS );

            WHEN( "The Overlay Sink is removed the Pipeline" ) 
            {
                REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(), 
                    overlaySinkName.c_str()) == DSL_RESULT_SUCCESS );

                THEN( "The Overlay Sink can be deleted" ) 
                {
                    REQUIRE( dsl_component_delete(overlaySinkName.c_str()) == DSL_RESULT_SUCCESS );

                    REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                    REQUIRE( dsl_pipeline_list_size() == 0 );
                    REQUIRE( dsl_component_list_size() == 0 );
                }
            }
        }
    }
}

SCENARIO( "An Overlay Sink in use can't be added to a second Pipeline", "[sink-api]" )
{
    GIVEN( "A new Overlay Sink and two new Pipelines" ) 
    {
        // Get the Device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        
        if (deviceProp.integrated)
        {
            std::wstring pipelineName1(L"test-pipeline-1");
            std::wstring pipelineName2(L"test-pipeline-2");
            std::wstring overlaySinkName = L"overlay-sink";
            uint displayId(0);
            uint depth(0);
            uint offsetX(0);
            uint offsetY(0);
            uint sinkW(1280);
            uint sinkH(720);

            REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), displayId, depth,
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_new(pipelineName1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_new(pipelineName2.c_str()) == DSL_RESULT_SUCCESS );

            WHEN( "The Overlay Sink is added to the first Pipeline" ) 
            {
                REQUIRE( dsl_pipeline_component_add(pipelineName1.c_str(), 
                    overlaySinkName.c_str()) == DSL_RESULT_SUCCESS );

                THEN( "The Overlay Sink can't be added to the second Pipeline" ) 
                {
                    REQUIRE( dsl_pipeline_component_add(pipelineName2.c_str(), 
                        overlaySinkName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );

                    REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                    REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                }
            }
        }
    }
}

SCENARIO( "The Components container is updated correctly on new Window Sink", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring windowSinkName(L"window-sink");
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Window Sink is created" ) 
        {

            REQUIRE( dsl_sink_window_new(windowSinkName.c_str(),
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 0 );
    }
}    

SCENARIO( "The Components container is updated correctly on Window Sink delete", "[sink-api]" )
{
    GIVEN( "An Window Sink Component" ) 
    {
        std::wstring windowSinkName = L"window-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_window_new(windowSinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        WHEN( "A new Window Sink is deleted" ) 
        {
            REQUIRE( dsl_component_delete(windowSinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Window Sink can update its Sync/Async attributes", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring windowSinkName = L"window-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_window_new(windowSinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        WHEN( "A the Window Sink's attributes are updated from the default" ) 
        {
            boolean newSync(false);
            REQUIRE( dsl_sink_sync_enabled_set(windowSinkName.c_str(), 
                newSync) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retSync(true);
                REQUIRE( dsl_sink_sync_enabled_get(windowSinkName.c_str(), 
                    &retSync) == DSL_RESULT_SUCCESS );
                REQUIRE( retSync == newSync );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "A Window Sink can update its force-aspect-ratio setting", "[sink-api]" )
{
    GIVEN( "A new window sink" ) 
    {
        std::wstring windowSinkName = L"window-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);
        boolean force(1);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_window_new(windowSinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        WHEN( "A Window Sink's force-aspect-ratio is set" ) 
        {
            REQUIRE( dsl_sink_window_force_aspect_ratio_set(windowSinkName.c_str(), 
                force) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retForce(false);
                REQUIRE( dsl_sink_window_force_aspect_ratio_get(windowSinkName.c_str(), &retForce) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( retForce == force );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "A Window Sink can be Reset", "[sink-api]" )
{
    GIVEN( "Given attributes for a new window sink" ) 
    {
        std::wstring windowSinkName = L"window-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);
        boolean force(1);

        WHEN( "The Window Sink is created" ) 
        {
            REQUIRE( dsl_sink_window_new(windowSinkName.c_str(), 
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

            THEN( "The Window Sink can be reset after creation" ) 
            {
                REQUIRE( dsl_sink_render_reset(windowSinkName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "A Window Sink in use can't be deleted", "[sink-api]" )
{
    GIVEN( "A new Window Sink and new Pipeline" ) 
    {
        std::wstring pipelineName  = L"test-pipeline";
        std::wstring windowSinkName = L"window-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_sink_window_new(windowSinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        WHEN( "The Window Sink is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                windowSinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Window Sink can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(windowSinkName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );
                
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Window Sink, once removed from a Pipeline, can be deleted", "[sink-api]" )
{
    GIVEN( "A new Sink owned by a new pPipeline" ) 
    {
        std::wstring pipelineName  = L"test-pipeline";
        std::wstring windowSinkName = L"window-sink";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_sink_window_new(windowSinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            windowSinkName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Window Sink is removed the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(), 
                windowSinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Window Sink can be deleted" ) 
            {
                REQUIRE( dsl_component_delete(windowSinkName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Window Sink in use can't be added to a second Pipeline", "[sink-api]" )
{
    GIVEN( "A new Window Sink and two new pPipelines" ) 
    {
        std::wstring pipelineName1(L"test-pipeline-1");
        std::wstring pipelineName2(L"test-pipeline-2");
        std::wstring windowSinkName = L"window-sink";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_sink_window_new(windowSinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName2.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Window Sink is added to the first Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName1.c_str(), 
                windowSinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Window Sink can't be added to the second Pipeline" ) 
            {
                REQUIRE( dsl_pipeline_component_add(pipelineName2.c_str(), 
                    windowSinkName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Window Sink's Offsets can be updated", "[sink-api]" )
{
    GIVEN( "A new Render Sink in memory" ) 
    {
        std::wstring sinkName = L"window-sink";
        uint sinkW(1280);
        uint sinkH(720);
        uint offsetX(0), offsetY(0);

        uint preOffsetX(100), preOffsetY(100);
        uint retOffsetX(0), retOffsetY(0);
        REQUIRE( dsl_sink_window_new(sinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        WHEN( "The Window Sink's Offsets are Set" ) 
        {
            REQUIRE( dsl_sink_render_offsets_set(sinkName.c_str(), 
                preOffsetX, preOffsetY) == DSL_RESULT_SUCCESS);
            
            THEN( "The correct values are returned on Get" ) 
            {
                dsl_sink_render_offsets_get(sinkName.c_str(), &retOffsetX, &retOffsetY);
                REQUIRE( preOffsetX == retOffsetX);
                REQUIRE( preOffsetY == retOffsetY);

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Window Sink's Dimensions can be updated", "[sink-api]" )
{
    GIVEN( "A new Window Sink in memory" ) 
    {
        std::wstring sinkName = L"window-sink";
        uint offsetX(100), offsetY(100);
        uint sinkW(1920), sinkH(1080);

        uint preSinkW(1280), preSinkH(720);
        uint retSinkW(0), retSinkH(0);

        REQUIRE( dsl_sink_window_new(sinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        WHEN( "The Window Sink's Dimensions are Set" ) 
        {
            REQUIRE( dsl_sink_render_dimensions_set(sinkName.c_str(), 
                preSinkW, preSinkH) == DSL_RESULT_SUCCESS);
            
            THEN( "The correct values are returned on Get" ) 
            {
                dsl_sink_render_dimensions_get(sinkName.c_str(), &retSinkW, &retSinkH);
                REQUIRE( preSinkW == retSinkW);
                REQUIRE( preSinkH == retSinkH);

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Components container is updated correctly on new File Sink", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring fileSinkName(L"file-sink");
        std::wstring filePath(L"./output.mp4");
        uint codec(DSL_CODEC_H265);
        uint container(DSL_CONTAINER_MP4);
        uint bitrate(2000000);
        uint interval(0);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new File Sink is created" ) 
        {
            REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
                codec, container, bitrate, interval) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                uint retCodec(0), retBitrate(0), retInterval(0);
                REQUIRE( dsl_sink_encode_settings_get(fileSinkName.c_str(), &retCodec, 
                    &retBitrate, &retInterval) == DSL_RESULT_SUCCESS );
                REQUIRE( retCodec == codec );
                REQUIRE( retBitrate == bitrate );
                REQUIRE( retInterval == interval );
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}    

SCENARIO( "The Components container is updated correctly on File Sink delete", "[sink-api]" )
{
    GIVEN( "An File Sink Component" ) 
    {
        std::wstring fileSinkName(L"file-sink");
        std::wstring filePath(L"./output.mp4");
        uint codec(DSL_CODEC_H265);
        uint container(DSL_CONTAINER_MP4);
        uint bitrate(2000000);
        uint interval(0);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
            codec, container, bitrate, interval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "A new File Sink is deleted" ) 
        {
            REQUIRE( dsl_component_delete(fileSinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "Creating a new File Sink with an invalid Codec will fail", "[sink-api]" )
{
    GIVEN( "Attributes for a new File Sink" ) 
    {
        std::wstring fileSinkName(L"file-sink");
        std::wstring filePath(L"./output.mp4");
        uint codec(DSL_CODEC_MPEG4 + 1);
        uint container(DSL_CONTAINER_MP4);
        uint bitrate(2000000);
        uint interval(0);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When creating a new File Sink with an invalid Codec" ) 
        {
            REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
                codec, container, bitrate, interval) == DSL_RESULT_SINK_CODEC_VALUE_INVALID );

            THEN( "The list size is left unchanged" ) 
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "Creating a new File Sink with an invalid Container will fail", "[sink-api]" )
{
    GIVEN( "Attributes for a new File Sink" ) 
    {
        std::wstring fileSinkName(L"file-sink");
        std::wstring filePath(L"./output.mp4");
        uint codec(DSL_CODEC_MPEG4);
        uint container(DSL_CONTAINER_MKV + 1);
        uint bitrate(2000000);
        uint interval(0);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When creating a new File Sink with an invalid Container" ) 
        {
            REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
                codec, container, bitrate, interval) == DSL_RESULT_SINK_CONTAINER_VALUE_INVALID );

            THEN( "The list size is left unchanged" ) 
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "A File Sink's Encoder settings can be updated", "[sink-api]" )
{
    GIVEN( "A new File Sink" ) 
    {
        std::wstring fileSinkName(L"file-sink");
        std::wstring filePath(L"./output.mp4");
        uint codec(DSL_CODEC_H265);
        uint container(DSL_CONTAINER_MP4);
        uint initBitrate(2000000);
        uint initInterval(0);

        REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
            codec, container, initBitrate, initInterval) == DSL_RESULT_SUCCESS );
            
        uint currCodec(0);
        uint currBitrate(0);
        uint currInterval(0);
    
        REQUIRE( dsl_sink_encode_settings_get(fileSinkName.c_str(), 
            &currCodec, &currBitrate, &currInterval) == DSL_RESULT_SUCCESS);
        REQUIRE( currCodec == codec );
        REQUIRE( currBitrate == initBitrate );
        REQUIRE( currInterval == initInterval );

        WHEN( "The FileSinkBintr's Encoder settings are Set" )
        {
            uint newCodec(DSL_CODEC_H264);
            uint newBitrate(2500000);
            uint newInterval(10);
            
            REQUIRE( dsl_sink_encode_settings_set(fileSinkName.c_str(), 
                newCodec, newBitrate, newInterval) == DSL_RESULT_SUCCESS);

            THEN( "The FileSinkBintr's new Encoder settings are returned on Get")
            {
                REQUIRE( dsl_sink_encode_settings_get(fileSinkName.c_str(), 
                    &currCodec, &currBitrate, &currInterval) == DSL_RESULT_SUCCESS);
                REQUIRE( currCodec == newCodec );
                REQUIRE( currBitrate == newBitrate );
                REQUIRE( currInterval == newInterval );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An invalid File Sink is caught on Encoder settings Get and Set", "[sink-api]" )
{
    GIVEN( "A new Fake Sink as incorrect Sink Type" ) 
    {
        std::wstring fakeSinkName(L"fake-sink");
            
        uint currCodec(0);
        uint currBitrate(0);
        uint currInterval(0);
    
        uint newCodec(1);
        uint newBitrate(2500000);
        uint newInterval(10);

        WHEN( "The File Sink Get-Set API called with a Fake sink" )
        {
            
            REQUIRE( dsl_sink_fake_new(fakeSinkName.c_str()) == DSL_RESULT_SUCCESS);

            THEN( "The File Sink encoder settings APIs fail correctly")
            {
                REQUIRE( dsl_sink_encode_settings_get(fakeSinkName.c_str(), &currCodec, &currBitrate, 
                    &currInterval) == DSL_RESULT_SINK_COMPONENT_IS_NOT_ENCODE_SINK);
                REQUIRE( dsl_sink_encode_settings_set(fakeSinkName.c_str(), newCodec,
                    newBitrate, newInterval) == DSL_RESULT_SINK_COMPONENT_IS_NOT_ENCODE_SINK);

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The Components container is updated correctly on new Record Sink", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring recordSinkName(L"record-sink");
        std::wstring outdir(L"./");
        uint container(DSL_CONTAINER_MP4);
        uint codec(DSL_CODEC_H264);
        uint bitrate(2000000);
        uint interval(0);

        dsl_record_client_listener_cb client_listener;

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Record Sink is created" ) 
        {
            REQUIRE( dsl_sink_record_new(recordSinkName.c_str(), outdir.c_str(),
                codec, container, bitrate, interval, client_listener) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                uint ret_cache_size(0);
                uint ret_width(0), ret_height(0);
                REQUIRE( dsl_sink_record_cache_size_get(recordSinkName.c_str(), 
                    &ret_cache_size) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_cache_size == DSL_DEFAULT_VIDEO_RECORD_CACHE_IN_SEC );
                REQUIRE( dsl_sink_record_dimensions_get(recordSinkName.c_str(), 
                    &ret_width, &ret_height) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_width == 0 );
                REQUIRE( ret_height == 0 );
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}    

SCENARIO( "The Components container is updated correctly on Record Sink delete", "[sink-api]" )
{
    GIVEN( "A Record Sink Component" ) 
    {
        std::wstring recordSinkName(L"record-sink");
        std::wstring outdir(L"./");
        uint container(DSL_CONTAINER_MP4);
        uint codec(DSL_CODEC_H264);
        uint bitrate(2000000);
        uint interval(0);

        dsl_record_client_listener_cb client_listener;

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_record_new(recordSinkName.c_str(), outdir.c_str(),
            codec, container, bitrate, interval, client_listener) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "A new Record Sink is deleted" ) 
        {
            REQUIRE( dsl_component_delete(recordSinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Player can be added to and removed from a Record Sink", "[sink-api]" )
{
    GIVEN( "A new Record Sink and Video Player" )
    {
        std::wstring recordSinkName(L"record-sink");
        std::wstring outdir(L"./");
        uint container(DSL_CONTAINER_MP4);
        uint codec(DSL_CODEC_H264);
        uint bitrate(2000000);
        uint interval(0);

        dsl_record_client_listener_cb client_listener;

        REQUIRE( dsl_sink_record_new(recordSinkName.c_str(), outdir.c_str(),
            codec, container, bitrate, interval, client_listener) == DSL_RESULT_SUCCESS );

        std::wstring player_name(L"player");
        std::wstring file_path = L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4";
        
        REQUIRE( dsl_player_render_video_new(player_name.c_str(),file_path.c_str(), 
            DSL_RENDER_TYPE_OVERLAY, 10, 10, 75, 0) == DSL_RESULT_SUCCESS );

        WHEN( "A capture-complete-listner is added" )
        {
            REQUIRE( dsl_sink_record_video_player_add(recordSinkName.c_str(),
                player_name.c_str()) == DSL_RESULT_SUCCESS );

            // ensure the same listener twice fails
            REQUIRE( dsl_sink_record_video_player_add(recordSinkName.c_str(),
                player_name.c_str()) == DSL_RESULT_SINK_PLAYER_ADD_FAILED );

            THEN( "The same listner can be remove" ) 
            {
                REQUIRE( dsl_sink_record_video_player_remove(recordSinkName.c_str(),
                    player_name.c_str()) == DSL_RESULT_SUCCESS );

                // calling a second time must fail
                REQUIRE( dsl_sink_record_video_player_remove(recordSinkName.c_str(),
                    player_name.c_str()) == DSL_RESULT_SINK_PLAYER_REMOVE_FAILED );
                    
                REQUIRE( dsl_component_delete(recordSinkName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_player_delete(player_name.c_str()) == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "A Mailer can be added to and removed from a Record Sink", "[sink-api]" )
{
    GIVEN( "A new Record Sink and Mailer" )
    {
        std::wstring recordSinkName(L"record-sink");
        std::wstring outdir(L"./");
        uint container(DSL_CONTAINER_MP4);
        uint codec(DSL_CODEC_H264);
        uint bitrate(2000000);
        uint interval(0);

        dsl_record_client_listener_cb client_listener;

        REQUIRE( dsl_sink_record_new(recordSinkName.c_str(), outdir.c_str(),
            codec, container, bitrate, interval, client_listener) == DSL_RESULT_SUCCESS );

        std::wstring mailer_name(L"mailer");
        std::wstring subject(L"Subject line");
        
        REQUIRE( dsl_mailer_new(mailer_name.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A Mailer is added" )
        {
            REQUIRE( dsl_sink_record_mailer_add(recordSinkName.c_str(),
                mailer_name.c_str(), subject.c_str()) == DSL_RESULT_SUCCESS );

            // ensure the same Mailer twice fails
            REQUIRE( dsl_sink_record_mailer_add(recordSinkName.c_str(),
                mailer_name.c_str(), subject.c_str()) == DSL_RESULT_SINK_MAILER_ADD_FAILED );

            THEN( "The Mailer can be removed" ) 
            {
                REQUIRE( dsl_sink_record_mailer_remove(recordSinkName.c_str(),
                    mailer_name.c_str()) == DSL_RESULT_SUCCESS );

                // calling a second time must fail
                REQUIRE( dsl_sink_record_mailer_remove(recordSinkName.c_str(),
                    mailer_name.c_str()) == DSL_RESULT_SINK_MAILER_REMOVE_FAILED );
                    
                REQUIRE( dsl_component_delete(recordSinkName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
                REQUIRE( dsl_mailer_delete(mailer_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_mailer_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The Components container is updated correctly on new DSL_CODEC_H264 RTSP Sink", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring rtspSinkName(L"rtsp-sink");
        std::wstring host(L"224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint codec(DSL_CODEC_H264);
        uint bitrate(4000000);
        uint interval(0);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new RTSP Sink is created" ) 
        {
            REQUIRE( dsl_sink_rtsp_new(rtspSinkName.c_str(), host.c_str(),
                udpPort, rtspPort, codec, bitrate, interval) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                uint retUdpPort(0), retRtspPort(0);
                dsl_sink_rtsp_server_settings_get(rtspSinkName.c_str(), &retUdpPort, &retRtspPort);
                REQUIRE( retUdpPort == udpPort );
                REQUIRE( retRtspPort == rtspPort );
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}    

SCENARIO( "The Components container is updated correctly on DSL_CODEC_H264 RTSP Sink delete", "[sink-api]" )
{
    GIVEN( "An RTSP Sink Component" ) 
    {
        std::wstring rtspSinkName(L"rtsp-sink");
        std::wstring host(L"224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint codec(DSL_CODEC_H264);
        uint bitrate(4000000);
        uint interval(0);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_rtsp_new(rtspSinkName.c_str(), host.c_str(),
            udpPort, rtspPort, codec, bitrate, interval) == DSL_RESULT_SUCCESS );

        WHEN( "A new RTSP Sink is deleted" ) 
        {
            REQUIRE( dsl_component_delete(rtspSinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The Components container is updated correctly on new DSL_CODEC_H265 RTSP Sink", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring rtspSinkName(L"rtsp-sink");
        std::wstring host(L"224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint codec(DSL_CODEC_H265);
        uint bitrate(4000000);
        uint interval(0);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new RTSP Sink is created" ) 
        {
            REQUIRE( dsl_sink_rtsp_new(rtspSinkName.c_str(), host.c_str(),
                udpPort, rtspPort, codec, bitrate, interval) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                uint retUdpPort(0), retRtspPort(0);
                dsl_sink_rtsp_server_settings_get(rtspSinkName.c_str(), &retUdpPort, &retRtspPort);
                REQUIRE( retUdpPort == udpPort );
                REQUIRE( retRtspPort == rtspPort );
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}    

SCENARIO( "The Components container is updated correctly on DSL_CODEC_H265 RTSP Sink delete", "[sink-api]" )
{
    GIVEN( "An RTSP Sink Component" ) 
    {
        std::wstring rtspSinkName(L"rtsp-sink");
        std::wstring host(L"224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint codec(DSL_CODEC_H265);
        uint bitrate(4000000);
        uint interval(0);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_rtsp_new(rtspSinkName.c_str(), host.c_str(),
            udpPort, rtspPort, codec, bitrate, interval) == DSL_RESULT_SUCCESS );

        WHEN( "A new RTSP Sink is deleted" ) 
        {
            REQUIRE( dsl_component_delete(rtspSinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
SCENARIO( "An RTSP Sink's Encoder settings can be updated", "[sink-api]" )
{
    GIVEN( "A new RTSP Sink" ) 
    {
        std::wstring rtspSinkName(L"rtsp-sink");
        std::wstring host(L"224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint codec(DSL_CODEC_H265);
        uint initBitrate(4000000);
        uint initInterval(0);

        REQUIRE( dsl_sink_rtsp_new(rtspSinkName.c_str(), host.c_str(),
            udpPort, rtspPort, codec, initBitrate, initInterval) == DSL_RESULT_SUCCESS );
            
        uint currCodec(99);
        uint currBitrate(0);
        uint currInterval(0);
    
        REQUIRE( dsl_sink_encode_settings_get(rtspSinkName.c_str(), 
            &currCodec, &currBitrate, &currInterval) == DSL_RESULT_SUCCESS);
        REQUIRE( currCodec == codec );
        REQUIRE( currBitrate == initBitrate );
        REQUIRE( currInterval == initInterval );

        WHEN( "The RTSP Sink's Encoder settings are Set" )
        {
            uint newCodec(DSL_CODEC_H265);
            uint newBitrate(2500000);
            uint newInterval(10);
            
            REQUIRE( dsl_sink_encode_settings_set(rtspSinkName.c_str(), 
                newCodec, newBitrate, newInterval) == DSL_RESULT_SUCCESS);

            THEN( "The RTSP Sink's new Encoder settings are returned on Get")
            {
                REQUIRE( dsl_sink_encode_settings_get(rtspSinkName.c_str(), 
                    &currCodec, &currBitrate, &currInterval) == DSL_RESULT_SUCCESS);
                REQUIRE( currCodec == newCodec);
                REQUIRE( currBitrate == newBitrate );
                REQUIRE( currInterval == newInterval );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An invalid RTSP Sink is caught on Encoder settings Get and Set", "[sink-api]" )
{
    GIVEN( "A new Fake Sink as incorrect Sink Type" ) 
    {
        std::wstring fakeSinkName(L"fake-sink");
            
        uint currCodec(DSL_CODEC_H264);
        uint currBitrate(0);
        uint currInterval(0);
    
        uint newCodec(DSL_CODEC_H265);
        uint newBitrate(2500000);
        uint newInterval(10);

        WHEN( "The RTSP Sink Get-Set API called with a Fake sink" )
        {
            REQUIRE( dsl_sink_fake_new(fakeSinkName.c_str()) == DSL_RESULT_SUCCESS);

            THEN( "The RTSP Sink encoder settings APIs fail correctly")
            {
                REQUIRE( dsl_sink_encode_settings_get(fakeSinkName.c_str(), 
                    &currCodec, &currBitrate, &currInterval) == DSL_RESULT_SINK_COMPONENT_IS_NOT_ENCODE_SINK);
                REQUIRE( dsl_sink_encode_settings_set(fakeSinkName.c_str(), 
                    currCodec, newBitrate, newInterval) == DSL_RESULT_SINK_COMPONENT_IS_NOT_ENCODE_SINK);

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Client is able to update the Sink in-use max", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_num_in_use_max_get() == DSL_DEFAULT_SINK_IN_USE_MAX );
        REQUIRE( dsl_sink_num_in_use_get() == 0 );
        
        WHEN( "The in-use-max is updated by the client" )   
        {
            uint new_max = 128;
            
            REQUIRE( dsl_sink_num_in_use_max_set(new_max) == true );
            
            THEN( "The new in-use-max will be returned to the client on get" )
            {
                REQUIRE( dsl_sink_num_in_use_max_get() == new_max );
            }
        }
    }
}

SCENARIO( "A Sink added to a Pipeline updates the in-use number", "[sink-api]" )
{
    GIVEN( "A new Sink and new Pipeline" )
    {
        std::wstring pipelineName  = L"test-pipeline";
        std::wstring windowSinkName = L"window-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_sink_window_new(windowSinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_num_in_use_get() == 0 );

        WHEN( "The Window Sink is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                windowSinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct in-use number is returned to the client" )
            {
                REQUIRE( dsl_sink_num_in_use_get() == 1 );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Sink removed from a Pipeline updates the in-use number", "[sink-api]" )
{
    GIVEN( "A new Pipeline with a Sink" ) 
    {
        std::wstring pipelineName  = L"test-pipeline";
        std::wstring windowSinkName = L"window-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_sink_window_new(windowSinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            windowSinkName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_sink_num_in_use_get() == 1 );

        WHEN( "The Source is removed from, the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(),
                windowSinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct in-use number is returned to the client" )
            {
                REQUIRE( dsl_sink_num_in_use_get() == 0 );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "Adding multiple Sinks to multiple Pipelines updates the in-use number", "[sink-api]" )
{
    GIVEN( "Two new Sinks and two new Pipeline" )
    {
        std::wstring sinkName1 = L"window-sink1";
        std::wstring pipelineName1  = L"test-pipeline1";
        std::wstring sinkName2 = L"window-sink2";
        std::wstring pipelineName2  = L"test-pipeline2";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_sink_window_new(sinkName1.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_sink_window_new(sinkName2.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName2.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_sink_num_in_use_get() == 0 );

        WHEN( "Each Sink is added to a different Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName1.c_str(), 
                sinkName1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_component_add(pipelineName2.c_str(), 
                sinkName2.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct in-use number is returned to the client" )
            {
                REQUIRE( dsl_sink_num_in_use_get() == 2 );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_source_num_in_use_get() == 0 );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "Adding greater than max Sinks to all Pipelines fails", "[sink-api]" )
{
    std::wstring sinkName1  = L"fake-sink1";
    std::wstring pipelineName1  = L"test-pipeline1";
    std::wstring sinkName2  = L"fake-sink2";
    std::wstring pipelineName2  = L"test-pipeline2";
    std::wstring sinkName3  = L"fake-sink3";
    std::wstring pipelineName3  = L"test-pipeline3";

    GIVEN( "Two new Sources and two new Pipeline" )
    {
        REQUIRE( dsl_sink_fake_new(sinkName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_sink_fake_new(sinkName2.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName2.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_sink_fake_new(sinkName3.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName3.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_sink_num_in_use_get() == 0 );

        // Reduce the max to less than 3
        REQUIRE( dsl_sink_num_in_use_max_set(2) == true );

        WHEN( "The max number of sinks are added to Pipelines" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName1.c_str(), 
                sinkName1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_component_add(pipelineName2.c_str(), 
                sinkName2.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "Adding an additional Source to a Pipeline will fail" )
            {
                REQUIRE( dsl_pipeline_component_add(pipelineName3.c_str(), 
                    sinkName3.c_str()) == DSL_RESULT_PIPELINE_SINK_MAX_IN_USE_REACHED );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                
                // restore the default for other scenarios
                REQUIRE( dsl_sink_num_in_use_max_set(DSL_DEFAULT_SINK_IN_USE_MAX) == true );
                
                REQUIRE( dsl_sink_num_in_use_get() == 0 );
            }
        }
    }
}

SCENARIO( "The Sink API checks for NULL input parameters", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring sinkName(L"test-sink");
        std::wstring otherName(L"other");
        
        uint cache_size(0), width(0), height(0), codec(0), container(0), bitrate(0), interval(0), udpPort(0), rtspPort(0);
        boolean is_on(0), reset_done(0), sync(0), async(0);
        
        std::wstring mailerName(L"mailer");
        
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_sink_app_new(NULL, 0, NULL, NULL) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_app_new(sinkName.c_str(), 0, NULL, NULL) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_fake_new(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                
                REQUIRE( dsl_sink_overlay_new(NULL, 0, 0, 0, 0, 0, 0 ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_window_new(NULL, 0, 0, 0, 0 ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_window_force_aspect_ratio_get(NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_window_force_aspect_ratio_get(sinkName.c_str(), 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_window_force_aspect_ratio_set(NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                
                REQUIRE( dsl_sink_render_reset(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                
                REQUIRE( dsl_sink_file_new(NULL, NULL, 0, 0, 0, 0 ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_file_new(sinkName.c_str(), NULL, 0, 0, 0, 0 ) == DSL_RESULT_INVALID_INPUT_PARAM );
                
                REQUIRE( dsl_sink_record_new(NULL, NULL, 0, 0, 0, 0, NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_new(sinkName.c_str(), NULL, 0, 0, 0, 0, NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_session_start(NULL, 0, 0, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_session_stop(NULL, false) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_cache_size_get(NULL, &cache_size) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_cache_size_set(NULL, cache_size) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_record_dimensions_get(NULL, &width, &height) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_dimensions_set(NULL, width, height) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_record_is_on_get(NULL, &is_on) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_record_reset_done_get(NULL, &reset_done) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_record_video_player_add(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_video_player_add(sinkName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_video_player_remove(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_video_player_remove(sinkName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_record_mailer_add(NULL, NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_mailer_add(sinkName.c_str(), NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_mailer_add(sinkName.c_str(), mailerName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_mailer_remove(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_mailer_remove(sinkName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_encode_settings_get(NULL, &codec, &bitrate, &interval) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_encode_settings_set(NULL, codec, bitrate, interval) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_rtsp_new(NULL, NULL, 0, 0, 0, 0, 0 ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtsp_new(sinkName.c_str(), NULL, 0, 0, 0, 0, 0 ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtsp_server_settings_get(NULL, &udpPort, &rtspPort) == DSL_RESULT_INVALID_INPUT_PARAM );
                
                REQUIRE( dsl_sink_pph_add(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_pph_add(sinkName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_sync_enabled_get(NULL, &sync) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_sync_enabled_set(NULL, sync) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
