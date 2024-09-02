/*
The MIT License

Copyright (c) 2019-2024, Prominence AI, Inc.

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
        std::wstring sink_name = L"app-sink";

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new App Sink is created" ) 
        {
            REQUIRE( dsl_sink_app_new(sink_name.c_str(), DSL_SINK_APP_DATA_TYPE_BUFFER, 
                new_buffer_cb, NULL) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean sync(false);
                REQUIRE( dsl_sink_sync_enabled_get(sink_name.c_str(), 
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
        std::wstring sink_name = L"app-sink";

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When an invalid data type is specified" ) 
        {
            uint invalid_data_type(DSL_SINK_APP_DATA_TYPE_BUFFER+1);

            THEN( "The Sink must fail to create" ) 
            {
                REQUIRE( dsl_sink_app_new(sink_name.c_str(), invalid_data_type, 
                    new_buffer_cb, NULL) == DSL_RESULT_SINK_SET_FAILED );
            }
        }
    }
}

SCENARIO( "An App Sink can update it Sync setting correctly", "[sink-api]" )
{
    GIVEN( "A new App Sink component" ) 
    {
        std::wstring sink_name = L"app-sink";

        REQUIRE( dsl_sink_app_new(sink_name.c_str(), DSL_SINK_APP_DATA_TYPE_BUFFER, 
            new_buffer_cb, NULL) == DSL_RESULT_SUCCESS );

        // check the default
        boolean retSync(TRUE);
        REQUIRE( dsl_sink_sync_enabled_get(sink_name.c_str(), 
            &retSync) == DSL_RESULT_SUCCESS );
        REQUIRE( retSync == TRUE );

        WHEN( "The App Sink's sync value is updated" ) 
        {
            boolean newSync(FALSE);
            REQUIRE( dsl_sink_sync_enabled_set(sink_name.c_str(), 
                newSync) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is retruned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                REQUIRE( dsl_sink_sync_enabled_get(sink_name.c_str(), 
                    &retSync) == DSL_RESULT_SUCCESS );
                REQUIRE( retSync == newSync );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "An App Sink can update its data-type setting correctly", "[sink-api]" )
{
    GIVEN( "A new App Sink Component" ) 
    {
        std::wstring sink_name = L"app-sink";
        uint init_data_type(DSL_SINK_APP_DATA_TYPE_BUFFER);
        
        REQUIRE( dsl_sink_app_new(sink_name.c_str(), init_data_type, 
            new_buffer_cb, NULL) == DSL_RESULT_SUCCESS );

        // Check the intial value
        uint ret_data_type;
        REQUIRE( dsl_sink_app_data_type_get(sink_name.c_str(), 
            &ret_data_type) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_data_type == init_data_type );

        WHEN( "The App Sink's data-type is updated" ) 
        {
            uint new_data_type(DSL_SINK_APP_DATA_TYPE_SAMPLE);
            
            REQUIRE( dsl_sink_app_data_type_set(sink_name.c_str(), 
                new_data_type) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_sink_app_data_type_get(sink_name.c_str(), 
                    &ret_data_type) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_data_type == new_data_type );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "An ivalid data-type is provided" ) 
        {
            uint new_data_type(DSL_SINK_APP_DATA_TYPE_BUFFER+1);

            THEN( "The set data-type service must fail" ) 
            {
                REQUIRE( dsl_sink_app_data_type_set(sink_name.c_str(), 
                    new_data_type) == DSL_RESULT_SINK_SET_FAILED);
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "The Components container is updated correctly on new and delete Frame-Capture Sink", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring action_name(L"capture-action");
        std::wstring outdir(L"./");

        std::wstring sink_name = L"frame-capture-sink";

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_ode_action_capture_frame_new(action_name.c_str(), 
            outdir.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A new App Sink is created" ) 
        {
            REQUIRE( dsl_sink_frame_capture_new(sink_name.c_str(), 
                action_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean sync(false);
                REQUIRE( dsl_sink_sync_enabled_get(sink_name.c_str(), 
                    &sync) == DSL_RESULT_SUCCESS );
                REQUIRE( sync == true );

                // delete and check the component count
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}
    
SCENARIO( "The Components container is updated correctly on new and delete Custom Sink",
    "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring sink_name = L"custom-sink";
        std::wstring element_name_1 = L"element-1";
        std::wstring element_name_2 = L"element-2";
        std::wstring factory_name_1 = L"capsfilter";
        std::wstring factory_name_2 = L"xvimagesink";


        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Custom Sink is created" ) 
        {
            REQUIRE( dsl_sink_custom_new(sink_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );

                // All get calls should fail .. since it's not derived from GST Base-Sink
                boolean enabled(false);
                int64_t max_lateness(99);
                REQUIRE( dsl_sink_sync_enabled_get(sink_name.c_str(), 
                    &enabled) == DSL_RESULT_SINK_GET_FAILED );
                REQUIRE( dsl_sink_async_enabled_get(sink_name.c_str(), 
                    &enabled) == DSL_RESULT_SINK_GET_FAILED );
                REQUIRE( dsl_sink_max_lateness_get(sink_name.c_str(), 
                    &max_lateness) == DSL_RESULT_SINK_GET_FAILED );
                REQUIRE( dsl_sink_qos_enabled_get(sink_name.c_str(), 
                    &enabled) == DSL_RESULT_SINK_GET_FAILED );
                REQUIRE( dsl_component_delete(sink_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "A new Custom Sink is created with multiple Elements" ) 
        {
            REQUIRE( dsl_gst_element_new(element_name_2.c_str(),
                factory_name_2.c_str()) == DSL_RESULT_SUCCESS );


            REQUIRE( dsl_sink_custom_new_element_add(sink_name.c_str(),
                element_name_2.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );

                REQUIRE( dsl_component_delete(sink_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_gst_element_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "A new Custom Sink is created with multiple Elements" ) 
        {
            REQUIRE( dsl_gst_element_new(element_name_1.c_str(),
                factory_name_1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_gst_element_new(element_name_2.c_str(),
                factory_name_2.c_str()) == DSL_RESULT_SUCCESS );

            const wchar_t* elements[] = {element_name_1.c_str(), 
                element_name_2.c_str(), NULL};
            
            REQUIRE( dsl_sink_custom_new_element_add_many(sink_name.c_str(),
                elements) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );

                REQUIRE( dsl_component_delete(sink_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_gst_element_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "A new Custom Sink can Add and Remove multiple GST Elements", 
    "[sink-api]" )
{
    GIVEN( "A new Custom Component and new GST Element" ) 
    {

        std::wstring sink_name = L"custom-sink";
        std::wstring element_name_1 = L"element-1";
        std::wstring element_name_2 = L"element-2";
        std::wstring factory_name_1 = L"capsfilter";
        std::wstring factory_name_2 = L"xvimagesink";
        
        REQUIRE( dsl_sink_custom_new(sink_name.c_str()) 
            == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_gst_element_new(element_name_1.c_str(),
            factory_name_1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_gst_element_new(element_name_2.c_str(),
            factory_name_2.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The GST Elements are added to the Custom Sink" ) 
        {
            REQUIRE( dsl_sink_custom_element_add(sink_name.c_str(), 
                element_name_1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_sink_custom_element_add(sink_name.c_str(), 
                element_name_2.c_str()) == DSL_RESULT_SUCCESS );
             
            THEN( "The same GST Elements can be removed correctly" ) 
            {
                REQUIRE( dsl_sink_custom_element_remove(sink_name.c_str(), 
                    element_name_1.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_sink_custom_element_remove(sink_name.c_str(), 
                    element_name_2.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_gst_element_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Components container is updated correctly on new and delete Fake Sink", 
    "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring sink_name = L"fake-sink";

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Fake Sink is created" ) 
        {
            REQUIRE( dsl_sink_fake_new(sink_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean sync(false);
                REQUIRE( dsl_sink_sync_enabled_get(sink_name.c_str(), 
                    &sync) == DSL_RESULT_SUCCESS );
                REQUIRE( sync == false );
                REQUIRE( dsl_component_delete(sink_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "The Components container is updated correctly on Fake Sink delete", "[sink-api]" )
{
    GIVEN( "A Fake Sink Component" ) 
    {
        std::wstring sink_name = L"fake-sink";


        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_fake_new(sink_name.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A new Fake Sink is deleted" ) 
        {
            REQUIRE( dsl_component_delete(sink_name.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Fake Sink can update it's common properties correctly", 
    "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring sink_name = L"fake-sink";

        REQUIRE( dsl_sink_fake_new(sink_name.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Fake Sink's sync property is updated from its default" ) 
        {
            boolean newSync(true); // default == false
            REQUIRE( dsl_sink_sync_enabled_set(sink_name.c_str(), 
                newSync) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retSync(false);
                REQUIRE( dsl_sink_sync_enabled_get(sink_name.c_str(), 
                    &retSync) == DSL_RESULT_SUCCESS );
                REQUIRE( retSync == newSync );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "The Fake Sink's async property is updated from its default" ) 
        {
            boolean newAsync(true);  // default == false
            REQUIRE( dsl_sink_async_enabled_set(sink_name.c_str(), 
                newAsync) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retAsync(false);
                REQUIRE( dsl_sink_async_enabled_get(sink_name.c_str(), 
                    &retAsync) == DSL_RESULT_SUCCESS );
                REQUIRE( retAsync == newAsync );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "The Fake Sink's max-lateness property is updated from its default" ) 
        {
            int64_t newMaxLateness(1);  // default == -1
            REQUIRE( dsl_sink_max_lateness_set(sink_name.c_str(), 
                newMaxLateness) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                int64_t retMaxLateness(12345678);
                REQUIRE( dsl_sink_max_lateness_get(sink_name.c_str(), 
                    &retMaxLateness) == DSL_RESULT_SUCCESS );
                REQUIRE( retMaxLateness == newMaxLateness );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "The Fake Sink's qos property is updated from its default" ) 
        {
            boolean newQos(true);  // default == false
            REQUIRE( dsl_sink_qos_enabled_set(sink_name.c_str(), 
                newQos) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retQos(false);
                REQUIRE( dsl_sink_qos_enabled_get(sink_name.c_str(), 
                    &retQos) == DSL_RESULT_SUCCESS );
                REQUIRE( retQos == newQos );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "The Components container is updated correctly on new 3D Sink", 
    "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        // Get the Device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        
        if (deviceProp.integrated)
        {
            std::wstring three_d_sink_name = L"3d-sink";
            uint offsetX(0);
            uint offsetY(0);
            uint sinkW(1280);
            uint sinkH(720);

            REQUIRE( dsl_component_list_size() == 0 );

            WHEN( "A new 3D Sink is created" ) 
            {

                REQUIRE( dsl_sink_window_3d_new(three_d_sink_name.c_str(),
                    offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

                THEN( "The list size is updated correctly" ) 
                {
                    REQUIRE( dsl_component_list_size() == 1 );

                    REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                    REQUIRE( dsl_component_list_size() == 0 );
                }
            }
        }
    }
}    

SCENARIO( "The Components container is updated correctly on 3D Sink delete", "[sink-api]" )
{
    GIVEN( "An 3D Sink Component" ) 
    {
        // Get the Device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        
        if (deviceProp.integrated)
        {
            std::wstring three_d_sink_name = L"3d-sink";
            uint offsetX(0);
            uint offsetY(0);
            uint sinkW(0);
            uint sinkH(0);

            REQUIRE( dsl_component_list_size() == 0 );
            REQUIRE( dsl_sink_window_3d_new(three_d_sink_name.c_str(),
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

            WHEN( "A new 3D Sink is deleted" ) 
            {
                REQUIRE( dsl_component_delete(three_d_sink_name.c_str()) == DSL_RESULT_SUCCESS );
                
                THEN( "The list size updated correctly" )
                {
                    REQUIRE( dsl_component_list_size() == 0 );
                }
            }
        }
    }
}

SCENARIO( "A 3D Sink can update it's common properties correctly", 
    "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        // Get the Device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        
        if (deviceProp.integrated)
        {
            std::wstring sink_name = L"3d-sink";
            uint offsetX(0);
            uint offsetY(0);
            uint sinkW(0);
            uint sinkH(0);

            REQUIRE( dsl_component_list_size() == 0 );
            REQUIRE( dsl_sink_window_3d_new(sink_name.c_str(),
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

            WHEN( "The 3D Sink's sync property is updated from its default" ) 
            {
                boolean newSync(false); // default == true
                REQUIRE( dsl_sink_sync_enabled_set(sink_name.c_str(), 
                    newSync) == DSL_RESULT_SUCCESS );

                THEN( "The correct value is returned on get" ) 
                {
                    REQUIRE( dsl_component_list_size() == 1 );
                    boolean retSync(true);
                    REQUIRE( dsl_sink_sync_enabled_get(sink_name.c_str(), 
                        &retSync) == DSL_RESULT_SUCCESS );
                    REQUIRE( retSync == newSync );
                    REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                    REQUIRE( dsl_component_list_size() == 0 );
                }
            }
            WHEN( "The 3D Sink's async property is updated from its default" ) 
            {
                boolean newAsync(true);  // default == false
                REQUIRE( dsl_sink_async_enabled_set(sink_name.c_str(), 
                    newAsync) == DSL_RESULT_SUCCESS );

                THEN( "The correct value is returned on get" ) 
                {
                    REQUIRE( dsl_component_list_size() == 1 );
                    boolean retAsync(false);
                    REQUIRE( dsl_sink_async_enabled_get(sink_name.c_str(), 
                        &retAsync) == DSL_RESULT_SUCCESS );
                    REQUIRE( retAsync == newAsync );
                    REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                    REQUIRE( dsl_component_list_size() == 0 );
                }
            }
            WHEN( "The 3D Sink's max-lateness property is updated from its default" ) 
            {
                int64_t newMaxLateness(-1);  // default == 20000000
                REQUIRE( dsl_sink_max_lateness_set(sink_name.c_str(), 
                    newMaxLateness) == DSL_RESULT_SUCCESS );

                THEN( "The correct value is returned on get" ) 
                {
                    REQUIRE( dsl_component_list_size() == 1 );
                    int64_t retMaxLateness(12345678);
                    REQUIRE( dsl_sink_max_lateness_get(sink_name.c_str(), 
                        &retMaxLateness) == DSL_RESULT_SUCCESS );
                    REQUIRE( retMaxLateness == newMaxLateness );
                    REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                    REQUIRE( dsl_component_list_size() == 0 );
                }
            }
            WHEN( "The 3D Sink's qos property is updated from its default" ) 
            {
                boolean newQos(true);  // default == false
                REQUIRE( dsl_sink_qos_enabled_set(sink_name.c_str(), 
                    newQos) == DSL_RESULT_SUCCESS );

                THEN( "The correct value is returned on get" ) 
                {
                    REQUIRE( dsl_component_list_size() == 1 );
                    boolean retQos(false);
                    REQUIRE( dsl_sink_qos_enabled_get(sink_name.c_str(), 
                        &retQos) == DSL_RESULT_SUCCESS );
                    REQUIRE( retQos == newQos );
                    REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                    REQUIRE( dsl_component_list_size() == 0 );
                }
            }
        }
    }
}    

SCENARIO( "A 3D Sink's Offsets can be updated", "[sink-api]" )
{
    GIVEN( "A new 3D Sink in memory" ) 
    {
        // Get the Device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        
        if (deviceProp.integrated)
        {
            std::wstring sink_name = L"3d-sink";
            uint offsetX(0);
            uint offsetY(0);
            uint sinkW(0);
            uint sinkH(0);

            uint preOffsetX(100), preOffsetY(100);
            uint retOffsetX(0), retOffsetY(0);

            REQUIRE( dsl_sink_window_3d_new(sink_name.c_str(),
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

            WHEN( "The Window Sink's Offsets are Set" ) 
            {
                REQUIRE( dsl_sink_window_offsets_set(sink_name.c_str(), 
                    preOffsetX, preOffsetY) == DSL_RESULT_SUCCESS);
                
                THEN( "The correct values are returned on Get" ) 
                {
                    dsl_sink_window_offsets_get(sink_name.c_str(), &retOffsetX, &retOffsetY);
                    REQUIRE( preOffsetX == retOffsetX);
                    REQUIRE( preOffsetY == retOffsetY);

                    REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                    REQUIRE( dsl_component_list_size() == 0 );
                }
            }
        }
    }
}

SCENARIO( "A 3D Sink's Dimensions can be updated", "[sink-api]" )
{
    GIVEN( "A new 3D Sink in memory" ) 
    {
        // Get the Device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        
        if (deviceProp.integrated)
        {
            std::wstring sink_name = L"3d-sink";
            uint offsetX(0);
            uint offsetY(0);
            uint sinkW(0);
            uint sinkH(0);

            uint preSinkW(1280), preSinkH(720);
            uint retSinkW(0), retSinkH(0);

            REQUIRE( dsl_sink_window_3d_new(sink_name.c_str(),
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

            WHEN( "The 3D Sink's Dimensions are Set" ) 
            {
                REQUIRE( dsl_sink_window_dimensions_set(sink_name.c_str(), 
                    preSinkW, preSinkH) == DSL_RESULT_SUCCESS);
                
                THEN( "The correct values are returned on Get" ) 
                {
                    dsl_sink_window_dimensions_get(sink_name.c_str(), &retSinkW, &retSinkH);
                    REQUIRE( preSinkW == retSinkW);
                    REQUIRE( preSinkH == retSinkH);

                    REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                    REQUIRE( dsl_component_list_size() == 0 );
                }
            }
        }
    }
}

SCENARIO( "An 3D Sink in use can't be deleted", "[sink-api]" )
{
    GIVEN( "A new 3D Sink and new pPipeline" ) 
    {
        // Get the Device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        
        if (deviceProp.integrated)
        {
            std::wstring pipelineName  = L"test-pipeline";
            std::wstring three_d_sink_name = L"3d-sink";
            uint offsetX(0);
            uint offsetY(0);
            uint sinkW(1280);
            uint sinkH(720);

            REQUIRE( dsl_sink_window_3d_new(three_d_sink_name.c_str(),
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_component_list_size() == 1 );
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_list_size() == 1 );

            WHEN( "The 3D Sink is added to the Pipeline" ) 
            {
                REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                    three_d_sink_name.c_str()) == DSL_RESULT_SUCCESS );

                THEN( "The 3D Sink can't be deleted" ) 
                {
                    REQUIRE( dsl_component_delete(three_d_sink_name.c_str()) == DSL_RESULT_COMPONENT_IN_USE );
                    
                    REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                    REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                    REQUIRE( dsl_pipeline_list_size() == 0 );
                    REQUIRE( dsl_component_list_size() == 0 );
                }
            }
        }
    }
}

SCENARIO( "An 3D Sink, once removed from a Pipeline, can be deleted", "[sink-api]" )
{
    GIVEN( "A new Sink owned by a new pPipeline" ) 
    {
        // Get the Device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        
        if (deviceProp.integrated)
        {
            std::wstring pipelineName  = L"test-pipeline";
            std::wstring three_d_sink_name = L"3d-sink";
            uint offsetX(0);
            uint offsetY(0);
            uint sinkW(1280);
            uint sinkH(720);

            REQUIRE( dsl_sink_window_3d_new(three_d_sink_name.c_str(),
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
                
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                three_d_sink_name.c_str()) == DSL_RESULT_SUCCESS );

            WHEN( "The 3D Sink is removed the Pipeline" ) 
            {
                REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(), 
                    three_d_sink_name.c_str()) == DSL_RESULT_SUCCESS );

                THEN( "The 3D Sink can be deleted" ) 
                {
                    REQUIRE( dsl_component_delete(three_d_sink_name.c_str()) == DSL_RESULT_SUCCESS );

                    REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                    REQUIRE( dsl_pipeline_list_size() == 0 );
                    REQUIRE( dsl_component_list_size() == 0 );
                }
            }
        }
    }
}

SCENARIO( "An 3D Sink in use can't be added to a second Pipeline", "[sink-api]" )
{
    GIVEN( "A new 3D Sink and two new Pipelines" ) 
    {
        // Get the Device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        
        if (deviceProp.integrated)
        {
            std::wstring pipelineName1(L"test-pipeline-1");
            std::wstring pipelineName2(L"test-pipeline-2");
            std::wstring three_d_sink_name = L"3d-sink";
            uint offsetX(0);
            uint offsetY(0);
            uint sinkW(1280);
            uint sinkH(720);

            REQUIRE( dsl_sink_window_3d_new(three_d_sink_name.c_str(),
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_new(pipelineName1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_new(pipelineName2.c_str()) == DSL_RESULT_SUCCESS );

            WHEN( "The 3D Sink is added to the first Pipeline" ) 
            {
                REQUIRE( dsl_pipeline_component_add(pipelineName1.c_str(), 
                    three_d_sink_name.c_str()) == DSL_RESULT_SUCCESS );

                THEN( "The 3D Sink can't be added to the second Pipeline" ) 
                {
                    REQUIRE( dsl_pipeline_component_add(pipelineName2.c_str(), 
                        three_d_sink_name.c_str()) == DSL_RESULT_COMPONENT_IN_USE );

                    REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                    REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                    REQUIRE( dsl_component_list_size() == 0 );
                }
            }
        }
    }
}

SCENARIO( "The Components container is updated correctly on new Window Sink", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring windowSinkName(L"egl-sink");
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Window Sink is created" ) 
        {

            REQUIRE( dsl_sink_window_egl_new(windowSinkName.c_str(),
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "The Components container is updated correctly on Window Sink delete", "[sink-api]" )
{
    GIVEN( "An Window Sink Component" ) 
    {
        std::wstring windowSinkName = L"egl-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_window_egl_new(windowSinkName.c_str(), 
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

SCENARIO( "A Window Sink can update it's common properties correctly", 
    "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        
        std::wstring sink_name = L"egl-sink";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_window_egl_new(sink_name.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        WHEN( "The Window Sink's sync property is updated from its default" ) 
        {
            boolean newSync(false); // default == true
            REQUIRE( dsl_sink_sync_enabled_set(sink_name.c_str(), 
                newSync) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retSync(true);
                REQUIRE( dsl_sink_sync_enabled_get(sink_name.c_str(), 
                    &retSync) == DSL_RESULT_SUCCESS );
                REQUIRE( retSync == newSync );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "The Window Sink's async property is updated from its default" ) 
        {
            boolean newAsync(true);  // default == false
            REQUIRE( dsl_sink_async_enabled_set(sink_name.c_str(), 
                newAsync) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retAsync(false);
                REQUIRE( dsl_sink_async_enabled_get(sink_name.c_str(), 
                    &retAsync) == DSL_RESULT_SUCCESS );
                REQUIRE( retAsync == newAsync );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "The Window Sink's max-lateness property is updated from its default" ) 
        {
            int64_t newMaxLateness(-1);  // default == 20000000
            REQUIRE( dsl_sink_max_lateness_set(sink_name.c_str(), 
                newMaxLateness) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                int64_t retMaxLateness(12345678);
                REQUIRE( dsl_sink_max_lateness_get(sink_name.c_str(), 
                    &retMaxLateness) == DSL_RESULT_SUCCESS );
                REQUIRE( retMaxLateness == newMaxLateness );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "The Window Sink's qos property is updated from its default" ) 
        {
            boolean newQos(true);  // default == false
            REQUIRE( dsl_sink_qos_enabled_set(sink_name.c_str(), 
                newQos) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retQos(false);
                REQUIRE( dsl_sink_qos_enabled_get(sink_name.c_str(), 
                    &retQos) == DSL_RESULT_SUCCESS );
                REQUIRE( retQos == newQos );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
    

SCENARIO( "A Window Sink can update its force-aspect-ratio setting", "[sink-api]" )
{
    GIVEN( "A new window sink" ) 
    {
        std::wstring windowSinkName = L"egl-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);
        boolean force(1);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_window_egl_new(windowSinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        WHEN( "A Window Sink's force-aspect-ratio is set" ) 
        {
            REQUIRE( dsl_sink_window_egl_force_aspect_ratio_set(windowSinkName.c_str(), 
                force) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retForce(false);
                REQUIRE( dsl_sink_window_egl_force_aspect_ratio_get(windowSinkName.c_str(), &retForce) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( retForce == force );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    
 

SCENARIO( "A Window Sinks full-screen-enabled setting can be Set/Get", "[sink-api]" )
{
    GIVEN( "A new Window Sink" ) 
    {
        std::wstring windowSinkName = L"egl-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);
        boolean defFullScreenEnabled(0);
        boolean retFullScreenEnabled(99);

        REQUIRE( dsl_sink_window_egl_new(windowSinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_fullscreen_enabled_get(windowSinkName.c_str(), 
            &retFullScreenEnabled) == DSL_RESULT_SUCCESS );
            
        // must be initialized false
        REQUIRE( retFullScreenEnabled == defFullScreenEnabled );

        WHEN( "When the Window Sinks full-screen-enabled setting is updated" ) 
        {
            boolean newFullScreenEnabled(1);
            
            REQUIRE( dsl_sink_window_fullscreen_enabled_set(windowSinkName.c_str(), 
                newFullScreenEnabled) == DSL_RESULT_SUCCESS );
                
            THEN( "The new values are returned on get" )
            {
                REQUIRE( dsl_sink_window_fullscreen_enabled_get(windowSinkName.c_str(), 
                    &retFullScreenEnabled) == DSL_RESULT_SUCCESS );
                    
                REQUIRE( retFullScreenEnabled == newFullScreenEnabled );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Window Sink's Handle can be Set/Get", "[sink-api]" )
{
    GIVEN( "A new Window Sink" ) 
    {
        std::wstring windowSinkName = L"egl-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);
        boolean defFullScreenEnabled(0);
        boolean retFullScreenEnabled(99);

        REQUIRE( dsl_sink_window_egl_new(windowSinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        
        uint64_t retHandle(0);
        
        REQUIRE( dsl_sink_window_handle_get(windowSinkName.c_str(), 
            &retHandle) == DSL_RESULT_SUCCESS );
            
        // must be initialized to NULL
        REQUIRE( retHandle == 0 );

        WHEN( "When the Window Sink's Handle is updated" ) 
        {
            uint64_t newHandle = 0x1234567812345678;
            
            REQUIRE( dsl_sink_window_handle_set(windowSinkName.c_str(), 
                newHandle) == DSL_RESULT_SUCCESS );
                
            THEN( "The new handle value is returned on get" )
            {
                REQUIRE( dsl_sink_window_handle_get(windowSinkName.c_str(), 
                    &retHandle) == DSL_RESULT_SUCCESS );
                    
                REQUIRE( retHandle == newHandle );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Window Sink in use can't be deleted", "[sink-api]" )
{
    GIVEN( "A new Window Sink and new Pipeline" ) 
    {
        std::wstring pipelineName  = L"test-pipeline";
        std::wstring windowSinkName = L"egl-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_sink_window_egl_new(windowSinkName.c_str(), 
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
        std::wstring windowSinkName = L"egl-sink";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_sink_window_egl_new(windowSinkName.c_str(), 
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
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
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
        std::wstring windowSinkName = L"egl-sink";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_sink_window_egl_new(windowSinkName.c_str(), 
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
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Window Sink's Offsets can be updated", "[sink-api]" )
{
    GIVEN( "A new Render Sink in memory" ) 
    {
        std::wstring sink_name = L"egl-sink";
        uint sinkW(1280);
        uint sinkH(720);
        uint offsetX(0), offsetY(0);

        uint preOffsetX(100), preOffsetY(100);
        uint retOffsetX(0), retOffsetY(0);
        REQUIRE( dsl_sink_window_egl_new(sink_name.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        WHEN( "The Window Sink's Offsets are Set" ) 
        {
            REQUIRE( dsl_sink_window_offsets_set(sink_name.c_str(), 
                preOffsetX, preOffsetY) == DSL_RESULT_SUCCESS);
            
            THEN( "The correct values are returned on Get" ) 
            {
                dsl_sink_window_offsets_get(sink_name.c_str(), &retOffsetX, &retOffsetY);
                REQUIRE( preOffsetX == retOffsetX);
                REQUIRE( preOffsetY == retOffsetY);

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Window Sink's Dimensions can be updated", "[sink-api]" )
{
    GIVEN( "A new Window Sink in memory" ) 
    {
        std::wstring sink_name = L"egl-sink";
        uint offsetX(100), offsetY(100);
        uint sinkW(1920), sinkH(1080);

        uint preSinkW(1280), preSinkH(720);
        uint retSinkW(0), retSinkH(0);

        REQUIRE( dsl_sink_window_egl_new(sink_name.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        WHEN( "The Window Sink's Dimensions are Set" ) 
        {
            REQUIRE( dsl_sink_window_dimensions_set(sink_name.c_str(), 
                preSinkW, preSinkH) == DSL_RESULT_SUCCESS);
            
            THEN( "The correct values are returned on Get" ) 
            {
                dsl_sink_window_dimensions_get(sink_name.c_str(), &retSinkW, &retSinkH);
                REQUIRE( preSinkW == retSinkW);
                REQUIRE( preSinkH == retSinkH);

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

static void my_sink_window_key_event_handler_cb(const wchar_t* key, void* client_data)
{
}

SCENARIO( "Window Sink Key Event Handlers are added and removed correctly ", 
    "[sink-api]" )
{
    GIVEN( "A Pipeline in memory" ) 
    {
        std::wstring sink_name = L"egl-sink";
        uint offsetX(100), offsetY(100);
        uint sinkW(1920), sinkH(1080);
        
        REQUIRE( dsl_sink_window_egl_new(sink_name.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        
        WHEN( "A XWindow Key Event Handler is added" )
        {
            REQUIRE( dsl_sink_window_key_event_handler_add(sink_name.c_str(),
                my_sink_window_key_event_handler_cb, 
                (void*)0x12345678) == DSL_RESULT_SUCCESS );

            // second attempt must fail
            REQUIRE( dsl_sink_window_key_event_handler_add(sink_name.c_str(),
                my_sink_window_key_event_handler_cb, 
                NULL) == DSL_RESULT_SINK_HANDLER_ADD_FAILED );

            THEN( "The same handler can be successfully removed" ) 
            {
                REQUIRE( dsl_sink_window_key_event_handler_remove(sink_name.c_str(),
                    my_sink_window_key_event_handler_cb) == DSL_RESULT_SUCCESS );

                // second attempt must fail
                REQUIRE( dsl_sink_window_key_event_handler_remove(sink_name.c_str(),
                    my_sink_window_key_event_handler_cb) 
                    == DSL_RESULT_SINK_HANDLER_REMOVE_FAILED );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
   
static void my_sink_window_button_event_handler_cb(uint button, 
    int xpos, int ypos, void* client_data)
{
}

SCENARIO( "Window Sink Button Event Handler are added and removded correctly", 
    "[sink-api]" )
{
    GIVEN( "A Pipeline in memory" ) 
    {
        std::wstring sink_name = L"egl-sink";
        uint offsetX(100), offsetY(100);
        uint sinkW(1920), sinkH(1080);
        
        REQUIRE( dsl_sink_window_egl_new(sink_name.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        
        WHEN( "A XWindow Button Event Handler is added" )
        {
            REQUIRE( dsl_sink_window_button_event_handler_add(sink_name.c_str(),
                my_sink_window_button_event_handler_cb, 
                (void*)0x12345678) == DSL_RESULT_SUCCESS );

            // second attempt must fail
            REQUIRE( dsl_sink_window_button_event_handler_add(sink_name.c_str(),
                my_sink_window_button_event_handler_cb, 
                (void*)0x12345678) == DSL_RESULT_SINK_HANDLER_ADD_FAILED );

            THEN( "The same handler can't be added again" ) 
            {
                REQUIRE( dsl_sink_window_button_event_handler_remove(sink_name.c_str(),
                    my_sink_window_button_event_handler_cb) == DSL_RESULT_SUCCESS );

                // second attempt must fail
                REQUIRE( dsl_sink_window_button_event_handler_remove(sink_name.c_str(),
                    my_sink_window_button_event_handler_cb) 
                    == DSL_RESULT_SINK_HANDLER_REMOVE_FAILED );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

static void my_sink_window_delete_event_handler_cb(void* client_data)
{
}

   
SCENARIO( "A XWindow Delete Event Handler must be unique", "[sink-api]" )
{
    GIVEN( "A Pipeline in memory" ) 
    {
        std::wstring sink_name = L"egl-sink";
        uint offsetX(100), offsetY(100);
        uint sinkW(1920), sinkH(1080);

        REQUIRE( dsl_sink_window_egl_new(sink_name.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        
        WHEN( "A XWindow Delete Event Handler is added" )
        {
            REQUIRE( dsl_sink_window_delete_event_handler_add(sink_name.c_str(),
                my_sink_window_delete_event_handler_cb, 
                (void*)0x12345678) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_sink_window_delete_event_handler_add(sink_name.c_str(),
                my_sink_window_delete_event_handler_cb, 
                (void*)0x12345678) == DSL_RESULT_SINK_HANDLER_ADD_FAILED );

            THEN( "The same handler can't be added again" ) 
            {
                REQUIRE( dsl_sink_window_delete_event_handler_remove(sink_name.c_str(),
                    my_sink_window_delete_event_handler_cb) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_sink_window_delete_event_handler_remove(sink_name.c_str(),
                    my_sink_window_delete_event_handler_cb) == DSL_RESULT_SINK_HANDLER_REMOVE_FAILED );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
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
        uint encoder(DSL_ENCODER_HW_H265);
        uint container(DSL_CONTAINER_MP4);
        uint bitrate(2000000);
        uint iframe_interval(30);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new File Sink is created" ) 
        {
            REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
                encoder, container, bitrate, iframe_interval) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                uint retEncoder(0), retBitrate(0), retInterval(0);
                REQUIRE( dsl_sink_encode_settings_get(fileSinkName.c_str(), &retEncoder, 
                    &retBitrate, &retInterval) == DSL_RESULT_SUCCESS );
                REQUIRE( retEncoder == encoder );
                REQUIRE( retBitrate == bitrate );
                REQUIRE( retInterval == iframe_interval );
                
                uint retHeight(99), retWidth(99);
                REQUIRE( dsl_sink_encode_dimensions_get(fileSinkName.c_str(), 
                    &retWidth, &retHeight) == DSL_RESULT_SUCCESS );
                REQUIRE( retWidth == 0 );
                REQUIRE( retHeight == 0 );

                REQUIRE( dsl_component_list_size() == 1 );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "The Components container is updated correctly on File Sink delete", "[sink-api]" )
{
    GIVEN( "An File Sink Component" ) 
    {
        std::wstring fileSinkName(L"file-sink");
        std::wstring filePath(L"./output.mp4");
        uint encoder(DSL_ENCODER_HW_H265);
        uint container(DSL_CONTAINER_MP4);
        uint bitrate(2000000);
        uint iframe_interval(30);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
            encoder, container, bitrate, iframe_interval) == DSL_RESULT_SUCCESS );
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

SCENARIO( "Creating a new File Sink with an invalid Encoder will fail", "[sink-api]" )
{
    GIVEN( "Attributes for a new File Sink" ) 
    {
        std::wstring fileSinkName(L"file-sink");
        std::wstring filePath(L"./output.mp4");
        uint encoder(DSL_ENCODER_SW_MPEG4 + 1);
        uint container(DSL_CONTAINER_MP4);
        uint bitrate(2000000);
        uint iframe_interval(30);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When creating a new File Sink with an invalid Encoder" ) 
        {
            REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
                encoder, container, bitrate, iframe_interval) == DSL_RESULT_SINK_ENCODER_VALUE_INVALID );

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
        uint encoder(DSL_ENCODER_HW_H265);
        uint container(DSL_CONTAINER_MKV + 1);
        uint bitrate(0);
        uint iframe_interval(30);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When creating a new File Sink with an invalid Container" ) 
        {
            REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
                encoder, container, bitrate, iframe_interval) == DSL_RESULT_SINK_CONTAINER_VALUE_INVALID );

            THEN( "The list size is left unchanged" ) 
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "A File Sink can update it's common properties correctly", 
    "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        
        std::wstring sink_name(L"file-sink");
        std::wstring file_path(L"./output.mp4");
        uint encoder(DSL_ENCODER_HW_H265);
        uint container(DSL_CONTAINER_MKV);
        uint bitrate(0);
        uint iframe_interval(30);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_file_new(sink_name.c_str(), file_path.c_str(),
            encoder, container, bitrate, iframe_interval) == DSL_RESULT_SUCCESS );

        WHEN( "The File Sink's sync property is updated from its default" ) 
        {
            boolean newSync(false); // default == true
            REQUIRE( dsl_sink_sync_enabled_set(sink_name.c_str(), 
                newSync) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retSync(true);
                REQUIRE( dsl_sink_sync_enabled_get(sink_name.c_str(), 
                    &retSync) == DSL_RESULT_SUCCESS );
                REQUIRE( retSync == newSync );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "The File Sink's async property is updated from its default" ) 
        {
            boolean newAsync(true);  // default == false
            REQUIRE( dsl_sink_async_enabled_set(sink_name.c_str(), 
                newAsync) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retAsync(false);
                REQUIRE( dsl_sink_async_enabled_get(sink_name.c_str(), 
                    &retAsync) == DSL_RESULT_SUCCESS );
                REQUIRE( retAsync == newAsync );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "The File Sink's max-lateness property is updated from its default" ) 
        {
            int64_t newMaxLateness(1);  // default == -1
            REQUIRE( dsl_sink_max_lateness_set(sink_name.c_str(), 
                newMaxLateness) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                int64_t retMaxLateness(12345678);
                REQUIRE( dsl_sink_max_lateness_get(sink_name.c_str(), 
                    &retMaxLateness) == DSL_RESULT_SUCCESS );
                REQUIRE( retMaxLateness == newMaxLateness );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "The File Sink's qos property is updated from its default" ) 
        {
            boolean newQos(true);  // default == false
            REQUIRE( dsl_sink_qos_enabled_set(sink_name.c_str(), 
                newQos) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retQos(false);
                REQUIRE( dsl_sink_qos_enabled_get(sink_name.c_str(), 
                    &retQos) == DSL_RESULT_SUCCESS );
                REQUIRE( retQos == newQos );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
    
SCENARIO( "A File Sink's dimensions can be updated", "[sink-api]" )
{
    GIVEN( "A new File Sink" ) 
    {
        std::wstring fileSinkName(L"file-sink");
        std::wstring filePath(L"./output.mp4");
        uint encoder(DSL_ENCODER_HW_H265);
        uint container(DSL_CONTAINER_MP4);
        uint initBitrate(4000000);
        uint initInterval(0);

        REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
            encoder, container, initBitrate, initInterval) == DSL_RESULT_SUCCESS );
            
        uint ret_width(99), ret_height(99);
    
        REQUIRE( dsl_sink_encode_dimensions_get(fileSinkName.c_str(), 
            &ret_width, &ret_height) == DSL_RESULT_SUCCESS);
        REQUIRE( ret_width == 0 );
        REQUIRE( ret_height == 0 );

        WHEN( "The FileSinkBintr's dimensions settings are Set" )
        {
            uint new_width(1280), new_height(720);
            
            REQUIRE( dsl_sink_encode_dimensions_set(fileSinkName.c_str(), 
                new_width, new_height) == DSL_RESULT_SUCCESS);

            THEN( "The FileSinkBintr's new Encoder settings are returned on Get")
            {
                REQUIRE( dsl_sink_encode_dimensions_get(fileSinkName.c_str(), 
                    &ret_width, &ret_height) == DSL_RESULT_SUCCESS);
                REQUIRE( ret_width == new_width );
                REQUIRE( ret_height == ret_height );

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
        uint encoder(DSL_ENCODER_HW_H264);
        uint bitrate(4000000);
        uint iframe_interval(30);

        dsl_record_client_listener_cb client_listener;

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Record Sink is created" ) 
        {
            REQUIRE( dsl_sink_record_new(recordSinkName.c_str(), outdir.c_str(),
                encoder, container, bitrate, iframe_interval, client_listener) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                uint ret_max_size(0);
                uint ret_cache_size(0);
                uint ret_width(0), ret_height(0);
                REQUIRE( dsl_sink_record_max_size_get(recordSinkName.c_str(), 
                    &ret_max_size) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == DSL_DEFAULT_VIDEO_RECORD_MAX_SIZE_IN_SEC );
                REQUIRE( dsl_sink_record_cache_size_get(recordSinkName.c_str(), 
                    &ret_cache_size) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_cache_size == DSL_DEFAULT_VIDEO_RECORD_CACHE_SIZE_IN_SEC );
                REQUIRE( dsl_sink_record_dimensions_get(recordSinkName.c_str(), 
                    &ret_width, &ret_height) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_width == 0 );
                REQUIRE( ret_height == 0 );
                REQUIRE( dsl_component_list_size() == 1 );
    
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "The Components container is updated correctly on Record Sink delete", "[sink-api]" )
{
    GIVEN( "A Record Sink Component" ) 
    {
        std::wstring recordSinkName(L"record-sink");
        std::wstring outdir(L"./");
        uint container(DSL_CONTAINER_MP4);
        uint encoder(DSL_ENCODER_HW_H264);
        uint bitrate(4000000);
        uint iframe_interval(30);

        dsl_record_client_listener_cb client_listener;

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_record_new(recordSinkName.c_str(), outdir.c_str(),
            encoder, container, bitrate, iframe_interval, client_listener) == DSL_RESULT_SUCCESS );
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
        uint encoder(DSL_ENCODER_HW_H264);
        uint bitrate(4000000);
        uint iframe_interval(30);

        dsl_record_client_listener_cb client_listener;

        REQUIRE( dsl_sink_record_new(recordSinkName.c_str(), outdir.c_str(),
            encoder, container, bitrate, iframe_interval, client_listener) == DSL_RESULT_SUCCESS );

        std::wstring player_name(L"player");
        std::wstring file_path = L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4";
        
        REQUIRE( dsl_player_render_video_new(player_name.c_str(),file_path.c_str(), 
            DSL_RENDER_TYPE_EGL, 10, 10, 75, 0) == DSL_RESULT_SUCCESS );

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

                REQUIRE( dsl_component_list_size() == 0 );
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
        uint encoder(DSL_ENCODER_HW_H264);
        uint bitrate(4000000);
        uint iframe_interval(30);

        dsl_record_client_listener_cb client_listener;

        REQUIRE( dsl_sink_record_new(recordSinkName.c_str(), outdir.c_str(),
            encoder, container, bitrate, iframe_interval, client_listener) == DSL_RESULT_SUCCESS );

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

SCENARIO( "The Components container is updated correctly on new RTMP Sink",
    "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring sink_name(L"rtmp-sink");
        std::wstring uri(L"rtmp://localhost/path/to/stream");
        uint encoder(DSL_ENCODER_SW_H264);
        uint bitrate(0);
        uint iframe_interval(30);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new RTMP Sink is created" ) 
        {
            REQUIRE( dsl_sink_rtmp_new(sink_name.c_str(),uri.c_str(),
                encoder, bitrate, iframe_interval) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                
                const wchar_t * c_ret_uri;
                REQUIRE( dsl_sink_rtmp_uri_get(sink_name.c_str(),
                    &c_ret_uri) == DSL_RESULT_SUCCESS );
                std::wstring ret_uri(c_ret_uri);
                REQUIRE( ret_uri == uri );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An RTMP Sink can update it's common properties correctly", 
    "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring sink_name(L"rtmp-sink");
        std::wstring uri(L"rtmp://localhost/path/to/stream");
        uint encoder(DSL_ENCODER_SW_H264);
        uint bitrate(0);
        uint iframe_interval(30);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_rtmp_new(sink_name.c_str(),uri.c_str(),
            encoder, bitrate, iframe_interval) == DSL_RESULT_SUCCESS );
        
        WHEN( "The RTMP Sink's sync property is updated from its default" ) 
        {
            boolean newSync(false); // default == true
            REQUIRE( dsl_sink_sync_enabled_set(sink_name.c_str(), 
                newSync) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retSync(true);
                REQUIRE( dsl_sink_sync_enabled_get(sink_name.c_str(), 
                    &retSync) == DSL_RESULT_SUCCESS );
                REQUIRE( retSync == newSync );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "The RTMP Sink's async property is updated from its default" ) 
        {
            boolean newAsync(true);  // default == false
            REQUIRE( dsl_sink_async_enabled_set(sink_name.c_str(), 
                newAsync) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retAsync(false);
                REQUIRE( dsl_sink_async_enabled_get(sink_name.c_str(), 
                    &retAsync) == DSL_RESULT_SUCCESS );
                REQUIRE( retAsync == newAsync );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "The RTMP Sink's max-lateness property is updated from its default" ) 
        {
            int64_t newMaxLateness(-1);  // default == 20000000
            REQUIRE( dsl_sink_max_lateness_set(sink_name.c_str(), 
                newMaxLateness) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                int64_t retMaxLateness(12345678);
                REQUIRE( dsl_sink_max_lateness_get(sink_name.c_str(), 
                    &retMaxLateness) == DSL_RESULT_SUCCESS );
                REQUIRE( retMaxLateness == newMaxLateness );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "The RTMP Sink's qos property is updated from its default" ) 
        {
            boolean newQos(true);  // default == false
            REQUIRE( dsl_sink_qos_enabled_set(sink_name.c_str(), 
                newQos) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retQos(false);
                REQUIRE( dsl_sink_qos_enabled_get(sink_name.c_str(), 
                    &retQos) == DSL_RESULT_SUCCESS );
                REQUIRE( retQos == newQos );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An RTMP Sink can update it's URI correctly", 
    "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring sink_name(L"rtmp-sink");
        std::wstring uri(L"rtmp://localhost/path/to/stream");
        uint encoder(DSL_ENCODER_SW_H264);
        uint bitrate(0);
        uint iframe_interval(30);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_rtmp_new(sink_name.c_str(),uri.c_str(),
            encoder, bitrate, iframe_interval) == DSL_RESULT_SUCCESS );
        
        WHEN( "The RTMP URI is updated" ) 
        {
            std::wstring new_uri(L"rtmp://0.0.0.0/new/path/to/stream"); 
            REQUIRE( dsl_sink_rtmp_uri_set(sink_name.c_str(), 
                new_uri.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );

                const wchar_t * c_ret_uri;
                REQUIRE( dsl_sink_rtmp_uri_get(sink_name.c_str(),
                    &c_ret_uri) == DSL_RESULT_SUCCESS );
                std::wstring ret_uri(c_ret_uri);
                REQUIRE( ret_uri == new_uri );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The Components container is updated correctly on new DSL_ENCODER_HW_H264 RTSP Sink", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring rtspSinkName(L"rtsp-sink");
        std::wstring host(L"224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint encoder(DSL_ENCODER_HW_H264);
        uint bitrate(4000000);
        uint iframe_interval(30);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new RTSP Sink is created" ) 
        {
            REQUIRE( dsl_sink_rtsp_server_new(rtspSinkName.c_str(), host.c_str(),
                udpPort, rtspPort, encoder, bitrate, iframe_interval) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                uint retUdpPort(0), retRtspPort(0);
                dsl_sink_rtsp_server_settings_get(rtspSinkName.c_str(), &retUdpPort, &retRtspPort);
                REQUIRE( retUdpPort == udpPort );
                REQUIRE( retRtspPort == rtspPort );
                REQUIRE( dsl_component_list_size() == 1 );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "The Components container is updated correctly on DSL_ENCODER_HW_H264 RTSP Sink delete", "[sink-api]" )
{
    GIVEN( "An RTSP Sink Component" ) 
    {
        std::wstring rtspSinkName(L"rtsp-sink");
        std::wstring host(L"224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint encoder(DSL_ENCODER_HW_H264);
        uint bitrate(4000000);
        uint iframe_interval(30);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_rtsp_server_new(rtspSinkName.c_str(), host.c_str(),
            udpPort, rtspPort, encoder, bitrate, iframe_interval) == DSL_RESULT_SUCCESS );

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

SCENARIO( "The Components container is updated correctly on new DSL_ENCODER_HW_H265 RTSP Sink", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring rtspSinkName(L"rtsp-sink");
        std::wstring host(L"224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint encoder(DSL_ENCODER_HW_H265);
        uint bitrate(4000000);
        uint iframe_interval(30);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new RTSP Sink is created" ) 
        {
            REQUIRE( dsl_sink_rtsp_server_new(rtspSinkName.c_str(), host.c_str(),
                udpPort, rtspPort, encoder, bitrate, iframe_interval) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                uint retUdpPort(0), retRtspPort(0);
                dsl_sink_rtsp_server_settings_get(rtspSinkName.c_str(), &retUdpPort, &retRtspPort);
                REQUIRE( retUdpPort == udpPort );
                REQUIRE( retRtspPort == rtspPort );
                REQUIRE( dsl_component_list_size() == 1 );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "The Components container is updated correctly on DSL_ENCODER_HW_H265 RTSP Sink delete", "[sink-api]" )
{
    GIVEN( "An RTSP Sink Component" ) 
    {
        std::wstring rtspSinkName(L"rtsp-sink");
        std::wstring host(L"224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint encoder(DSL_ENCODER_HW_H265);
        uint bitrate(4000000);
        uint iframe_interval(30);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_rtsp_server_new(rtspSinkName.c_str(), host.c_str(),
            udpPort, rtspPort, encoder, bitrate, iframe_interval) == DSL_RESULT_SUCCESS );

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

SCENARIO( "A RTSP Sink can update it's common properties correctly", 
    "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring sink_name(L"rtsp-sink");
        std::wstring host(L"224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint encoder(DSL_ENCODER_HW_H265);
        uint initBitrate(4000000);
        uint initInterval(0);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_rtsp_server_new(sink_name.c_str(), host.c_str(),
            udpPort, rtspPort, encoder, initBitrate, initInterval) == DSL_RESULT_SUCCESS );

        WHEN( "The RTSP Sink's sync property is updated from its default" ) 
        {
            boolean newSync(false); // default == true
            REQUIRE( dsl_sink_sync_enabled_set(sink_name.c_str(), 
                newSync) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retSync(true);
                REQUIRE( dsl_sink_sync_enabled_get(sink_name.c_str(), 
                    &retSync) == DSL_RESULT_SUCCESS );
                REQUIRE( retSync == newSync );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "The RTSP Sink's async property is updated from its default" ) 
        {
            boolean newAsync(true);  // default == false
            REQUIRE( dsl_sink_async_enabled_set(sink_name.c_str(), 
                newAsync) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retAsync(false);
                REQUIRE( dsl_sink_async_enabled_get(sink_name.c_str(), 
                    &retAsync) == DSL_RESULT_SUCCESS );
                REQUIRE( retAsync == newAsync );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "The RTSP Sink's max-lateness property is updated from its default" ) 
        {
            int64_t newMaxLateness(1);  // default == -1
            REQUIRE( dsl_sink_max_lateness_set(sink_name.c_str(), 
                newMaxLateness) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                int64_t retMaxLateness(12345678);
                REQUIRE( dsl_sink_max_lateness_get(sink_name.c_str(), 
                    &retMaxLateness) == DSL_RESULT_SUCCESS );
                REQUIRE( retMaxLateness == newMaxLateness );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "The RTSP Sink's qos property is updated from its default" ) 
        {
            boolean newQos(true);  // default == false
            REQUIRE( dsl_sink_qos_enabled_set(sink_name.c_str(), 
                newQos) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retQos(false);
                REQUIRE( dsl_sink_qos_enabled_get(sink_name.c_str(), 
                    &retQos) == DSL_RESULT_SUCCESS );
                REQUIRE( retQos == newQos );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
    

SCENARIO( "The Components container is updated correctly on new RTSP-Client Sink", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring rtspClientSinkName(L"rtsp-client-sink");
        std::wstring uri(L"rtsp://server_endpoint/stream");
        uint encoder(DSL_ENCODER_HW_H264);
        uint bitrate(2000000);
        uint iframe_interval(30);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new File Sink is created" ) 
        {
            REQUIRE( dsl_sink_rtsp_client_new(rtspClientSinkName.c_str(), 
                uri.c_str(), encoder, bitrate, iframe_interval) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                uint retEncoder(0), retBitrate(0), retInterval(0);
                REQUIRE( dsl_sink_encode_settings_get(rtspClientSinkName.c_str(), 
                    &retEncoder, &retBitrate, &retInterval) == DSL_RESULT_SUCCESS );
                REQUIRE( retEncoder == encoder );
                REQUIRE( retBitrate == bitrate );
                REQUIRE( retInterval == iframe_interval );
                
                uint retHeight(99), retWidth(99);
                REQUIRE( dsl_sink_encode_dimensions_get(rtspClientSinkName.c_str(), 
                    &retWidth, &retHeight) == DSL_RESULT_SUCCESS );
                REQUIRE( retWidth == 0 );
                REQUIRE( retHeight == 0 );

                REQUIRE( dsl_component_list_size() == 1 );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "A new RTSP-Client Sink can set its credentials", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring rtspClientSinkName(L"rtsp-client-sink");
        std::wstring uri(L"rtsp://server_endpoint/stream");
        uint encoder(DSL_ENCODER_HW_H264);
        uint bitrate(0);
        uint iframe_interval(30);
        
        std::wstring user_id(L"admin");
        std::wstring user_pw(L"123456");

        REQUIRE( dsl_sink_rtsp_client_new(rtspClientSinkName.c_str(), 
            uri.c_str(), encoder, bitrate, iframe_interval) == DSL_RESULT_SUCCESS );

        WHEN( "A the RTSP-Client Sink's credentials are upaded" ) 
        {
            REQUIRE( dsl_sink_rtsp_client_credentials_set(rtspClientSinkName.c_str(), 
                user_id.c_str(), user_pw.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                // Needs to be verified by using the logs level=4

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "A new RTSP-Client Sink can update its properties correctly", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring rtspClientSinkName(L"rtsp-client-sink");
        std::wstring uri(L"rtsp://server_endpoint/stream");
        uint encoder(DSL_ENCODER_HW_H264);
        uint bitrate(0);
        uint iframe_interval(30);
        
        uint def_latency(2000);
        uint def_profiles(DSL_RTSP_PROFILE_AVP);
        uint def_protocols(DSL_RTSP_LOWER_TRANS_TCP |
            DSL_RTSP_LOWER_TRANS_UDP_MCAST | DSL_RTSP_LOWER_TRANS_UDP);
        uint def_flags(DSL_TLS_CERTIFICATE_VALIDATE_ALL);
        
        REQUIRE( dsl_sink_rtsp_client_new(rtspClientSinkName.c_str(), 
            uri.c_str(), encoder, bitrate, iframe_interval) == DSL_RESULT_SUCCESS );

        uint ret_latency(123);
        REQUIRE( dsl_sink_rtsp_client_latency_get(rtspClientSinkName.c_str(), 
            &ret_latency) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_latency == def_latency );

        uint ret_profiles(0);
        REQUIRE( dsl_sink_rtsp_client_profiles_get(rtspClientSinkName.c_str(), 
            &ret_profiles) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_profiles == def_profiles );

        uint ret_protocols(0);
        REQUIRE( dsl_sink_rtsp_client_protocols_get(rtspClientSinkName.c_str(), 
            &ret_protocols) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_protocols == def_protocols );

        uint ret_flags(0);
        REQUIRE( dsl_sink_rtsp_client_tls_validation_flags_get(
            rtspClientSinkName.c_str(), &ret_flags) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_flags == def_flags );

        WHEN( "A new RTSP-Client Sink's latency is updated" ) 
        {
            uint new_latency(1000);
            REQUIRE( dsl_sink_rtsp_client_latency_set(rtspClientSinkName.c_str(), 
                new_latency) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_sink_rtsp_client_latency_get(rtspClientSinkName.c_str(), 
                    &ret_latency) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_latency == new_latency );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "A new RTSP-Client Sink's profiles are updated" ) 
        {
            uint new_profiles(DSL_RTSP_PROFILE_SAVPF);
            REQUIRE( dsl_sink_rtsp_client_profiles_set(rtspClientSinkName.c_str(), 
                new_profiles) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_sink_rtsp_client_profiles_get(rtspClientSinkName.c_str(), 
                    &ret_profiles) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_profiles == new_profiles );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "A new RTSP-Client Sink's protocols are updated" ) 
        {
            uint new_protocols(DSL_RTSP_LOWER_TRANS_HTTP);
            REQUIRE( dsl_sink_rtsp_client_protocols_set(rtspClientSinkName.c_str(), 
                new_protocols) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_sink_rtsp_client_protocols_get(rtspClientSinkName.c_str(), 
                    &ret_protocols) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_protocols == new_protocols );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "A new RTSP-Client Sink's tls-validation-flags are updated" ) 
        {
            uint new_flags(DSL_TLS_CERTIFICATE_BAD_IDENTITY);
            REQUIRE( dsl_sink_rtsp_client_tls_validation_flags_set(
                rtspClientSinkName.c_str(), new_flags) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_sink_rtsp_client_tls_validation_flags_get(
                    rtspClientSinkName.c_str(), &ret_flags) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_flags == new_flags );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "The Components container is updated correctly on new and delete Multi-Image Sink", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring sink_name = L"multi-image-sink";
        std::wstring file_path(L"./frame-%05d.jpg");

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Multi-Image Sink is created" ) 
        {
            REQUIRE( dsl_sink_image_multi_new(sink_name.c_str(),
                file_path.c_str(), 0, 0, 1, 1) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );

                // delete and check the component count
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An Multi-Image Sink can update its property settings correctly", "[sink-api]" )
{
    GIVEN( "A new Multi-Image Sink component" ) 
    {
        std::wstring sink_name = L"multi-image-sink";
        std::wstring file_path(L"./frame-%05d.jpg");
        
        uint width(0), height(0);
        uint fps_n(1), fps_d(2);

        REQUIRE( dsl_sink_image_multi_new(sink_name.c_str(),
            file_path.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );

        WHEN( "The Multi-Image Sinks's file_path is updated" ) 
        {
            // check the default
            const wchar_t* ret_c_file_path;
            REQUIRE( dsl_sink_image_multi_file_path_get(sink_name.c_str(), 
                &ret_c_file_path) == DSL_RESULT_SUCCESS );
            std::wstring ret_file_path(ret_c_file_path);
            REQUIRE( ret_file_path == file_path );

            std::wstring new_file_path(L"./new/path/new-file-%2d.jpeg");
            REQUIRE( dsl_sink_image_multi_file_path_set(sink_name.c_str(), 
                new_file_path.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct values are returned on get" ) 
            {
                REQUIRE( dsl_sink_image_multi_file_path_get(sink_name.c_str(), 
                    &ret_c_file_path) == DSL_RESULT_SUCCESS );
                std::wstring ret_file_path(ret_c_file_path);
                REQUIRE( ret_file_path == new_file_path );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "The Multi-Image Sinks's dimensions are updated" ) 
        {
            // check the default
            uint ret_width(123), ret_height(123);
            REQUIRE( dsl_sink_image_multi_dimensions_get(sink_name.c_str(), 
                &ret_width, &ret_height) == DSL_RESULT_SUCCESS );
            REQUIRE( ret_width == width );
            REQUIRE( ret_height == height );

            uint new_width(1280), new_height(720);
            REQUIRE( dsl_sink_image_multi_dimensions_set(sink_name.c_str(), 
                new_width, new_height) == DSL_RESULT_SUCCESS );

            THEN( "The correct values are returned on get" ) 
            {
                REQUIRE( dsl_sink_image_multi_dimensions_get(sink_name.c_str(), 
                    &ret_width, &ret_height) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_width == new_width );
                REQUIRE( ret_height == new_height );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "The Multi-Image Sink's frame-rate is updated" ) 
        {
            // check the default
            uint ret_fps_n(123), ret_fps_d(123);
            REQUIRE( dsl_sink_image_multi_frame_rate_get(sink_name.c_str(), 
                &ret_fps_n, &ret_fps_d) == DSL_RESULT_SUCCESS );
            REQUIRE( ret_fps_n == fps_n );
            REQUIRE( ret_fps_d == fps_d );

            uint new_fps_n(2), new_fps_d(6);
            REQUIRE( dsl_sink_image_multi_frame_rate_set(sink_name.c_str(), 
                new_fps_n, new_fps_d) == DSL_RESULT_SUCCESS );

            THEN( "The correct values are returned on get" ) 
            {
                REQUIRE( dsl_sink_image_multi_frame_rate_get(sink_name.c_str(), 
                    &ret_fps_n, &ret_fps_d) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_fps_n == new_fps_n );
                REQUIRE( ret_fps_d == new_fps_d );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "The Multi-Image Sink's max-file setting is updated" ) 
        {
            // check the default
            uint ret_max_file(123);
            REQUIRE( dsl_sink_image_multi_file_max_get(sink_name.c_str(), 
                &ret_max_file) == DSL_RESULT_SUCCESS );
            REQUIRE( ret_max_file == 0 );

            uint new_max_file(123);
            REQUIRE( dsl_sink_image_multi_file_max_set(sink_name.c_str(), 
                new_max_file) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_sink_image_multi_file_max_get(sink_name.c_str(), 
                    &ret_max_file) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_file == new_max_file );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "A Multi-Image Sink can update it's common properties correctly", 
    "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring sink_name = L"multi-image-sink";
        std::wstring file_path(L"./frame-%05d.jpg");
        
        uint width(0), height(0);
        uint fps_n(1), fps_d(2);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_image_multi_new(sink_name.c_str(),
            file_path.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );

        WHEN( "The Multi-Image Sink's sync property is updated from its default" ) 
        {
            boolean newSync(false); // default == true
            REQUIRE( dsl_sink_sync_enabled_set(sink_name.c_str(), 
                newSync) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retSync(true);
                REQUIRE( dsl_sink_sync_enabled_get(sink_name.c_str(), 
                    &retSync) == DSL_RESULT_SUCCESS );
                REQUIRE( retSync == newSync );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "The Multi-Image Sink's async property is updated from its default" ) 
        {
            boolean newAsync(true);  // default == false
            REQUIRE( dsl_sink_async_enabled_set(sink_name.c_str(), 
                newAsync) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retAsync(false);
                REQUIRE( dsl_sink_async_enabled_get(sink_name.c_str(), 
                    &retAsync) == DSL_RESULT_SUCCESS );
                REQUIRE( retAsync == newAsync );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "The Multi-Image Sink's max-lateness property is updated from its default" ) 
        {
            int64_t newMaxLateness(1);  // default == -1
            REQUIRE( dsl_sink_max_lateness_set(sink_name.c_str(), 
                newMaxLateness) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                int64_t retMaxLateness(12345678);
                REQUIRE( dsl_sink_max_lateness_get(sink_name.c_str(), 
                    &retMaxLateness) == DSL_RESULT_SUCCESS );
                REQUIRE( retMaxLateness == newMaxLateness );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "The Multi-Image Sink's qos property is updated from its default" ) 
        {
            boolean newQos(true);  // default == false
            REQUIRE( dsl_sink_qos_enabled_set(sink_name.c_str(), 
                newQos) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retQos(true);
                REQUIRE( dsl_sink_qos_enabled_get(sink_name.c_str(), 
                    &retQos) == DSL_RESULT_SUCCESS );
                REQUIRE( retQos == newQos );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The Components container is updated correctly on new V4L2 Sink", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring v4l2_sink_name(L"v4l2-sink");
        std::wstring device_location(L"/dev/video10");

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new V4L2 Sink is created" ) 
        {
            REQUIRE( dsl_sink_v4l2_new(v4l2_sink_name.c_str(), 
                device_location.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                const wchar_t* c_ret_device_location;
                REQUIRE( dsl_sink_v4l2_device_location_get(v4l2_sink_name.c_str(), 
                    &c_ret_device_location) == DSL_RESULT_SUCCESS );
                std::wstring ret_device_location(c_ret_device_location);
                REQUIRE( ret_device_location == device_location );

                const wchar_t* c_ret_device_name;
                REQUIRE( dsl_sink_v4l2_device_name_get(v4l2_sink_name.c_str(), 
                    &c_ret_device_name) == DSL_RESULT_SUCCESS );
                std::wstring ret_device_name(c_ret_device_name);
                REQUIRE( ret_device_name == L"" );

                int ret_device_fd;
                REQUIRE( dsl_sink_v4l2_device_fd_get(v4l2_sink_name.c_str(), 
                    &ret_device_fd) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_device_fd == -1 );

                uint ret_device_flags;
                REQUIRE( dsl_sink_v4l2_device_flags_get(v4l2_sink_name.c_str(), 
                    &ret_device_flags) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_device_flags == DSL_V4L2_DEVICE_TYPE_NONE );

                const wchar_t* c_bufffer_in_format;
                REQUIRE( dsl_sink_v4l2_buffer_in_format_get(v4l2_sink_name.c_str(), 
                    &c_bufffer_in_format) == DSL_RESULT_SUCCESS );
                std::wstring ret_buffer_in_format(c_bufffer_in_format);
                REQUIRE( ret_buffer_in_format == DSL_VIDEO_FORMAT_I420 );

                int retBrightness(0), retContrast(0), retSaturation(0);
                REQUIRE( dsl_sink_v4l2_picture_settings_get(v4l2_sink_name.c_str(), 
                    &retBrightness, &retContrast, &retSaturation) == DSL_RESULT_SUCCESS );
                REQUIRE( retBrightness == 0 );
                REQUIRE( retContrast == 0 );
                REQUIRE( retSaturation == 0 );

                REQUIRE( dsl_component_list_size() == 1 );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "The Components container is updated correctly on V4L2 Sink delete", "[sink-api]" )
{
    GIVEN( "An V4L2 Sink Component" ) 
    {
        std::wstring v4l2_sink_name(L"v4l2-sink");
        std::wstring device_location(L"/dev/video10");

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_v4l2_new(v4l2_sink_name.c_str(), 
            device_location.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "A new V4L2 Sink is deleted" ) 
        {
            REQUIRE( dsl_component_delete(v4l2_sink_name.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new V4L2 Sink can update its properties correctly", "[sink-api]" )
{
    GIVEN( "An V4L2 Sink Component" ) 
    {
        std::wstring v4l2_sink_name(L"v4l2-sink");
        std::wstring device_location(L"/dev/video10");

        REQUIRE( dsl_sink_v4l2_new(v4l2_sink_name.c_str(), 
            device_location.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A V4L2 Sink's device locaton is set" ) 
        {
            const wchar_t* c_ret_device_location;
            REQUIRE( dsl_sink_v4l2_device_location_get(v4l2_sink_name.c_str(), 
                &c_ret_device_location) == DSL_RESULT_SUCCESS );
            std::wstring ret_device_location(c_ret_device_location);
            REQUIRE( ret_device_location == device_location );

            std::wstring new_device_location(L"/dev/video0");
            REQUIRE( dsl_sink_v4l2_device_location_set(v4l2_sink_name.c_str(), 
                new_device_location.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" )
            {
                REQUIRE( dsl_sink_v4l2_device_location_get(v4l2_sink_name.c_str(), 
                    &c_ret_device_location) == DSL_RESULT_SUCCESS );
                ret_device_location = c_ret_device_location;
                REQUIRE( ret_device_location == new_device_location );

                REQUIRE( dsl_component_delete(v4l2_sink_name.c_str()) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "A V4L2 Sink's buffer-in-format is set" ) 
        {
            const wchar_t* c_bufffer_in_format;
            REQUIRE( dsl_sink_v4l2_buffer_in_format_get(v4l2_sink_name.c_str(), 
                &c_bufffer_in_format) == DSL_RESULT_SUCCESS );
            std::wstring ret_buffer_in_format(c_bufffer_in_format);
            REQUIRE( ret_buffer_in_format == DSL_VIDEO_FORMAT_I420 );

            REQUIRE( dsl_sink_v4l2_buffer_in_format_set(v4l2_sink_name.c_str(), 
                DSL_VIDEO_FORMAT_YVYU) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" )
            {
                REQUIRE( dsl_sink_v4l2_buffer_in_format_get(v4l2_sink_name.c_str(), 
                    &c_bufffer_in_format) == DSL_RESULT_SUCCESS );
                ret_buffer_in_format = c_bufffer_in_format;
                REQUIRE( ret_buffer_in_format == DSL_VIDEO_FORMAT_YVYU );

                REQUIRE( dsl_component_delete(v4l2_sink_name.c_str()) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "A V4L2 Sink's picture settings are set" ) 
        {
            int retBrightness(0), retContrast(0), retSaturation(0);
            REQUIRE( dsl_sink_v4l2_picture_settings_get(v4l2_sink_name.c_str(), 
                &retBrightness, &retContrast, &retSaturation) == DSL_RESULT_SUCCESS );
            REQUIRE( retBrightness == 0 );
            REQUIRE( retContrast == 0 );
            REQUIRE( retSaturation == 0 );

            int newBrightness(12), newContrast(-34), newSaturation(99);

            REQUIRE( dsl_sink_v4l2_picture_settings_set(v4l2_sink_name.c_str(), 
                newBrightness, newContrast, newSaturation) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" )
            {
                REQUIRE( dsl_sink_v4l2_picture_settings_get(v4l2_sink_name.c_str(), 
                    &retBrightness, &retContrast, &retSaturation) == DSL_RESULT_SUCCESS );
                REQUIRE( retBrightness == newBrightness );
                REQUIRE( retContrast == newContrast );
                REQUIRE( retSaturation == newSaturation );

                REQUIRE( dsl_component_delete(v4l2_sink_name.c_str()) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The Sink API checks for NULL input parameters", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring sink_name(L"test-sink");
        std::wstring otherName(L"other");
        
        uint max_size(0), cache_size(0), width(0), height(0), encoder(0), container(0), 
        offset_x(0), offset_y(0),
        bitrate(0), iframe_interval(30), udpPort(0), rtspPort(0), fps_n(0), fps_d(0);
        boolean is_on(0), reset_done(0), sync(0), async(0);
        
        std::wstring mailerName(L"mailer");
        
        std::wstring user_id(L"admin");
        
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_sink_app_new(NULL, 0, NULL, NULL) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_app_new(sink_name.c_str(), 0, NULL, NULL) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_custom_new(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_custom_new_element_add(
                    NULL, NULL)
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_custom_new_element_add(
                    sink_name.c_str(), NULL)
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_custom_new_element_add_many(
                    NULL, NULL)
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_custom_new_element_add_many(
                    sink_name.c_str(), NULL)
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_custom_element_add(
                    NULL, NULL)
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_custom_element_add(
                    sink_name.c_str(), NULL)
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_custom_element_add_many(NULL, NULL)
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_custom_element_add_many(
                    sink_name.c_str(), NULL)
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_custom_element_remove(
                    NULL, NULL)
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_custom_element_remove(
                    sink_name.c_str(), NULL)
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_custom_element_remove_many(
                    NULL, NULL)
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_custom_element_remove_many(
                    sink_name.c_str(), NULL)
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                
                REQUIRE( dsl_sink_fake_new(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                
                REQUIRE( dsl_sink_window_3d_new(NULL, 
                    0, 0, 0, 0 ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_window_egl_new(NULL, 
                    0, 0, 0, 0 ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_window_offsets_get(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_window_offsets_get(sink_name.c_str(), 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_window_offsets_get(sink_name.c_str(), 
                    &offset_x, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_window_offsets_set(NULL, 
                    0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_window_dimensions_get(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_window_dimensions_get(sink_name.c_str(), 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_window_dimensions_get(sink_name.c_str(), 
                    &width, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_window_dimensions_set(NULL, 
                    0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_sink_window_egl_force_aspect_ratio_get(NULL, 
                    0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_window_egl_force_aspect_ratio_get(sink_name.c_str(), 
                    0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_window_egl_force_aspect_ratio_set(NULL, 
                    0) == DSL_RESULT_INVALID_INPUT_PARAM );
                
                REQUIRE( dsl_sink_file_new(NULL, 
                    NULL, 0, 0, 0, 0 ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_file_new(sink_name.c_str(), 
                    NULL, 0, 0, 0, 0 ) == DSL_RESULT_INVALID_INPUT_PARAM );
                
                REQUIRE( dsl_sink_record_new(NULL, 
                    NULL, 0, 0, 0, 0, NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_new(sink_name.c_str(), 
                    NULL, 0, 0, 0, 0, NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_session_start(NULL, 
                    0, 0, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_session_stop(NULL, 
                    false) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_record_max_size_get(NULL, 
                    &max_size) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_max_size_get(sink_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_max_size_set(NULL, 
                    max_size) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_record_cache_size_get(NULL, 
                    &cache_size) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_cache_size_get(sink_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_cache_size_set(NULL, 
                    cache_size) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_record_dimensions_get(NULL, 
                    &width, &height) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_dimensions_set(NULL, 
                    width, height) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_record_is_on_get(NULL, 
                    &is_on) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_record_reset_done_get(NULL, 
                    &reset_done) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_record_video_player_add(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_video_player_add(sink_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_video_player_remove(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_video_player_remove(sink_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_record_mailer_add(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_mailer_add(sink_name.c_str(), 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_mailer_add(sink_name.c_str(), 
                    mailerName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_mailer_remove(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_record_mailer_remove(sink_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_encode_settings_get(NULL, 
                    &encoder, &bitrate, &iframe_interval) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_encode_dimensions_get(NULL, 
                    &width, &height) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_encode_dimensions_get(sink_name.c_str(), 
                    NULL, &height) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_encode_dimensions_get(sink_name.c_str(), 
                    &width, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_encode_dimensions_set(NULL, 
                    0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_rtmp_new(NULL, NULL,
                    0, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtmp_new(NULL, sink_name.c_str(),
                    0, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtmp_uri_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtmp_uri_get(sink_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtmp_uri_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtmp_uri_set(sink_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_rtsp_server_new(NULL, 
                    NULL, 0, 0, 0, 0, 0 ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtsp_server_new(sink_name.c_str(), 
                    NULL, 0, 0, 0, 0, 0 ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtsp_server_settings_get(NULL, 
                    &udpPort, &rtspPort) == DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_sink_rtsp_client_new(NULL, 
                    NULL, 0, 0, 0 ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtsp_client_new(sink_name.c_str(), 
                    NULL, 0, 0, 0 ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtsp_client_credentials_set(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtsp_client_credentials_set(sink_name.c_str(), 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtsp_client_credentials_set(sink_name.c_str(), 
                    user_id.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtsp_client_latency_get(NULL, 
                    NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtsp_client_latency_get(sink_name.c_str(), 
                    NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtsp_client_latency_set(NULL, 
                    2000) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtsp_client_profiles_get(NULL, 
                    NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtsp_client_profiles_get(sink_name.c_str(), 
                    NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtsp_client_profiles_set(NULL, 
                    0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtsp_client_protocols_get(NULL, 
                    NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtsp_client_protocols_get(sink_name.c_str(), 
                    NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtsp_client_protocols_set(NULL, 
                    0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtsp_client_tls_validation_flags_get(NULL, 
                    NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtsp_client_tls_validation_flags_get(sink_name.c_str(), 
                    NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_rtsp_client_tls_validation_flags_set(NULL, 
                    0) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_image_multi_new(NULL,
                    NULL, 0, 0, 1, 1) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_image_multi_new(sink_name.c_str(),
                    NULL, 0, 0, 1, 1) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_image_multi_file_path_get(sink_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_image_multi_file_path_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_image_multi_file_path_set(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_image_multi_file_path_set(sink_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_image_multi_dimensions_get(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_image_multi_dimensions_get(sink_name.c_str(), 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_image_multi_dimensions_get(sink_name.c_str(), 
                    &width, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_image_multi_dimensions_set(NULL, 
                    0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_image_multi_frame_rate_get(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_image_multi_frame_rate_get(sink_name.c_str(), 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_image_multi_frame_rate_get(sink_name.c_str(), 
                    &fps_n, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_image_multi_frame_rate_set(NULL, 
                    1, 1) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_image_multi_file_max_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_image_multi_file_max_get(sink_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_image_multi_file_max_set(NULL, 
                    1) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_frame_capture_new(NULL,
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_frame_capture_new(sink_name.c_str(),
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_frame_capture_initiate(NULL) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_frame_capture_schedule(NULL, 0) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_v4l2_new(NULL, 
                    NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_v4l2_new(sink_name.c_str(), 
                    NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_v4l2_device_location_get(NULL, 
                    NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_v4l2_device_location_get(sink_name.c_str(), 
                    NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_v4l2_device_location_set(NULL, 
                    NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_v4l2_device_location_set(sink_name.c_str(), 
                    NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_v4l2_device_name_get(NULL, 
                    NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_v4l2_device_name_get(sink_name.c_str(), 
                    NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_v4l2_device_fd_get(NULL, 
                    NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_v4l2_device_fd_get(sink_name.c_str(), 
                    NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_v4l2_device_flags_get(NULL, 
                    NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_v4l2_device_flags_get(sink_name.c_str(), 
                    NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_v4l2_buffer_in_format_get(NULL, 
                    NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_v4l2_buffer_in_format_get(sink_name.c_str(), 
                    NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_v4l2_buffer_in_format_set(NULL, 
                    NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_v4l2_buffer_in_format_set(sink_name.c_str(), 
                    NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                
                REQUIRE( dsl_sink_sync_enabled_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_sync_enabled_get(sink_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_sync_enabled_set(NULL, 
                    1) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_async_enabled_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_async_enabled_get(sink_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_async_enabled_set(NULL, 
                    1) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_max_lateness_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_max_lateness_get(sink_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_max_lateness_set(NULL, 
                    1) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_qos_enabled_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_qos_enabled_get(sink_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_qos_enabled_set(NULL, 
                    1) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_pph_add(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_pph_add(sink_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_pph_remove(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_pph_remove(sink_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
