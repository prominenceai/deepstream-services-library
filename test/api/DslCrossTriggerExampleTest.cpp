/*
The MIT License

Copyright (c) 2022, Prominence AI, Inc.

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


// Unique Pipeline name
static const std::wstring pipeline_name(L"test-pipeline");

// File Source name and pathspec to the NVIDIA's H265 test stream
static const std::wstring source_name(L"uri-source");
static const std::wstring uri(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");

// Filespecs for the Primary GIE    
static const std::wstring primary_gie_name(L"primary-gie");
static const std::wstring primary_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt");
static const std::wstring primary_model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine");
static const std::wstring tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");

// Tracker name and input dimensions.
static const std::wstring tracker_name(L"iou-tracker");
static const uint tracker_width(480);
static const uint tracker_height(272);

// Object Detection Event (ODE) Pad Probe Handler (PPH) name.
static const std::wstring ode_pph_name(L"ode-handler");

// On-Screen Display (OSD) name.        
static const std::wstring osd_name(L"osd");
        
// Polygon name, coordinates and line_with. ** IMPORT ** the defines hysteresis 
// for the Line ODE Cross Trigger. 
static const std::wstring polygon(L"polygon");
static const dsl_coordinate coordinates[4] = {{290,650},{600,660},{620, 770},{180,750}};
static const uint num_coordinates(4);
static const uint line_width(10);  

// RGBA Colors and Polygon Display Type names.
static const std::wstring light_yellow(L"light-yellow");
static const std::wstring yellow(L"solid-yellow");
static const std::wstring heavy_opaque_black(L"heavy-opaque-black");
static const std::wstring black_background(L"black-background");
static const std::wstring random_color(L"random-color");
static const std::wstring verdana(L"verdana bold");
static const std::wstring verdana_16_yellow(L"arial-16-yellow");
static const std::wstring area_name  = L"polygon-area";

// ODE Trigger names and class id.
static const std::wstring every_frame_trigger(L"every-frame-trigger");
static const std::wstring every_occurrence_trigger(L"every-occurrence");
static const std::wstring person_cross_trigger(L"person-cross-trigger");
static const uint person_class_id(2);

// ODE Action names. 
static const std::wstring add_text_background_action(L"add-background-action");
static const std::wstring exclude_bbox_action(L"exclude-bbox-action");
static const std::wstring ode_print_action_name(L"print-action");

// Window Sink name and attributes.
static const std::wstring window_sink_name(L"window-sink");
static const uint offsetX(0);
static const uint offsetY(0);
static const uint sinkW(DSL_STREAMMUX_DEFAULT_WIDTH);
static const uint sinkH(DSL_STREAMMUX_DEFAULT_HEIGHT);

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(2000)

// 
// Function to be called on XWindow KeyRelease event
// 
void xwindow_key_event_handler(const wchar_t* in_key, void* client_data)
{   
    std::wstring wkey(in_key); 
    std::string key(wkey.begin(), wkey.end());
    std::cout << "key released = " << key << std::endl;
    key = std::toupper(key[0]);
    if(key == "P"){
        dsl_pipeline_pause(pipeline_name.c_str());
    } else if (key == "R"){
        dsl_pipeline_play(pipeline_name.c_str());
    } else if (key == "Q"){
        std::cout << "Main Loop Quit" << std::endl;
        dsl_pipeline_stop(pipeline_name.c_str());
        dsl_main_loop_quit();
    }
}

// 
// Function to be called on XWindow Delete event
//
void xwindow_delete_event_handler(void* client_data)
{
    std::cout<<"delete window event"<<std::endl;
    dsl_pipeline_stop(pipeline_name.c_str());
    dsl_main_loop_quit();
}
    
// 
// Function to be called on End-of-Stream (EOS) event
// 
void eos_event_listener(void* client_data)
{
    std::cout<<"Pipeline EOS event"<<std::endl;
    dsl_pipeline_stop(pipeline_name.c_str());
    dsl_main_loop_quit();
}    

// 
// Function to be called on every change of Pipeline state
// 
void state_change_listener(uint old_state, uint new_state, void* client_data)
{
    std::cout<<"previous state = " << dsl_state_value_to_string(old_state) 
        << ", new state = " << dsl_state_value_to_string(new_state) << std::endl;
}

//int main(int argc, char** argv)
int test()
{
    DslReturnType retval = DSL_RESULT_FAILURE;

    // Since we're not using args, we can Let DSL initialize GST on first call    
    while(true) 
    {    
        retval = dsl_display_type_rgba_color_predefined_new(light_yellow.c_str(), 
             DSL_COLOR_PREDEFINED_LIGHT_YELLOW, 1.0);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_display_type_rgba_color_predefined_new(heavy_opaque_black.c_str(), 
            DSL_COLOR_PREDEFINED_BLACK, 0.3);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_display_type_rgba_font_new(verdana_16_yellow.c_str(), 
            verdana.c_str(), 16, light_yellow.c_str());
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_display_type_rgba_rectangle_new(black_background.c_str(),
            1190, 30, 300, 120, 1, heavy_opaque_black.c_str(), true, 
            heavy_opaque_black.c_str());
        if (retval != DSL_RESULT_SUCCESS) break;

        // Create a list of Display Types to display on every frame.
        const wchar_t* display_types[] = {black_background.c_str(), NULL};

        // Create an ODE Acton to add the display metadata to a frame's metadata
        retval = dsl_ode_action_display_meta_add_many_new(
            add_text_background_action.c_str(),
            display_types);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Create a Trigger to trigger on every frame - alays.
        retval = dsl_ode_trigger_always_new(every_frame_trigger.c_str(), 
            DSL_ODE_ANY_SOURCE, DSL_ODE_PRE_OCCURRENCE_CHECK);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the add-metatdata Action to the Always Trigger. We add
        // the Trigger to the ODE Pad Probe Handler further below.
        retval = dsl_ode_trigger_action_add(every_frame_trigger.c_str(),
            add_text_background_action.c_str());
        if (retval != DSL_RESULT_SUCCESS) break;
        
        
        // ---------------------------------------------------------------------------
        // New Display Types used to define a Polygon Area and Cross Trigger
        
        // New predefined light-yellow color with alpha=0.8 for the Polygon Area
        retval = dsl_display_type_rgba_color_predefined_new(yellow.c_str(), 
            DSL_COLOR_PREDEFINED_LIGHT_YELLOW, 0.8);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Polygon Display Type - will be used to define an ODE Polygon Area.
        // **** IMPORTANT **** the line_width provided will be used for hysteresis.
        // The test point on the object's bounding box must fully cross the line
        // to trigger an ODE occurrence. 
        retval = dsl_display_type_rgba_polygon_new(polygon.c_str(), 
            coordinates, num_coordinates, line_width, yellow.c_str());
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Random Color - will be used to give each tracked Person Object a
        // uniquely colored bounding box and object trace. 
        retval = dsl_display_type_rgba_color_random_new(random_color.c_str(), 
            DSL_COLOR_HUE_RANDOM, DSL_COLOR_LUMINOSITY_BRIGHT, 1.0, 123);
        if (retval != DSL_RESULT_SUCCESS) break;
        

        // ---------------------------------------------------------------------------
        // We want to remove/hide all object bounding boxes for all classes by default
        // and then let the Cross Trigger re-format the bounding boxes and object traces
        // for the People that it trackes with the random color defined above. 

        // Create an Any-Class Occurrence Trigger for our format bounding box action.
        retval = dsl_ode_trigger_occurrence_new(every_occurrence_trigger.c_str(), 
            DSL_ODE_ANY_SOURCE, DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE);
        if (retval != DSL_RESULT_SUCCESS) return retval;

        // Create the format bounding box action to add to the every occurrence trigger.
        retval = dsl_ode_action_bbox_format_new(exclude_bbox_action.c_str(), 0,
            NULL, false, NULL);
        if (retval != DSL_RESULT_SUCCESS) return retval;

        // Add the Format Action to the every occurrence trigger. We will add the trigger
        // to the Object Detection Event (ODE) Pad Probe Handerl (PPH) below.
        retval = dsl_ode_trigger_action_add(every_occurrence_trigger.c_str(), 
            exclude_bbox_action.c_str());
        if (retval != DSL_RESULT_SUCCESS) break;


        // ---------------------------------------------------------------------------
        // Next, setup the Cross Trigger and Polygon Area for object tracking.
        
        // New ODE Polygon Inclusion Area created with the light-yellow polygon
        // Display Type defined above.
        retval = dsl_ode_area_inclusion_new(area_name.c_str(), polygon.c_str(), 
            true, DSL_BBOX_POINT_SOUTH);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New ODE Cross Trigger to track Person Objects and triggers on the 
        // occurrence that an object crosses the trigger's Polygon Area.
        retval = dsl_ode_trigger_cross_new(person_cross_trigger.c_str(), 
            NULL, person_class_id, DSL_ODE_TRIGGER_LIMIT_NONE, 2, 200, 
                DSL_OBJECT_TRACE_TEST_METHOD_END_POINTS);
        if (retval != DSL_RESULT_SUCCESS) break;

        // ********** IMPORT **********  need to set a minimum confidence level to
        // avoid false triggers when the bounding box coordinates are accurate - 40%.
        retval = dsl_ode_trigger_infer_confidence_min_set(person_cross_trigger.c_str(), 
            0.40);
        if (retval != DSL_RESULT_SUCCESS) break;
            
        // Set the Cross Trigger's view settings so we can see the objects trace
        // as it is tracked through successive frames. Use the random color so the
        // trigger can generate a new color for each new object tracked. 
        retval = dsl_ode_trigger_cross_view_settings_set(person_cross_trigger.c_str(),
            true, random_color.c_str(), 4);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the ODE Polygon Area to the Cross Trigger as criteria for line cross.
        retval = dsl_ode_trigger_area_add(person_cross_trigger.c_str(), 
            area_name.c_str());
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Print Action to print the line crossing events to the console.
        retval = dsl_ode_action_print_new(ode_print_action_name.c_str(), 
            false);
        if (retval != DSL_RESULT_SUCCESS) break;
        
        // Add the Print Action the Cross Trigger to be invoked on ODE occurrence.
        retval = dsl_ode_trigger_action_add(person_cross_trigger.c_str(), 
            ode_print_action_name.c_str());
        if (retval != DSL_RESULT_SUCCESS) break;
        

        // ---------------------------------------------------------------------------
        // Next, setup the Object Detection Event (ODE) Pad Probe Hander

        // Create the ODE PPH and then increase the Display Meta allocation size so
        // that we can display the Polygon and Object traces which can require many 
        // lines. This is especially true if the Cross Trigger test method is set
        // to DSL_OBJECT_TRACE_TEST_METHOD_ALL_POINTS. Size accordingly
        retval = dsl_pph_ode_new(ode_pph_name.c_str());
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_pph_ode_display_meta_alloc_size_set(
            ode_pph_name.c_str(), 2);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Create a list of our trigger names to add to the ODE PPH
        // ** NOTE ** order is important - remove bounding boxes first.
        const wchar_t* triggers[] ={every_frame_trigger.c_str(), 
            every_occurrence_trigger.c_str(), person_cross_trigger.c_str(), NULL};

        // Add the Triggers to the ODE PPH which will add to our OSD below.
        retval = dsl_pph_ode_trigger_add_many(ode_pph_name.c_str(), triggers);
        if (retval != DSL_RESULT_SUCCESS) break;
        

        // ---------------------------------------------------------------------------
        // Next, create all Pipeline components starting from Source to Sink
    
        // New File Source for for the Pipeline
        retval = dsl_source_file_new(source_name.c_str(), uri.c_str(), false);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Primary GIE using NVIDIA's provided config and model-engine files.
        retval = dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            primary_infer_config_file.c_str(), primary_model_engine_file.c_str(), 0);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Multi Object Tracker - required when using a Cross Trigger
        REQUIRE( dsl_tracker_new(tracker_name.c_str(), tracker_config_file.c_str(),
            tracker_width, tracker_height) == DSL_RESULT_SUCCESS );
        if (retval != DSL_RESULT_SUCCESS) break;

        // New On-Screen Display (OSD) to display bounding boxes and object traces.
        REQUIRE( dsl_osd_new(osd_name.c_str(), false, false,
            true, false) == DSL_RESULT_SUCCESS );
        if (retval != DSL_RESULT_SUCCESS) break;
        
        // ** IMPORT ** add the ODE PPH created above to the Sink Pad of the OSD.
        retval = dsl_osd_pph_add(osd_name.c_str(), 
            ode_pph_name.c_str(), DSL_PAD_SINK);
        if (retval != DSL_RESULT_SUCCESS) break;
        
        // New Window Sink to render the video stream. 
        retval = dsl_sink_window_new(window_sink_name.c_str(),
            offsetX, offsetY, sinkW, sinkH);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Create a list of Pipeline Components to add to the new Pipeline.
        const wchar_t* components[] = {L"uri-source", 
            L"primary-gie", L"iou-tracker", L"osd", L"window-sink", NULL};
        
        // Create a new Pipeline and add the above components in the next call.
        retval = dsl_pipeline_new_component_add_many(
            pipeline_name.c_str(), components);
        if (retval != DSL_RESULT_SUCCESS) break;

        // ---------------------------------------------------------------------------
        // Add all Client callback functions to the Pipeline.
        
        // Add the XWindow key event handler callback function.
        retval = dsl_pipeline_xwindow_key_event_handler_add(pipeline_name.c_str(), 
            xwindow_key_event_handler, nullptr);    
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the XWindow delete window event handler function.
        retval = dsl_pipeline_xwindow_delete_event_handler_add(pipeline_name.c_str(), 
            xwindow_delete_event_handler, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the state-change listener callback function.
        retval = dsl_pipeline_state_change_listener_add(pipeline_name.c_str(), 
            state_change_listener, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the end of stream (EOS) event listener callback function.
        retval = dsl_pipeline_eos_listener_add(pipeline_name.c_str(), 
            eos_event_listener, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;


        // ---------------------------------------------------------------------------
        // Start the Pipeline and join the g-main-loop
        
        retval = dsl_pipeline_play(pipeline_name.c_str());
        if (retval != DSL_RESULT_SUCCESS) break;

        std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
        dsl_pipeline_stop(pipeline_name.c_str());
        break;
    }
    dsl_delete_all();
    return retval;
}

SCENARIO( "A new Pipeline with a Cross ODE Trigger using an ODE Polygon Area can provide example", 
    "[cross-trigger]" )
{
    GIVEN( "A Pipeline, ODE Handler, Cross ODE Trigger, Line ODE Area, and Fill ODE Action" ) 
    {
        REQUIRE( test() == DSL_RESULT_SUCCESS );
    }
}
