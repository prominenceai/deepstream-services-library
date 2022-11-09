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

static const std::wstring source_name(L"uri-source");
static const std::wstring uri(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");
static const uint intr_decode(false);
static const uint drop_frame_interval(0);

static const std::wstring primary_gie_name(L"primary-gie");
static const std::wstring infer_config_file(
            L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt");
static const std::wstring model_engine_file(
            L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine");
        
static const std::wstring tracker_name(L"iou-tracker");
static const uint tracker_width(480);
static const uint tracker_height(272);
static const std::wstring tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");

static const std::wstring tiler_name(L"tiler");
static const uint width(1280);
static const uint height(720);
        
static const std::wstring window_sink_name(L"window-sink");
static const uint offsetX(0);
static const uint offsetY(0);
static const uint sinkW(1280);
static const uint sinkH(720);

static const std::wstring osd_name(L"osd");
static const boolean text_enabled(true);
static const boolean clock_enabled(false);
static const boolean bbox_enabled(true);
static const boolean mask_enabled(false);

static const uint PGIE_CLASS_ID_VEHICLE = 0;
static const uint PGIE_CLASS_ID_BICYCLE = 1;
static const uint PGIE_CLASS_ID_PERSON = 2;
static const uint PGIE_CLASS_ID_ROADSIGN = 3;


static const std::wstring ode_pph_name(L"ode-handler");

static const std::wstring pipeline_name(L"test-pipeline");

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(2000)

SCENARIO( "All DisplayTypes can be displayed by an ODE Action", "[display-types-behavior]" )
{
    GIVEN( "A Pipeline, ODE Handler, Always ODE Trigger, and Display Meta Action" ) 
    {
        std::wstring odeAlwaysTriggerName(L"always-trigger");

        std::wstring odeDisplayMetaActionName(L"display-meta");
        
        uint limit(0);


        REQUIRE( dsl_source_file_new(source_name.c_str(), uri.c_str(),
            true) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            infer_config_file.c_str(), model_engine_file.c_str(), 
            0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tracker_new(tracker_name.c_str(), NULL,
            tracker_width, tracker_height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_pph_ode_new(ode_pph_name.c_str()) == DSL_RESULT_SUCCESS );

        // ------------------------------------------------------------------------
        
        REQUIRE( dsl_display_type_rgba_color_custom_new(L"red-50", 
            1.0, 0.2, 0.3, 0.5) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_color_predefined_new(L"dark-red", 
            DSL_COLOR_PREDEFINED_DARK_RED, 1.0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_color_custom_new(L"green", 
            0.2, 1.0, 0.3, 0.5) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_color_predefined_new(L"purple", 
            DSL_COLOR_PREDEFINED_PURPLE, 1.0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_color_predefined_new(L"turquoise", 
            DSL_COLOR_PREDEFINED_TURQUOISE, 1.0) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_color_predefined_new(L"indigo", 
            DSL_COLOR_PREDEFINED_INDIGO, 1.0) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_color_predefined_new(L"white", 
            DSL_COLOR_PREDEFINED_WHITE, 1.0) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_color_predefined_new(L"black", 
            DSL_COLOR_PREDEFINED_BLACK, 1.0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_color_predefined_new(L"yellow", 
            DSL_COLOR_PREDEFINED_LIGHT_YELLOW, 1.0) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_color_predefined_new(L"orange", 
            DSL_COLOR_PREDEFINED_ORANGE, 1.0) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_color_predefined_new(L"black-50", 
            DSL_COLOR_PREDEFINED_BLACK, 0.5) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_color_random_new(L"random-magenta", 
            DSL_COLOR_HUE_MAGENTA, DSL_COLOR_LUMINOSITY_DARK, 
            1.0, 0) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_color_random_new(L"random-magenta-pink", 
            DSL_COLOR_HUE_MAGENTA_PINK, DSL_COLOR_LUMINOSITY_DARK, 
            0.5, 0) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_color_random_new(L"random-green", 
            DSL_COLOR_HUE_GREEN, DSL_COLOR_LUMINOSITY_LIGHT, 
            1.0, 0) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_color_random_new(L"random-cyan", 
            DSL_COLOR_HUE_CYAN, DSL_COLOR_LUMINOSITY_LIGHT, 
            1.0, 0) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_color_random_new(L"random-red", 
            DSL_COLOR_HUE_RED, DSL_COLOR_LUMINOSITY_DARK, 
            0.8, 0) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_color_predefined_new(L"white-65", 
            DSL_COLOR_PREDEFINED_WHITE, 0.65) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_color_custom_new(L"dark-blue", 
            0.2, 0.0, 1.0, 1.0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_color_custom_new(L"dark-grey", 
            0.2, 0.2, 0.2, 1.0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_color_random_new(L"random-color", 
            DSL_COLOR_HUE_RANDOM, DSL_COLOR_LUMINOSITY_BRIGHT, 
            1.0, 200) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_font_new(L"arial-bold-purple", 
            L"arial bold", 32, L"purple") == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_font_new(L"arial-bold-dark-grey", 
            L"arial bold", 32, L"dark-grey") == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_font_new(L"verdana-italic", 
            L"verdana italic bold", 36, L"dark-red") == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_font_new(L"georgia", 
            L"georgia bold", 32, L"white") == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_font_new(L"verdana-bold", 
            L"verdana bold", 32, L"yellow") == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_font_new(L"impact", 
            L"impact bold", 32, L"dark-blue") == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_text_new(L"colors", L"COLORS", 
            140, 100, L"arial-bold-purple", true, L"yellow") == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_text_new(L"fonts", L"Fonts", 
            385, 95, L"verdana-italic", false, NULL) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_text_new(L"text", L"Text with background!", 
            605, 95, L"georgia", true, L"black-50") == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_text_new(L"shadows", L"shadows", 
            1225, 95, L"verdana-bold", true, L"black-50") == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_text_shadow_add(L"shadows", 
            10, 10,  L"black-50") == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_text_new(L"shapes", L"SHAPES", 
            1520, 96, L"impact", true, L"white-65") == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_line_new(L"line", 
            200, 300, 380, 300, 10, L"turquoise")== DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_arrow_new(L"arrow", 
            460, 300, 640, 300, 10, DSL_ARROW_END_HEAD, L"indigo")== DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_text_new(L"source", L"Camera / Location", 
            40, 980, L"arial-bold-dark-grey", true, L"yellow") == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_text_shadow_add(L"source", 
            10, 10,  L"black-50") == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_source_dimensions_new(L"dimensions", 
            1610, 980, L"arial-bold-dark-grey", true, L"yellow") == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_text_shadow_add(L"dimensions", 
            10, 10,  L"black-50") == DSL_RESULT_SUCCESS );
            
        dsl_coordinate coordinates1[] = 
            {{710,280},{750,320},{790,280},{830,320},{870,280}};
            
        REQUIRE( dsl_display_type_rgba_line_multi_new(L"multi-line", 
            coordinates1, 5, 10, L"orange")== DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_rectangle_new(L"rectangle", 
            980, 260, 180, 80, 10, L"random-magenta", 
            true, L"random-magenta-pink")== DSL_RESULT_SUCCESS );

        dsl_coordinate coordinates2[] = 
            {{1240,280},{1330,230},{1420,280},{1380,370},{1280,370}};

        REQUIRE( dsl_display_type_rgba_polygon_new(L"polygon", 
            coordinates2, 5, 10, L"white")== DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_circle_new(L"circle", 
            1580, 300, 60, L"random-cyan", false, NULL)== DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_line_new(L"line-area", 
            355, 605, 570, 610, 8, L"green")== DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_text_new(L"ode-areas", L"ODE Areas",
            800, 780, L"arial-bold-dark-grey", true, L"yellow") == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_arrow_new(L"arrow-1", 
            585, 625, 795, 775, 5, DSL_ARROW_START_HEAD, L"white")== DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_arrow_new(L"arrow-2", 
            625, 747, 795, 775, 5, DSL_ARROW_START_HEAD, L"white")== DSL_RESULT_SUCCESS );

        const wchar_t* display_types[] = {
            L"colors", L"fonts", L"text", L"shadows", L"shapes", L"line", L"arrow", 
            L"multi-line", L"rectangle", L"polygon", L"circle", L"source", L"dimensions", 
            L"line-area", L"ode-areas", L"arrow-1", L"arrow-2", NULL
        };

        REQUIRE( dsl_ode_action_display_meta_add_many_new(odeDisplayMetaActionName.c_str(), 
            display_types) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_trigger_always_new(odeAlwaysTriggerName.c_str(), 
            NULL, DSL_ODE_PRE_OCCURRENCE_CHECK) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_trigger_action_add(odeAlwaysTriggerName.c_str(), 
            odeDisplayMetaActionName.c_str()) == DSL_RESULT_SUCCESS );


        REQUIRE( dsl_ode_action_label_format_new(L"format-label-action",  
            NULL, false, NULL) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_trigger_occurrence_new(L"every-object-trigger", 
            NULL, DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_action_bbox_format_new(L"format-bbox-action1",  
            0, NULL, false, NULL) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_trigger_action_add(L"every-object-trigger", 
            L"format-label-action") == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_trigger_action_add(L"every-object-trigger", 
            L"format-bbox-action1") == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_action_bbox_format_new(L"format-bbox-action2",  
            3, L"dark-blue", false, NULL) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_trigger_occurrence_new(L"every-vehicle-trigger", 
            NULL, PGIE_CLASS_ID_VEHICLE, DSL_ODE_TRIGGER_LIMIT_NONE) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_trigger_action_add(L"every-vehicle-trigger", 
            L"format-bbox-action2") == DSL_RESULT_SUCCESS );

        dsl_coordinate coordinates3[] = {{300,650},{580,660},{605, 740},{190,725}};

        REQUIRE( dsl_display_type_rgba_polygon_new(L"polygon-shape", 
            coordinates3, 4, 8, L"red-50")== DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_area_inclusion_new(L"polygon-area", L"polygon-shape", 
            true, DSL_BBOX_POINT_SOUTH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_trigger_cross_new(L"person-cross-trigger", 
            NULL, PGIE_CLASS_ID_PERSON, DSL_ODE_TRIGGER_LIMIT_NONE, 2, 200, 
            DSL_OBJECT_TRACE_TEST_METHOD_END_POINTS) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_trigger_infer_confidence_min_set(L"person-cross-trigger", 
            0.40) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_ode_trigger_cross_view_settings_set(L"person-cross-trigger",
            true, L"random-color", 4) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_trigger_area_add(L"person-cross-trigger", 
            L"polygon-area") == DSL_RESULT_SUCCESS );
        
        const wchar_t* triggers[] = {
            odeAlwaysTriggerName.c_str(), L"every-object-trigger", 
            L"every-vehicle-trigger", L"person-cross-trigger", NULL
        };
        
        REQUIRE( dsl_pph_ode_trigger_add_many(ode_pph_name.c_str(), 
            triggers) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_osd_new(osd_name.c_str(), 
            text_enabled, clock_enabled, bbox_enabled, mask_enabled) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_pph_add(osd_name.c_str(), 
            ode_pph_name.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source", 
            L"primary-gie", L"iou-tracker", L"osd", L"window-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pph_ode_display_meta_alloc_size_set(ode_pph_name.c_str(), 
                70) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );
                
            REQUIRE( dsl_pipeline_xwindow_fullscreen_enabled_set(
                pipeline_name.c_str(), true) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_list_size() == 0 );
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_ode_area_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_area_list_size() == 0 );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "When the PPH ODE Display Meta allocation size is 0" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );
            
            REQUIRE( dsl_pph_ode_display_meta_alloc_size_set(ode_pph_name.c_str(), 
                0) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_list_size() == 0 );
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_ode_area_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_area_list_size() == 0 );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}


SCENARIO( "DisplayTypes with a Random Color can be displayed by an ODE Action", 
    "[display-types-behavior]" )
{
    GIVEN( "A Pipeline, ODE Handler, Always ODE Trigger, and Display Meta Action" ) 
    {
        std::wstring odeAlwaysTriggerName(L"always-trigger");

        std::wstring odeDisplayMetaActionName(L"display-meta");
        
        uint limit(0);
        
        std::wstring lineName(L"line");
        uint x1(300), y1(600), x2(600), y2(620);
        uint line_width(5);

        std::wstring polygonName(L"polygon");
        std::wstring multiLineName(L"multi-line");
        dsl_coordinate coordinates1[4] = {{100,100},{210,110},{220,300},{110,330}};
        dsl_coordinate coordinates2[4] = {{100,400},{210,410},{220,500},{110,530}};
        uint num_coordinates(4);
        uint border_width(3);

        std::wstring colorName(L"random-color");
        double alpha(1.0);
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name.c_str(), uri.c_str(),
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            infer_config_file.c_str(), model_engine_file.c_str(), 
            0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tracker_new(tracker_name.c_str(), NULL,
            tracker_width, tracker_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name.c_str(), 
            width, height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_pph_ode_new(ode_pph_name.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tiler_pph_add(tiler_name.c_str(), 
            ode_pph_name.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_color_random_new(colorName.c_str(), 
            DSL_COLOR_HUE_RANDOM, DSL_COLOR_LUMINOSITY_RANDOM, alpha, 123) == 
                DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_line_new(lineName.c_str(), 
            x1, y1, x2, y2, line_width, colorName.c_str())== DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_polygon_new(polygonName.c_str(), 
            coordinates1, num_coordinates, border_width, 
            colorName.c_str())== DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_line_multi_new(multiLineName.c_str(), 
            coordinates2, num_coordinates, border_width, 
            colorName.c_str())== DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_trigger_always_new(odeAlwaysTriggerName.c_str(), 
            NULL, DSL_ODE_PRE_OCCURRENCE_CHECK) == DSL_RESULT_SUCCESS );

        const wchar_t* display_types[] = {L"line", L"polygon", L"multi-line", NULL};

        REQUIRE( dsl_ode_action_display_meta_add_many_new(odeDisplayMetaActionName.c_str(), 
            display_types) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_trigger_action_add(odeAlwaysTriggerName.c_str(), 
            odeDisplayMetaActionName.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_pph_ode_trigger_add(ode_pph_name.c_str(), 
            odeAlwaysTriggerName.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_osd_new(osd_name.c_str(), 
            text_enabled, clock_enabled, bbox_enabled, mask_enabled) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source", 
            L"primary-gie", L"iou-tracker", L"tiler", L"osd", L"window-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled and played" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The random color can be regenerated with set-next" )
            {
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_display_type_rgba_color_next_set(colorName.c_str()) 
                    == DSL_RESULT_SUCCESS );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_display_type_rgba_color_next_set(colorName.c_str()) 
                    == DSL_RESULT_SUCCESS );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_display_type_rgba_color_next_set(colorName.c_str()) 
                    == DSL_RESULT_SUCCESS );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_display_type_rgba_color_next_set(colorName.c_str()) 
                    == DSL_RESULT_SUCCESS );
                
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_list_size() == 0 );
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "DisplayTypes with a RGBA Palette color can be displayed by an ODE Action", 
    "[display-types-behavior]" )
{
    GIVEN( "A Pipeline, ODE Handler, Always ODE Trigger, and Display Meta Action" ) 
    {
        std::wstring odeAlwaysTriggerName(L"always-trigger");

        std::wstring odeDisplayMetaActionName(L"display-meta");
        
        uint limit(0);
        
        std::wstring lineName(L"line");
        uint x1(300), y1(600), x2(600), y2(620);
        uint line_width(5);

        std::wstring polygonName(L"polygon");
        std::wstring multiLineName(L"multi-line");
        dsl_coordinate coordinates1[4] = {{100,100},{210,110},{220,300},{110,330}};
        dsl_coordinate coordinates2[4] = {{100,400},{210,410},{220,500},{110,530}};
        uint num_coordinates(4);
        uint border_width(3);

        std::wstring colorName(L"palette-color");
        std::wstring colorName1(L"random-color-1");
        std::wstring colorName2(L"random-color-2");
        std::wstring colorName3(L"random-color-3");
        double alpha(1.0);
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name.c_str(), uri.c_str(),
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            infer_config_file.c_str(), model_engine_file.c_str(), 
            0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tracker_new(tracker_name.c_str(), NULL,
            tracker_width, tracker_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name.c_str(), 
            width, height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_pph_ode_new(ode_pph_name.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tiler_pph_add(tiler_name.c_str(), 
            ode_pph_name.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_color_random_new(colorName1.c_str(), 
            DSL_COLOR_HUE_RANDOM, DSL_COLOR_LUMINOSITY_RANDOM, alpha, 123) == 
                DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_color_random_new(colorName2.c_str(), 
            DSL_COLOR_HUE_RANDOM, DSL_COLOR_LUMINOSITY_RANDOM, alpha, 555) == 
                DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_color_random_new(colorName3.c_str(), 
            DSL_COLOR_HUE_RANDOM, DSL_COLOR_LUMINOSITY_RANDOM, alpha, 99999) == 
                DSL_RESULT_SUCCESS );

        const wchar_t* colors[] = 
            {colorName1.c_str(), colorName2.c_str(), colorName3.c_str(), NULL};

        REQUIRE( dsl_display_type_rgba_color_palette_new(colorName.c_str(), 
            colors) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_line_new(lineName.c_str(), 
            x1, y1, x2, y2, line_width, colorName.c_str())== DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_polygon_new(polygonName.c_str(), 
            coordinates1, num_coordinates, border_width, 
            colorName.c_str())== DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_line_multi_new(multiLineName.c_str(), 
            coordinates2, num_coordinates, border_width, 
            colorName.c_str())== DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_trigger_always_new(odeAlwaysTriggerName.c_str(), 
            NULL, DSL_ODE_PRE_OCCURRENCE_CHECK) == DSL_RESULT_SUCCESS );

        const wchar_t* display_types[] = {L"line", L"polygon", L"multi-line", NULL};

        REQUIRE( dsl_ode_action_display_meta_add_many_new(odeDisplayMetaActionName.c_str(), 
            display_types) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_trigger_action_add(odeAlwaysTriggerName.c_str(), 
            odeDisplayMetaActionName.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_pph_ode_trigger_add(ode_pph_name.c_str(), 
            odeAlwaysTriggerName.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_osd_new(osd_name.c_str(), 
            text_enabled, clock_enabled, bbox_enabled, mask_enabled) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source", 
            L"primary-gie", L"iou-tracker", L"tiler", L"osd", L"window-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled and played" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The next palette color is displayed on set-next" )
            {
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_display_type_rgba_color_next_set(colorName.c_str()) 
                    == DSL_RESULT_SUCCESS );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_display_type_rgba_color_next_set(colorName.c_str()) 
                    == DSL_RESULT_SUCCESS );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_display_type_rgba_color_next_set(colorName.c_str()) 
                    == DSL_RESULT_SUCCESS );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_display_type_rgba_color_next_set(colorName.c_str()) 
                    == DSL_RESULT_SUCCESS );
                
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_list_size() == 0 );
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "When the Pipeline is Assembled and played" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The color palette color is updated on set-index" )
            {
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_display_type_rgba_color_palette_index_set(colorName.c_str(), 2) 
                    == DSL_RESULT_SUCCESS );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_display_type_rgba_color_palette_index_set(colorName.c_str(), 0) 
                    == DSL_RESULT_SUCCESS );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_display_type_rgba_color_palette_index_set(colorName.c_str(), 2) 
                    == DSL_RESULT_SUCCESS );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_display_type_rgba_color_palette_index_set(colorName.c_str(), 1) 
                    == DSL_RESULT_SUCCESS );
                
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_list_size() == 0 );
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

static void color_provider_cb(double* red, 
    double* green, double* blue, double* alpha, void* client_data)
{
    static double _red(0.1), _green(0.1), _blue(0.1), _alpha(0.1);
    *red = _red;
    *green = _green;
    *blue = _blue;
    *alpha = _alpha;
    _red = _red+0.3;
    _green = _green+0.2;
    _blue = _blue+0.1;
    _alpha = _alpha+0.1;
}
    
SCENARIO( "DisplayTypes with an On-Deman Color can be displayed by an ODE Action", 
    "[display-types-behavior]" )
{
    GIVEN( "A Pipeline, ODE Handler, Always ODE Trigger, and Display Meta Action" ) 
    {
        std::wstring odeAlwaysTriggerName(L"always-trigger");

        std::wstring odeDisplayMetaActionName(L"display-meta");
        
        uint limit(0);
        
        std::wstring ode_pph_name(L"ode-handler");
        
        std::wstring lineName(L"line");
        uint x1(300), y1(600), x2(600), y2(620);
        uint line_width(5);

        std::wstring polygonName(L"polygon");
        std::wstring multiLineName(L"multi-line");
        dsl_coordinate coordinates1[4] = {{100,100},{210,110},{220,300},{110,330}};
        dsl_coordinate coordinates2[4] = {{100,400},{210,410},{220,500},{110,530}};
        uint num_coordinates(4);
        uint border_width(3);

        std::wstring colorName(L"on-demand-color");
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name.c_str(), uri.c_str(),
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            infer_config_file.c_str(), model_engine_file.c_str(), 
            0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tracker_new(tracker_name.c_str(), tracker_config_file.c_str(),
            tracker_width, tracker_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name.c_str(), 
            width, height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_pph_ode_new(ode_pph_name.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tiler_pph_add(tiler_name.c_str(), 
            ode_pph_name.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_color_on_demand_new(colorName.c_str(), 
            color_provider_cb, NULL) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_line_new(lineName.c_str(), 
            x1, y1, x2, y2, line_width, colorName.c_str())== DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_polygon_new(polygonName.c_str(), 
            coordinates1, num_coordinates, border_width, 
            colorName.c_str())== DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_line_multi_new(multiLineName.c_str(), 
            coordinates2, num_coordinates, border_width, 
            colorName.c_str())== DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_trigger_always_new(odeAlwaysTriggerName.c_str(), 
            NULL, DSL_ODE_PRE_OCCURRENCE_CHECK) == DSL_RESULT_SUCCESS );

        const wchar_t* display_types[] = {L"line", L"polygon", L"multi-line", NULL};

        REQUIRE( dsl_ode_action_display_meta_add_many_new(odeDisplayMetaActionName.c_str(), 
            display_types) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_trigger_action_add(odeAlwaysTriggerName.c_str(), 
            odeDisplayMetaActionName.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_pph_ode_trigger_add(ode_pph_name.c_str(), 
            odeAlwaysTriggerName.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_osd_new(osd_name.c_str(), 
            text_enabled, clock_enabled, bbox_enabled, mask_enabled) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source", 
            L"primary-gie", L"iou-tracker", L"tiler", L"osd", L"window-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled and played" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The random color can be regenerated with set-next" )
            {
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_display_type_rgba_color_next_set(colorName.c_str()) 
                    == DSL_RESULT_SUCCESS );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_display_type_rgba_color_next_set(colorName.c_str()) 
                    == DSL_RESULT_SUCCESS );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_display_type_rgba_color_next_set(colorName.c_str()) 
                    == DSL_RESULT_SUCCESS );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_display_type_rgba_color_next_set(colorName.c_str()) 
                    == DSL_RESULT_SUCCESS );
                
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_list_size() == 0 );
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}


SCENARIO( "A Format BBox ODE Action works correctly with a Random Color Palette", 
    "[display-types-behavior]" )
{
    GIVEN( "A Pipeline, ODE Handler, Occurrence ODE Trigger, and Format BBox ODE Action" ) 
    {
        std::wstring ode_trigger_name(L"occurrence-trigger");
        std::wstring ode_action_name(L"format-bbox-action");
        uint border_width(8);

        std::wstring border_color_name(L"my-border-color");
        std::wstring bg_color_name(L"my-bg-color");
        
        boolean has_bg_color(true);
        
        REQUIRE( dsl_display_type_rgba_color_palette_random_new(border_color_name.c_str(), 
            4, DSL_COLOR_HUE_RANDOM, DSL_COLOR_LUMINOSITY_RANDOM, 0.78, 123) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_color_palette_random_new(bg_color_name.c_str(), 
            4, DSL_COLOR_HUE_RANDOM, DSL_COLOR_LUMINOSITY_RANDOM, 0.43, 456) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_source_uri_new(source_name.c_str(), uri.c_str(), 
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            infer_config_file.c_str(), model_engine_file.c_str(), 0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tracker_new(tracker_name.c_str(), tracker_config_file.c_str(),
            tracker_width, tracker_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_pph_ode_new(ode_pph_name.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tiler_pph_add(tiler_name.c_str(), 
            ode_pph_name.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_trigger_occurrence_new(ode_trigger_name.c_str(), 
            NULL, DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_action_bbox_format_new(ode_action_name.c_str(), border_width, 
            border_color_name.c_str(), has_bg_color, 
            bg_color_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(ode_trigger_name.c_str(), 
            ode_action_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pph_ode_trigger_add(ode_pph_name.c_str(), 
            ode_trigger_name.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osd_name.c_str(), text_enabled, clock_enabled,
            bbox_enabled, mask_enabled) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source", 
            L"primary-gie", L"iou-tracker", L"tiler", L"osd", L"window-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_list_size() == 0 );
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Format Label ODE Action works correctly with a Random Color Palette",
    "[display-types-behavior]" )
{
    GIVEN( "A Pipeline, ODE Handler, Occurrence ODE Trigger, and Format Label ODE Action" ) 
    {
        std::wstring ode_trigger_name(L"occurrence-trigger");
        std::wstring ode_action_name(L"format-label-action");

        std::wstring bg_color_name(L"my-bg-color");
        
        std::wstring font_name(L"font-name");
        std::wstring font(L"arial");
        std::wstring font_color_name(L"font-color");
        uint size(14);

        double redFont(0.0), greenFont(0.0), blueFont(0.0), alphaFont(1.0);
        boolean has_bg_color(true);
        
        REQUIRE( dsl_display_type_rgba_color_custom_new(font_color_name.c_str(), 
            redFont, greenFont, blueFont, alphaFont) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_color_palette_random_new(bg_color_name.c_str(), 
            4, DSL_COLOR_HUE_RANDOM, DSL_COLOR_LUMINOSITY_RANDOM, 0.43, 456) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_font_new(font_name.c_str(), 
            font.c_str(), size, font_color_name.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_action_label_format_new(ode_action_name.c_str(),  
            font_name.c_str(), has_bg_color, bg_color_name.c_str()) 
                == DSL_RESULT_SUCCESS );

        
        REQUIRE( dsl_source_uri_new(source_name.c_str(), uri.c_str(), 
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            infer_config_file.c_str(), model_engine_file.c_str(), 0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tracker_new(tracker_name.c_str(), tracker_config_file.c_str(),
            tracker_width, tracker_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_pph_ode_new(ode_pph_name.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tiler_pph_add(tiler_name.c_str(), 
            ode_pph_name.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_trigger_occurrence_new(ode_trigger_name.c_str(), 
            NULL, DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_trigger_action_add(ode_trigger_name.c_str(), 
            ode_action_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pph_ode_trigger_add(ode_pph_name.c_str(), 
            ode_trigger_name.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osd_name.c_str(), text_enabled, clock_enabled,
            bbox_enabled, mask_enabled) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source", 
            L"primary-gie", L"iou-tracker", L"tiler", L"osd", L"window-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_list_size() == 0 );
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}
