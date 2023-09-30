/*
The MIT License

Copyright (c) 2023, Prominence AI, Inc.

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

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(1000)

// ---------------------------------------------------------------------------
// Shared Test Inputs 

static const std::wstring pipeline_name(L"test-pipeline");

static const std::wstring uri(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4");

static const std::wstring source_name1(L"file-source-1");
static const std::wstring source_name2(L"file-source-2");
static const std::wstring source_name3(L"file-source-3");
static const std::wstring source_name4(L"file-source-4");

static const std::wstring demuxer_name(L"demuxer");

static const std::wstring sink_name1(L"sink-1");
static const std::wstring sink_name2(L"sink-2");
static const std::wstring sink_name3(L"sink-3");
static const std::wstring sink_name4(L"sink-4");

static const uint offest_x(0);
static const uint offest_y(0);
static const uint sink_width(640);
static const uint sink_height(360);

// -----------------------------------------------------------------------------------

SCENARIO( "Two File Sources, Demuxer, and two Fake-Sinks can play", 
    "[demuxer-behavior]")
{
    GIVEN( "A Pipeline, two File sources, Dewarper, and two Fake-Sinks" ) 
    {

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_file_new(source_name1.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_file_new(source_name2.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_fake_new(sink_name1.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_fake_new(sink_name2.c_str()) == DSL_RESULT_SUCCESS );
        
        const wchar_t* demuxer_branches[] = {
            sink_name1.c_str(), sink_name2.c_str(), NULL};

        REQUIRE( dsl_tee_demuxer_new_branch_add_many(demuxer_name.c_str(), 2,
            demuxer_branches) == DSL_RESULT_SUCCESS );

        WHEN( "When the Pipeline is assembled" ) 
        {
            const wchar_t* components[] = {
                source_name1.c_str(), source_name2.c_str(), 
                demuxer_name.c_str(), NULL};
            
            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline is able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}

SCENARIO( "Two File Sources, Demuxer, and two Overlay-Sinks can play", 
    "[demuxer-behavior]")
{
    GIVEN( "A Pipeline, two File sources, Demuxer, and two Overlay-Sinks" ) 
    {

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_file_new(source_name1.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_file_new(source_name2.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_overlay_new(sink_name1.c_str(), 0, 0,
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_overlay_new(sink_name2.c_str(), 0, 1,
            offest_x+300, offest_y+300, sink_width, sink_height) == DSL_RESULT_SUCCESS );
        
        const wchar_t* demuxer_branches[] = {
            sink_name1.c_str(), sink_name2.c_str(), NULL};

        REQUIRE( dsl_tee_demuxer_new_branch_add_many(demuxer_name.c_str(), 2,
            demuxer_branches) == DSL_RESULT_SUCCESS );

        WHEN( "When the Pipeline is assembled" ) 
        {
            const wchar_t* components[] = {
                source_name1.c_str(), source_name2.c_str(), 
                demuxer_name.c_str(), NULL};
            
            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline is able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}

SCENARIO( "A Pipeline can add a Source and Overlay-Sink dynamically", 
    "[demuxer-behavior]")
{
    GIVEN( "A Pipeline, with a File Source, Demuxer, and Overlay-Sink" ) 
    {

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_file_new(source_name1.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_file_new(source_name2.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_overlay_new(sink_name1.c_str(), 0, 0,
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_overlay_new(sink_name2.c_str(), 0, 1,
            offest_x+300, offest_y+300, sink_width, sink_height) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_sink_sync_enabled_set(sink_name2.c_str(), false) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tee_demuxer_new(demuxer_name.c_str(), 2) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_tee_branch_add(demuxer_name.c_str(), 
            sink_name1.c_str()) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {
            source_name1.c_str(), demuxer_name.c_str(), NULL};
        
        REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
            components) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_streammux_batch_properties_set(pipeline_name.c_str(), 
            2, 40000) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) 
            == DSL_RESULT_SUCCESS );

        std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

        WHEN( "When a new Source and Sink are added" ) 
        {
            REQUIRE( dsl_tee_branch_add(demuxer_name.c_str(), 
                sink_name2.c_str()) == DSL_RESULT_SUCCESS );
            
            REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
                source_name2.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Pipeline continues to play " )
            {
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}

SCENARIO( "A Pipeline can remove a Source and Overlay-Sink dynamically", 
    "[demuxer-behavior]")
{
    GIVEN( "A Pipeline, with a File Source, Demuxer, and Overlay-Sink" ) 
    {

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_file_new(source_name1.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_file_new(source_name2.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_overlay_new(sink_name1.c_str(), 0, 0,
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_overlay_new(sink_name2.c_str(), 0, 1,
            offest_x+300, offest_y+300, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        const wchar_t* demuxer_branches[] = {
            sink_name1.c_str(), sink_name2.c_str(), NULL};

        REQUIRE( dsl_tee_demuxer_new_branch_add_many(demuxer_name.c_str(), 2,
            demuxer_branches) == DSL_RESULT_SUCCESS );
            
        const wchar_t* components[] = {
            source_name1.c_str(), source_name2.c_str(), 
            demuxer_name.c_str(), NULL};
        
        REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
            components) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) 
            == DSL_RESULT_SUCCESS );

        std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

        WHEN( "When a Source and Sink are removed" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipeline_name.c_str(), 
                source_name2.c_str()) == DSL_RESULT_SUCCESS );
            
            REQUIRE( dsl_tee_branch_remove(demuxer_name.c_str(), 
                sink_name2.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Pipeline is able to LinkAll and Play" )
            {

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}
SCENARIO( "A Pipeline can add and remove Sources and Overlay-Sinks dynamically multiple times", 
    "[demuxer-behavior]")
{
    GIVEN( "A Pipeline, with a File Source, Demuxer, and Overlay-Sink" ) 
    {

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_file_new(source_name1.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_file_new(source_name2.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_overlay_new(sink_name1.c_str(), 0, 0,
            offest_x, offest_y, sink_width, sink_height) == 
                DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_overlay_new(sink_name2.c_str(), 0, 1,
            offest_x+300, offest_y+300, sink_width, sink_height) == 
                DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_sink_sync_enabled_set(sink_name1.c_str(), 
            false) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_sync_enabled_set(sink_name2.c_str(), 
            false) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tee_demuxer_new(demuxer_name.c_str(), 2) == 
            DSL_RESULT_SUCCESS );
        REQUIRE( dsl_tee_branch_add(demuxer_name.c_str(), 
            sink_name1.c_str()) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {
            source_name1.c_str(), demuxer_name.c_str(), NULL};
        
        REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
            components) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_streammux_batch_properties_set(pipeline_name.c_str(), 
            3, 400000) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) 
            == DSL_RESULT_SUCCESS );

        std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

        WHEN( "When a new Source and Sink are added" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
                source_name2.c_str()) == DSL_RESULT_SUCCESS );

            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
            
            REQUIRE( dsl_tee_branch_add(demuxer_name.c_str(), 
                sink_name2.c_str()) == DSL_RESULT_SUCCESS );
            
            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

            REQUIRE( dsl_tee_branch_remove(demuxer_name.c_str(), 
                sink_name2.c_str()) == DSL_RESULT_SUCCESS );
            
            REQUIRE( dsl_pipeline_component_remove(pipeline_name.c_str(), 
                source_name2.c_str()) == DSL_RESULT_SUCCESS );
            
            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

            REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
                source_name2.c_str()) == DSL_RESULT_SUCCESS );

            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

            REQUIRE( dsl_tee_branch_add(demuxer_name.c_str(), 
                sink_name2.c_str()) == DSL_RESULT_SUCCESS );
            
            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

            REQUIRE( dsl_tee_branch_remove(demuxer_name.c_str(), 
                sink_name1.c_str()) == DSL_RESULT_SUCCESS );
            
            REQUIRE( dsl_pipeline_component_remove(pipeline_name.c_str(), 
                source_name1.c_str()) == DSL_RESULT_SUCCESS );
            
            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

            REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
                source_name1.c_str()) == DSL_RESULT_SUCCESS );

            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

            REQUIRE( dsl_tee_branch_add(demuxer_name.c_str(), 
                sink_name1.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Pipeline continues to play " )
            {
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}

SCENARIO( "A Pipeline can add and remove three Sources and Window-Sinks", 
    "[demuxer-behavior]")
{
    GIVEN( "A Pipeline, with a File Source, Demuxer, and Overlay-Sink" ) 
    {

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_file_new(source_name1.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_file_new(source_name2.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_file_new(source_name3.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_file_new(source_name4.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_new(sink_name1.c_str(),
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_sync_enabled_set(sink_name1.c_str(), false) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(sink_name2.c_str(),
            offest_x+300, offest_y+300, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_sync_enabled_set(sink_name2.c_str(), false) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_sink_window_new(sink_name3.c_str(), 
            offest_x+600, offest_y+600, sink_width, sink_height) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_sink_sync_enabled_set(sink_name3.c_str(), false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_new(sink_name4.c_str(), 
            offest_x+900, offest_y+300, sink_width, sink_height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_sync_enabled_set(sink_name4.c_str(), false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tee_demuxer_new(demuxer_name.c_str(), 4) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_tee_branch_add(demuxer_name.c_str(), 
            sink_name1.c_str()) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {
            source_name1.c_str(), demuxer_name.c_str(), NULL};
        
        REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
            components) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_streammux_batch_properties_set(pipeline_name.c_str(), 
            4, 400000) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) 
            == DSL_RESULT_SUCCESS );

        std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

        WHEN( "When a new Sources and Sinks are added and removed" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
                source_name2.c_str()) == DSL_RESULT_SUCCESS );

            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

            REQUIRE( dsl_tee_branch_add(demuxer_name.c_str(), 
                sink_name2.c_str()) == DSL_RESULT_SUCCESS );
            
            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
            
            REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
                source_name3.c_str()) == DSL_RESULT_SUCCESS );

            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

            REQUIRE( dsl_tee_branch_add(demuxer_name.c_str(), 
                sink_name3.c_str()) == DSL_RESULT_SUCCESS );
            
            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

//            REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
//                source_name4.c_str()) == DSL_RESULT_SUCCESS );
//
//            REQUIRE( dsl_tee_branch_add(demuxer_name.c_str(), 
//                sink_name4.c_str()) == DSL_RESULT_SUCCESS );
//            
//            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

            REQUIRE( dsl_tee_branch_remove(demuxer_name.c_str(), 
                sink_name1.c_str()) == DSL_RESULT_SUCCESS );
            
            REQUIRE( dsl_pipeline_component_remove(pipeline_name.c_str(), 
                source_name1.c_str()) == DSL_RESULT_SUCCESS );
            
            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

            REQUIRE( dsl_tee_branch_remove(demuxer_name.c_str(), 
                sink_name3.c_str()) == DSL_RESULT_SUCCESS );
                
            REQUIRE( dsl_pipeline_component_remove(pipeline_name.c_str(), 
                source_name3.c_str()) == DSL_RESULT_SUCCESS );
            
            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

            REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
                source_name3.c_str()) == DSL_RESULT_SUCCESS );

            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

            REQUIRE( dsl_tee_branch_add(demuxer_name.c_str(), 
                sink_name3.c_str()) == DSL_RESULT_SUCCESS );
            
            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

            REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
                source_name1.c_str()) == DSL_RESULT_SUCCESS );

            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

            REQUIRE( dsl_tee_branch_add(demuxer_name.c_str(), 
                sink_name1.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Pipeline continues to play " )
            {
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}

SCENARIO( "A Pipeline can have multiple Sources with a Demuxer and single dynamic Branch", 
    "[demuxer-behavior]")
{
    GIVEN( "A Pipeline, with a File Source, Demuxer, and Overlay-Sinks" ) 
    {

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_file_new(source_name1.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_file_new(source_name2.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_file_new(source_name3.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_file_new(source_name4.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_new(sink_name1.c_str(),
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_sync_enabled_set(sink_name1.c_str(), false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tee_demuxer_new(demuxer_name.c_str(), 4) == DSL_RESULT_SUCCESS );
        
        //
        REQUIRE( dsl_tee_demuxer_branch_add_to(demuxer_name.c_str(), 
            sink_name1.c_str(), 3) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {
            source_name1.c_str(), source_name2.c_str(), source_name3.c_str(), 
            source_name4.c_str(), demuxer_name.c_str(), NULL};
        
        REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
            components) == DSL_RESULT_SUCCESS );

        WHEN( "When the Pipeline transitions to a state of Playing" ) 
        {
            REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) 
                == DSL_RESULT_SUCCESS );

            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
            
            THEN( "The Pipeline continues to play " )
            {
                dsl_tee_branch_remove(demuxer_name.c_str(), 
                    sink_name1.c_str());

                g_usleep(1000);

                REQUIRE( dsl_tee_demuxer_branch_add_to(demuxer_name.c_str(), 
                    sink_name1.c_str(), 2) == DSL_RESULT_SUCCESS );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                
                dsl_tee_branch_remove(demuxer_name.c_str(), 
                    sink_name1.c_str());

                g_usleep(1000);

                REQUIRE( dsl_tee_demuxer_branch_add_to(demuxer_name.c_str(), 
                    sink_name1.c_str(), 1) == DSL_RESULT_SUCCESS );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                
                dsl_tee_branch_remove(demuxer_name.c_str(), 
                    sink_name1.c_str());

                g_usleep(1000);

                REQUIRE( dsl_tee_demuxer_branch_add_to(demuxer_name.c_str(), 
                    sink_name1.c_str(), 0) == DSL_RESULT_SUCCESS );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}
