
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
/*
valgrind command
valgrind --leak-check=yes --suppressions=./dsl.supp --suppressions=./gst.supp --suppressions=./glib.supp -s ./dsl-test-app.exe [memtest_0]
*/

#include "catch.hpp"
#include "DslApi.h"
#include "DslSinkBintr.h"
#include "DslSourceBintr.h"
#include "DslPipelineSourcesBintr.h"
#include "DslPipelineBintr.h"

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(1000)

static std::wstring pipeline_name(L"test-pipeline");

static const std::string pipelineSourcesName("pipeline-sources");

static std::wstring uri_source_name1(L"uri-source-1");
static std::wstring uri_source_name2(L"uri-source-2");
static std::wstring uri(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");
static std::wstring jpg_uri(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.jpg");

static uint skip_frames(0);
static uint drop_frame_interval(0);

static std::wstring osd_name(L"osd");

static std::wstring fake_sink_name(L"fake-sink");
static std::wstring app_sink_name(L"app-sink");
static std::wstring capture_sink_name(L"capture-sink");
static std::wstring window_sink_name(L"window-sink");
static std::wstring file_sink_name(L"file-sink");
static std::wstring record_sink_name(L"record-sink");
static std::wstring rtmp_sink_name(L"rtmp-sink");
static std::wstring rtsp_server_sink_name(L"rtsp-server-sink");
static std::wstring rtsp_client_sink_name(L"rtsp-client-sink");
static std::wstring multi_image_sink_name(L"multi-image-sink");

static const std::wstring rtmp_uri(L"rtmp://localhost/path/to/stream");

static const uint init_data_type(DSL_SINK_APP_DATA_TYPE_BUFFER);

static const std::string sourceName("source");
static const std::string filePath(
    "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");   
static const std::string pipelineName("pipeline");
static const std::string sinkName("sink");         
static const uint offsetX(300);
static const uint offsetY(0);        
// static const uint windowW(1280);
// static const uint windowH(720);
static const uint windowW(1280);
static const uint windowH(720);

static const std::wstring output_file_path(L"./output.mp4");
static const uint codec(DSL_CODEC_H265);
static const uint container(DSL_CONTAINER_MP4);
static const uint bitrate(2000000);
static const uint interval(0);

static const std::wstring host(L"224.224.255.255");
static const uint udp_port(5400);
static const uint rtsp_port(8554);

static const std::wstring rtsp_client_uri(L"rtsp://server_endpoint/stream");

static const uint sink_fps_n(1), sink_fps_d(2);
static const std::wstring multi_image_file_path(L"./frame-%05d.jpg");
static const uint imageW(1280);
static const uint imageH(720);

static const std::wstring custom_ppm_name1(L"custom-ppm-1");

static const std::wstring action_name(L"capture-action");
static const std::wstring outdir(L"./");

static const std::wstring ode_pph_name(L"ode-handler");

static uint new_buffer_cb(uint data_type,
    void* buffer, void* client_data)
{
    return DSL_FLOW_OK;
}

using namespace DSL;

static GThread* main_loop_thread_1(NULL);

static void* main_loop_thread_func_1(void *data)
{
    dsl_main_loop_run();
    
    return NULL;
}

SCENARIO( "A Pipeline that has played frees all memory",  "[memtest_0]" )
{
    GIVEN( "A new UriSourceBintr" ) 
    {
        std::wcout << dsl_info_version_get() << std::endl;

        std::wcout << L"gpu-type = " << dsl_info_gpu_type_get(0) << std::endl;

        REQUIRE( dsl_source_uri_new(uri_source_name1.c_str(), uri.c_str(),
            false, skip_frames, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_fake_new(fake_sink_name.c_str()) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source-1", L"fake-sink", NULL};

        REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(),
            components) == DSL_RESULT_SUCCESS );

        WHEN( "The UriSourceBintr is deleted " )
        {
            REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

            main_loop_thread_1 = g_thread_new("main-loop-1", 
                main_loop_thread_func_1, NULL);

            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
            REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "All memory is freed up correctly" )
            {
                dsl_main_loop_quit();                
                REQUIRE( dsl_pipeline_component_remove_many(pipeline_name.c_str(),
                    components) == DSL_RESULT_SUCCESS );

                dsl_delete_all();
                // requires inspection of the valgrind report
            }
        }
    }
}

SCENARIO( "A PipelineBintr that links and unlinks frees all memory",  "[memtest_1]" )
{
    GIVEN( "A new UriSourceBintr" ) 
    {
        dsl_record_client_listener_cb client_listener;

        REQUIRE( dsl_sink_fake_new(fake_sink_name.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_app_new(app_sink_name.c_str(), init_data_type, 
            new_buffer_cb, NULL) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_action_capture_frame_new(action_name.c_str(), 
            outdir.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_sink_frame_capture_new(capture_sink_name.c_str(), 
            action_name.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(window_sink_name.c_str(), 
            offsetX, offsetY, windowW, windowH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_file_new(file_sink_name.c_str(), output_file_path.c_str(),
            codec, container, bitrate, interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_record_new(record_sink_name.c_str(), outdir.c_str(),
            codec, container, bitrate, interval, client_listener) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_rtmp_new(rtmp_sink_name.c_str(), rtmp_uri.c_str(),
                bitrate, interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_rtsp_server_new(rtsp_server_sink_name.c_str(), host.c_str(),
            udp_port, rtsp_port, codec, bitrate, interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_rtsp_client_new(rtsp_client_sink_name.c_str(), 
            rtsp_client_uri.c_str(), codec, bitrate, interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_image_multi_new(multi_image_sink_name.c_str(),
            multi_image_file_path.c_str(), imageW, imageH, sink_fps_n, sink_fps_d) 
            == DSL_RESULT_SUCCESS );

        WHEN( "All components are added to the PipelineBintr" )
        {

            THEN( "The Pipeline components are Linked correctly" )
            {

                dsl_delete_all();
            }
        }
    }
}

static boolean pad_probe_handler_cb1(void* buffer, void* user_data)
{
    std::cout << "Custom Pad Probe Handler callback #1 called " << std::endl;
    return DSL_PAD_PROBE_OK;
}

static void eos_event_listener(void* client_data)
{
    std::cout<<"Pipeline EOS event"<<std::endl;
    dsl_pipeline_stop(pipeline_name.c_str());
    dsl_main_loop_quit();
}   

SCENARIO( "A Pipeline Component releases all PPH memeory correctly",  "[memtest_2]" )
{
    GIVEN( "A new OSD Component" ) 
    {

        // REQUIRE( dsl_source_uri_new(uri_source_name1.c_str(), uri.c_str(),
        //     false, skip_frames, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_image_single_new(uri_source_name1.c_str(), 
            jpg_uri.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osd_name.c_str(), 
            false, false, false, false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pph_custom_new(custom_ppm_name1.c_str(), 
            pad_probe_handler_cb1, NULL) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_osd_pph_add(osd_name.c_str(), 
            custom_ppm_name1.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(window_sink_name.c_str(), 
            offsetX, offsetY, windowW, windowH) == DSL_RESULT_SUCCESS );

        // REQUIRE( dsl_sink_fake_new(window_sink_name.c_str()) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {
                uri_source_name1.c_str(), osd_name.c_str(), window_sink_name.c_str(), NULL};

        REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(),
            components) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_eos_listener_add(pipeline_name.c_str(), 
            eos_event_listener, NULL) == DSL_RESULT_SUCCESS );

        WHEN( "The OSD Component is deleted " )
        {
            for (auto i=0;i<10;i++)
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                
                dsl_main_loop_run();
            }

            THEN( "All memory is freed up correctly" )
            {
                dsl_delete_all();
                // requires inspection of the valgrind report
            }
        }
    }
}

SCENARIO( "A Pipeline that links and unlinks frees all memory",  "[memtest_3]" )
{
    GIVEN( "A new UriSourceBintr" ) 
    {
        std::string pipelineName = "pipeline";

        DSL_PIPELINE_PTR pPipelineBintr = 
            DSL_PIPELINE_NEW(pipelineName.c_str());

        WHEN( "The UriSourceBintr is deleted " )
        {

            THEN( "All memory is freed up correctly" )
            {

                dsl_delete_all();
                // requires inspection of the valgrind report
            }
        }
    }
}

SCENARIO( "A EGL Window Sink frees all memory",  "[memtest_4]" )
{
    GIVEN( "A new UriSourceBintr" ) 
    {
        std::string sinkName("egl-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        WHEN( "The EglSinkBintr is created " )
        {
            DSL_EGL_SINK_PTR pSinkBintr = 
                DSL_EGL_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

            THEN( "All memory is freed up correctly" )
            {
                pSinkBintr->LinkAll();
                pSinkBintr->UnlinkAll();
                
                dsl_delete_all();
                // requires inspection of the valgrind report
            }
        }
    }
}

SCENARIO( "A Pipeline and Components releases all memeory on stop correctly",  "[memtest_5]" )
{
    GIVEN( "A new OSD Component" ) 
    {

        REQUIRE( dsl_source_uri_new(uri_source_name1.c_str(), uri.c_str(),
            false, skip_frames, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osd_name.c_str(), 
            false, false, false, false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pph_custom_new(custom_ppm_name1.c_str(), 
            pad_probe_handler_cb1, NULL) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_osd_pph_add(osd_name.c_str(), 
            custom_ppm_name1.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(window_sink_name.c_str(), 
            offsetX, offsetY, windowW, windowH) == DSL_RESULT_SUCCESS );

        // REQUIRE( dsl_sink_fake_new(window_sink_name.c_str()) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {
                uri_source_name1.c_str(), osd_name.c_str(), window_sink_name.c_str(), NULL};

        REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(),
            components) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_eos_listener_add(pipeline_name.c_str(), 
            eos_event_listener, NULL) == DSL_RESULT_SUCCESS );

        WHEN( "The OSD Component is deleted " )
        {
            for (auto i=0;i<100;i++)
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR*2);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
            }

            THEN( "All memory is freed up correctly" )
            {
                dsl_delete_all();
                // requires inspection of the valgrind report
            }
        }
    }
}

