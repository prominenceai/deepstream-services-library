/*
The MIT License

Copyright (c) 2024, Prominence AI, Inc.

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
#
# This example demonstrates the use of a Frame-Capture Sink to encode and
# save video frames to JPEG files on client/viewer demand.
#
# An ODE Frame-Capture Action is provided to The Frame-Capture Sink on creation.
# A client "capture_complete_listener" is added to the the Action to be notified
# when each new file is saved (the ODE Action performs the actual frame-capture).
#
# Child Players (to play the captured image) and Mailers (to mail the image) can
# be added to the ODE Frame-Capture action as well (not shown).
#
# A Custom Pad Probe Handler (PPH) is added to the sink-pad of the OSD component
# to process every buffer flowing over the pad by:
#    - Retrieving the batch-metadata and its list of frame metadata structures
#      (only one frame per batched-buffer with 1 Source)
#    - Retrieving the list of object metadata structures from the frame metadata.
#    - Iterating through the list of objects looking for the first occurrence of
#      a bicycle. 
#    - If detected, the current frame-number is schedule to be captured by the
#      Frame-Capture Sink using its Frame-Capture Action.
#
#          dsl_sink_frame_capture_schedule('frame-capture-sink', 
#                   frame_meta.frame_num)
#
# Note: The Custom PPH will schedule every frame with a bicycle to be captured!
#
# IMPORT All captured frames are copied and buffered in the Sink's processing
# thread. The encoding and saving of each buffered frame is done in the 
# g-idle-thread, therefore, the capture-complete notification is asynchronous.
#
*/

#include <iostream>
#include <glib.h>
#include <gst/gst.h>
#include <gstnvdsmeta.h>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "DslApi.h"

// Use the DSL Surface Transform utility class
#include "DslSurfaceTransform.h"

uint device_type;
uint unique_id=0;

std::wstring uri_h265(
    L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");

// Config and model-engine files - Jetson and dGPU
std::wstring primary_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt");
std::wstring primary_model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine");

// Config file used by the IOU Tracker    
std::wstring iou_tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");

// Same dimensions as Pipeline Streammuxer 
int TILER_WIDTH = DSL_1K_HD_WIDTH;
int TILER_HEIGHT = DSL_1K_HD_HEIGHT;

uint WINDOW_WIDTH = 1280;
uint WINDOW_HEIGHT = 720;

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
        dsl_pipeline_pause(L"pipeline");
    } else if (key == "R"){
        dsl_pipeline_play(L"pipeline");
    } else if (key == "Q" or key == "" or key == ""){
        std::cout << "Main Loop Quit" << std::endl;
        dsl_pipeline_stop(L"pipeline");
        dsl_main_loop_quit();
    }
}

// 
// Function to be called on XWindow Delete event
//
void xwindow_delete_event_handler(void* client_data)
{
    std::cout << "delete window event" <<std::endl;
    dsl_pipeline_stop(L"pipeline");
    dsl_main_loop_quit();
}
    
// 
// Function to be called on End-of-Stream (EOS) event
// 
void eos_event_listener(void* client_data)
{
    std::cout <<"Pipeline EOS event" <<std::endl;
    dsl_pipeline_stop(L"pipeline");
    dsl_main_loop_quit();
}    

// 
// Function to be called on every change of Pipeline state
// 
void state_change_listener(uint old_state, uint new_state, void* client_data)
{
    std::cout << "previous state = " << dsl_state_value_to_string(old_state) 
        << ", new state = " << dsl_state_value_to_string(new_state) << std::endl;
}

// 
// Custom Pad Probe Handler function called with every buffer
// 
uint custom_pad_probe_handler(void* buffer, void* user_data)
{
    // Retrieve batch metadata from the gst_buffer
    NvDsBatchMeta* pBatchMeta = gst_buffer_get_nvds_batch_meta((GstBuffer*)buffer);
    
    // For each frame in the batched meta data
    for (NvDsMetaList* pFrameMetaList = pBatchMeta->frame_meta_list; 
        pFrameMetaList; pFrameMetaList = pFrameMetaList->next)
    {
        // Check for valid frame data
        NvDsFrameMeta* pFrameMeta = (NvDsFrameMeta*)(pFrameMetaList->data);
        if (pFrameMeta != NULL)
        {
            
            // Map the current buffer
            std::unique_ptr<DSL::DslMappedBuffer> pMappedBuffer = 
                std::unique_ptr<DSL::DslMappedBuffer>(new DSL::DslMappedBuffer(
                    (GstBuffer*)buffer));
                
            NvBufSurfaceMemType transformMemType = (
                device_type == DSL_GPU_TYPE_INTEGRATED)
                ? NVBUF_MEM_DEFAULT
                : NVBUF_MEM_CUDA_PINNED;
        
            // Transforming only one frame in the batch, so create a copy of the single 
            // surface ... becoming our new source surface. This creates a new mono 
            // (non-batched) surface copied from the "batched frames" using the batch id 
            // as the index
            DSL::DslMonoSurface monoSurface(pMappedBuffer->pSurface, 
                pFrameMeta->batch_id);

            // Coordinates and dimensions for our destination surface.
            int left(0), top(0);

            // capturing full frame
            int width = pMappedBuffer->GetWidth(pFrameMeta->batch_id);
            int height = pMappedBuffer->GetHeight(pFrameMeta->batch_id);
            
            // New "create params" for our destination surface. we only need one 
            // surface so set memory allocation (for the array of surfaces) size to 0
            DSL::DslSurfaceCreateParams surfaceCreateParams(monoSurface.gpuId, 
                width, height, 0, NVBUF_COLOR_FORMAT_RGBA, transformMemType);
            
            // New Destination surface with a batch size of 1 for transforming 
            // the single surface. Use batch-id as unique id. 
            std::unique_ptr<DSL::DslBufferSurface> pBufferSurface = 
                std::unique_ptr<DSL::DslBufferSurface>(new DSL::DslBufferSurface(1, 
                surfaceCreateParams, pFrameMeta->batch_id));

            // New "transform params" for the surface transform, croping
            DSL::DslTransformParams transformParams(left, top, width, height);
            
            // New "Cuda stream" for the surface transform
            DSL::DslCudaStream dslCudaStream(monoSurface.gpuId);
            
            // New "Transform Session" config params using the new Cuda stream
            DSL::DslSurfaceTransformSessionParams dslTransformSessionParams(
                monoSurface.gpuId, dslCudaStream);
            
            // Set the "Transform Params" for the current tranform session
            if (!dslTransformSessionParams.Set())
            {
                std::cout << "failed to set Transform Params" << std::endl;
                return DSL_PAD_PROBE_REMOVE;
            }
            
            // We can now transform our Mono Source surface to the first (and only) 
            // surface in the batched buffer.
            if (!pBufferSurface->TransformMonoSurface(monoSurface, 0, transformParams))
            {
                std::cout << "failed to transform Mono Surface" << std::endl;
                return DSL_PAD_PROBE_REMOVE;
            }

            // Map the tranformed surface for read
            if (!pBufferSurface->Map())
            {
                std::cout << "failed to map the transformed buffer" << std::endl;
                return DSL_PAD_PROBE_REMOVE;
            }

            std::cout << "New Buffer Surface mapped succesfully" << std::endl;

            // New background Mat for our image
            cv::Mat bgrFrame = cv::Mat(cv::Size(width, height), CV_8UC3);

            // Use openCV to remove padding
            cv::Mat inMat = cv::Mat(height, width, CV_8UC4, 
                (&(*pBufferSurface))->surfaceList[0].mappedAddr.addr[0],
                (&(*pBufferSurface))->surfaceList[0].pitch);

            cv::cvtColor (inMat, bgrFrame, cv::COLOR_RGBA2BGR);
            
            // Do something with the cv::mat - the below saves the mat to jpeg file

            // Generate the image file name from the date-time string
            std::ostringstream fileNameStream;
            fileNameStream << "batch_" 
                << std::setw(2) << std::setfill('0') << pBufferSurface->GetUniqueId()
                << "_" << pBufferSurface->GetDateTimeStr() << ".jpeg";
                
            // Generate the filespec from the output dir and file name
            std::string filespec = "./" + 
                fileNameStream.str();
            cv::imwrite(filespec.c_str(), bgrFrame);
            
        }
    }
    return DSL_PAD_PROBE_OK;
}

int main(int argc, char** argv)
{
    DslReturnType retval = DSL_RESULT_FAILURE;

    // Since we're not using args, we can Let DSL initialize GST on first call    
    while(true) 
    {    
        // Get the Device type - DSL_GPU_TYPE_INTEGRATED | DSL_GPU_TYPE_DISCRETE
        device_type = dsl_info_gpu_type_get(0);

        // 4 New File Sources
        retval = dsl_source_file_new(L"file-source-0", uri_h265.c_str(), true);
        if (retval != DSL_RESULT_SUCCESS) break;
        retval = dsl_source_file_new(L"file-source-1", uri_h265.c_str(), true);
        if (retval != DSL_RESULT_SUCCESS) break;
        retval = dsl_source_file_new(L"file-source-2", uri_h265.c_str(), true);
        if (retval != DSL_RESULT_SUCCESS) break;
        retval = dsl_source_file_new(L"file-source-3", uri_h265.c_str(), true);
        if (retval != DSL_RESULT_SUCCESS) break;

        // lower the frame-rates - this trivial example saves every frame to jpeg.
        retval = dsl_source_video_buffer_out_frame_rate_set(L"file-source-0", 1, 1);
        if (retval != DSL_RESULT_SUCCESS) break;
        retval = dsl_source_video_buffer_out_frame_rate_set(L"file-source-1", 1, 1);
        if (retval != DSL_RESULT_SUCCESS) break;
        retval = dsl_source_video_buffer_out_frame_rate_set(L"file-source-2", 1, 1);
        if (retval != DSL_RESULT_SUCCESS) break;
        retval = dsl_source_video_buffer_out_frame_rate_set(L"file-source-3", 1, 1);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Primary GIE using the filespecs defined above, with interval = 4
        retval = dsl_infer_gie_primary_new(L"primary-gie", 
            primary_infer_config_file.c_str(), 
            primary_model_engine_file.c_str(), 4);
        if (retval != DSL_RESULT_SUCCESS) break;
        
        // New IOU Tracker, setting operational width and height of input frame
        retval = dsl_tracker_new(L"iou-tracker", 
            iou_tracker_config_file.c_str(), 480, 272);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Tiler, setting width and height
        retval = dsl_tiler_new(L"tiler", TILER_WIDTH, TILER_HEIGHT);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Custom Pad Probe Handler (PPH) to call our handler function above
        retval = dsl_pph_custom_new(L"custom-pph", custom_pad_probe_handler, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the Custom PPH to the sink (input) pad of the Tiler component.
        retval = dsl_tiler_pph_add(L"tiler", L"custom-pph", DSL_PAD_SINK);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new(L"on-screen-display", true, true, true, false);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Window Sink, 0 x/y offsets.
        retval = dsl_sink_window_egl_new(L"egl-sink", 0, 0, 
            WINDOW_WIDTH, WINDOW_HEIGHT);
        if (retval != DSL_RESULT_SUCCESS) break;
    
        // Add the XWindow event handler functions defined above
        retval = dsl_sink_window_key_event_handler_add(L"egl-sink", 
            xwindow_key_event_handler, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_sink_window_delete_event_handler_add(L"egl-sink", 
            xwindow_delete_event_handler, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Create a list of Pipeline Components to add to the new Pipeline.
        const wchar_t* components[] = {
            L"file-source-0", L"file-source-1", L"file-source-2", L"file-source-3",
            L"primary-gie", L"iou-tracker", L"tiler", L"on-screen-display", 
            L"egl-sink", NULL};
        
        // Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many(L"pipeline", components);
        if (retval != DSL_RESULT_SUCCESS) break;
            
        // Add the EOS listener function defined above
        retval = dsl_pipeline_eos_listener_add(L"pipeline", eos_event_listener, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Play the pipeline
        retval = dsl_pipeline_play(L"pipeline");
        if (retval != DSL_RESULT_SUCCESS) break;

        // Start and join the main-loop
        dsl_main_loop_run();
        break;
    }
    
    // Print out the final result
    std::wcout << dsl_return_value_to_string(retval) << std::endl;

    dsl_delete_all();

    std::cout<<"Goodbye!"<<std::endl;  
    return 0;
}
            