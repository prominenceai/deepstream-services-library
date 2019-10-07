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

#include "DslApi.h"

GST_DEBUG_CATEGORY(NVDS_APP);

using namespace DSL;

#define INIT_MEMORY(m) memset(&m, 0, sizeof(m));

#define INIT_STRUCT(type, name) struct type name; INIT_MEMORY(name) 

/**
 * Function to handle program interrupt signal.
 * It installs default handler after handling the interrupt.
 */
static void PrgItrSigIsr(int signum)
{
    LOG_FUNC();
    
    INIT_STRUCT(sigaction, sa);

    sa.sa_handler = SIG_DFL;

    sigaction(SIGINT, &sa, NULL);

    g_main_loop_quit(Services::GetServices()->m_pMainLoop);
}

/**
 * Function to install custom handler for program interrupt signal.
 */
static void PrgItrSigIsrInstall(void)
{
    LOG_FUNC();

    INIT_STRUCT(sigaction, sa);

    sa.sa_handler = PrgItrSigIsr;

    sigaction(SIGINT, &sa, NULL);
}

 
int main(int argc, char **argv)
{
    LOG_FUNC();
    
    int returnValue = EXIT_FAILURE;
    
    // initialize the GStreamer library
    gst_init(&argc, &argv);
    
    // Install the custom Program Interrupt Signal ISR
    PrgItrSigIsrInstall();    
    
//    if (dsl_source_csi_new(
//        "csi-source1",              // name for new Source 
//        1280, 720,                  // width and height in pixels
//        30, 1)                      // fps-numerator and fps-denominator
//        != DSL_RESULT_SUCCESS)
//    {
//        return EXIT_FAILURE;
//    }
    
    if (dsl_source_uri_new(
        "uri_file_1280_720_30fs",           // name for new Source 
        "sample_1080p_h264.mp4",            // URI source file - using default stream director
        DSL_CUDADEC_MEMTYPE_DEVICE,         // Cudadec memory type to use
        0)                                  // Intra decode
        != DSL_RESULT_SUCCESS)
    {
        return EXIT_FAILURE;
    }
    
//    if (dsl_sink_new(
//        "sink1",                    // name for new Sink
//        0,                          // display Id
//        1,                          // overlay Id
//        0, 0,                       // X and Y offsets
//        0, 0)                       // width and height in pixels
//        != DSL_RESULT_SUCCESS)
//    {
//        return EXIT_FAILURE;
//    }

//    if (dsl_osd_new(
//        "osd1",                     // name for new OSD
//        TRUE)                       // clock enabled?
//        != DSL_RESULT_SUCCESS)
//    {
//        return EXIT_FAILURE;
//    }

//    if (dsl_gie_new(
//        "gie1",                                 // name for new GIE
//        "config_infer_primary.txt",             // Config File with defaults
//        1, 1,                                   // batch size and interval
//        1, 0,                                   // unique Id and gpu Id
//        "resnet10.caffemodel_b1_fp16.engine",   // model engine file
//        "")                                     // raw output folder
//        != DSL_RESULT_SUCCESS)
//    {
//        return EXIT_FAILURE;
//    }

//    if (dsl_display_new(
//        "display",                  // name for new display
//        1, 1,                       // rows and columns
//        1280, 720)                  // width and height in pixels
//        != DSL_RESULT_SUCCESS)
//    {
//        return EXIT_FAILURE;
//    }
    
    if (dsl_pipeline_new(
        "pipeline1")                // name for new pipeline
        != DSL_RESULT_SUCCESS)
    {
        return EXIT_FAILURE;
    }
    
//    const char* components[] = {"source1", "sink1", "osd1", "gie1", "display", NULL};
    const char* components[] = {"uri_file_1280_720_30fs", NULL};
    
    if (dsl_pipeline_components_add(
        "pipeline1",                // name of the Pipeline to update
        components)                 // NULL terminated array of component names
        != DSL_RESULT_SUCCESS)
    {
        return EXIT_FAILURE;
    }

    if (dsl_pipeline_streammux_properties_set(
        "pipeline1",                // name of the Pipeline to update
        FALSE,                       // are sources 1ive? 
        1,                          // batch size
        40000,                      // batch timeout
        1280, 720)                  // width and height in pixels
        != DSL_RESULT_SUCCESS)
    {
        return EXIT_FAILURE;
    }
    
    dsl_pipeline_play("pipeline1");
    
    // Run the main loop
    dsl_main_loop_run();
    
//    dsl_component_delete("display");
    
    dsl_component_delete("uri_file_1280_720_30fs");

    dsl_pipeline_delete("pipeline1");

    returnValue = EXIT_SUCCESS;
    
    // Clean-up 
    gst_deinit();
    
    // Main loop has terminated
    return returnValue;
}
