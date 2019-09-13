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

#include "DsdDriver.h"

GST_DEBUG_CATEGORY(NVDS_APP);

using namespace DSD;

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

    g_main_loop_quit(Driver::GetDriver()->m_pMainLoop);
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
    
    // Install the custom Program Interrup Signal ISR
    PrgItrSigIsrInstall();    

    
    // First call to GetDriver() for initialization
    Driver* pDriver = Driver::GetDriver();

    DSD_SOURCE_NEW("source1", DSD_SOURCE_TYPE_CSI, 1280, 720, 30, 1);
    DSD_SOURCE_NEW("source1", DSD_SOURCE_TYPE_CSI, 0, 0, 0, 1);

    DSD_STREAMMUX_NEW("streammux1", TRUE, 1, 40000, 1280, 720);
    DSD_STREAMMUX_NEW("streammux1", FALSE, 0, 0, 0, 0);

    DSD_DISPLAY_NEW("display1", 1, 1, 1280, 720);
    DSD_DISPLAY_NEW("display1", 1, 0, 0, 0);

    DSD_GIE_NEW("GIE1",
        "../../models/Primary_Detector/resnet10.caffemodel_b30_int8.engine", 
        "config_infer_primary.txt", 1, 1, 1, 1, 1);
    DSD_GIE_NEW("GIE1", "", "", 0, 0, 0, 0, 0);

    DSD_GIE_DELETE("GIE1");
    DSD_GIE_DELETE("GIE1");
    
    DSD_DISPLAY_DELETE("display1");
    DSD_DISPLAY_DELETE("display1");
    
    DSD_STREAMMUX_DELETE("streammux1");
    DSD_STREAMMUX_DELETE("streammux1");
    
    DSD_SOURCE_DELETE("source1");
    DSD_SOURCE_DELETE("source1");

        // Run the main loop
//        g_main_loop_run(pDrv->m_pMainLoop);

    returnValue = EXIT_SUCCESS;
    
    // Clean-up 
    gst_deinit();
    
    // Main loop has terminated
    return returnValue;
}
