/*
The MIT License

Copyright (c) 2021, Prominence AI, Inc.

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

#include <iostream>
#include <experimental/filesystem>
#include <sstream>
#include <fstream>
#include <dirent.h>
#include <regex>

#include "DslApi.h"


// Set Camera RTSP URI's - these must be set to valid rtsp uri's for camera's on your network
// RTSP Source URI
std::wstring rtsp_uri_1 = L"rtsp://admin:Password1!@10.0.0.37:554/cam/realmonitor?channel=1&subtype=1";
std::wstring rtsp_uri_2 = L"rtsp://admin:Password1!@10.0.0.37:554/cam/realmonitor?channel=2&subtype=1";


int WINDOW_WIDTH = DSL_STREAMMUX_DEFAULT_WIDTH;
int WINDOW_HEIGHT = DSL_STREAMMUX_DEFAULT_HEIGHT;

// ##     
// # Function to be called on XWindow KeyRelease event    
// ##     
void xwindow_key_event_handler(const wchar_t* in_key, void* client_data)
{   
    std::wstring wkey(in_key); 
    std::string key(wkey.begin(), wkey.end());
    std::cout << "key released = " << key << std::endl;
    key = std::toupper(key[0]);
    if(key == "P"){
        dsl_player_pause(L"player");
    } else if (key == "R"){
        dsl_player_play(L"player");   
    } else if (key == "Q"){
        dsl_player_stop(L"player");
        dsl_main_loop_quit();
    }
}

int main(int argc, char** argv)
{  
    DslReturnType retval;

    // # Since we're not using args, we can Let DSL initialize GST on first call    
    while(true){    

        // # For each camera, create a new RTSP Source for the specific RTSP URI    
        retval = dsl_source_rtsp_new(L"rtsp-source", rtsp_uri_1.c_str(), DSL_RTP_ALL,     
            false, 0, 100, 2);
        if (retval != DSL_RESULT_SUCCESS)    
            return retval;

        // # New Overlay Sink, 0 x/y offsets and same dimensions as Tiled Display    
        retval = dsl_sink_window_new(L"window-sink", 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_player_new(L"player", L"rtsp-source", L"window-sink");
        if (retval != DSL_RESULT_SUCCESS) break;
            
        // # Add the XWindow event handler functions defined above    
        retval = dsl_player_xwindow_key_event_handler_add(L"player", xwindow_key_event_handler, nullptr);    
        if (retval != DSL_RESULT_SUCCESS) break;    
        
        // # Play the player    
        retval = dsl_player_play(L"player");
        if (retval != DSL_RESULT_SUCCESS) break;

        dsl_main_loop_run();
        retval = DSL_RESULT_SUCCESS;
        break;
    }

    // # Print out the final result
    std::cout << dsl_return_value_to_string(retval) << std::endl;

    // # Cleanup all DSL/GST resources
    dsl_delete_all();

    std::cout<<"Goodbye!"<<std::endl;  
    return 0;
}
