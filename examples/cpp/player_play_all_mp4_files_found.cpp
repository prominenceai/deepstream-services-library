#include <iostream>
#include <experimental/filesystem>
#include <sstream>
#include <fstream>
#include <dirent.h>
#include <regex>

#include "DslApi.h"

// set path to your video files
std::string dir_path = "/root/Videos";

// Find all files in the given path that have the given extension
std::vector<std::string> FindAllFiles(std::string path, std::string type){
  DIR *dir;
  struct dirent *ent;
  std::vector<std::string> FileList;  

  if ((dir = opendir (path.c_str())) != NULL) {
    //Examine all files in this directory
    while ((ent = readdir (dir)) != NULL) {
        std::string filename = std::string(ent->d_name);
        bool dotFound = false;
        int extNdx = -1;
        int leng = 0;

        if(filename.size()>3){
            for(int i = (int)filename.size() - 1; i>0; --i){
                if(filename[i] == '.'){
                    extNdx = i + 1;
                    dotFound = true;
                    break;
                }
                leng++;
            }
        }

        if(dotFound)
        {
            std::string ext = filename.substr(extNdx,leng);
            if(ext == type){
            //store this file if it's correct type
            FileList.push_back(filename);
            }
        }
    }
    closedir (dir);
  } else {
    //Couldn't open dir
    std::cerr << "Could not open directory: " << path << "\n";    
  }

  return FileList;
}


// ## 
// # Function to be called on Player termination event
// ## 
void player_termination_event_listener(void* client_data)
{
    std::cout << "player termination event" << std::endl;
    dsl_main_loop_quit();
}
    
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
    } else if (key == "N"){
        dsl_player_render_next(L"player");
    } else if (key == "Q"){
        dsl_main_loop_quit();
    }
}
 
int main(int argc, char** argv)
{  
    DslReturnType retval;

    std::vector<std::string> mp4files = FindAllFiles(dir_path, "mp4");

    for(auto& file:mp4files){
        file = dir_path + "/" + file;        
    }

    while(true)
    {
        for(auto& file:mp4files){            
            std::wstring wfile(file.begin(), file.end());
            // # create the Player on first file found
            if (!dsl_player_exists(L"player")){            
                // # New Video Render Player to play all the MP4 files found
                retval = dsl_player_render_video_new(L"player", wfile.c_str(), DSL_RENDER_TYPE_WINDOW, 0, 0, 50, false);
                if (retval != DSL_RESULT_SUCCESS) break;
            } else {
                retval = dsl_player_render_file_path_queue(L"player", wfile.c_str());
            }
        }

        // # Add the Termination listener callback to the Player
        retval = dsl_player_termination_event_listener_add(L"player", player_termination_event_listener, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_player_xwindow_key_event_handler_add(L"player", xwindow_key_event_handler, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;

        // # Play the Player until end-of-stream (EOS)
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
