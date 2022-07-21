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

SCENARIO( "The ODE Actions container is updated correctly on multiple new ODE Action", "[ode-action-api]" )
{
    GIVEN( "An empty list of Events" ) 
    {
        std::wstring action_name1(L"log-action-1");
        std::wstring action_name2(L"log-action-2");
        std::wstring action_name3(L"log-action-3");
        
        REQUIRE( dsl_ode_action_list_size() == 0 );

        WHEN( "Several new Actions are created" ) 
        {
            REQUIRE( dsl_ode_action_log_new(action_name1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_action_log_new(action_name2.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_action_log_new(action_name3.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size and events are updated correctly" ) 
            {
                REQUIRE( dsl_ode_action_list_size() == 3 );

                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The ODE Actions container is updated correctly on Delete ODE Action", "[ode-action-api]" )
{
    GIVEN( "A list of several ODE Actions" ) 
    {
        std::wstring action_name1(L"action-1");
        std::wstring action_name2(L"action-2");
        std::wstring action_name3(L"action-3");
        boolean display(true);
        
        REQUIRE( dsl_ode_action_list_size() == 0 );

        REQUIRE( dsl_ode_action_log_new(action_name1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_action_log_new(action_name2.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_action_log_new(action_name3.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A single Action is deleted" ) 
        {
            REQUIRE( dsl_ode_action_delete(action_name1.c_str()) == DSL_RESULT_SUCCESS );
            THEN( "The list size and events are updated correctly" ) 
            {
                REQUIRE( dsl_ode_action_list_size() == 2 );

                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "Multiple Actions are deleted" ) 
        {
            const wchar_t* actions[] = {L"action-2", L"action-3", NULL};
            
            REQUIRE( dsl_ode_action_delete_many(actions) == DSL_RESULT_SUCCESS );
            THEN( "The list size and events are updated correctly" ) 
            {
                REQUIRE( dsl_ode_action_list_size() == 1 );

                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Format Bounding Box ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Format Bounding Box ODE Action" ) 
    {
        std::wstring action_name(L"format-bbox-action");
        uint border_width(5);

        std::wstring border_color_name(L"my-border-color");
        std::wstring bg_color_name(L"my-bg-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        
        boolean has_bg_color(true);
        
        REQUIRE( dsl_display_type_rgba_color_custom_new(border_color_name.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_color_custom_new(bg_color_name.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        WHEN( "A new Format Bounding Box Action is created" ) 
        {
            REQUIRE( dsl_ode_action_format_bbox_new(action_name.c_str(), border_width, 
                border_color_name.c_str(), has_bg_color, bg_color_name.c_str()) == DSL_RESULT_SUCCESS );

            // second attempt must fail
            REQUIRE( dsl_ode_action_format_bbox_new(action_name.c_str(), border_width, 
                border_color_name.c_str(), has_bg_color, bg_color_name.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
            
            THEN( "The Format Bounding Box Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                
                // second attempt must fail
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_FOUND );
                
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Format Bounding Box ODE Action with no Border or Background Color can be created", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Format Bounding Box ODE Action" ) 
    {
        std::wstring action_name(L"format-bbox-action");

        WHEN( "Using using input parameters border_width = 0 and has_bg_color = false" ) 
        {
            uint border_width(0);
            boolean has_bg_color(false);

            THEN( "The Format Bounding Box Action can be created" ) 
            {
                REQUIRE( dsl_ode_action_format_bbox_new(action_name.c_str(), border_width, 
                    NULL, has_bg_color, NULL) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Format Bounding Box ODE Action verifies its input parameters correctly", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Format Bounding Box ODE Action" ) 
    {
        std::wstring action_name(L"format-bbox-action");

        std::wstring border_color_name(L"my-border-color");
        std::wstring bg_color_name(L"my-bg-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        
        uint border_width(5);
        boolean has_bg_color(true);
        
        REQUIRE( dsl_display_type_rgba_color_custom_new(border_color_name.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_color_custom_new(bg_color_name.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        WHEN( "Using input parameters border_width > 0 and a border_coler = NULL" ) 
        {
            THEN( "The Format Bounding Box Action will fail to create" ) 
            {
                REQUIRE( dsl_ode_action_format_bbox_new(action_name.c_str(), border_width, 
                    NULL, has_bg_color, bg_color_name.c_str()) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_list_size() == 0 );
                
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "Using input parameters has_bg_color = true and a bg_coler = NULL" ) 
        {
            uint border_width(0);
            
            THEN( "The Format Bounding Box Action will fail to create" ) 
            {
                REQUIRE( dsl_ode_action_format_bbox_new(action_name.c_str(), border_width, 
                    NULL, has_bg_color, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_list_size() == 0 );
                
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Format Object Label ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Format Object Label ODE Action" ) 
    {
        std::wstring action_name(L"format-label-action");

        std::wstring font_name(L"font-name");
        std::wstring font(L"arial");
        uint size(14);

        std::wstring font_color_name(L"font-color");
        std::wstring font_bg_color_name(L"font-bg-color");

        double redFont(0.0), greenFont(0.0), blueFont(0.0), alphaFont(1.0);
        double redBgColor(0.12), greenBgColor(0.34), blueBgColor(0.56), alphaBgColor(0.78);
        
        boolean has_bg_color(true);
        
        REQUIRE( dsl_display_type_rgba_color_custom_new(font_color_name.c_str(), 
            redFont, greenFont, blueFont, alphaFont) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_color_custom_new(font_bg_color_name.c_str(), 
            redBgColor, greenBgColor, blueBgColor, alphaBgColor) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_font_new(font_name.c_str(), 
            font.c_str(), size, font_color_name.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A new Format Bounding Box Action is created" ) 
        {
            REQUIRE( dsl_ode_action_format_label_new(action_name.c_str(),  
                font_name.c_str(), has_bg_color, font_bg_color_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

            // second attempt must fail
            REQUIRE( dsl_ode_action_format_label_new(action_name.c_str(),  
                font_name.c_str(), has_bg_color, font_bg_color_name.c_str()) 
                    == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
            
            THEN( "The Format Bounding Box Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                
                // second attempt must fail
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) 
                    == DSL_RESULT_ODE_ACTION_NAME_NOT_FOUND );
                
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Format Object Label ODE Action checks its input parameters correctly", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Format Object Label ODE Action" ) 
    {
        std::wstring action_name(L"format-label-action");

        std::wstring font_name(L"font-name");
        std::wstring font(L"arial");
        uint size(14);

        std::wstring font_color_name(L"font-color");
        std::wstring font_bg_color_name(L"font-bg-color");

        double redFont(0.0), greenFont(0.0), blueFont(0.0), alphaFont(1.0);
        double redBgColor(0.12), greenBgColor(0.34), blueBgColor(0.56), alphaBgColor(0.78);
        
        REQUIRE( dsl_display_type_rgba_color_custom_new(font_color_name.c_str(), 
            redFont, greenFont, blueFont, alphaFont) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_color_custom_new(font_bg_color_name.c_str(), 
            redBgColor, greenBgColor, blueBgColor, alphaBgColor) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_font_new(font_name.c_str(), 
            font.c_str(), size, font_color_name.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A new Format ObjectLabel Action is created with NO Font" ) 
        {
            REQUIRE( dsl_ode_action_format_label_new(action_name.c_str(),  
                NULL, true, font_bg_color_name.c_str()) 
                    == DSL_RESULT_SUCCESS );
            
            THEN( "The Format Bounding Box Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                
                // second attempt must fail
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) 
                    == DSL_RESULT_ODE_ACTION_NAME_NOT_FOUND );
                
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new Format Bounding Box Action is created with NO Background Color" ) 
        {
            REQUIRE( dsl_ode_action_format_label_new(action_name.c_str(),  
                font_name.c_str(), false, NULL) 
                    == DSL_RESULT_SUCCESS );
            
            THEN( "The Format Bounding Box Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                
                // second attempt must fail
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) 
                    == DSL_RESULT_ODE_ACTION_NAME_NOT_FOUND );
                
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Custom ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Custom ODE Action" ) 
    {
        std::wstring action_name(L"custom-action");
        dsl_ode_handle_occurrence_cb client_handler;

        WHEN( "A new Custom ODE Action is created" ) 
        {
            REQUIRE( dsl_ode_action_custom_new(action_name.c_str(), client_handler, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "The Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Custom ODE Action is created" ) 
        {
            REQUIRE( dsl_ode_action_custom_new(action_name.c_str(), client_handler, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "A second custom of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_custom_new(action_name.c_str(), 
                    client_handler, NULL) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Monitor ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Monitor ODE Action" ) 
    {
        std::wstring action_name(L"custom-action");
        dsl_ode_monitor_occurrence_cb client_monitor;

        WHEN( "A new Monitor ODE Action is created" ) 
        {
            REQUIRE( dsl_ode_action_monitor_new(action_name.c_str(), 
                client_monitor, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "The Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Monitor ODE Action is created" ) 
        {
            REQUIRE( dsl_ode_action_monitor_new(action_name.c_str(), 
                client_monitor, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "A second custom of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_monitor_new(action_name.c_str(), 
                    client_monitor, NULL) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Frame Capture ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Frame Capture ODE Action" ) 
    {
        std::wstring action_name(L"capture-action");
        std::wstring outdir(L"./");
        boolean annotate(true);

        WHEN( "A new Frame Capture Action is created" ) 
        {
            REQUIRE( dsl_ode_action_capture_frame_new(action_name.c_str(), 
                outdir.c_str(), annotate) == DSL_RESULT_SUCCESS );
            
            THEN( "The Frame Capture Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Frame Capture Action is created" ) 
        {
            REQUIRE( dsl_ode_action_capture_frame_new(action_name.c_str(), 
                outdir.c_str(), annotate) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Frame Capture Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_capture_frame_new(action_name.c_str(), 
                    outdir.c_str(), annotate) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "An invalid Output Directory is specified" ) 
        {
            std::wstring invalidOutDir(L"/invalid/output/directory");
            
            THEN( "A new Frame Capture Action fails to create" ) 
            {
                REQUIRE( dsl_ode_action_capture_frame_new(action_name.c_str(),
                    invalidOutDir.c_str(), annotate) == DSL_RESULT_ODE_ACTION_FILE_PATH_NOT_FOUND );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Object Capture ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Object Capture ODE Action" ) 
    {
        std::wstring action_name(L"capture-action");
        std::wstring outdir(L"./");

        WHEN( "A new Object Capture Action is created" ) 
        {
            REQUIRE( dsl_ode_action_capture_object_new(action_name.c_str(), 
                outdir.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Object Capture Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Object Capture Action is created" ) 
        {
            REQUIRE( dsl_ode_action_capture_object_new(action_name.c_str(), 
                outdir.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Object Capture Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_capture_object_new(action_name.c_str(), 
                    outdir.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "An invalid Output Directory is specified" ) 
        {
            std::wstring invalidOutDir(L"/invalid/output/directory");
            
            THEN( "A new Object Capture Action fails to create" ) 
            {
                REQUIRE( dsl_ode_action_capture_object_new(action_name.c_str(), 
                    invalidOutDir.c_str()) == DSL_RESULT_ODE_ACTION_FILE_PATH_NOT_FOUND );
                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

static void capture_complete_cb(dsl_capture_info* pInfo, void* user_data)
{
}

SCENARIO( "A Capture Complete Listener can be added and removed", "[ode-action-api]" )
{
    GIVEN( "A new Capture Action and client listener callback" )
    {
        std::wstring action_name(L"capture-action");
        std::wstring outdir(L"./");
        
        REQUIRE( dsl_ode_action_capture_object_new(action_name.c_str(), 
            outdir.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A capture-complete-listner is added" )
        {
            REQUIRE( dsl_ode_action_capture_complete_listener_add(action_name.c_str(),
                capture_complete_cb, NULL) == DSL_RESULT_SUCCESS );

            // ensure the same listener twice fails
            REQUIRE( dsl_ode_action_capture_complete_listener_add(action_name.c_str(),
                capture_complete_cb, NULL) == DSL_RESULT_ODE_ACTION_CALLBACK_ADD_FAILED );

            THEN( "The same listner can be remove" ) 
            {
                REQUIRE( dsl_ode_action_capture_complete_listener_remove(action_name.c_str(),
                    capture_complete_cb) == DSL_RESULT_SUCCESS );

                // calling a second time must fail
                REQUIRE( dsl_ode_action_capture_complete_listener_remove(action_name.c_str(),
                    capture_complete_cb) == DSL_RESULT_ODE_ACTION_CALLBACK_REMOVE_FAILED );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "A Player can be added and removed from a Capture Action", "[ode-action-api]" )
{
    GIVEN( "A new Capture Action and Image Player" )
    {
        std::wstring action_name(L"capture-action");
        std::wstring outdir(L"./");

        std::wstring player_name(L"player");
        std::wstring file_path(L"/opt/nvidia/deepstream/deepstream/samples/streams/yoga.jpg");
        
        REQUIRE( dsl_ode_action_capture_object_new(action_name.c_str(), 
            outdir.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_player_render_image_new(player_name.c_str(),file_path.c_str(), 
            DSL_RENDER_TYPE_OVERLAY, 10, 10, 75, 0) == DSL_RESULT_SUCCESS );

        WHEN( "A Player is added" )
        {
            REQUIRE( dsl_ode_action_capture_image_player_add(action_name.c_str(),
                player_name.c_str()) == DSL_RESULT_SUCCESS );

            // ensure the same listener twice fails
            REQUIRE( dsl_ode_action_capture_image_player_add(action_name.c_str(),
                player_name.c_str()) == DSL_RESULT_ODE_ACTION_PLAYER_ADD_FAILED );

            THEN( "The same Player can be removed" ) 
            {
                REQUIRE( dsl_ode_action_capture_image_player_remove(action_name.c_str(),
                    player_name.c_str()) == DSL_RESULT_SUCCESS );

                // calling a second time must fail
                REQUIRE( dsl_ode_action_capture_image_player_remove(action_name.c_str(),
                    player_name.c_str()) == DSL_RESULT_ODE_ACTION_PLAYER_REMOVE_FAILED );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_player_delete(player_name.c_str()) == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "A Mailer can be added and removed from a Capture Action", "[ode-action-api]" )
{
    GIVEN( "A new Capture Action and Mailer" )
    {
        std::wstring action_name(L"capture-action");
        std::wstring outdir(L"./");

        std::wstring mailer_name(L"mailer");
        
        std::wstring subject(L"Subject line");
        
        REQUIRE( dsl_ode_action_capture_object_new(action_name.c_str(), 
            outdir.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_mailer_new(mailer_name.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A capture-complete-listner is added" )
        {
            REQUIRE( dsl_ode_action_capture_mailer_add(action_name.c_str(),
                mailer_name.c_str(), subject.c_str(), false) == DSL_RESULT_SUCCESS );

            // ensure the same listener twice fails
            REQUIRE( dsl_ode_action_capture_mailer_add(action_name.c_str(),
                mailer_name.c_str(), subject.c_str(), false) == DSL_RESULT_ODE_ACTION_MAILER_ADD_FAILED );

            THEN( "The same listner can be removed" ) 
            {
                REQUIRE( dsl_ode_action_capture_mailer_remove(action_name.c_str(),
                    mailer_name.c_str()) == DSL_RESULT_SUCCESS );

                // calling a second time must fail
                REQUIRE( dsl_ode_action_capture_mailer_remove(action_name.c_str(),
                    mailer_name.c_str()) == DSL_RESULT_ODE_ACTION_MAILER_REMOVE_FAILED );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_mailer_delete(mailer_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_mailer_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "A new Customize Label ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Display ODE Action" ) 
    {
        std::wstring action_name(L"customize-label-action");
        uint label_types[] = {DSL_METRIC_OBJECT_LOCATION,
            DSL_METRIC_OBJECT_DIMENSIONS, DSL_METRIC_OBJECT_CONFIDENCE_INFERENCE,
            DSL_METRIC_OBJECT_PERSISTENCE};

        uint size(4);

        WHEN( "A new Customize Label is created" ) 
        {
            REQUIRE( dsl_ode_action_customize_label_new(action_name.c_str(),
                label_types, size) == DSL_RESULT_SUCCESS );

            // second attempt must fail
            REQUIRE( dsl_ode_action_customize_label_new(action_name.c_str(),
                label_types, size) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
            
            THEN( "The Customize Label can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );

                // second attempt must fail
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == 
                    DSL_RESULT_ODE_ACTION_NAME_NOT_FOUND );
                
            }
        }
    }
}

SCENARIO( "Parameters for a new Customize Label ODE Action are checked on construction", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Customize Label ODE Action" ) 
    {
        std::wstring action_name(L"customize-label-action");
        uint label_types[] = {DSL_METRIC_OBJECT_LOCATION,
            DSL_METRIC_OBJECT_DIMENSIONS, DSL_METRIC_OBJECT_CONFIDENCE_INFERENCE,
            DSL_METRIC_OBJECT_PERSISTENCE};

        WHEN( "The size parameter is out of range" ) 
        {
            uint size(5);
            
            THEN( "The Customize Label Action fails to create" ) 
            {
                REQUIRE( dsl_ode_action_customize_label_new(action_name.c_str(),
                    label_types, size) == DSL_RESULT_ODE_ACTION_PARAMETER_INVALID );

                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Customize Label ODE Action can be updated", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Display ODE Action" ) 
    {
        std::wstring action_name(L"customize-label-action");
        uint label_types[] = {DSL_METRIC_OBJECT_LOCATION,
            DSL_METRIC_OBJECT_DIMENSIONS, DSL_METRIC_OBJECT_CONFIDENCE_INFERENCE,
            DSL_METRIC_OBJECT_PERSISTENCE};
            
        uint size(4);

        REQUIRE( dsl_ode_action_customize_label_new(action_name.c_str(),
            label_types, size) == DSL_RESULT_SUCCESS );

        WHEN( "A Customize Label is updated" ) 
        {
            uint ret_label_types[DSL_METRIC_OBJECT_PERSISTENCE+1] = {0};
            uint in_out_size(DSL_METRIC_OBJECT_PERSISTENCE+1);

            // test initial condition first
            REQUIRE( dsl_ode_action_customize_label_get(action_name.c_str(),
                ret_label_types, &in_out_size) == DSL_RESULT_SUCCESS );
            REQUIRE( in_out_size == size );
            REQUIRE( label_types[0] == ret_label_types[0] );
            REQUIRE( label_types[1] == ret_label_types[1] );
            REQUIRE( label_types[2] == ret_label_types[2] );
            REQUIRE( label_types[3] == ret_label_types[3] );

            uint new_label_types[] = {DSL_METRIC_OBJECT_CLASS};
            size = 1;

            REQUIRE( dsl_ode_action_customize_label_set(action_name.c_str(),
                new_label_types, size) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct values are returned on get" ) 
            {
                REQUIRE( dsl_ode_action_customize_label_get(action_name.c_str(),
                    ret_label_types, &in_out_size) == DSL_RESULT_SUCCESS );
                REQUIRE( in_out_size == size );
                
                REQUIRE( new_label_types[0] == ret_label_types[0] );
                
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Customize Label ODE Action checks its parameters correctly", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Display ODE Action" ) 
    {
        std::wstring action_name(L"customize-label-action");
        uint label_types[] = {DSL_METRIC_OBJECT_LOCATION,
            DSL_METRIC_OBJECT_DIMENSIONS, DSL_METRIC_OBJECT_CONFIDENCE_INFERENCE,
            DSL_METRIC_OBJECT_PERSISTENCE};
            
        uint size(4);

        REQUIRE( dsl_ode_action_customize_label_new(action_name.c_str(),
            label_types, size) == DSL_RESULT_SUCCESS );

        WHEN( "When insufficient memory is passed in on Get" ) 
        {
            uint ret_label_types[2] = {0};
            uint in_out_size(2);
            
            THEN( "The Get call fails correctly" ) 
            {
                REQUIRE( dsl_ode_action_customize_label_get(action_name.c_str(),
                    ret_label_types, &in_out_size) == DSL_RESULT_ODE_ACTION_PARAMETER_INVALID );

                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "When the size parameter is NULL" ) 
        {
            uint* pSize(NULL);
            uint ret_label_types[2] = {0};
            
            THEN( "The ODE Actions container is unchanged" ) 
            {
                REQUIRE( dsl_ode_action_customize_label_get(action_name.c_str(),
                    ret_label_types, pSize) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Display ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Display ODE Action" ) 
    {
        std::wstring action_name(L"display-action");
        
        std::wstring font(L"arial");
        std::wstring fontName(L"arial-20");
        uint size(20);

        std::wstring fontColorName(L"my-font-color");
        std::wstring bg_color_name(L"my-bg-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        
        std::wstring format_string(L"Class: %0");
        
        REQUIRE( dsl_display_type_rgba_color_custom_new(fontColorName.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_font_new(fontName.c_str(), font.c_str(),
            size, fontColorName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_color_custom_new(bg_color_name.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        WHEN( "A new Display Action is created" ) 
        {
            REQUIRE( dsl_ode_action_display_new(action_name.c_str(), 
                format_string.c_str(), 10, 10, fontName.c_str(), true, 
                bg_color_name.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Display Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
                
            }
        }
        WHEN( "A new Display Action is created" ) 
        {
            REQUIRE( dsl_ode_action_display_new(action_name.c_str(), 
                format_string.c_str(), 10, 10, fontName.c_str(), true, 
                bg_color_name.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Display Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_display_new(action_name.c_str(), 
                    format_string.c_str(), 10, 10, fontName.c_str(), true, 
                    bg_color_name.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Add Display Meta ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Add Display Meta ODE Action" ) 
    {
        std::wstring action_name(L"display-meta-action");
        
        std::wstring font(L"arial");
        std::wstring fontName(L"arial-20");
        uint size(20);
        

        std::wstring fontColorName(L"my-font-color");
        std::wstring bg_color_name(L"my-bg-color");
        
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        std::wstring sourceDisplay(L"source-display");
        uint x_offset(10), y_offset(10);
        
        
        REQUIRE( dsl_display_type_rgba_color_custom_new(fontColorName.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_font_new(fontName.c_str(), font.c_str(),
            size, fontColorName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_source_name_new(sourceDisplay.c_str(), 
            x_offset, y_offset, fontName.c_str(), false, NULL) == DSL_RESULT_SUCCESS );
            
        WHEN( "A new Add Display Meta Action is created" ) 
        {
            REQUIRE( dsl_ode_action_display_meta_add_new(action_name.c_str(), sourceDisplay.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Add Display Meta Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
                
            }
        }
        WHEN( "A new Add Display Meta Action is created" ) 
        {
            REQUIRE( dsl_ode_action_display_meta_add_new(action_name.c_str(), sourceDisplay.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Display Action of the same name fails to create" ) 
            {
            REQUIRE( dsl_ode_action_display_meta_add_new(action_name.c_str(), sourceDisplay.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Add Many Display Meta ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Add Many Display Meta ODE Action" ) 
    {
        std::wstring action_name(L"display-meta-action");
        
        std::wstring font(L"arial");
        std::wstring fontName(L"arial-20");
        uint size(20);
        

        std::wstring fontColorName(L"my-font-color");
        std::wstring bg_color_name(L"my-bg-color");
        
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        std::wstring sourceName(L"source-name");
        std::wstring sourceNumber(L"source-number");
        std::wstring sourceDimensions(L"source-dimensions");
        uint x_offset(10), y_offset(10);
        
        
        REQUIRE( dsl_display_type_rgba_color_custom_new(fontColorName.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_font_new(fontName.c_str(), font.c_str(),
            size, fontColorName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_source_name_new(sourceName.c_str(), 
            x_offset, y_offset, fontName.c_str(), false, NULL) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_source_number_new(sourceNumber.c_str(), 
            x_offset, y_offset+30, fontName.c_str(), false, NULL) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_source_dimensions_new(sourceDimensions.c_str(), 
            x_offset, y_offset+60, fontName.c_str(), false, NULL) == DSL_RESULT_SUCCESS );

        WHEN( "A new Add Display Meta Action is created" ) 
        {
            const wchar_t* display_types[] = {L"source-name", L"source-number", L"source-dimensions", NULL};
            REQUIRE( dsl_ode_action_display_meta_add_many_new(action_name.c_str(), display_types) == DSL_RESULT_SUCCESS );
            
            THEN( "The Add Display Meta Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
                
            }
        }
    }
}

SCENARIO( "A new Text File ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new File ODE Action" ) 
    {
        std::wstring action_name(L"file-action");
        std::wstring file_path(L"./file-action.txt");
        uint mode(DSL_WRITE_MODE_TRUNCATE);
        uint format(DSL_EVENT_FILE_FORMAT_TEXT);
        boolean force_flush(true);

        WHEN( "A new File Action is created" ) 
        {
            REQUIRE( dsl_ode_action_file_new(action_name.c_str(),
                file_path.c_str(), mode, format, force_flush) == DSL_RESULT_SUCCESS );
            
            THEN( "The File Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new File Action is created" ) 
        {
            REQUIRE( dsl_ode_action_file_new(action_name.c_str(),
                file_path.c_str(), mode, format, force_flush) == DSL_RESULT_SUCCESS );
            
            THEN( "A second File Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_file_new(action_name.c_str(),
                    file_path.c_str(), mode, format, force_flush) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new CSV File ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new File ODE Action" ) 
    {
        std::wstring action_name(L"file-action");
        std::wstring file_path(L"./file-action.txt");
        uint mode(DSL_WRITE_MODE_TRUNCATE);
        uint format(DSL_EVENT_FILE_FORMAT_CSV);
        boolean force_flush(true);

        WHEN( "A new File Action is created" ) 
        {
            REQUIRE( dsl_ode_action_file_new(action_name.c_str(),
                file_path.c_str(), mode, format, force_flush) == DSL_RESULT_SUCCESS );
            
            THEN( "The File Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new File Action is created" ) 
        {
            REQUIRE( dsl_ode_action_file_new(action_name.c_str(),
                file_path.c_str(), mode, format, force_flush) == DSL_RESULT_SUCCESS );
            
            THEN( "A second File Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_file_new(action_name.c_str(),
                    file_path.c_str(), mode, format, force_flush) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new MOT Challenge File ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new File ODE Action" ) 
    {
        std::wstring action_name(L"file-action");
        std::wstring file_path(L"./file-action.txt");
        uint mode(DSL_WRITE_MODE_TRUNCATE);
        uint format(DSL_EVENT_FILE_FORMAT_MOTC);
        boolean force_flush(true);

        WHEN( "A new File Action is created" ) 
        {
            REQUIRE( dsl_ode_action_file_new(action_name.c_str(),
                file_path.c_str(), mode, format, force_flush) == DSL_RESULT_SUCCESS );
            
            THEN( "The File Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new File Action is created" ) 
        {
            REQUIRE( dsl_ode_action_file_new(action_name.c_str(),
                file_path.c_str(), mode, format, force_flush) == DSL_RESULT_SUCCESS );
            
            THEN( "A second File Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_file_new(action_name.c_str(),
                    file_path.c_str(), mode, format, force_flush) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "Parameters for a new File ODE Action are checked on construction", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new File ODE Action" ) 
    {
        std::wstring action_name(L"file-action");
        std::wstring file_path(L"./file-action.txt");
        boolean force_flush(true);

        WHEN( "The mode parameter is out of range" ) 
        {
            uint mode(DSL_WRITE_MODE_TRUNCATE+1);
            uint format(DSL_EVENT_FILE_FORMAT_TEXT);
            
            THEN( "The File Action fails to create" ) 
            {
                REQUIRE( dsl_ode_action_file_new(action_name.c_str(),
                    file_path.c_str(), mode, format, force_flush) == DSL_RESULT_ODE_ACTION_PARAMETER_INVALID );

                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "The format parameter is out of range" ) 
        {
            uint mode(DSL_WRITE_MODE_TRUNCATE);
            uint format(DSL_EVENT_FILE_FORMAT_MOTC+1);
            
            THEN( "The File Action fails to create" ) 
            {
                REQUIRE( dsl_ode_action_file_new(action_name.c_str(),
                    file_path.c_str(), mode, format, force_flush) == DSL_RESULT_ODE_ACTION_PARAMETER_INVALID );

                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Fill Frame ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Fill Frame ODE Action" ) 
    {
        std::wstring action_name(L"fill-frame-action");
        
        std::wstring colorName(L"my-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        WHEN( "A new Fill Frame Action is created" ) 
        {
            REQUIRE( dsl_ode_action_fill_frame_new(action_name.c_str(), colorName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Fill Frame Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new Fill Frame Action is created" ) 
        {
            REQUIRE( dsl_ode_action_fill_frame_new(action_name.c_str(), colorName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Fill Frame Action of the same name fails to create" ) 
            {
                REQUIRE( dsl_ode_action_fill_frame_new(action_name.c_str(), colorName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}


SCENARIO( "A new Fill Surroundings ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Fill Surroundings ODE Action" ) 
    {
        std::wstring action_name(L"fill-frame-action");
        
        std::wstring colorName(L"my-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        WHEN( "A new Fill Surroundings Action is created" ) 
        {
            REQUIRE( dsl_ode_action_fill_surroundings_new(action_name.c_str(), colorName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Fill Frame Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new Fill Frame Action is created" ) 
        {
            REQUIRE( dsl_ode_action_fill_surroundings_new(action_name.c_str(), colorName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Fill Frame Action of the same name fails to create" ) 
            {
                REQUIRE( dsl_ode_action_fill_surroundings_new(action_name.c_str(), colorName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Handler Disable ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Handler Disable ODE Action" ) 
    {
        std::wstring action_name(L"handler-disable-action");
        std::wstring handlerName(L"handler");

        WHEN( "A new Handler Disable Action is created" ) 
        {
            REQUIRE( dsl_ode_action_handler_disable_new(action_name.c_str(), handlerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Handler Disable Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Handler Disable Action is created" ) 
        {
            REQUIRE( dsl_ode_action_handler_disable_new(action_name.c_str(), handlerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Handler Disable Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_handler_disable_new(action_name.c_str(), handlerName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Log ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Log ODE Action" ) 
    {
        std::wstring action_name(L"log-action");
        boolean offsetY_with_classId(true);

        WHEN( "A new Log Action is created" ) 
        {
            REQUIRE( dsl_ode_action_log_new(action_name.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Log Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Log Action is created" ) 
        {
            REQUIRE( dsl_ode_action_log_new(action_name.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Log Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_log_new(action_name.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Pause ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Pause ODE Action" ) 
    {
        std::wstring action_name(L"pause-action");
        std::wstring pipelineName(L"pipeline");
        boolean offsetY_with_classId(true);

        WHEN( "A new Pause Action is created" ) 
        {
            REQUIRE( dsl_ode_action_pause_new(action_name.c_str(), pipelineName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Pause Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Pause Action is created" ) 
        {
            REQUIRE( dsl_ode_action_pause_new(action_name.c_str(), pipelineName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Pause Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_pause_new(action_name.c_str(), pipelineName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Print ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Print ODE Action" ) 
    {
        std::wstring action_name(L"print-action");

        WHEN( "A new Print Action is created" ) 
        {
            REQUIRE( dsl_ode_action_print_new(action_name.c_str(), false) == DSL_RESULT_SUCCESS );
            
            THEN( "The Print Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Print Action is created" ) 
        {
            REQUIRE( dsl_ode_action_print_new(action_name.c_str(), false) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Print Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_print_new(action_name.c_str(), false) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Redact ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Redact ODE Action" ) 
    {
        std::wstring action_name(L"redact-action");

        WHEN( "A new Redact Action is created" ) 
        {
            REQUIRE( dsl_ode_action_redact_new(action_name.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Redact Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Redact Action is created" ) 
        {
            REQUIRE( dsl_ode_action_redact_new(action_name.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Redact Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_redact_new(action_name.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Add Sink ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Add Sink ODE Action" ) 
    {
        std::wstring action_name(L"sink_add-action");
        std::wstring pipelineName(L"pipeline");
        std::wstring sinkName(L"sink");

        WHEN( "A new Add Sink Action is created" ) 
        {
            REQUIRE( dsl_ode_action_sink_add_new(action_name.c_str(), pipelineName.c_str(), sinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Add Sink Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Add Sink Action is created" ) 
        {
            REQUIRE( dsl_ode_action_sink_add_new(action_name.c_str(), pipelineName.c_str(), sinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Add Sink Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_sink_add_new(action_name.c_str(), pipelineName.c_str(), sinkName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Remove Sink ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Remove Sink ODE Action" ) 
    {
        std::wstring action_name(L"sink_add-action");
        std::wstring pipelineName(L"pipeline");
        std::wstring sinkName(L"sink");

        WHEN( "A new Remove Sink Action is created" ) 
        {
            REQUIRE( dsl_ode_action_sink_remove_new(action_name.c_str(), pipelineName.c_str(), sinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Remove Sink Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Remove Sink Action is created" ) 
        {
            REQUIRE( dsl_ode_action_sink_remove_new(action_name.c_str(), pipelineName.c_str(), sinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Remove Sink Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_sink_remove_new(action_name.c_str(), pipelineName.c_str(), sinkName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Add Source ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Add Source ODE Action" ) 
    {
        std::wstring action_name(L"source_add-action");
        std::wstring pipelineName(L"pipeline");
        std::wstring sourceName(L"source");

        WHEN( "A new Add Source Action is created" ) 
        {
            REQUIRE( dsl_ode_action_source_add_new(action_name.c_str(), pipelineName.c_str(), sourceName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Add Source Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Add Source Action is created" ) 
        {
            REQUIRE( dsl_ode_action_source_add_new(action_name.c_str(), pipelineName.c_str(), sourceName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Add Source Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_source_add_new(action_name.c_str(), pipelineName.c_str(), sourceName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Remove Source ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Remove Source ODE Action" ) 
    {
        std::wstring action_name(L"source_add-action");
        std::wstring pipelineName(L"pipeline");
        std::wstring sourceName(L"source");

        WHEN( "A new Remove Source Action is created" ) 
        {
            REQUIRE( dsl_ode_action_source_remove_new(action_name.c_str(), pipelineName.c_str(), sourceName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Remove Source Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Remove Source Action is created" ) 
        {
            REQUIRE( dsl_ode_action_source_remove_new(action_name.c_str(), pipelineName.c_str(), sourceName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Remove Source Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_source_remove_new(action_name.c_str(), pipelineName.c_str(), sourceName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Reset Trigger ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Reset Trigger ODE Action" ) 
    {
        std::wstring action_name(L"trigger-reset-action");
        std::wstring triggerName(L"trigger");

        WHEN( "A new Reset Trigger Action is created" ) 
        {
            REQUIRE( dsl_ode_action_trigger_reset_new(action_name.c_str(), triggerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Reset Trigger Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Reset Trigger Action is created" ) 
        {
            REQUIRE( dsl_ode_action_trigger_reset_new(action_name.c_str(), triggerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Reset Trigger Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_trigger_reset_new(action_name.c_str(), triggerName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Disable Trigger ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Disable Trigger ODE Action" ) 
    {
        std::wstring action_name(L"trigger_disable-action");
        std::wstring triggerName(L"trigger");

        WHEN( "A new Disable Trigger Action is created" ) 
        {
            REQUIRE( dsl_ode_action_trigger_disable_new(action_name.c_str(), triggerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Disable Trigger Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Disable Trigger Action is created" ) 
        {
            REQUIRE( dsl_ode_action_trigger_disable_new(action_name.c_str(), triggerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Disable Trigger Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_trigger_disable_new(action_name.c_str(), triggerName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Enable Trigger ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Enable Trigger ODE Action" ) 
    {
        std::wstring action_name(L"trigger-enable-action");
        std::wstring triggerName(L"trigger");

        WHEN( "A new Enable Trigger Action is created" ) 
        {
            REQUIRE( dsl_ode_action_trigger_enable_new(action_name.c_str(), triggerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Enable Trigger Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Enable Trigger Action is created" ) 
        {
            REQUIRE( dsl_ode_action_trigger_enable_new(action_name.c_str(), triggerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Enable Trigger Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_trigger_enable_new(action_name.c_str(), triggerName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Disable Action ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Disable Action ODE Action" ) 
    {
        std::wstring action_name(L"action_disable-action");
        std::wstring slaveActionName(L"action");

        WHEN( "A new Disable Action Action is created" ) 
        {
            REQUIRE( dsl_ode_action_action_disable_new(action_name.c_str(), slaveActionName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Disable Action Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Disable Action Action is created" ) 
        {
            REQUIRE( dsl_ode_action_action_disable_new(action_name.c_str(), slaveActionName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Disable Action Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_action_disable_new(action_name.c_str(), slaveActionName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Enable Action ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Enable Action ODE Action" ) 
    {
        std::wstring action_name(L"action-enable-action");
        std::wstring slaveActionName(L"action");

        WHEN( "A new Enable Action Action is created" ) 
        {
            REQUIRE( dsl_ode_action_action_enable_new(action_name.c_str(), slaveActionName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Enable Action Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Enable Action Action is created" ) 
        {
            REQUIRE( dsl_ode_action_action_enable_new(action_name.c_str(), slaveActionName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Enable Action Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_action_enable_new(action_name.c_str(), slaveActionName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Start Record Sink ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Start Record Sink ODE Action" ) 
    {
        std::wstring action_name(L"start-record-action");
        std::wstring recordSinkName(L"record-sink");

        std::wstring outdir(L"./");
        uint container(DSL_CONTAINER_MP4);
        uint codec(DSL_CODEC_H264);
        uint bitrate(2000000);
        uint interval(0);

        dsl_record_client_listener_cb client_listener;

        REQUIRE( dsl_sink_record_new(recordSinkName.c_str(), outdir.c_str(),
            codec, container, bitrate, interval, client_listener) == DSL_RESULT_SUCCESS );

        WHEN( "A new Start Record Sink Action is created" ) 
        {
            REQUIRE( dsl_ode_action_sink_record_start_new(action_name.c_str(), 
                recordSinkName.c_str(), 1, 1, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "The Start Record Sink Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "A new Start Record Action is created" ) 
        {
            REQUIRE( dsl_ode_action_sink_record_start_new(action_name.c_str(), recordSinkName.c_str(), 1, 1, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Start Record Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_sink_record_start_new(action_name.c_str(), recordSinkName.c_str(), 1, 1, NULL) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A new Stop Record Sink ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Stop Record Sink ODE Action" ) 
    {
        std::wstring action_name(L"stop-record-action");
        std::wstring recordSinkName(L"record-sink");

        std::wstring outdir(L"./");
        uint container(DSL_CONTAINER_MP4);
        uint codec(DSL_CODEC_H264);
        uint bitrate(2000000);
        uint interval(0);

        dsl_record_client_listener_cb client_listener;

        REQUIRE( dsl_sink_record_new(recordSinkName.c_str(), outdir.c_str(),
            codec, container, bitrate, interval, client_listener) == DSL_RESULT_SUCCESS );

        WHEN( "A new Stop Record Sink Action is created" ) 
        {
            REQUIRE( dsl_ode_action_sink_record_stop_new(action_name.c_str(), 
                recordSinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Stop Record Sink Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "A new Stop Record Action is created" ) 
        {
            REQUIRE( dsl_ode_action_sink_record_stop_new(action_name.c_str(), 
                recordSinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Stop Record Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_sink_record_stop_new(action_name.c_str(), 
                    recordSinkName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A new Start Record Tap ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Start Record Tap ODE Action" ) 
    {
        std::wstring action_name(L"start-record-action");
        std::wstring recordTapName(L"record-sink");

        std::wstring outdir(L"./");
        uint container(DSL_CONTAINER_MP4);

        dsl_record_client_listener_cb client_listener;

        REQUIRE( dsl_tap_record_new(recordTapName.c_str(), outdir.c_str(),
            container, client_listener) == DSL_RESULT_SUCCESS );

        WHEN( "A new Start Record Sink Action is created" ) 
        {
            REQUIRE( dsl_ode_action_tap_record_start_new(action_name.c_str(), 
                recordTapName.c_str(), 1, 1, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "The Start Record Tap Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "A new Start Record Tap Action is created" ) 
        {
            REQUIRE( dsl_ode_action_tap_record_start_new(action_name.c_str(), 
                recordTapName.c_str(), 1, 1, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Start Record Tap Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_tap_record_start_new(action_name.c_str(), 
                    recordTapName.c_str(), 1, 1, NULL) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A new Stop Record Tap ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Stop Record Tap ODE Action" ) 
    {
        std::wstring action_name(L"stop-record-action");
        std::wstring recordTapName(L"record-sink");

        std::wstring outdir(L"./");
        uint container(DSL_CONTAINER_MP4);

        dsl_record_client_listener_cb client_listener;

        REQUIRE( dsl_tap_record_new(recordTapName.c_str(), outdir.c_str(),
            container, client_listener) == DSL_RESULT_SUCCESS );

        WHEN( "A new Stop Record Tap Action is created" ) 
        {
            REQUIRE( dsl_ode_action_tap_record_start_new(action_name.c_str(), 
                recordTapName.c_str(), 1, 1, NULL) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_action_list_size() == 1 );
            
            THEN( "The Stop Record Tap Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "A new Stop Record Tap Action is created" ) 
        {
            REQUIRE( dsl_ode_action_tap_record_start_new(action_name.c_str(), 
                recordTapName.c_str(), 1, 1, NULL) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_action_list_size() == 1 );
            
            THEN( "A second Stop Record Tap Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_tap_record_start_new(action_name.c_str(), 
                    recordTapName.c_str(), 1, 1, NULL) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A new Tiler Show Source ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Tiler Show Source ODE Action" ) 
    {
        std::wstring action_name(L"tiler-show-source-action");
        std::wstring tilerName(L"action");
        uint timeout(2);
        boolean has_precedence(true);

        WHEN( "A new Tiler Source Show Action is created" ) 
        {
            REQUIRE( dsl_ode_action_tiler_source_show_new(action_name.c_str(), 
                tilerName.c_str(), timeout, has_precedence) == DSL_RESULT_SUCCESS );
            
            THEN( "The Tiler Show Source Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Tiler Source Show Action is created" ) 
        {
            REQUIRE( dsl_ode_action_tiler_source_show_new(action_name.c_str(), 
                tilerName.c_str(), timeout, has_precedence) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Enable Action Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_tiler_source_show_new(action_name.c_str(), 
                    tilerName.c_str(), timeout, has_precedence) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                
                REQUIRE( dsl_ode_action_delete(action_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

static void enabled_state_change_listener(boolean enabled, void* client_data)
{
    
}

SCENARIO( "An ODE Action can add/remove an enabled-state-change-listener", "[ode-action-api]" )
{
    GIVEN( "A new Capture Action and client listener callback" )
    {
        std::wstring action_name(L"capture-action");
        std::wstring outdir(L"./");
        
        REQUIRE( dsl_ode_action_capture_object_new(action_name.c_str(), 
            outdir.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "When an enabled-state-change-listener is added" )         
        {
            REQUIRE( dsl_ode_action_enabled_state_change_listener_add(action_name.c_str(),
                enabled_state_change_listener, NULL) == DSL_RESULT_SUCCESS );

            // second call must fail
            REQUIRE( dsl_ode_action_enabled_state_change_listener_add(action_name.c_str(),
                enabled_state_change_listener, NULL) == 
                DSL_RESULT_ODE_ACTION_CALLBACK_ADD_FAILED );
            
            THEN( "The same listener function can be removed" ) 
            {
                REQUIRE( dsl_ode_action_enabled_state_change_listener_remove(action_name.c_str(),
                    enabled_state_change_listener) == DSL_RESULT_SUCCESS );

                // second call must fail
                REQUIRE( dsl_ode_action_enabled_state_change_listener_remove(action_name.c_str(),
                    enabled_state_change_listener) == DSL_RESULT_ODE_ACTION_CALLBACK_REMOVE_FAILED );
                    
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "The ODE Action API checks for NULL input parameters", "[ode-action-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring action_name  = L"test-action";
        std::wstring otherName  = L"other";
        
        uint interval(0);
        boolean enabled(0);
        
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_ode_action_custom_new(NULL, NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_custom_new(action_name.c_str(), NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_monitor_new(NULL, NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_monitor_new(action_name.c_str(), NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_capture_frame_new(NULL, NULL, true) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_capture_frame_new(action_name.c_str(), NULL, true) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_capture_object_new(NULL, NULL) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_capture_object_new(action_name.c_str(), NULL) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_customize_label_new(NULL,
                    NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_display_new(NULL, 0, 0, false, NULL, false, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_display_new(action_name.c_str(), 0, 0, false, NULL, false, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_file_new(NULL, NULL, DSL_WRITE_MODE_APPEND, 
                    DSL_EVENT_FILE_FORMAT_TEXT, false) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_file_new(action_name.c_str(), NULL, DSL_WRITE_MODE_APPEND,
                    DSL_EVENT_FILE_FORMAT_TEXT, false) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_fill_frame_new(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_fill_frame_new(action_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_fill_surroundings_new(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_fill_surroundings_new(action_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_handler_disable_new(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_handler_disable_new(action_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                
                REQUIRE( dsl_ode_action_log_new(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_display_meta_add_new(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_display_meta_add_new(action_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_pause_new(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_pause_new(action_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_print_new(NULL, false) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_redact_new(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_sink_add_new(NULL, NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_sink_add_new(action_name.c_str(), NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_sink_add_new(action_name.c_str(), otherName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_sink_remove_new(NULL, NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_sink_remove_new(action_name.c_str(), NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_sink_remove_new(action_name.c_str(), otherName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_sink_record_start_new(NULL, NULL, 0, 0, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_sink_record_start_new(action_name.c_str(), NULL, 0, 0, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_source_add_new(NULL, NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_source_add_new(action_name.c_str(), NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_source_add_new(action_name.c_str(), otherName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_source_remove_new(NULL, NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_source_remove_new(action_name.c_str(), NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_source_remove_new(action_name.c_str(), otherName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_tap_record_start_new(NULL, NULL, 0, 0, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_tap_record_start_new(action_name.c_str(), NULL, 0, 0, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_area_add_new(NULL, NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_area_add_new(action_name.c_str(), NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_area_add_new(action_name.c_str(), otherName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_area_remove_new(NULL, NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_area_remove_new(action_name.c_str(), NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_area_remove_new(action_name.c_str(), otherName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_trigger_disable_new(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_trigger_disable_new(action_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_trigger_enable_new(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_trigger_enable_new(action_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_trigger_reset_new(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_trigger_reset_new(action_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_action_disable_new(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_action_disable_new(action_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_action_enable_new(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_action_enable_new(action_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_tiler_source_show_new(NULL, NULL, 1, true) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_tiler_source_show_new(action_name.c_str(), NULL, 1, true) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_enabled_get(NULL, &enabled) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_enabled_set(NULL, false) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_action_delete(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_action_delete_many(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

