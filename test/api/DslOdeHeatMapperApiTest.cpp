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

static const std::wstring color_palette_name(L"spectral-color-palette");
static const std::wstring ode_heat_mapper_name(L"ode-heat-mapper");


SCENARIO( "The ODE Heat-Mappers container is updated correctly on multiple new ODE Heat-Mappers", 
    "[ode-heat-mapper-api]" )
{
    GIVEN( "An empty list of Heat-Mappers" ) 
    {
        std::wstring odeHeatMapperName1(L"heat-mapper-1");
        std::wstring odeHeatMapperName2(L"heat-mapper-2");
        std::wstring odeHeatMapperName3(L"heat-mapper-3");
        
        REQUIRE( dsl_display_type_rgba_color_palette_predefined_new(
            color_palette_name.c_str(), DSL_COLOR_PREDEFINED_PALETTE_SPECTRAL, 
            0.5) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_heat_mapper_list_size() == 0 );

        WHEN( "Several new Heat-Mappers are created" ) 
        {
                
            REQUIRE( dsl_ode_heat_mapper_new(odeHeatMapperName1.c_str(),
                64, 36, DSL_BBOX_POINT_SOUTH, color_palette_name.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_heat_mapper_new(odeHeatMapperName2.c_str(),
                64, 36, DSL_BBOX_POINT_SOUTH, color_palette_name.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_heat_mapper_new(odeHeatMapperName3.c_str(),
                64, 36, DSL_BBOX_POINT_SOUTH, color_palette_name.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size and events are updated correctly" ) 
            {
                REQUIRE( dsl_ode_heat_mapper_list_size() == 3 );

                REQUIRE( dsl_ode_heat_mapper_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_heat_mapper_list_size() == 0 );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "The Heat-Mappers container is updated correctly on ODE Heat-Mapper deletion", 
    "[ode-heat-mapper-api]" )
{
    GIVEN( "A list of Heat-Mappers" ) 
    {
        std::wstring odeHeatMapperName1(L"heat-mapper-1");
        std::wstring odeHeatMapperName2(L"heat-mapper-2");
        std::wstring odeHeatMapperName3(L"heat-mapper-3");

        REQUIRE( dsl_display_type_rgba_color_palette_predefined_new(
            color_palette_name.c_str(), DSL_COLOR_PREDEFINED_PALETTE_SPECTRAL, 
            0.5) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_heat_mapper_new(odeHeatMapperName1.c_str(),
            64, 36, DSL_BBOX_POINT_SOUTH, color_palette_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_heat_mapper_new(odeHeatMapperName2.c_str(),
            64, 36, DSL_BBOX_POINT_SOUTH, color_palette_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_heat_mapper_new(odeHeatMapperName3.c_str(),
            64, 36, DSL_BBOX_POINT_SOUTH, color_palette_name.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "When Heat-Mappers are deleted" )         
        {
            REQUIRE( dsl_ode_heat_mapper_list_size() == 3 );
            REQUIRE( dsl_ode_heat_mapper_delete(odeHeatMapperName1.c_str()) == 
                DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_heat_mapper_list_size() == 2 );

            const wchar_t* eventList[] = 
                {odeHeatMapperName2.c_str(), odeHeatMapperName3.c_str(), NULL};
            REQUIRE( dsl_ode_heat_mapper_delete_many(eventList) == 
                DSL_RESULT_SUCCESS );
            
            THEN( "The list size and events are updated correctly" ) 
            {
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_heat_mapper_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "A new Heat-Mapper can handle calls to its Metrics services", 
    "[ode-heat-mapper-api]" )
{
    GIVEN( "A list of Heat-Mappers" ) 
    {
        REQUIRE( dsl_display_type_rgba_color_palette_predefined_new(
            color_palette_name.c_str(), DSL_COLOR_PREDEFINED_PALETTE_SPECTRAL, 
            0.5) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_heat_mapper_new(ode_heat_mapper_name.c_str(),
            16, 9, DSL_BBOX_POINT_SOUTH, color_palette_name.c_str()) == 
                DSL_RESULT_SUCCESS );

        // Note: this is just a simple test to ensure that the services can be called
        // successfully - manual verification of output is required.
        WHEN( "When the Heat-Mappers metrics services are called" )
        {
            REQUIRE( dsl_ode_heat_mapper_metrics_print(
                ode_heat_mapper_name.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_heat_mapper_metrics_log(
                ode_heat_mapper_name.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_heat_mapper_metrics_clear(
                ode_heat_mapper_name.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Heat-Mapper can be deleted successfully" ) 
            {
                REQUIRE( dsl_ode_heat_mapper_delete(ode_heat_mapper_name.c_str()) == 
                    DSL_RESULT_SUCCESS );
                    
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_heat_mapper_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "The ODE Heat-Mapper API checks for NULL input parameters", "[ode-heat-mapper-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_ode_heat_mapper_new(NULL, 
                    0, 0 , 0, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_heat_mapper_new(ode_heat_mapper_name.c_str(), 
                    0, 0 , 0, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_heat_mapper_color_palette_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_heat_mapper_color_palette_get(ode_heat_mapper_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_heat_mapper_color_palette_set(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_heat_mapper_color_palette_set(ode_heat_mapper_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_heat_mapper_legend_settings_get(NULL, 
                    NULL, NULL, NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_ode_heat_mapper_legend_settings_get(NULL, 
                    0, 0, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_ode_heat_mapper_metrics_get(ode_heat_mapper_name.c_str(), 
                    NULL, NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_heat_mapper_metrics_get(NULL,
                    NULL, NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_heat_mapper_metrics_clear(NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_heat_mapper_metrics_print(NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_ode_heat_mapper_metrics_log(NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_heat_mapper_metrics_file(NULL,
                    NULL, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_ode_heat_mapper_metrics_file(ode_heat_mapper_name.c_str(),
                    NULL, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
            }
        }
    }
}