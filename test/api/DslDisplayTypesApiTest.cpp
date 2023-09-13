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

SCENARIO( "A new RGBA Custom Color can be created and deleted",
    "[display-types-api]" )
{
    GIVEN( "Attributes for a new RGBA Custom Color" ) 
    {
        std::wstring colorName(L"my-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        REQUIRE( dsl_display_type_list_size() == 0 );
        WHEN( "A new RGBA Custom Color is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
                red, green, blue, alpha) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_display_type_list_size() == 1 );
            
            THEN( "The RGBA Custom Color can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(colorName.c_str()) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new RGBA Custom Color is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
                red, green, blue, alpha) == DSL_RESULT_SUCCESS );
            
            THEN( "A second RGBA Custom Color of the same name fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
                    red, green, blue, alpha) == 
                        DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE );

                REQUIRE( dsl_display_type_delete(colorName.c_str()) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new RGBA Custom Color checks input parameter ranges correctly",
    "[display-types-api]" )
{
    GIVEN( "Attributes for a new RGBA Custom Color" ) 
    {
        std::wstring colorName(L"my-color");

        WHEN( "An invalid red parameter is issued" ) 
        {
            double red(1.1), green(0.34), blue(0.56), alpha(0.78);
            
            THEN( "The RGBA Custom Color fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
                    red, green, blue, alpha) == DSL_RESULT_DISPLAY_PARAMETER_INVALID );
                    
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "An invalid green parameter is issued" ) 
        {
            double red(0.12), green(1.1), blue(0.56), alpha(0.78);
            
            THEN( "The RGBA Custom Color fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
                    red, green, blue, alpha) == DSL_RESULT_DISPLAY_PARAMETER_INVALID );
                    
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "An invalid blue parameter is issued" ) 
        {
            double red(0.12), green(0.34), blue(1.10), alpha(0.78);
            
            THEN( "The RGBA Custom Color fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
                    red, green, blue, alpha) == DSL_RESULT_DISPLAY_PARAMETER_INVALID );
                    
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "An invalid alpah parameter is issued" ) 
        {
            double red(0.12), green(0.34), blue(0.56), alpha(1.01);
            
            THEN( "The RGBA Custom Color fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
                    red, green, blue, alpha) == DSL_RESULT_DISPLAY_PARAMETER_INVALID );
                    
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new RGBA Predefined Color can be created and deleted", 
    "[display-types-api]" )
{
    GIVEN( "Attributes for a new RGBA Predefined Color" ) 
    {
        std::wstring colorName(L"my-color");
        uint color_id(DSL_COLOR_HUE_MAGENTA_PINK);
        double alpha(0.78);

        REQUIRE( dsl_display_type_list_size() == 0 );
        WHEN( "A new RGBA Predefined Color is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_color_predefined_new(colorName.c_str(), 
                color_id, alpha) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_display_type_list_size() == 1 );
            
            THEN( "The RGBA Predefined Color can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(colorName.c_str()) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new RGBA Predefined Color is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_color_predefined_new(colorName.c_str(), 
                color_id, alpha) == DSL_RESULT_SUCCESS );
            
            THEN( "A second RGBA Predefined Color of the same name fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_color_predefined_new(colorName.c_str(), 
                    color_id, alpha) == 
                        DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE );

                REQUIRE( dsl_display_type_delete(colorName.c_str()) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new RGBA Predefined Color checks input parmeter ranges corectly", 
    "[display-types-api]" )
{
    GIVEN( "Attributes for a new RGBA Predefined Color" ) 
    {
        std::wstring colorName(L"my-color");

        REQUIRE( dsl_display_type_list_size() == 0 );

        WHEN( "An invalid color_id parameter is issued" ) 
        {
            uint color_id(88);
            double alpha(0.78);
            
            THEN( "The RGBA Predefined Color fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_color_predefined_new(colorName.c_str(), 
                    color_id, alpha) == DSL_RESULT_DISPLAY_PARAMETER_INVALID );

                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "An invalid alpah parameter is issued" ) 
        {
            uint color_id(0);
            double alpha(1.10);
            
            THEN( "The RGBA Predefined Color fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_color_predefined_new(colorName.c_str(), 
                    color_id, alpha) == DSL_RESULT_DISPLAY_PARAMETER_INVALID );

                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new RGBA Random Color can be created and deleted", 
    "[display-types-api]" )
{
    GIVEN( "Attributes for a new RGBA Random Color" ) 
    {
        std::wstring colorName(L"my-color");
        uint color_id(DSL_COLOR_HUE_MAGENTA_PINK);
        uint hue(DSL_COLOR_HUE_MAGENTA_PINK);
        uint luminosity(DSL_COLOR_LUMINOSITY_DARK);
        double alpha(0.78);
        uint seed(444);

        REQUIRE( dsl_display_type_list_size() == 0 );
        WHEN( "A new RGBA Random Color is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_color_random_new(colorName.c_str(), 
                hue, luminosity, alpha, seed) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_display_type_list_size() == 1 );
            
            THEN( "The RGBA Random Color can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(colorName.c_str()) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new RGBA Random Color is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_color_random_new(colorName.c_str(), 
                hue, luminosity, alpha, seed) == DSL_RESULT_SUCCESS );
            
            THEN( "A second RGBA Random Color of the same name fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_color_random_new(colorName.c_str(), 
                    hue, luminosity, alpha, seed) == 
                        DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE );

                REQUIRE( dsl_display_type_delete(colorName.c_str()) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new RGBA Random Color checks input parameter ranges correctly", 
    "[display-types-api]" )
{
    GIVEN( "Attributes for a new RGBA Random Color" ) 
    {
        std::wstring colorName(L"my-color");
        uint seed(444);

        REQUIRE( dsl_display_type_list_size() == 0 );

        WHEN( "When an invalid hue parameter is providied" ) 
        {
            uint hue(99);
            uint luminosity(DSL_COLOR_LUMINOSITY_DARK);
            double alpha(0.78);
            
            THEN( "RGBA Random Color fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_color_random_new(colorName.c_str(), 
                    hue, luminosity, alpha, seed) == 
                        DSL_RESULT_DISPLAY_PARAMETER_INVALID );

                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "When an invalid luminosity parameter is providied" ) 
        {
            uint hue(DSL_COLOR_HUE_MAGENTA_PINK);
            uint luminosity(6);
            double alpha(0.78);
            
            THEN( "RGBA Random Color fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_color_random_new(colorName.c_str(), 
                    hue, luminosity, alpha, seed) == 
                        DSL_RESULT_DISPLAY_PARAMETER_INVALID );

                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "When an invalid alpha parameter is providied" ) 
        {
            uint hue(DSL_COLOR_HUE_MAGENTA_PINK);
            uint luminosity(DSL_COLOR_LUMINOSITY_DARK);
            double alpha(1.01);
            
            THEN( "RGBA Random Color fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_color_random_new(colorName.c_str(), 
                    hue, luminosity, alpha, seed) == 
                        DSL_RESULT_DISPLAY_PARAMETER_INVALID );

                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

static void color_provider(double* red, 
    double* green, double* blue, double* alpha, void* client_data)
{
    *red = 0.123;
    *green = 0.456;
    *blue = 0.789;
    *alpha = 0.444;
}

SCENARIO( "A new RGBA On-Demand Color can be created and deleted", 
    "[display-types-api]" )
{
    GIVEN( "Attributes for a new RGBA On-Demand Color" ) 
    {
        std::wstring colorName(L"my-color");

        REQUIRE( dsl_display_type_list_size() == 0 );
        WHEN( "A new RGBA Random Color is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_color_on_demand_new(colorName.c_str(), 
                color_provider, NULL) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_display_type_list_size() == 1 );
            
            THEN( "The RGBA Random Color can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(colorName.c_str()) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new RGBA Random Color is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_color_on_demand_new(colorName.c_str(), 
                color_provider, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "A second RGBA Random Color of the same name fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_color_on_demand_new(colorName.c_str(), 
                    color_provider, NULL) == 
                        DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE );

                REQUIRE( dsl_display_type_delete(colorName.c_str()) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new RGBA Color Palette can be created and deleted", 
    "[display-types-api]" )
{
    GIVEN( "Attributes for a new RGBA Color Palette" ) 
    {
        std::wstring colorName1(L"color1");
        std::wstring colorName2(L"color2");
        std::wstring colorName3(L"color3");
        std::wstring colorPaletteName(L"color-palette");
        uint color_id1(DSL_COLOR_HUE_MAGENTA_PINK);
        uint color_id2(DSL_COLOR_HUE_BLACK_AND_WHITE);
        uint color_id3(DSL_COLOR_HUE_BLUE);
        double alpha(0.78);

        REQUIRE( dsl_display_type_list_size() == 0 );
        
        REQUIRE( dsl_display_type_rgba_color_predefined_new(colorName1.c_str(), 
            color_id1, alpha) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_display_type_rgba_color_predefined_new(colorName2.c_str(), 
            color_id2, alpha) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_display_type_rgba_color_predefined_new(colorName3.c_str(), 
            color_id3, alpha) == DSL_RESULT_SUCCESS );
            
        const wchar_t* colors[] = 
            {colorName1.c_str(), colorName2.c_str(), colorName3.c_str(), NULL};

        WHEN( "A new RGBA Color Palette is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_color_palette_new(
                colorPaletteName.c_str(), colors) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_display_type_list_size() == 4 );
            
            THEN( "The RGBA Color Palette can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete_all() == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new RGBA Color Palette is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_color_palette_new(
                colorPaletteName.c_str(), colors) == DSL_RESULT_SUCCESS );
            
            THEN( "A second RGBA Color Palette of the same name fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_color_palette_new(
                    colorPaletteName.c_str(), colors) == 
                        DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE );

                REQUIRE( dsl_display_type_delete_all() == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new RGBA Predefined Color Palette can be created and deleted", 
    "[display-types-api]" )
{
    GIVEN( "Attributes for a new RGBA Predefined Color Palete" ) 
    {
        std::wstring colorName(L"my-color");
        uint palette_id(DSL_COLOR_PREDEFINED_PALETTE_SPECTRAL);
        double alpha(0.78);

        REQUIRE( dsl_display_type_list_size() == 0 );
        WHEN( "A new RGBA Predefined Color Palette is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_color_palette_predefined_new(
                colorName.c_str(), palette_id, alpha) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_display_type_list_size() == 1 );
            
            THEN( "The RGBA Predefined Color Palette can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(colorName.c_str()) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new RGBA Predefined Color Palette is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_color_palette_predefined_new(
                colorName.c_str(), palette_id, alpha) == DSL_RESULT_SUCCESS );
            
            THEN( "A second RGBA Predefined Color of the same name fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_color_palette_predefined_new(
                    colorName.c_str(), palette_id, alpha) == 
                        DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE );

                REQUIRE( dsl_display_type_delete(colorName.c_str()) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}
    
SCENARIO( "A new RGBA Random Color Palette can be created and deleted", 
    "[display-types-api]" )
{
    GIVEN( "Attributes for a new RGBA Random Color Palette" ) 
    {
        std::wstring colorName(L"my-color");
        uint size(10);
        uint color_id(DSL_COLOR_HUE_MAGENTA_PINK);
        uint hue(DSL_COLOR_HUE_MAGENTA_PINK);
        uint luminosity(DSL_COLOR_LUMINOSITY_DARK);
        double alpha(0.78);
        uint seed(444);

        REQUIRE( dsl_display_type_list_size() == 0 );
        WHEN( "A new RGBA Random Color Palette is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_color_palette_random_new(colorName.c_str(), 
                size, hue, luminosity, alpha, seed) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_display_type_list_size() == 1 );
            
            THEN( "The RGBA Random Color Palette can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(colorName.c_str()) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new RGBA Random Color Palette is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_color_palette_random_new(colorName.c_str(), 
                size, hue, luminosity, alpha, seed) == DSL_RESULT_SUCCESS );
            
            THEN( "A second RGBA Random Color Palette of the same name fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_color_palette_random_new(colorName.c_str(), 
                    size, hue, luminosity, alpha, seed) == 
                        DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE );

                REQUIRE( dsl_display_type_delete(colorName.c_str()) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new RGBA Random Color Palette checks input parameter ranges correctly", 
    "[display-types-api]" )
{
    GIVEN( "Attributes for a new RGBA Random Color Palette" ) 
    {
        std::wstring colorName(L"my-color");
        uint seed(444);

        REQUIRE( dsl_display_type_list_size() == 0 );

        WHEN( "When an invalid size parameter is providied" ) 
        {
            uint size(1);
            uint hue(DSL_COLOR_HUE_MAGENTA_PINK);
            uint luminosity(DSL_COLOR_LUMINOSITY_DARK);
            double alpha(0.78);
            
            THEN( "RGBA Random Color Palette fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_color_palette_random_new(colorName.c_str(), 
                    size, hue, luminosity, alpha, seed) == 
                        DSL_RESULT_DISPLAY_PARAMETER_INVALID );

                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "When an invalid hue parameter is providied" ) 
        {
            uint size(10);
            uint hue(99);
            uint luminosity(DSL_COLOR_LUMINOSITY_DARK);
            double alpha(0.78);
            
            THEN( "RGBA Random Color Palette fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_color_palette_random_new(colorName.c_str(), 
                    size, hue, luminosity, alpha, seed) == 
                        DSL_RESULT_DISPLAY_PARAMETER_INVALID );

                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "When an invalid luminosity parameter is providied" ) 
        {
            uint size(10);
            uint hue(DSL_COLOR_HUE_MAGENTA_PINK);
            uint luminosity(6);
            double alpha(0.78);
            
            THEN( "RGBA Random Color Palette fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_color_palette_random_new(colorName.c_str(), 
                    size, hue, luminosity, alpha, seed) == 
                        DSL_RESULT_DISPLAY_PARAMETER_INVALID );

                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "When an invalid alpha parameter is providied" ) 
        {
            uint size(10);
            uint hue(DSL_COLOR_HUE_MAGENTA_PINK);
            uint luminosity(DSL_COLOR_LUMINOSITY_DARK);
            double alpha(1.01);
            
            THEN( "RGBA Random Color Palette fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_color_palette_random_new(colorName.c_str(), 
                    size, hue, luminosity, alpha, seed) == 
                        DSL_RESULT_DISPLAY_PARAMETER_INVALID );

                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}
    
SCENARIO( "A new RGBA Font can be created and deleted", "[display-types-api]" )
{
    GIVEN( "Attributes for a new RGBA Font" ) 
    {
        std::wstring fontName(L"arial-20");
        std::wstring font(L"arial");
        uint size(20);

        std::wstring colorName(L"my-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        WHEN( "A new RGBA Font is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_font_new(fontName.c_str(), font.c_str(),
                size, colorName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The RGBA Font can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(fontName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new RGBA Font is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_font_new(fontName.c_str(), font.c_str(),
                size, colorName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second RGBA Font of the same name fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_font_new(fontName.c_str(), font.c_str(),
                    size, colorName.c_str()) == DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE );

                REQUIRE( dsl_display_type_delete(fontName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new RGBA Text can be created and deleted", "[display-types-api]" )
{
    GIVEN( "Attributes for a new RGBA Text" ) 
    {
        std::wstring textName(L"display-text");
        std::wstring text(L"this is text to display");
        uint xOffset(100), yOffset(100);

        std::wstring fontName(L"arial-20");
        std::wstring font(L"arial");
        uint size(20);

        std::wstring colorName(L"my-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_font_new(fontName.c_str(), font.c_str(),
            size, colorName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A new RGBA Text is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_text_new(textName.c_str(), text.c_str(), 
                xOffset, yOffset, fontName.c_str(), true, colorName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The RGBA Text can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(textName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new RGBA Text is created without a background color" ) 
        {
            REQUIRE( dsl_display_type_rgba_text_new(textName.c_str(), text.c_str(), 
                xOffset, yOffset, fontName.c_str(), false, NULL)== DSL_RESULT_SUCCESS );

            THEN( "The RGBA Text can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(textName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new RGBA Text is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_text_new(textName.c_str(), text.c_str(), 
                xOffset, yOffset, fontName.c_str(), true, colorName.c_str())== DSL_RESULT_SUCCESS );
            
            THEN( "A second RGBA Text of the same name fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_text_new(textName.c_str(), text.c_str(), 
                    xOffset, yOffset, fontName.c_str(), true, colorName.c_str()) == 
                        DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE );

                REQUIRE( dsl_display_type_delete(textName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new RGBA Line can be created and deleted", "[display-types-api]" )
{
    GIVEN( "Attributes for a new RGBA Line" ) 
    {
        std::wstring lineName(L"line");
        uint x1(100), y1(100), x2(200), y2(200);
        uint width(20);

        std::wstring colorName(L"my-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        WHEN( "A new RGBA Line is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_line_new(lineName.c_str(), 
                x1, y1, x2, y2, width, colorName.c_str())== DSL_RESULT_SUCCESS );

            THEN( "The RGBA Line can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(lineName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new RGBA Line is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_line_new(lineName.c_str(), 
                x1, y1, x2, y2, width, colorName.c_str())== DSL_RESULT_SUCCESS );
            
            THEN( "A second RGBA Line of the same name fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_line_new(lineName.c_str(), 
                    x1, y1, x2, y2, width, colorName.c_str()) == DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE );

                REQUIRE( dsl_display_type_delete(lineName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new RGBA Arrow can be created and deleted", "[display-types-api]" )
{
    GIVEN( "Attributes for a new RGBA Arrow" ) 
    {
        std::wstring arrowName(L"arrow");
        uint x1(100), y1(100), x2(200), y2(200);
        uint width(20);
        uint head(DSL_ARROW_BOTH_HEAD);

        std::wstring colorName(L"my-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        WHEN( "A new RGBA Arrow is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_arrow_new(arrowName.c_str(), 
                x1, y1, x2, y2, width, head, colorName.c_str())== DSL_RESULT_SUCCESS );

            THEN( "The RGBA Arrow can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(arrowName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new RGBA Arrow is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_arrow_new(arrowName.c_str(), 
                x1, y1, x2, y2, width, head, colorName.c_str())== DSL_RESULT_SUCCESS );
            
            THEN( "A second RGBA Arrow of the same name fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_arrow_new(arrowName.c_str(), 
                    x1, y1, x2, y2, width, head, colorName.c_str()) == 
                    DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE );

                REQUIRE( dsl_display_type_delete(arrowName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "An invalid Arrow Head type is specified " ) 
        {
            head = DSL_ARROW_BOTH_HEAD + 1;
            
            THEN( "The RGBA Arrow fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_arrow_new(arrowName.c_str(), 
                    x1, y1, x2, y2, width, head, colorName.c_str()) ==
                        DSL_RESULT_DISPLAY_PARAMETER_INVALID );

                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new RGBA Rectangle can be created and deleted", "[display-types-api]" )
{
    GIVEN( "Attributes for a new RGBA Rectangle" ) 
    {
        std::wstring rectangleName(L"rectangle");
        uint left(100), top(100), width(123), height(456);
        uint border_width(3);

        std::wstring colorName(L"my-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        WHEN( "A new RGBA Rectangle is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_rectangle_new(rectangleName.c_str(), left, top, width, height, 
                border_width, colorName.c_str(), true, colorName.c_str())== DSL_RESULT_SUCCESS );

            THEN( "The RGBA Rectangle can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(rectangleName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new RGBA Rectangle is created without a background color" ) 
        {
            REQUIRE( dsl_display_type_rgba_rectangle_new(rectangleName.c_str(), left, top, width, height, 
                border_width, colorName.c_str(), false, NULL)== DSL_RESULT_SUCCESS );

            THEN( "The RGBA Rectangle can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(rectangleName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new RGBA Rectangle is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_rectangle_new(rectangleName.c_str(), left, top, width, height, 
                border_width, colorName.c_str(), true, colorName.c_str())== DSL_RESULT_SUCCESS );
            
            THEN( "A second RGBA Rectangle of the same name fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_rectangle_new(rectangleName.c_str(), left, top, width, height, 
                    border_width, colorName.c_str(), true, colorName.c_str()) == 
                        DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE );

                REQUIRE( dsl_display_type_delete(rectangleName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new RGBA Polygon can be created and deleted", "[display-types-api]" )
{
    GIVEN( "Attributes for a new RGBA Polygon" ) 
    {
        std::wstring polygonName(L"polygon");
        uint border_width(3);

        std::wstring colorName(L"my-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        dsl_coordinate coordinates[4] = {{100,100},{210,110},{220, 300},{110,330}};
        uint num_coordinates(4);

        REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        WHEN( "A new RGBA Polygon is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_polygon_new(polygonName.c_str(), coordinates, num_coordinates, 
                border_width, colorName.c_str())== DSL_RESULT_SUCCESS );

            THEN( "The RGBA Rectangle can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(polygonName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new RGBA Polygon is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_polygon_new(polygonName.c_str(), coordinates, num_coordinates, 
                border_width, colorName.c_str())== DSL_RESULT_SUCCESS );
            
            THEN( "A second RGBA Polygon of the same name fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_polygon_new(polygonName.c_str(), coordinates, num_coordinates, 
                    border_width, colorName.c_str()) == 
                        DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE );

                REQUIRE( dsl_display_type_delete(polygonName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "An invalid 'num_coordinates' value is used to create an RGBA Polygon" ) 
        {
            num_coordinates = DSL_MAX_POLYGON_COORDINATES+1; 
            
            THEN( "A RGBA Polygon fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_polygon_new(polygonName.c_str(), coordinates, num_coordinates, 
                    border_width, colorName.c_str())== DSL_RESULT_DISPLAY_PARAMETER_INVALID );

                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new RGBA Mulit-Line can be created and deleted", "[display-types-api]" )
{
    GIVEN( "Attributes for a new RGBA Mulit-Line" ) 
    {
        std::wstring multiLineName(L"multi-line");
        uint border_width(3);

        std::wstring colorName(L"my-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        dsl_coordinate coordinates[4] = {{100,100},{210,110},{220, 300},{110,330}};
        uint num_coordinates(4);

        REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        WHEN( "A new RGBA Polygon is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_line_multi_new(multiLineName.c_str(), coordinates, num_coordinates, 
                border_width, colorName.c_str())== DSL_RESULT_SUCCESS );

            THEN( "The RGBA Rectangle can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(multiLineName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new RGBA Polygon is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_line_multi_new(multiLineName.c_str(), coordinates, num_coordinates, 
                border_width, colorName.c_str())== DSL_RESULT_SUCCESS );
            
            THEN( "A second RGBA Polygon of the same name fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_line_multi_new(multiLineName.c_str(), coordinates, num_coordinates, 
                    border_width, colorName.c_str()) == 
                        DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE );

                REQUIRE( dsl_display_type_delete(multiLineName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "An invalid 'num_coordinates' value is used to create an RGBA Polygon" ) 
        {
            num_coordinates = DSL_MAX_MULTI_LINE_COORDINATES+1; 
            
            THEN( "A RGBA Polygon fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_line_multi_new(multiLineName.c_str(), coordinates, num_coordinates, 
                    border_width, colorName.c_str())== DSL_RESULT_DISPLAY_PARAMETER_INVALID );

                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new RGBA Circle can be created and deleted", "[display-types-api]" )
{
    GIVEN( "Attributes for a new RGBA Circle" ) 
    {
        std::wstring circleName(L"display-text");
        uint x_center(100), y_center(100);
        uint radius(50);

        std::wstring colorName(L"my-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        WHEN( "A new RGBA Circle is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_circle_new(circleName.c_str(), x_center, y_center,
                radius, colorName.c_str(), true, colorName.c_str())== DSL_RESULT_SUCCESS );

            THEN( "The RGBA Circle can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(circleName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new RGBA Circle is created without a background color" ) 
        {
            REQUIRE( dsl_display_type_rgba_circle_new(circleName.c_str(), x_center, y_center,
                radius, colorName.c_str(), false, NULL)== DSL_RESULT_SUCCESS );

            THEN( "The RGBA Circle can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(circleName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new RGBA Circle is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_circle_new(circleName.c_str(), x_center, y_center,
                radius, colorName.c_str(), true, colorName.c_str())== DSL_RESULT_SUCCESS );
            
            THEN( "A second RGBA Circle of the same name fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_circle_new(circleName.c_str(), x_center, y_center,
                    radius, colorName.c_str(), true, colorName.c_str()) == 
                        DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE );

                REQUIRE( dsl_display_type_delete(circleName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Source Unique Id Display can be created and deleted", "[display-types-api]" )
{
    GIVEN( "Attributes for a new Source Unique Id Display" ) 
    {
        std::wstring displayName(L"source-number");
        uint xOffset(100), yOffset(100);

        std::wstring fontName(L"arial-20");
        std::wstring font(L"arial");
        uint size(20);

        std::wstring colorName(L"my-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_font_new(fontName.c_str(), font.c_str(),
            size, colorName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A new Source Unique Id Display is created" ) 
        {
            REQUIRE( dsl_display_type_source_unique_id_new(displayName.c_str(),
                xOffset, yOffset, fontName.c_str(), true, colorName.c_str())== DSL_RESULT_SUCCESS );

            THEN( "The Source Unique Id Display can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(displayName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new Source Unique Id Display is created without a background color" ) 
        {
            REQUIRE( dsl_display_type_source_unique_id_new(displayName.c_str(),
                xOffset, yOffset, fontName.c_str(), false, NULL)== DSL_RESULT_SUCCESS );

            THEN( "The Source Unique Id Display can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(displayName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new Source Unique Id Display is created" ) 
        {
            REQUIRE( dsl_display_type_source_unique_id_new(displayName.c_str(),
                xOffset, yOffset, fontName.c_str(), true, colorName.c_str())== DSL_RESULT_SUCCESS );
            
            THEN( "A second Source Unique Id Display of the same name fails to create" ) 
            {
                REQUIRE( dsl_display_type_source_unique_id_new(displayName.c_str(),
                    xOffset, yOffset, fontName.c_str(), true, colorName.c_str()) == 
                        DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE );

                REQUIRE( dsl_display_type_delete(displayName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Source Stream Id Display can be created and deleted", "[display-types-api]" )
{
    GIVEN( "Attributes for a new Source Stream Id Display" ) 
    {
        std::wstring displayName(L"source-number");
        uint xOffset(100), yOffset(100);

        std::wstring fontName(L"arial-20");
        std::wstring font(L"arial");
        uint size(20);

        std::wstring colorName(L"my-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_font_new(fontName.c_str(), font.c_str(),
            size, colorName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A new Source Stream Id Display is created" ) 
        {
            REQUIRE( dsl_display_type_source_stream_id_new(displayName.c_str(),
                xOffset, yOffset, fontName.c_str(), true, colorName.c_str())== DSL_RESULT_SUCCESS );

            THEN( "The Source Stream Id Display can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(displayName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new Source Stream Id Display is created without a background color" ) 
        {
            REQUIRE( dsl_display_type_source_stream_id_new(displayName.c_str(),
                xOffset, yOffset, fontName.c_str(), false, NULL)== DSL_RESULT_SUCCESS );

            THEN( "The Source Stream Id Display can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(displayName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new Source Stream Id Display is created" ) 
        {
            REQUIRE( dsl_display_type_source_stream_id_new(displayName.c_str(),
                xOffset, yOffset, fontName.c_str(), true, colorName.c_str())== DSL_RESULT_SUCCESS );
            
            THEN( "A second Source Stream Id Display of the same name fails to create" ) 
            {
                REQUIRE( dsl_display_type_source_stream_id_new(displayName.c_str(),
                    xOffset, yOffset, fontName.c_str(), true, colorName.c_str()) == 
                        DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE );

                REQUIRE( dsl_display_type_delete(displayName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Source Name Display can be created and deleted", "[display-types-api]" )
{
    GIVEN( "Attributes for a new Source Name Display" ) 
    {
        std::wstring displayName(L"source-name");
        uint xOffset(100), yOffset(100);

        std::wstring fontName(L"arial-20");
        std::wstring font(L"arial");
        uint size(20);

        std::wstring colorName(L"my-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_font_new(fontName.c_str(), font.c_str(),
            size, colorName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A new Source Name Display is created" ) 
        {
            REQUIRE( dsl_display_type_source_name_new(displayName.c_str(),
                xOffset, yOffset, fontName.c_str(), true, colorName.c_str())== DSL_RESULT_SUCCESS );

            THEN( "The Source Name Display can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(displayName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new Source Name Display is created without a background color" ) 
        {
            REQUIRE( dsl_display_type_source_name_new(displayName.c_str(),
                xOffset, yOffset, fontName.c_str(), false, NULL)== DSL_RESULT_SUCCESS );

            THEN( "The Source Name Display can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(displayName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new Source Name Display is created" ) 
        {
            REQUIRE( dsl_display_type_source_name_new(displayName.c_str(),
                xOffset, yOffset, fontName.c_str(), true, colorName.c_str())== DSL_RESULT_SUCCESS );
            
            THEN( "A second Source Name Display of the same name fails to create" ) 
            {
                REQUIRE( dsl_display_type_source_name_new(displayName.c_str(),
                    xOffset, yOffset, fontName.c_str(), true, colorName.c_str()) == 
                        DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE );

                REQUIRE( dsl_display_type_delete(displayName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Source Dimensions Display can be created and deleted", "[display-types-api]" )
{
    GIVEN( "Attributes for a new Source Dimensions Display" ) 
    {
        std::wstring displayName(L"source-dimensions");
        uint xOffset(100), yOffset(100);

        std::wstring fontName(L"arial-20");
        std::wstring font(L"arial");
        uint size(20);

        std::wstring colorName(L"my-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_font_new(fontName.c_str(), font.c_str(),
            size, colorName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A new Source Dimenions Display is created" ) 
        {
            REQUIRE( dsl_display_type_source_dimensions_new(displayName.c_str(),
                xOffset, yOffset, fontName.c_str(), true, colorName.c_str())== DSL_RESULT_SUCCESS );

            THEN( "The Source Dimenions Display can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(displayName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new Source Dimenions Display is created without a background color" ) 
        {
            REQUIRE( dsl_display_type_source_dimensions_new(displayName.c_str(),
                xOffset, yOffset, fontName.c_str(), false, NULL)== DSL_RESULT_SUCCESS );

            THEN( "The Source Dimenions Display can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(displayName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new Source Dimenions Display is created" ) 
        {
            REQUIRE( dsl_display_type_source_dimensions_new(displayName.c_str(),
                xOffset, yOffset, fontName.c_str(), true, colorName.c_str())== DSL_RESULT_SUCCESS );
            
            THEN( "A second Source Dimenions Display of the same name fails to create" ) 
            {
                REQUIRE( dsl_display_type_source_dimensions_new(displayName.c_str(),
                    xOffset, yOffset, fontName.c_str(), true, colorName.c_str()) == 
                        DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE );

                REQUIRE( dsl_display_type_delete(displayName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

//SCENARIO( "A new Source Frame Rate Display can be created and deleted", "[display-types-api]" )
//{
//    GIVEN( "Attributes for a new Frame Rate Display" ) 
//    {
//        std::wstring displayName(L"source-frame-rate");
//        uint xOffset(100), yOffset(100);
//
//        std::wstring fontName(L"arial-20");
//        std::wstring font(L"arial");
//        uint size(20);
//
//        std::wstring colorName(L"my-color");
//        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
//
//        REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
//            red, green, blue, alpha) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_display_type_rgba_font_new(fontName.c_str(), font.c_str(),
//            size, colorName.c_str()) == DSL_RESULT_SUCCESS );
//
//        WHEN( "A new Source Frame Rate Display is created" ) 
//        {
//            REQUIRE( dsl_display_type_source_frame_rate_new(displayName.c_str(),
//                xOffset, yOffset, fontName.c_str(), true, colorName.c_str())== DSL_RESULT_SUCCESS );
//
//            THEN( "The Source Frame Rate Display can be deleted" ) 
//            {
//                REQUIRE( dsl_display_type_delete(displayName.c_str()) == DSL_RESULT_SUCCESS );
//                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
//                REQUIRE( dsl_display_type_list_size() == 0 );
//            }
//        }
//        WHEN( "A new Source Frame Rate Display is created" ) 
//        {
//            REQUIRE( dsl_display_type_source_frame_rate_new(displayName.c_str(),
//                xOffset, yOffset, fontName.c_str(), true, colorName.c_str())== DSL_RESULT_SUCCESS );
//            
//            THEN( "A second Source Frame Rate Display of the same name fails to create" ) 
//            {
//                REQUIRE( dsl_display_type_source_frame_rate_new(displayName.c_str(),
//                    xOffset, yOffset, fontName.c_str(), true, colorName.c_str())== DSL_RESULT_DISPLAY_SOURCE_FRAMERATE_NAME_NOT_UNIQUE );
//
//                REQUIRE( dsl_display_type_delete(displayName.c_str()) == DSL_RESULT_SUCCESS );
//                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
//                REQUIRE( dsl_display_type_list_size() == 0 );
//            }
//        }
//    }
//}
