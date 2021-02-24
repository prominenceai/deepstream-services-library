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

SCENARIO( "A new RGBA Color can be created and deleted", "[display-types-api]" )
{
    GIVEN( "Attributes for a new RGBA Color" ) 
    {
        std::wstring colorName(L"my-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        REQUIRE( dsl_display_type_list_size() == 0 );
        WHEN( "A new RGBA Color is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_color_new(colorName.c_str(), 
                red, green, blue, alpha) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_display_type_list_size() == 1 );
            
            THEN( "The Action can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(colorName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new RGBA Color is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_color_new(colorName.c_str(), 
                red, green, blue, alpha) == DSL_RESULT_SUCCESS );
            
            THEN( "A second RGBA Color of the same name fails to create" ) 
            {
                REQUIRE( dsl_display_type_rgba_color_new(colorName.c_str(), 
                    red, green, blue, alpha) == DSL_RESULT_DISPLAY_RGBA_COLOR_NAME_NOT_UNIQUE );

                REQUIRE( dsl_display_type_delete(colorName.c_str()) == DSL_RESULT_SUCCESS );
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

        REQUIRE( dsl_display_type_rgba_color_new(colorName.c_str(), 
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
                    size, colorName.c_str()) == DSL_RESULT_DISPLAY_RGBA_FONT_NAME_NOT_UNIQUE );

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

        REQUIRE( dsl_display_type_rgba_color_new(colorName.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_font_new(fontName.c_str(), font.c_str(),
            size, colorName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A new RGBA Text is created" ) 
        {
            REQUIRE( dsl_display_type_rgba_text_new(textName.c_str(), text.c_str(), 
                xOffset, yOffset, fontName.c_str(), true, colorName.c_str())== DSL_RESULT_SUCCESS );

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
                    xOffset, yOffset, fontName.c_str(), true, colorName.c_str())== DSL_RESULT_DISPLAY_RGBA_TEXT_NAME_NOT_UNIQUE );

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

        REQUIRE( dsl_display_type_rgba_color_new(colorName.c_str(), 
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
                    x1, y1, x2, y2, width, colorName.c_str())== DSL_RESULT_DISPLAY_RGBA_LINE_NAME_NOT_UNIQUE );

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

        REQUIRE( dsl_display_type_rgba_color_new(colorName.c_str(), 
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
                    x1, y1, x2, y2, width, head, colorName.c_str())== DSL_RESULT_DISPLAY_RGBA_ARROW_NAME_NOT_UNIQUE );

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
                    x1, y1, x2, y2, width, head, colorName.c_str())== DSL_RESULT_DISPLAY_RGBA_ARROW_HEAD_INVALID );

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

        REQUIRE( dsl_display_type_rgba_color_new(colorName.c_str(), 
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
                    border_width, colorName.c_str(), true, colorName.c_str())== DSL_RESULT_DISPLAY_RGBA_RECTANGLE_NAME_NOT_UNIQUE );

                REQUIRE( dsl_display_type_delete(rectangleName.c_str()) == DSL_RESULT_SUCCESS );
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

        REQUIRE( dsl_display_type_rgba_color_new(colorName.c_str(), 
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
                    radius, colorName.c_str(), true, colorName.c_str())== DSL_RESULT_DISPLAY_RGBA_CIRCLE_NAME_NOT_UNIQUE );

                REQUIRE( dsl_display_type_delete(circleName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Source Number Display can be created and deleted", "[display-types-api]" )
{
    GIVEN( "Attributes for a new Source Number Display" ) 
    {
        std::wstring displayName(L"source-number");
        uint xOffset(100), yOffset(100);

        std::wstring fontName(L"arial-20");
        std::wstring font(L"arial");
        uint size(20);

        std::wstring colorName(L"my-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        REQUIRE( dsl_display_type_rgba_color_new(colorName.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_font_new(fontName.c_str(), font.c_str(),
            size, colorName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A new Source Number Display is created" ) 
        {
            REQUIRE( dsl_display_type_source_number_new(displayName.c_str(),
                xOffset, yOffset, fontName.c_str(), true, colorName.c_str())== DSL_RESULT_SUCCESS );

            THEN( "The Source Number Display can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(displayName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new Source Number Display is created without a background color" ) 
        {
            REQUIRE( dsl_display_type_source_number_new(displayName.c_str(),
                xOffset, yOffset, fontName.c_str(), false, NULL)== DSL_RESULT_SUCCESS );

            THEN( "The Source Number Display can be deleted" ) 
            {
                REQUIRE( dsl_display_type_delete(displayName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
        WHEN( "A new Source Number Display is created" ) 
        {
            REQUIRE( dsl_display_type_source_number_new(displayName.c_str(),
                xOffset, yOffset, fontName.c_str(), true, colorName.c_str())== DSL_RESULT_SUCCESS );
            
            THEN( "A second Source Number Display of the same name fails to create" ) 
            {
                REQUIRE( dsl_display_type_source_number_new(displayName.c_str(),
                    xOffset, yOffset, fontName.c_str(), true, colorName.c_str())== DSL_RESULT_DISPLAY_SOURCE_NUMBER_NAME_NOT_UNIQUE );

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

        REQUIRE( dsl_display_type_rgba_color_new(colorName.c_str(), 
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
                    xOffset, yOffset, fontName.c_str(), true, colorName.c_str())== DSL_RESULT_DISPLAY_SOURCE_NAME_NAME_NOT_UNIQUE );

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

        REQUIRE( dsl_display_type_rgba_color_new(colorName.c_str(), 
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
                    xOffset, yOffset, fontName.c_str(), true, colorName.c_str())== DSL_RESULT_DISPLAY_SOURCE_DIMENSIONS_NAME_NOT_UNIQUE );

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
//        REQUIRE( dsl_display_type_rgba_color_new(colorName.c_str(), 
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
