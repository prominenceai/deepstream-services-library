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

#include "catch.hpp"
#include "DslDisplayTypes.h"

using namespace DSL;

SCENARIO( "An RGBA Color is constructed correctly", "[DisplayTypes]" )
{
    GIVEN( "Attributes for a new RGBA Color" )
    {
        std::string colorName  = "my-custom-color";
        
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        WHEN( "The RGBA Color is created" )
        {
            DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), red, green, blue, alpha);
            
            THEN( "Its member variables are initialized correctly" )
            {
                REQUIRE( pColor->GetName() == colorName );
                REQUIRE( pColor->red == red );
                REQUIRE( pColor->blue == blue );
                REQUIRE( pColor->green == green );
                REQUIRE( pColor->alpha == alpha );
            }
        }
    }
}

SCENARIO( "A RGBA Font is constructed correctly", "[DisplayTypes]" )
{
    GIVEN( "Attrubutes for a new RGBA Font" )
    {

        std::string fontName  = "arial-10";
        std::string font  = "arial";
        uint size(10);
        std::string colorName  = "my-custom-color";
        
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), red, green, blue, alpha);
        
        WHEN( "The RGBA Font is created" )
        {
            DSL_RGBA_FONT_PTR pFont = DSL_RGBA_FONT_NEW(fontName.c_str(), font.c_str(), size, pColor);
            
            THEN( "Its member variables are initialized correctly" )
            {
                REQUIRE( pFont->GetName() == fontName );
                REQUIRE( pFont->font_name == NULL );
                REQUIRE( pFont->m_fontName == font );
                REQUIRE( pFont->font_size == size );
                REQUIRE( pFont->font_color.red == red );
                REQUIRE( pFont->font_color.blue == blue );
                REQUIRE( pFont->font_color.green == green );
                REQUIRE( pFont->font_color.alpha == alpha );
            }
        }
    }
}

SCENARIO( "A RGBA Text is constructed correctly", "[DisplayTypes]" )
{
    GIVEN( "Attrubutes for a new RGBA Text" )
    {

        std::string textName("my-custom-text");
        int xOffset(123), yOffset(456);
        std::string text("this is my custom display text");
        std::string fontName("arial-10");
        std::string font("arial");
        uint size(10);
        std::string colorName("my-custom-color");
        
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), red, green, blue, alpha);
        DSL_RGBA_FONT_PTR pFont = DSL_RGBA_FONT_NEW(fontName.c_str(), font.c_str(), size, pColor);
        
        WHEN( "The RGBA Text is created" )
        {
            DSL_RGBA_TEXT_PTR pText = DSL_RGBA_TEXT_NEW(textName.c_str(), text.c_str(), 
                xOffset, yOffset, pFont, true, pColor);
            
            THEN( "Its member variables are initialized correctly" )
            {
                REQUIRE( pText->GetName() == textName );
                REQUIRE( pText->display_text == NULL );
                REQUIRE( pText->m_text == text );
                REQUIRE( pText->font_params.font_size == size );
                REQUIRE( pText->font_params.font_color.red == red );
                REQUIRE( pText->font_params.font_color.blue == blue );
                REQUIRE( pText->font_params.font_color.green == green );
                REQUIRE( pText->font_params.font_color.alpha == alpha );
                REQUIRE( pText->set_bg_clr == (int)true );
                REQUIRE( pText->text_bg_clr.red == red );
                REQUIRE( pText->text_bg_clr.blue == blue );
                REQUIRE( pText->text_bg_clr.green == green );
                REQUIRE( pText->text_bg_clr.alpha == alpha );
            }
        }
    }
}

SCENARIO( "A RGBA Line is constructed correctly", "[DisplayTypes]" )
{
    GIVEN( "Attrubutes for a new RGBA Line" )
    {

        std::string lineName  = "my-line";
        uint x1(12), y1(34), x2(56), y2(78);
        uint width(4);

        std::string colorName  = "my-custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        

        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), red, green, blue, alpha);
        
        WHEN( "The RGBA Line is created" )
        {
            DSL_RGBA_LINE_PTR pLine = DSL_RGBA_LINE_NEW(lineName.c_str(), x1, y1, x2, y2, width, pColor);
            
            THEN( "Its member variables are initialized correctly" )
            {
                REQUIRE( pLine->GetName() == lineName );
                REQUIRE( pLine->x1 == x1 );
                REQUIRE( pLine->y1 == y1 );
                REQUIRE( pLine->x2 == x2 );
                REQUIRE( pLine->y2 == y2 );
                REQUIRE( pLine->line_width == width );
                REQUIRE( pLine->line_color.red == red );
                REQUIRE( pLine->line_color.blue == blue );
                REQUIRE( pLine->line_color.green == green );
                REQUIRE( pLine->line_color.alpha == alpha );
            }
        }
    }
}

SCENARIO( "A RGBA Arrow is constructed correctly", "[DisplayTypes]" )
{
    GIVEN( "Attrubutes for a new RGBA Arrow" )
    {

        std::string arrowName  = "my-double-arrow";
        uint x1(12), y1(34), x2(56), y2(78);
        uint width(4);
        uint head(DSL_ARROW_BOTH_HEAD);

        std::string colorName  = "my-custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        

        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), red, green, blue, alpha);
        
        WHEN( "The RGBA Line is created" )
        {
            DSL_RGBA_ARROW_PTR pArrow = DSL_RGBA_ARROW_NEW(arrowName.c_str(), x1, y1, x2, y2, width, head, pColor);
            
            THEN( "Its member variables are initialized correctly" )
            {
                REQUIRE( pArrow->GetName() == arrowName );
                REQUIRE( pArrow->x1 == x1 );
                REQUIRE( pArrow->y1 == y1 );
                REQUIRE( pArrow->x2 == x2 );
                REQUIRE( pArrow->y2 == y2 );
                REQUIRE( pArrow->arrow_width == width );
                REQUIRE( pArrow->arrow_head == head );
                REQUIRE( pArrow->arrow_color.red == red );
                REQUIRE( pArrow->arrow_color.blue == blue );
                REQUIRE( pArrow->arrow_color.green == green );
                REQUIRE( pArrow->arrow_color.alpha == alpha );
            }
        }
    }
}

SCENARIO( "A RGBA Rectangle is constructed correctly", "[DisplayTypes]" )
{
    GIVEN( "Attrubutes for a new RGBA Rectangle" )
    {

        std::string rectangleName  = "my-rectangle";
        uint left(12), top(34), width(56), height(78);
        uint borderWidth(4);

        std::string colorName  = "my-custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        

        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), red, green, blue, alpha);
        
        WHEN( "The RGBA Rectangle is created" )
        {
            DSL_RGBA_RECTANGLE_PTR pRectangle = DSL_RGBA_RECTANGLE_NEW(rectangleName.c_str(), 
                left, top, width, height, borderWidth, pColor, true, pColor);
            
            THEN( "Its member variables are initialized correctly" )
            {
                REQUIRE( pRectangle->GetName() == rectangleName );
                REQUIRE( pRectangle->left == (float)left );
                REQUIRE( pRectangle->top == (float)top );
                REQUIRE( pRectangle->width == (float)width );
                REQUIRE( pRectangle->height == (float)height );
                REQUIRE( pRectangle->border_width == borderWidth );
                REQUIRE( pRectangle->border_color.red == red );
                REQUIRE( pRectangle->border_color.blue == blue );
                REQUIRE( pRectangle->border_color.green == green );
                REQUIRE( pRectangle->border_color.alpha == alpha );
                REQUIRE( pRectangle->has_bg_color == true );
                REQUIRE( pRectangle->bg_color.red == red );
                REQUIRE( pRectangle->bg_color.blue == blue );
                REQUIRE( pRectangle->bg_color.green == green );
                REQUIRE( pRectangle->bg_color.alpha == alpha );
            }
        }
    }
}

SCENARIO( "A RGBA Circle is constructed correctly", "[DisplayTypes]" )
{
    GIVEN( "Attrubutes for a new Circle Line" )
    {

        std::string circleName  = "my-circle";
        uint xc(123), yc(456);
        uint radius(100);
        uint width(4);

        std::string colorName  = "my-custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        

        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), red, green, blue, alpha);
        
        WHEN( "The RGBA Circle is created" )
        {
            DSL_RGBA_CIRCLE_PTR pCircle = DSL_RGBA_CIRCLE_NEW(circleName.c_str(), xc, yc, radius, pColor, true, pColor);
            
            THEN( "Its member variables are initialized correctly" )
            {
                REQUIRE( pCircle->GetName() == circleName);
                REQUIRE( pCircle->xc == xc );
                REQUIRE( pCircle->yc == yc );
                REQUIRE( pCircle->radius == radius );
                REQUIRE( pCircle->circle_color.red == red );
                REQUIRE( pCircle->circle_color.blue == blue );
                REQUIRE( pCircle->circle_color.green == green );
                REQUIRE( pCircle->circle_color.alpha == alpha );
                REQUIRE( pCircle->has_bg_color == true );
                REQUIRE( pCircle->bg_color.red == red );
                REQUIRE( pCircle->bg_color.blue == blue );
                REQUIRE( pCircle->bg_color.green == green );
                REQUIRE( pCircle->bg_color.alpha == alpha );
            }
        }
    }
}

