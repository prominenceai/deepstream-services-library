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

#include "Dsl.h"
#include "DslOdeArea.h"

namespace DSL
{

    OdeArea::OdeArea(const char* name, 
        uint left, uint top, uint width, uint height, bool display)
        : Base(name)
        , m_rectParams{0}
        , m_display(display)
    {
        LOG_FUNC();
        
        m_rectParams.left = left;
        m_rectParams.top = top;
        m_rectParams.width = width;
        m_rectParams.height = height;
        
        // default area background color
        m_rectParams.has_bg_color = true;
        m_rectParams.bg_color.red = 1.0;
        m_rectParams.bg_color.green = 1.0;
        m_rectParams.bg_color.blue = 1.0;
        m_rectParams.bg_color.alpha = 0.2;

        g_mutex_init(&m_propertyMutex);
    }
    
    OdeArea::~OdeArea()
    {
        LOG_FUNC();

        g_mutex_clear(&m_propertyMutex);
    }
        
    void OdeArea::GetArea(uint* left, uint* top, uint* width, uint* height, bool* display)
    {
        LOG_FUNC();
        
        *left = m_rectParams.left;
        *top = m_rectParams.top;
        *width = m_rectParams.width;
        *height = m_rectParams.height;
        *display = m_display;
    }
    
    void OdeArea::SetArea(uint left, uint top, uint width, uint height, bool display)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_rectParams.left = left;
        m_rectParams.top = top;
        m_rectParams.width = width;
        m_rectParams.height = height;
        m_display = display;
    }
    
    void OdeArea::GetColor(double* red, double* green, double* blue, double* alpha)
    {
        LOG_FUNC();
        
        *red = m_rectParams.bg_color.red;
        *green = m_rectParams.bg_color.green;
        *blue = m_rectParams.bg_color.blue;
        *alpha = m_rectParams.bg_color.alpha;
    }

    void OdeArea::SetColor(double red, double green, double blue, double alpha)
    {
        LOG_FUNC();
        
        m_rectParams.bg_color.red = red;
        m_rectParams.bg_color.green = green;
        m_rectParams.bg_color.blue = blue;
        m_rectParams.bg_color.alpha = alpha;
    }
}