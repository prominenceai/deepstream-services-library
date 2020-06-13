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

#ifndef _DSL_ODE_AREA_H
#define _DSL_ODE_AREA_H

#include "Dsl.h"
#include "DslBase.h"
#include "DslApi.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_ODE_AREA_PTR std::shared_ptr<OdeArea>
    #define DSL_ODE_AREA_NEW(name, left, top, width, height, display) \
        std::shared_ptr<OdeArea>(new OdeArea(name, left, top, width, height, display))

    class OdeArea : public Base
    {
    public: 

        /**
         * @brief ctor for the OdeArea
         * @param[out] left location of the area's left side on the x-axis in pixels, from the left of the frame
         * @param[out] top location of the area's top on the y-axis in pixels, from the top of the frame
         * @param[out] width width of the area in pixels
         * @param[out] height of the area in pixels
         * @param[out] display if true, the area will be displayed by adding meta data
         */
        OdeArea(const char* name, 
            uint left, uint top, uint width, uint height, bool display);

        /**
         * @brief dtor for the OdeArea
         */
        ~OdeArea();

        /**
         * @brief Gets the current detection area 
         * @param[out] left location of the area's left side on the x-axis in pixels, from the left of the frame
         * @param[out] top location of the area's top on the y-axis in pixels, from the top of the frame
         * @param[out] width width of the area in pixels
         * @param[out] height of the area in pixels
         * @param[out] display if true, the area will be displayed by adding meta data
         */
        void GetArea(uint* left, uint* top, uint* width, uint* height, bool* display);

        /**
         * @brief sets the current detection area
         * @param[in] left location of the area's left side on the x-axis in pixels, from the left of the frame
         * @param[in] top location of the area's top on the y-axis in pixels, from the top of the frame
         * @param[in] width width of the area in pixels
         * @param[in] height of the area in pixels
         * @param[in] display if true, the area will be displayed by adding meta data
         */
        void SetArea(uint left, uint top, uint width, uint height, bool display);

        /**
         * @brief Gets the current detection area backtround color
         * @param[out] red red level for the area background color [0..1]
         * @param[out] blue blue level for the area background color [0..1]
         * @param[out] green green level for the area background color [0..1]
         * @param[out] alpha alpha level for the area background color [0..1]
         */
        void GetColor(double* red, double* green, double* blue, double* alpha);
        
        /**
         * @brief Sets the current detection area backtround color
         * @param[in] red red level for the area background color [0..1]
         * @param[in] blue blue level for the area background color [0..1]
         * @param[in] green green level for the area background color [0..1]
         * @param[in] alpha alpha level for the area background color [0..1]
         */
        void SetColor(double red, double green, double blue, double alpha);
        
        
       /**
         * @brief Area rectangle parameters for object detection 
         */
        NvOSD_RectParams m_rectParams;
        
        /**
         * @brief Display the area (add display meta) if true
         */
        bool m_display;
        
    private:

        /**
         * @brief Mutex to ensure mutual exlusion for propery get/sets
         */
        GMutex m_propertyMutex;
    
    };

}

#endif //_DSL_ODE_AREA_H
