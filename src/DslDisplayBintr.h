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

#ifndef _DSL_DISPLAY_BINTR_H
#define _DSL_DISPLAY_BINTR_H

#include "Dsl.h"
#include "DslBintr.h"

namespace DSL
{
    class DisplayBintr : public Bintr
    {
    public: 
    
        DisplayBintr(const std::string& display, Display* m_pXDisplay,
            guint rows, guint columns, guint width, guint height);

        ~DisplayBintr();

        void AddToParent(std::shared_ptr<Bintr> pParentBintr);
        
    private:
    
        /**
         @brief
         */
        guint m_rows; 
        
        /**
         @brief
         */
        guint m_columns;
        
        /**
         @brief
         */
        guint m_width; 
        
        /**
         @brief
         */
        guint m_height;
        
        /**
         @brief
         */
        gboolean m_enablePadding;

        /**
         @brief
         */
        GstElement* m_pQueue;
 
        /**
         @brief
         */
        GstElement* m_pTiler;
        
        Window m_window;

        /**
         * @brief a single display for the driver
        */
        Display* m_pXDisplay;
        
    };
    
}

#endif // _DSL_DISPLAY_BINTR_H
