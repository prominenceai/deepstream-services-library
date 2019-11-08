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
#include "DslElementr.h"
#include "DslBintr.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_DISPLAY_PTR std::shared_ptr<DisplayBintr>
    #define DSL_DISPLAY_NEW(name, width, height) \
        std::shared_ptr<DisplayBintr>(new DisplayBintr(name, width, height))
        
    class DisplayBintr : public Bintr
    {
    public: 
    
        DisplayBintr(const char* name, guint width, guint height);

        ~DisplayBintr();

        void AddToParent(DSL_NODETR_PTR pParentBintr);

        void SetTiles(uint rows, uint columns);
        
        void GetTiles(uint& rows, uint& columns);
        
        void SetDimensions(uint width, uint hieght);
        
        void GetDimensions(uint& width, uint& height);

        bool LinkAll();
        
        void UnlinkAll();
        
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
        DSL_ELEMENT_PTR m_pQueue;
 
        /**
         @brief
         */
        DSL_ELEMENT_PTR  m_pTiler;
        
        /**
         * @brief a single display for the driver
        */
        Display* m_pXDisplay;
    };
}

#endif // _DSL_DISPLAY_BINTR_H
