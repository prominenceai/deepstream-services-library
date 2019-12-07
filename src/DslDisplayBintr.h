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

        /**
         * @brief Adds the DisplayBintr to a Parent Pipeline Bintr
         * @param[in] pParentBintr Parent Pipeline to add this Bintr to
         */
        bool AddToParent(DSL_NODETR_PTR pParentBintr);

        /**
         * @brief Links all Child Elementrs owned by this Bintr
         * @return true if all links were succesful, false otherwise
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elemntrs owned by this Bintr
         * Calling UnlinkAll when in an unlinked state has no effect.
         */
        void UnlinkAll();
        
        /**
         * @brief Gets the current number of rows and columns for the DisplayBintr
         * @param[out] rows the current number of rows
         * @param[out] columns the current number of columns
         */
        void GetTiles(uint* rows, uint* columns);
        
        /**
         * @brief Sets the number of rows and columns for the DisplayBintr
         * The caller is required to provide valid row and column values
         * @param[in] rows the number of rows to set
         * @param[in] columns the number of columns to set
         * @return false if the Display is currently in Use. True otherwise
         */
        bool SetTiles(uint rows, uint columns);

        /**
         * @brief Gets the current width and height settings for this DisplayBintr
         * @param[out] width the current width setting in pixels
         * @param[out] height the current height setting in pixels
         */ 
        void GetDimensions(uint* width, uint* height);
        
        /**
         * @brief Sets the current width and height settings for this DisplayBintr
         * The caller is required to provide valid width and height values
         * @param[in] width the width value to set in pixels
         * @param[in] height the height value to set in pixels
         * @return false if the Display is currently in Use. True otherwise
         */ 
        bool SetDimensions(uint width, uint hieght);
        
    private:
    
        /**
         * @brief number of rows for the Tiled DisplayBintr
         */
        uint m_rows; 
        
        /**
         * @brief number of rows for the Tiled DisplayBintr
         */
        uint m_columns;
        
        /**
         * @brief width of the Tiled Display in pixels
         */
        uint m_width; 
        
        /**
         * @brief height of the Tiled Display in pixels
         */
        uint m_height;
        
        /**
         * @brief Queue Elementr as Sink for this Tiled DisplayBintr
         */
        DSL_ELEMENT_PTR m_pQueue;
 
        /**
         * @brief Tiler Elementr as Source for this Tiled DisplayBintr
         */
        DSL_ELEMENT_PTR  m_pTiler;
        
        /**
         * @brief a single display for the driver
         */
        Display* m_pXDisplay;
    };
}

#endif // _DSL_DISPLAY_BINTR_H
