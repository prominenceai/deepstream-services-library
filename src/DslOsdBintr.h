/*
The MIT License

Copyright (c) 2019-2024, Prominence AI, Inc.

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

#ifndef _DSL_OSD_BINTR_H
#define _DSL_OSD_BINTR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslElementr.h"
#include "DslQBintr.h"

namespace DSL
{
    
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_OSD_PTR std::shared_ptr<OsdBintr>
    #define DSL_OSD_NEW(name, textEnabled, clockEnabled, bboxEnabled, maskEnabled) \
        std::shared_ptr<OsdBintr>(new OsdBintr(name, \
            textEnabled, clockEnabled, bboxEnabled, maskEnabled))

    /**
     * @class OsdBintr
     * @brief Implements an On-Screen-Display bin container
     */
    class OsdBintr : public QBintr
    {
    public: 
    
        /**
         * @brief ctor for the OsdBintr class
         * @param[in] name name to give the new OsdBintr
         * @param[in] textEnabled true if object labels are to be displayed
         * @param[in] clockEnabled true if clock is to be displayed
         * @param[in] bboxEnabled true if bounding boxes are to be displayed
         * @param[in] maskEnabled true if segmentation mask is to be displayed
         */
        OsdBintr(const char* name, boolean textEnabled, boolean clockEnabled,
            boolean bboxEnabled, boolean maskEnabled);

        /**
         * @brief dtor for the OsdBintr class
         */
        ~OsdBintr();

        /**
         * @brief Adds this OsdBintr to a Parent Branch Bintr
         * @param[in] pParentBintr parent Pipeline to add to
         * @return true on successful add, false otherwise
         */
        bool AddToParent(DSL_BASE_PTR pParentBintr);
        
        /**
         * @brief Removes this OsdBintr to a Parent Branch Bintr
         * @param[in] pParentBintr parent Pipeline to remove from
         * @return true on successful add, false otherwise
         */
        bool RemoveFromParent(DSL_BASE_PTR pParentBintr);
        
        /**
         * @brief Links all child elements of this OsdBintr
         * @return true if all elements were succesfully linked, false otherwise.
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all child elements of the OsdBintr
         */
        void UnlinkAll();

        /**
         * @brief Gets the current display text enabled state for this OsdBintr.
         * @param[out] enabled true if text display is currently enabled, false otherwise.
         */
        void GetTextEnabled(boolean* enabled);

        /**
         * @brief Sets the display text enabled state for this OsdBintr.
         * @param[in] enabled set to true to enable text display, false otherwise.
         */
        bool SetTextEnabled(boolean enabled);
        
        /**
         * @brief Gets the current display clock enabled state for this OsdBintr.
         * @param[out] enabled true if clock display is currently enabled, false otherwise.
         */
        void GetClockEnabled(boolean* enabled);

        /**
         * @brief Sets the display clock enabled state for this OsdBintr.
         * @param[in] enabled set to true to enable clock display, false otherwise.
         */
        bool SetClockEnabled(boolean enabled);
        
        /**
         * @brief Gets the current X and Y offsets for the OSD clock
         * @param[out] offsetX current offset in the x direction in pixels
         * @param[out] offsetY current offset in the Y direction in pixels
         */
        void GetClockOffsets(uint* offsetX, uint* offsetY);
        
        /**
         * @brief Sets the current X and Y offsets for the OSD clock
         * @param[in] offsetX new offset in the Y direction in pixels
         * @param[in] offsetY new offset in the Y direction in pixels
         * @return true on successful update, false otherwise
         */
        bool SetClockOffsets(uint offsetX, uint offsetY);
        
        /**
         * @brief Gets the current font name and size for the OSD clock
         * @param[out] name name of the current font in use
         * @param[out] size soze of the current font in use
         */
        void GetClockFont(const char** name, uint *size);

        /**
         * @brief Sets the font name and size to use by the OSD clock
         * @param[in] name name of the new font to use
         * @param[in] size size of the new font to use
         * @return true on successful update, false otherwise
         */
        bool SetClockFont(const char* name, uint size);

        /**
         * @brief Gets the current RGB colors for the OSD clock
         * @param[out] red red value currently in use
         * @param[out] green green value curretly in use
         * @param[out] blue blue value currently in use
         */
        void GetClockColor(double* red, double* green, double* blue, double* alpha);

        /**
         * @brief Sets the current RGB colors for OSD clock
         * @param[in] red new red value to use
         * @param[in] green new green value to
         * @param[in] blue new blue value to use
         * @param[in] alpha new alpha value to use
         * @return true on successful update, false otherwise
         */
        bool SetClockColor(double red, double green, double blue, double alpha);

        /**
         * @brief Gets the current display bbox enabled state for this OsdBintr.
         * @param[out] enabled true if bbox display is currently enabled, false otherwise.
         */
        void GetBboxEnabled(boolean* enabled);

        /**
         * @brief Sets the display bbox enabled state for this OsdBintr.
         * @param[in] enabled set to true to enable bbox display, false otherwise.
         */
        bool SetBboxEnabled(boolean enabled);
        /**
         * @brief Gets the current display segmentation mask enabled state for this OsdBintr.
         * @param[out] enabled true if mask display is currently enabled, false otherwise.
         */
        void GetMaskEnabled(boolean* enabled);

        /**
         * @brief Sets the display segmentation mask enabled state for this OsdBintr.
         * @param[in] enabled set to true to enable mask display, false otherwise.
         */
        bool SetMaskEnabled(boolean enabled);
        
        /**
         * @brief Gets the current process mode for the OsdBintr.
         * @param[out] mode 0 = CPU mode, 1 = GPU mode.
         */
        void GetProcessMode(uint* mode);

        /**
         * @brief Gets the current process mode for the OsdBintr.
         * @param[in] mode set to 0 for CPU mode, 1 for GPU mode.
         */
        bool SetProcessMode(uint mode);

        /**
         * @brief Sets the GPU ID for all Elementrs
         * @return true if successfully set, false otherwise.
         */
        bool SetGpuId(uint gpuId);

        /**
         * @brief Sets the NVIDIA buffer memory type.
         * @brief nvbufMemType new memory type to use, one of the 
         * DSL_NVBUF_MEM_TYPE constant values.
         * @return true if successfully set, false otherwise.
         */
        bool SetNvbufMemType(uint nvbufMemType);

    private:
    
        /**
         * @brief specifies whether object labels are displayed or not.
         */
        boolean m_textEnabled;

        /**
         * @brief specifies whether the on-screen clock is displayed or not.
         */
        boolean m_clockEnabled;
        
        /**
         * @brief font name for the on-screen clock if enabled.
         */
        std::string m_clockFont;

        /**
         * @brief font size for the on-screen clock if enabled.
         */
        uint m_clockFontSize;

        /**
         * @brief X-offset for the on-screen clock if enabled.
         */
        uint m_clockOffsetX;

        /**
         * @brief X-offset for the on-screen clock if enabled.
         */
        uint m_clockOffsetY;
        
        /**
         * @brief RGBA clock color for the on-screen clock if enabled.
         */
        NvOSD_ColorParams m_clockColor;
        
        /**
         * @brief RGBA clock background color of the clock if enabled
         */
        uint m_clkRgbaColor;

        /**
         * @brief Indicates the mode used to draw the objects
         * 0: CPU mode
         * 1: GPU mode
         */
        guint m_processMode;

        /**
         * @brief specifies whether bounding-boxs are displayed or not.
         */
        boolean m_bboxEnabled;

        /**
         * @brief specifies whether the segmentation mask will be displayed or note.
         */
        boolean m_maskEnabled;

        /**
         * @brief Vidconverter plugin for the OsdBintr.
         */
        DSL_ELEMENT_PTR m_pVidConv;

        /**
         * @brief NVIDIA OSD plugin for the OsdBintr.
         */
        DSL_ELEMENT_PTR m_pOsd;
    
    };
    
}

#endif // _DSL_OSD_BINTR_H