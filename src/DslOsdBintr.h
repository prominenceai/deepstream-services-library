
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

#ifndef _DSL_OSD_BINTR_H
#define _DSL_OSD_BINTR_H

#include "Dsl.h"
#include "DslElementr.h"
#include "DslBintr.h"

namespace DSL
{
    
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_OSD_PTR std::shared_ptr<OsdBintr>
    #define DSL_OSD_NEW(name, isClockEnabled) \
        std::shared_ptr<OsdBintr>(new OsdBintr(name, isClockEnabled))

    /**
     * @class OsdBintr
     * @brief Implements an On-Screen-Display bin container
     */
    class OsdBintr : public Bintr
    {
    public: 
    
        /**
         * @brief ctor for the OsdBintr class
         * @param[in] name name to give the new OsdBintr
         * @param[in] isClockEnabled true if clock is to be displayed
         */
        OsdBintr(const char* osd, gboolean isClockEnabled);

        /**
         * @brief dtor for the OsdBintr class
         */
        ~OsdBintr();

        /**
         * @brief Adds this OsdBintr to a Parent Pipline Bintr
         * @param[in] pParentBintr
         */
        bool AddToParent(DSL_NODETR_PTR pParentBintr);
        
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
         * @brief Gets the current state of the On-screen clock
         * @param[out] enabled true if the OSD clock is current enabled, false otherwise
         */
        void GetClockEnabled(boolean* enabled);

        /**
         * @brief Sets the current state of the On-screen clock
         * @param[in] enabled true if the OSD clock is to be enabled, false otherwise
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
        void GetClockFont(const char** name, uint* size);

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
        void GetClockColor(uint* red, uint* green, uint* blue);

        /**
         * @brief Sets the current RGB colors for OSD clock
         * @param[in] red new red value to use
         * @param[in] green new green value to
         * @param[in] blue new blue value to use
         * @return true on successful update, false otherwise
         */
        bool SetClockColor(uint red, uint green, uint blue);
        
    private:

        boolean m_isClockEnabled;
        
        std::string m_clockFont;
        guint m_clockFontSize;
        guint m_clockOffsetX;
        guint m_clockOffsetY;
        guint m_clockColorRed;
        guint m_clockColorGreen;
        guint m_clockColorBlue;
        
        /**
         @brief
         */
        guint m_processMode;
        
        DSL_ELEMENT_PTR m_pQueue;
        DSL_ELEMENT_PTR m_pVidConv;
        DSL_ELEMENT_PTR m_pCapsFilter;
        DSL_ELEMENT_PTR m_pConvQueue;
        DSL_ELEMENT_PTR m_pOsd;
    
    };
}

#endif // _DSL_OSD_BINTR_H