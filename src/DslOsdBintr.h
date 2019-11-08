
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
         * @param name name to give the new OsdBintr
         * @param isClockEnabled true if clock is to be displayed
         */
        OsdBintr(const char* osd, gboolean isClockEnabled);

        /**
         * @brief dtor for the OsdBintr class
         */
        ~OsdBintr();

        /**
         * @brief Adds this OsdBintr to a Parent Pipline Bintr
         * @param pParentBintr
         */
        void AddToParent(DSL_NODETR_PTR pParentBintr);
        
        /**
         * @brief Links all child elements of this OsdBintr
         * @return true if all elements were succesfully linked, false otherwise.
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all child elements of the OsdBintr
         */
        void UnlinkAll();
        
    private:

        gboolean m_isClockEnabled;
        
        std::string m_sClockFont;
        guint m_sClockFontSize;
        guint m_sClockOffsetX;
        guint m_sClockOffsetY;
        guint m_sClockColor;
        
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