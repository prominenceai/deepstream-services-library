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

#ifndef _DSL_ODE_AREA_H
#define _DSL_ODE_AREA_H

#include "Dsl.h"
#include "DslBase.h"
#include "DslApi.h"
#include "DslDisplayTypes.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_ODE_AREA_PTR std::shared_ptr<OdeArea>

    #define DSL_ODE_AREA_INCLUSION_PTR std::shared_ptr<OdeInclusionArea>
    #define DSL_ODE_AREA_INCLUSION_NEW(name, pRectangle, display) \
        std::shared_ptr<OdeInclusionArea>(new OdeInclusionArea(name, pRectangle, display))

    #define DSL_ODE_AREA_EXCLUSION_PTR std::shared_ptr<OdeExclusionArea>
    #define DSL_ODE_AREA_EXCLUSION_NEW(name, pRectangle, display) \
        std::shared_ptr<OdeExclusionArea>(new OdeExclusionArea(name, pRectangle, display))

    class OdeArea : public Base
    {
    public: 

        /**
         * @brief ctor for the OdeArea
         * @param[in] pRectangle a shared pointer to a RGBA Rectangle Display Type.
         * @param[in] display if true, the area will be displayed by adding meta data
         */
        OdeArea(const char* name, DSL_RGBA_RECTANGLE_PTR pRectangle, bool display);

        /**
         * @brief dtor for the OdeArea
         */
        ~OdeArea();
        
        
        /**
         * @brief Adds metadata for the RGBA rectangle to pDisplayMeta to overlay the Area for display
         * @param[in] pDisplayMeta display metadata to add the Area to
         * @param[in] pFrameMeta the Frame metadata for the current Frame
         */
        void AddMeta(NvDsDisplayMeta* pDisplayMeta,  NvDsFrameMeta* pFrameMeta);
        
       /**
         * @brief Area rectangle parameters for object detection and display
         */
        DSL_RGBA_RECTANGLE_PTR m_pRectangle;
        
        /**
         * @brief Display the area (add display meta) if true
         */
        bool m_display;
        
        /**
         * @brief Updated for each source/frame-number. Allows multiple Triggers to share a single Area,
         * And although each Trigger will call OverlayFrame() the Area can check to see if the overlay
         * has occurred for the current source/frame-number. If not, the Area can do an actual Frame-Overlay 
         * once-per-frame-per-source
         */
        std::map<uint, uint64_t> m_frameNumPerSource;
    };

    class OdeInclusionArea : public OdeArea
    {
    public: 

        /**
         * @brief ctor for the OdeInclusionArea
         * @param[in] pRectangle a shared pointer to a RGBA Rectangle Display Type.
         * @param[in] display if true, the area will be displayed by adding meta data
         */
        OdeInclusionArea(const char* name, DSL_RGBA_RECTANGLE_PTR pRectangle, bool display);

        /**
         * @brief dtor for the InclusionOdeArea
         */
        ~OdeInclusionArea();
        
    };

    class OdeExclusionArea : public OdeArea
    {
    public: 

        /**
         * @brief ctor for the OdeExclusionArea
         * @param[in] pRectangle a shared pointer to a RGBA Rectangle Display Type.
         * @param[in] display if true, the area will be displayed by adding meta data
         */
        OdeExclusionArea(const char* name, DSL_RGBA_RECTANGLE_PTR pRectangle, bool display);

        /**
         * @brief dtor for the InclusionOdeArea
         */
        ~OdeExclusionArea();
        
    };
}

#endif //_DSL_ODE_AREA_H
