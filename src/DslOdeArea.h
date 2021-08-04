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
#include "DslGeosTypes.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_ODE_AREA_PTR std::shared_ptr<OdeArea>

    #define DSL_ODE_AREA_INCLUSION_PTR std::shared_ptr<OdeInclusionArea>
    #define DSL_ODE_AREA_INCLUSION_NEW(name, pDisplayType, show, bboxTestPoint) \
        std::shared_ptr<OdeInclusionArea>(new OdeInclusionArea( \
            name, pDisplayType, show, bboxTestPoint))

    #define DSL_ODE_AREA_EXCLUSION_PTR std::shared_ptr<OdeExclusionArea>
    #define DSL_ODE_AREA_EXCLUSION_NEW(name, pDisplayType, show, bboxTestPoint) \
        std::shared_ptr<OdeExclusionArea>(new OdeExclusionArea( \
            name, pDisplayType, show, bboxTestPoint))

    #define DSL_ODE_AREA_LINE_PTR std::shared_ptr<OdeLineArea>
    #define DSL_ODE_AREA_LINE_NEW(name, pLine, show, bboxTestEdge) \
        std::shared_ptr<OdeLineArea>(new OdeLineArea( \
            name, pLine, show, bboxTestEdge))

    class OdeArea : public Base
    {
    public: 

        /**
         * @brief ctor for the OdeArea
         * @param[in] name unique name for the Area
         * @param[in] pDisplayType a shared pointer to a RGBA Display Type.
         * @param[in] show if true, the area will be displayed by adding meta data
         * @param[in] 
         */
        OdeArea(const char* name, DSL_DISPLAY_TYPE_PTR pDisplayType, bool show);

        /**
         * @brief dtor for the OdeArea
         */
        ~OdeArea();
        
        /**
         * @brief Adds metadata for the RGBA rectangle to pDisplayMeta to overlay the Area for show
         * @param[in] pDisplayMeta show metadata to add the Area to
         * @param[in] pFrameMeta the Frame metadata for the current Frame
         */
        void AddMeta(NvDsDisplayMeta* pDisplayMeta,  NvDsFrameMeta* pFrameMeta);
        
        /**
         * @brief Checks if an bounding box edge (for Line Areas) or point (for Polygon Areas)  
         * is within this Area's underlying display type.
         * @param[in] pBox pointer to an object's bbox rectangle to check for within
         */
        virtual bool CheckForWithin(const NvOSD_RectParams& bbox) = 0;
        
        /**
         * @brief Display type used to define the Area's location, dimensions, and color
         */
        DSL_DISPLAY_TYPE_PTR m_pDisplayType;
        
        /**
         * @brief Display the area (add display meta) if true
         */
        bool m_show;
        
        /**
         * @brief Updated for each source/frame-number. Allows multiple Triggers to share a single Area,
         * And although each Trigger will call OverlayFrame() the Area can check to see if the overlay
         * has occurred for the current source/frame-number. If not, the Area can do an actual Frame-Overlay 
         * once-per-frame-per-source
         */
        std::map<uint, uint64_t> m_frameNumPerSource;

    };
    
    class OdePolygonArea : public OdeArea
    {
    public: 

        /**
         * @brief ctor for the OdePolygon Area
         * @param[in] name unique name for the Area
         * @param[in] pPolygon a shared pointer to a RGBA Polygon Type.
         * @param[in] show if true, the area will be displayed by adding meta data
         * @param[in] bboxTestPoint the bounding box point to check for Area overlap
         * @param[in] areaType either DSL_AREA_TYPE_INCLUSION or DSL_AREA_TYPE_EXCLUSION
         */
        OdePolygonArea(const char* name, DSL_RGBA_POLYGON_PTR pPolygon, 
            bool show, uint bboxTestPoint);

        /**
         * @brief dtor for the OdeInclusionArea
         */
        ~OdePolygonArea();

        /**
         * @brief Checks if the Area overlaps with the provided object bbox based on
         * the bboxTestPoint criteria set for the Area
         * @param[in] pBox pointer to an object's bbox rectangle to check for overlap
         */
        bool CheckForWithin(const NvOSD_RectParams& bbox);
        
        /**
         * @brief Bounding box test point to check for Area overlap. One of DSL_BBOX_POINT
         */
        uint m_bboxTestPoint;

        /**
         * @brief GeosPolygon type created with the Area's display type
         */
        GeosPolygon m_pGeosPolygon;
    };


    class OdeInclusionArea : public OdePolygonArea
    {
    public: 

        /**
         * @brief ctor for the OdeInclusionArea
         * @param[in] name unique name for the Area
         * @param[in] pPolygon a shared pointer to a RGBA Polygon Type.
         * @param[in] show if true, the area will be displayed by adding meta data
         * @param[in] bboxTestPoint one of DSL_BBOX_POINT defining the bounding box 
         * point to check for Area overlap
         */
        OdeInclusionArea(const char* name, DSL_RGBA_POLYGON_PTR pPolygon, 
            bool show, uint bboxTestPoint);

        /**
         * @brief dtor for the OdeInclusionArea
         */
        ~OdeInclusionArea();
        
    };

    class OdeExclusionArea : public OdePolygonArea
    {
    public: 

        /**
         * @brief ctor for the OdeExclusionArea
         * @param[in] pPolygon a shared pointer to a RGBA Polygon Display Type.
         * @param[in] show if true, the area will be displayed by adding meta data
         * @param[in] bboxTestPoint one of DSL_BBOX_POINT defining the bounding box 
         * point to check for Area overlap
         */
        OdeExclusionArea(const char* name, DSL_RGBA_POLYGON_PTR pPolygon, 
            bool show, uint bboxTestPoint);

        /**
         * @brief dtor for the InclusionOdeArea
         */
        ~OdeExclusionArea();

    };
    
    class OdeLineArea : public OdeArea
    {
    public: 

        /**
         * @brief ctor for the OdeExclusionArea
         * @param[in] pLine a shared pointer to a RGBA Line Display Type.
         * @param[in] show if true, the area will be displayed by adding meta data
         * @param[in] bbox_test_edge one of DSL_BBOX_EDGE values to define which edge
         * of the bounding box to test for lines crossing
         */
        OdeLineArea(const char* name, DSL_RGBA_LINE_PTR pLine, 
            bool show, uint bboxTestEdge);

        /**
         * @brief Checks if the Line Area overlaps (crosses) with the provided object 
         * rectangle based on the bboxTestEdge set for the Line Area
         * @param[in] pBox pointer to an object's bbox rectangle to check for overlap
         */
        bool CheckForWithin(const NvOSD_RectParams& bbox);

        /**
         * @brief dtor for the InclusionOdeArea
         */
        ~OdeLineArea();
        
        /**
         * @brief GeosLine type created with the Area's display type
         */
        GeosLine m_pGeosLine;
        
        /**
         * @brief one of DSL_BBOX_EDGE values defining which edge
         * of the bounding box to test for lines crossing
         */
        uint m_bboxTestEdge;
    };

}

#endif //_DSL_ODE_AREA_H
