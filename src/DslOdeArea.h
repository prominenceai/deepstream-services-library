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

#ifndef _DSL_ODE_AREA_H
#define _DSL_ODE_AREA_H

#include "Dsl.h"
#include "DslBase.h"
#include "DslApi.h"
#include "DslDisplayTypes.h"
#include "DslGeosTypes.h"
#include "DslOdeTrackedObject.h"

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
    #define DSL_ODE_AREA_LINE_NEW(name, pLine, show, bboxTestPoint) \
        std::shared_ptr<OdeLineArea>(new OdeLineArea( \
            name, pLine, show, bboxTestPoint))

    #define DSL_ODE_AREA_MULTI_LINE_PTR std::shared_ptr<OdeMultiLineArea>
    #define DSL_ODE_AREA_MULTI_LINE_NEW(name, pMultiLine, show, bboxTestPoint) \
        std::shared_ptr<OdeMultiLineArea>(new OdeMultiLineArea( \
            name, pMultiLine, show, bboxTestPoint))

    class OdeArea : public Base
    {
    public: 

        /**
         * @brief ctor for the OdeArea
         * @param[in] name unique name for the Area
         * @param[in] pDisplayType a shared pointer to a RGBA Display Type.
         * @param[in] show if true, the area will be displayed by adding meta data
         * @param[in] bboxTestPoint one of DSL_BBOX_POINT defining the bounding box 
         * point to check for Area inclusion or cross
         */
        OdeArea(const char* name, DSL_DISPLAY_TYPE_PTR pDisplayType, 
            bool show, uint bboxTestPoint);

        /**
         * @brief dtor for the OdeArea
         */
        ~OdeArea();
        
        /**
         * @brief Adds metadata for the RGBA rectangle to pDisplayMeta to overlay 
         * the Area for show
         * @param[in] displayMetaData vector of pointers to allocated Display Meta 
         * structures to add the Area's underliying Display Type to.
         * @param[in] pFrameMeta the Frame metadata for the current Frame
         */
        void AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData,  
            NvDsFrameMeta* pFrameMeta);
        
        /**
         * @brief Checks if a bounding box - using bboxTestPoint - is inside 
         * the Area's RGBA Display Type.
         * @param[in] bbox object's bbox rectangle to test for inside
         * @return true if inside, false otherwise
         */
        virtual bool IsBboxInside(const NvOSD_RectParams& bbox) = 0;

        /**
         * @brief Checks if an x,y coordinate is inside the Area's Display Type.
         * @param[in] pBox pointer to an object's bbox rectangle to check for inside.
         * @return true if inside, false otherwise.
         */
        virtual bool IsPointInside(const dsl_coordinate& coordinate) = 0;

        /**
         * @brief returns the direction of an x, y point relative to the Area's
         * Display Type
         * @param[in] coordinate x,y coordinate for the point to test.
         * @return one DSL_AREA_POINT_LOCATION_IN or DSL_AREA_POINT_LOCATION_OUT
         */ 
        virtual uint GetPointLocation(const dsl_coordinate& coordinate) = 0;
        
        /**
         * @brief tests if a point is on the area's line(s) including line width,
         * @param[in] coordinate x,y coordinate for the point to test.
         * @return true if the point is located on a line, false otherwise.
         */
        virtual bool IsPointOnLine(const dsl_coordinate& coordinate) = 0;
        
        /**
         * @brief Checks if a bounding box trace crosses the Area's underlying 
         * Display Type.
         * @param[in] coordinates array of dsl_coordinates.
         * @param[in] numCoordinates size of the array.
         * @param[out] direction one of the DSL_AREA_CROSS_DIRECTION_* constants 
         * defining the direction of the cross, including DSL_AREA_CROSS_DIRECTION_NONE.
         * @return true if trace fully crosses the Area's Display Type including 
         * line-width, false otherwise.
         */
        virtual bool DoesTraceCrossLine(dsl_coordinate* coordinates, uint numCoordinates,
            uint& direction) = 0;
        
        /**
         * @brief Gets the bbox test-point for the defined for this area
         * @return one of the DSL_BBOX_POINT_* constants defining the test point.
         */
        uint GetBboxTestPoint(){return m_bboxTestPoint;};
        
    protected:
    
        /**
         * @brief Get an x,y coordinate from a Bbox based on this Trigger's
         * client specified test-point
         * @param[in] pBbox to optain the coordinate from
         * @param[out] coordinate x,y coordinage value.
         */
        void getCoordinate(const NvOSD_RectParams& bbox, 
            dsl_coordinate& coordinate);
    
        /**
         * @brief Display type used to define the Area's location, dimensions, and color
         */
        DSL_DISPLAY_TYPE_PTR m_pDisplayType;
        
        /**
         * @brief Display the area (add display meta) if true
         */
        bool m_show;

        /**
         * @brief Bounding box test point to check for Area overlap/cross. One of DSL_BBOX_POINT
         */
        uint m_bboxTestPoint;
        
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
         * @param[in] bboxTestPoint the bounding box point to check for Area criteria
         */
        OdePolygonArea(const char* name, DSL_RGBA_POLYGON_PTR pPolygon, 
            bool show, uint bboxTestPoint);

        /**
         * @brief dtor for the OdeInclusionArea
         */
        ~OdePolygonArea();

        /**
         * @brief Checks if a bounding box - using bboxTestPoint - is inside 
         * the Area's Multi-Line Display Type.
         * @param[in] bbox object's bbox rectangle to test for inside
         * @return true if inside, false otherwise
         */
        bool IsBboxInside(const NvOSD_RectParams& bbox);

        /**
         * @brief Checks if an x,y coordinate is inside the Area's Polygon 
         * Display Type.
         * @param[in] pBox pointer to an object's bbox rectangle to check for inside.
         * @return true if inside, false otherwise.
         */
        bool IsPointInside(const dsl_coordinate& coordinate);

        /**
         * @brief Returns the location of an x, y point relative to the Area's 
         * Polygon Display Type.
         * @param[in] coordinate x,y coordinate of the point to test for location.
         * @return DSL_AREA_POINT_LOCATION_INSIDE or DSL_AREA_POINT_LOCATION_OUTSIDE.
         */ 
        uint GetPointLocation(const dsl_coordinate& coordinate);
        
        /**
         * @brief Tests if a point is on the Area's Polygon Display Type 
         * including line width,
         * @param[in] coordinate x,y coordinate for the point to test for on-line
         * @return true if the point is located on the line including line-wide, 
         * false otherwise.
         */
        bool IsPointOnLine(const dsl_coordinate& coordinate);

        /**
         * @brief Checks if a bounding box trace crosses the Area's Polygon 
         * Display Type.
         * @param[in] coordinates array of dsl_coordinates.
         * @param[in] numCoordinates size of the array.
         * @param[out] direction one of the DSL_AREA_CROSS_DIRECTION_* constants 
         * defining the direction of the cross, including DSL_AREA_CROSS_DIRECTION_NONE.
         * @return true if trace fully crosses one of the sides of the Area's 
         * Polygon including line-width, false otherwise.
         */
        bool DoesTraceCrossLine(dsl_coordinate* coordinates, uint numCoordinates,
            uint& direction);

        /**
         * @brief Polygon display type used to define the Area's location, dimensions, and color
         */
        DSL_RGBA_POLYGON_PTR m_pPolygon;
        
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
         * @brief ctor for the OdeLineArea
         * @param[in] pLine a shared pointer to a RGBA Line Display Type.
         * @param[in] show if true, the area will be displayed by adding meta data
         * @param[in] bbox_test_edge one of DSL_BBOX_EDGE values to define which edge
         * of the bounding box to test for lines crossing
         */
        OdeLineArea(const char* name, DSL_RGBA_LINE_PTR pLine, 
            bool show, uint bboxTestPoint);

        /**
         * @brief dtor for the OdeLineArea
         */
        ~OdeLineArea();
        
        /**
         * @brief Checks if a bounding box - using bboxTestPoint - is inside 
         * the Area's Line Display Type.
         * @param[in] bbox object's bbox rectangle to test for inside
         * @return true if inside, false otherwise
         */
        bool IsBboxInside(const NvOSD_RectParams& bbox);

        /**
         * @brief Checks if an x,y coordinate is inside the Area's Line Display Type.
         * @param[in] pBox pointer to an object's bbox rectangle to check for inside.
         * @return true if inside, false otherwise.
         */
        bool IsPointInside(const dsl_coordinate& coordinate);

        /**
         * @brief Returns the location of an x, y point relative to the Area's 
         * Line Display Type.
         * @param[in] coordinate x,y coordinate of the point to test for location.
         * @return DSL_AREA_POINT_LOCATION_INSIDE or DSL_AREA_POINT_LOCATION_OUTSIDE.
         */ 
        uint GetPointLocation(const dsl_coordinate& coordinate);
        
        /**
         * @brief Tests if a point is on the Area's Line Display Type 
         * including line width,
         * @param[in] coordinate x,y coordinate for the point to test for on-line
         * @return true if the point is located on the line including line-wide, 
         * false otherwise.
         */
        bool IsPointOnLine(const dsl_coordinate& coordinate);

        /**
         * @brief Checks if a bounding box trace crosses the Area's Line 
         * Display Type.
         * @param[in] coordinates array of dsl_coordinates.
         * @param[in] numCoordinates size of the array.
         * @param[out] direction one of the DSL_AREA_CROSS_DIRECTION_* constants 
         * defining the direction of the cross, including DSL_AREA_CROSS_DIRECTION_NONE.
         * @return true if trace fully crosses the Area's Line Display Type including 
         * line-width, false otherwise.
         */
        bool DoesTraceCrossLine(dsl_coordinate* coordinates, uint numCoordinates,
            uint& direction);
            
        /**
         * @brief RGBA Line Display Type used to define the Area's location, 
         * dimensions, and color
         */
        DSL_RGBA_LINE_PTR m_pLine;

        /**
         * @brief one of DSL_BBOX_EDGE values defining which edge
         * of the bounding box to test for lines crossing
         */
        uint m_bboxTestEdge;
    };

    class OdeMultiLineArea : public OdeArea
    {
    public: 

        /**
         * @brief ctor for the OdeMultiLineArea
         * @param[in] pMultiLine a shared pointer to a RGBA Multi Line Display Type.
         * @param[in] show if true, the area will be displayed by adding meta data
         * @param[in] bbox_test_edge one of DSL_BBOX_EDGE values to define which edge
         * of the bounding box to test for lines crossing
         */
        OdeMultiLineArea(const char* name, DSL_RGBA_MULTI_LINE_PTR pMultiLine, 
            bool show, uint bboxTestPoint);

        /**
         * @brief dtor for the OdeLineArea
         */
        ~OdeMultiLineArea();
        
        /**
         * @brief Checks if a bounding box - using bboxTestPoint - is inside 
         * the Area's Multi-Line Display Type.
         * @param[in] bbox object's bbox rectangle to test for inside
         * @return true if inside, false otherwise
         */
        bool IsBboxInside(const NvOSD_RectParams& bbox);

        /**
         * @brief Checks if an x,y coordinate is inside the Area's Multi-Line 
         * Display Type.
         * @param[in] pBox pointer to an object's bbox rectangle to check for inside.
         * @return true if inside, false otherwise.
         */
        bool IsPointInside(const dsl_coordinate& coordinate);

        /**
         * @brief Returns the location of an x, y point relative to the Area's 
         * Multi-Line Display Type.
         * @param[in] coordinate x,y coordinate of the point to test for location.
         * @return DSL_AREA_POINT_LOCATION_INSIDE or DSL_AREA_POINT_LOCATION_OUTSIDE.
         */ 
        uint GetPointLocation(const dsl_coordinate& coordinate);
        
        /**
         * @brief Tests if a point is on the Area's Multi-Line Display Type 
         * including line width,
         * @param[in] coordinate x,y coordinate for the point to test for on-line
         * @return true if the point is located on the line including line-wide, 
         * false otherwise.
         */
        bool IsPointOnLine(const dsl_coordinate& coordinate);

        /**
         * @brief Checks if a bounding box trace crosses the Area's Multi-Line 
         * Display Type.
         * @param[in] coordinates array of dsl_coordinates.
         * @param[in] numCoordinates size of the array.
         * @param[out] direction one of the DSL_AREA_CROSS_DIRECTION_* constants 
         * defining the direction of the cross, including DSL_AREA_CROSS_DIRECTION_NONE.
         * @return true if trace fully crosses the Area's Multi-Line Display Type 
         * including line-width, false otherwise.
         */
        bool DoesTraceCrossLine(dsl_coordinate* coordinates, uint numCoordinates,
            uint& direction);

        /**
         * @brief RGBA Multi-Line Display Type used to define the Area's location, 
         * dimensions, and color
         */
        DSL_RGBA_MULTI_LINE_PTR m_pMultiLine;

        /**
         * @brief one of DSL_BBOX_EDGE values defining which edge
         * of the bounding box to test for lines crossing
         */
        uint m_bboxTestEdge;
    };
}

#endif //_DSL_ODE_AREA_H
