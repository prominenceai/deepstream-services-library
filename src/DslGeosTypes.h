/*
The MIT License

Copyright (c) 2021, Prominence AI, Inc.

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

#ifndef _DSL_GEOS_TYPES_H
#define _DSL_GEOS_TYPES_H

#include "Dsl.h"
#include "DslDisplayTypes.h"

namespace DSL
{
    /**
     * @class GeosPoint 
     * @file DslGeosTypes.h
     * @brief Implements a GEOS Point object with x and y coordinates
     */
    class GeosPoint
    {
    public: 

        /**
         * @brief ctor for the GeosPoint class
         * @param[in] x x coordinate for the point
         * @param[in] y y coordinate for the point
         */
        GeosPoint(uint x, uint y);

        /**
         * @brief dtor for the GeosPoint class
         */
        ~GeosPoint();
        
        /**
         * @brief function to determine the distance between two GEOS Points
         * @param testPoint GEOS Point to test for distance
         * @return distance between the two points in units of pixels
         */
        uint Distance(const GeosPoint& testPoint);
        
        /**
         * @brief Actual GEOS Point for this class.
         */
        GEOSGeometry* m_pGeosPoint;
    };
        
    /**
     * @class GeosLine
     * @file DslGeosTypes.h
     * @brief Implements a GEOS Line object that can test for
     * intersection with another GEOS Line object
     */
    class GeosLine
    {
    public: 

        /**
         * @brief ctor 1 of 2 for the GeosLine class
         * @param[in] line reference to a Nvidia OSD Line Structure.
         */
        GeosLine(const NvOSD_LineParams& line);

        /**
         * @brief ctor 1 of 2 for the GeosLine class
         * @param[in] x1 x-position for the start point of the line
         * @param[in] y1 y-position for the start point of the line
         * @param[in] x2 x-position for the end point of the line
         * @param[in] y2 y-position for the end point of the line
         */
        GeosLine(uint x1, uint y1, uint x2, uint y2);
        
        /**
         * @brief dtor for the GeosLine class
         */
        ~GeosLine();

        /**
         * @brief function to determine if two GEOS Lines intersect
         * @param[in] testLine GEOS line to test for intersection
         * @return true if lines intersect, false otherwise
         */
        bool Intersects(const GeosLine& testLine);
        
        /**
         * @brief function to determine the distance from a Point.
         * @param[in] testPoint GEOS point to calculate for distance. 
         * @return the distance in pixels. 
         */
        uint Distance(const GeosPoint& testPoint);
        
        /**
         * @brief Actual GEOS Line for this class.
         */
        GEOSGeometry* m_pGeosLine;
    };

    /**
     * @class GeosRectangle
     * @file DslGeosTypes.h
     * @brief Implements a GEOS Rectangle object that can test for
     * intersection with other GEOS Rectangle objects
     */
    class GeosRectangle
    {
    public: 

        /**
         * @brief ctor for the GeosRectangle class
         * @param[in] rectangle reference to a Nvidia OSD Rectangle Structure.
         */
        GeosRectangle(const NvOSD_RectParams& rectangle);

        /**
         * @brief dtor for the GeosRectangle class
         */
        ~GeosRectangle();

        /**
         * @brief function to determine the distance between two GEOS Rectangels
         * @param testRectangle GEOS rectaangle to test for distance
         * @return shortest distance between any two points on the rectangles
         */
        uint Distance(const GeosRectangle& testRectangle);
        
        /**
         * @brief function to determine if two GEOS Rectangles overlap
         * @param[in] testRectangle GEOS rectangle to test for overlap
         * @return true if rectangles intersect, false otherwise
         */
        bool Overlaps(const GeosRectangle& testRectangle);
        
        /**
         * @brief Actual GEOS Rectangle for this class.
         */
        GEOSGeometry* m_pGeosRectangle;

    };

    /**
     * @class GeosPolygon
     * @file DslGeosTypes.h
     * @brief Implements a GEOS Polygon object that can test for
     * overlap and containment with other GEOS Types
     */
    class GeosPolygon
    {
    public: 

        /**
         * @brief ctor for the GeosPolygon class
         * @param[in] polygon reference to a DSL Polygon Structure (interim).
         */
        GeosPolygon(const dsl_polygon_params& polygon);
        
        /**
         * @brief ctor for the GeosPolygon class
         * @param[in] rectangle reference to a Nvidia OSD Rectancle Structure
         */
        GeosPolygon(const NvOSD_RectParams& rectangle);

        /**
         * @brief dtor for the GeosPolygon class
         */
        ~GeosPolygon();

        /**
         * @brief function to determine the distance from a Point.
         * @param[in] testPoint GEOS point to calculate for distance. 
         * @return the distance in pixels. 
         */
        uint Distance(const GeosPoint& testPoint);

        /**
         * @brief function to determine if two GEOS Polygons overlap
         * @param[in] testPolygon polygon to test for intersection with polygon
         * @return true if polygons intersect, false otherwise
         */
        bool Overlaps(const GeosPolygon& testPolygon);

        /**
         * @brief function to determine if the polygon contains a polygon
         * @param[in] testPolygon polygon to test
         * @return true if the polygon contains the test polygon, false otherwise
         */
        bool Contains(const GeosPolygon& testPolygon);

        /**
         * @brief function to determine if the polygon contains a point
         * @param[in] testPoint GeosPoint to test
         * @return true if the polygon contains the point, false otherwise
         */
        bool Contains(const GeosPoint& testPoint);
        
        /**
         * @brief Actual GEOS Polygon for this class.
         */
        GEOSGeometry* m_pGeosPolygon;

    };

    /**
     * @class GeosMultiLine
     * @file DslGeosTypes.h
     * @brief Implements a GEOS Multi-Line object that can test for
     * intersection with a GEOS Line object
     */
    class GeosMultiLine
    {
    public: 

        /**
         * @brief ctor for the GeosMultiLine class
         * @param[in] multi-line reference to a DSL Multi-Line Structure.
         */ 
        GeosMultiLine(const dsl_multi_line_params& multiLine);

        /**
         * @brief dtor for the GeosMultiLine class
         */
        ~GeosMultiLine();

        /**
         * @brief function to determine if a GEOS Line cross with this Multi-Line
         * @param[in] testLine GEOS line to test for cross
         * @return true if lines intersect, false otherwise
         */
        bool Crosses(const GeosLine& testLine);
        
        /**
         * @brief function to determine a 
         * @param[in] testLine GEOS line to test for intersection
         * @return true if lines intersect, false otherwise
         */
        bool Crosses(const GeosPolygon& testPolygon);

        /**
         * @brief function to determine a 
         * @param[in] testLine GEOS line to test for intersection
         * @return true if lines intersect, false otherwise
         */
        bool Crosses(const GeosMultiLine& testMultLine);

        /**
         * @brief function to determine the distance from a Point.
         * @param[in] testPoint GEOS point to calculate for distance. 
         * @return the distance in pixels. 
         */
        uint Distance(const GeosPoint& testPoint);
        
        /**
         * @brief Actual GEOS Multi-Line for this class.
         */
        GEOSGeometry* m_pGeosMultiLine;
    };


}

#endif //_DSL_GEOS_TYPES_H
