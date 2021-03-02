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

#ifndef _DSL_GEOS_TYPES_H
#define _DSL_GEOS_TYPES_H

#include "Dsl.h"
#include "DslDisplayTypes.h"

namespace DSL
{

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
         * @brief Actual GEOS Point for this class.
         */
        GEOSGeometry* m_pGeosPoint;
    };
        
    
    class GeosLine
    {
    public: 

        /**
         * @brief ctor for the GeosLine class
         * @param[in] line reference to a Nvidia OSD Line Structure.
         */
        GeosLine(const NvOSD_LineParams& line);

        /**
         * @brief dtor for the GeosLine class
         */
        ~GeosLine();

        /**
         * @brief function to determine if two GEOS Lines intersect
         * @param[in] testLine RGBA line to test for intersection
         * @return true if lines intersect, false otherwise
         */
        bool Intersects(const GeosLine& testLine);
        
        /**
         * @brief Actual GEOS Line for this class.
         */
        GEOSGeometry* m_pGeosLine;
    };

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

}

#endif //_DSL_GEOS_TYPES_H
