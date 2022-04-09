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

#include "DslGeosTypes.h"

namespace DSL
{

    GeosPoint::GeosPoint(uint x, uint y)
        : m_pGeosPoint(NULL)
    {
        // Don't log function entry/exit
        
        GEOSCoordSequence* geosCoordSequence = GEOSCoordSeq_create(1, 2);
        if (!geosCoordSequence)
        {
            LOG_ERROR("Exception when creating GEOS Coordinate Sequence for GEOS Point");
            throw;
        }
        if (!GEOSCoordSeq_setX(geosCoordSequence, 0, double(x)) or 
            !GEOSCoordSeq_setY(geosCoordSequence, 0, double(y))) 
        {
            LOG_ERROR("Exception when setting GEOS Coordinate Sequence for GEOS Point");
            throw;
        }
        
        m_pGeosPoint = GEOSGeom_createPoint(geosCoordSequence);
        if (!m_pGeosPoint)
        {
            LOG_ERROR("Exception when setting GEOS Point");
            throw;
        }
    }

    GeosPoint::~GeosPoint()
    {
        // Don't log function entry/exit
        
        if (m_pGeosPoint)
        {
            GEOSGeom_destroy(m_pGeosPoint);
        }
    }

    uint GeosPoint::Distance(const GeosPoint& testPoint)
    {
        // Don't log function entry/exit
        double distance;
        
        if (!GEOSDistance(m_pGeosPoint, testPoint.m_pGeosPoint, &distance))
        {
            LOG_ERROR("Exception when calling GEOS Distance");
            throw;
        }
        return (uint)round(distance);
    }
    
    //******************************************************************************
    
    GeosLine::GeosLine(const NvOSD_LineParams& line)
        : m_pGeosLine(NULL)
    {
        // Don't log function entry/exit
        
        GEOSCoordSequence* geosCoordSequence = GEOSCoordSeq_create(2, 2);
        if (!geosCoordSequence)
        {
            LOG_ERROR("Exception when creating GEOS Coordinate Sequence");
            throw;
        }
        if (!GEOSCoordSeq_setX(geosCoordSequence, 0, double(line.x1)) or 
            !GEOSCoordSeq_setY(geosCoordSequence, 0, double(line.y1)) or
            !GEOSCoordSeq_setX(geosCoordSequence, 1, double(line.x2)) or
            !GEOSCoordSeq_setY(geosCoordSequence, 1, double(line.y2))) 
        {
            LOG_ERROR("Exception when setting GEOS Coordinate Sequence");
            throw;
        }
        
        // once created, m_pGeosLine will own the memory of geosCoordSequence
        // and will free it when GEOSGeom_destroy is called
        m_pGeosLine = GEOSGeom_createLineString(geosCoordSequence);
        if (!m_pGeosLine)
        {
            LOG_ERROR("Exception when creating GEOS Line String");
            throw;
        }
    }
    
    GeosLine::~GeosLine()
    {
        // Don't log function entry/exit
        
        if (m_pGeosLine)
        {
            GEOSGeom_destroy(m_pGeosLine);
        }
    }

    bool GeosLine::Intersects(const GeosLine& testLine)
    {
        // Don't log function entry/exit
        
        char result = GEOSIntersects(m_pGeosLine, testLine.m_pGeosLine);
        if (result == 2)
        {
            LOG_ERROR("Exception when testing if GEOS Line Strings cross");
            throw;
        }
        return bool(result);
    }

    //******************************************************************************
    
    GeosMultiLine::GeosMultiLine(const dsl_multi_line_params& multiLine)
        : m_pGeosMultiLine(NULL)
    {
        // Don't log function entry/exit
        
        GEOSCoordSequence* geosCoordSequence = GEOSCoordSeq_create(multiLine.num_coordinates+1, 2);
        if (!geosCoordSequence)
        {
            LOG_ERROR("Exception when creating GEOS Coordinate Sequence");
            throw;
        }
        for (uint i = 0; i < multiLine.num_coordinates; i++)
        {
            if (!GEOSCoordSeq_setX(geosCoordSequence, i, 
                    double(multiLine.coordinates[i].x)) or   
                !GEOSCoordSeq_setY(geosCoordSequence, i, 
                    double(multiLine.coordinates[i].y))) 
            {
                LOG_ERROR("Exception when setting GEOS Coordinate Sequence");
                throw;
            }
        }
        
        
        // once created, m_pGeosMultiLine will own the memory of geosCoordSequence
        // and will free it when GEOSGeom_destroy is called
        m_pGeosMultiLine = GEOSGeom_createLineString(geosCoordSequence);
        if (!m_pGeosMultiLine)
        {
            LOG_ERROR("Exception when creating GEOS Line String");
            throw;
        }
    }
    
    GeosMultiLine::~GeosMultiLine()
    {
        // Don't log function entry/exit
        
        if (m_pGeosMultiLine)
        {
            GEOSGeom_destroy(m_pGeosMultiLine);
        }
    }

    bool GeosMultiLine::Intersects(const GeosLine& testLine)
    {
        // Don't log function entry/exit
        
        char result = GEOSIntersects(m_pGeosMultiLine, testLine.m_pGeosLine);
        if (result == 2)
        {
            LOG_ERROR("Exception when testing if GEOS Line Strings cross");
            throw;
        }
        return bool(result);
    }

    // *****************************************************************************

    GeosRectangle::GeosRectangle(const NvOSD_RectParams& rectangle)
        : m_pGeosRectangle(NULL)
    {
        // Don't log function entry/exit
        
        GEOSCoordSequence* geosCoordSequence = GEOSCoordSeq_create(5, 2);
        if (!geosCoordSequence)
        {
            LOG_ERROR("Exception when creating GEOS Coordinate Sequence");
            throw;
        }
        if (!GEOSCoordSeq_setX(geosCoordSequence, 0, double(rectangle.left)) or   
            !GEOSCoordSeq_setY(geosCoordSequence, 0, double(rectangle.top)) or
            !GEOSCoordSeq_setX(geosCoordSequence, 1, double(rectangle.left + rectangle.width)) or   
            !GEOSCoordSeq_setY(geosCoordSequence, 1, double(rectangle.top)) or
            !GEOSCoordSeq_setX(geosCoordSequence, 2, double(rectangle.left + rectangle.width)) or   
            !GEOSCoordSeq_setY(geosCoordSequence, 2, double(rectangle.top + rectangle.height)) or
            !GEOSCoordSeq_setX(geosCoordSequence, 3, double(rectangle.left)) or   
            !GEOSCoordSeq_setY(geosCoordSequence, 3, double(rectangle.top + rectangle.height)) or
            !GEOSCoordSeq_setX(geosCoordSequence, 4, double(rectangle.left)) or   
            !GEOSCoordSeq_setY(geosCoordSequence, 4, double(rectangle.top))) 
        {
            LOG_ERROR("Exception when setting GEOS Coordinate Sequence");
            throw;
        }
        
        GEOSGeometry* outerRing = GEOSGeom_createLinearRing(geosCoordSequence);
        if (!outerRing)
        {
            LOG_ERROR("Exception when creating GEOS outer ring");
            throw;
        }

        m_pGeosRectangle = GEOSGeom_createPolygon(outerRing, NULL, 0);
        if (!m_pGeosRectangle)
        {
            LOG_ERROR("Exception when creating GEOS Polygon");
            throw;
        }
    }

    GeosRectangle::~GeosRectangle()
    {
        // Don't log function entry/exit
        
        if (m_pGeosRectangle)
        {
            GEOSGeom_destroy(m_pGeosRectangle);
        }
    }

    uint GeosRectangle::Distance(const GeosRectangle& testRectangle)
    {
        // Don't log function entry/exit
        double distance;
        
        if (!GEOSDistance(m_pGeosRectangle, testRectangle.m_pGeosRectangle, &distance))
        {
            LOG_ERROR("Exception when calling GEOS Distance");
            throw;
        }
        return (uint)round(distance);
    }

    bool GeosRectangle::Overlaps(const GeosRectangle& testRectangle)
    {
        // Don't log function entry/exit

        char result = GEOSOverlaps(m_pGeosRectangle, testRectangle.m_pGeosRectangle);
        if (result == 2)
        {
            LOG_ERROR("Exception when testing if GEOS Rectangles overlap");
            throw;
        }
        return bool(result);
        
    }

    // *****************************************************************************

    GeosPolygon::GeosPolygon(const dsl_polygon_params& polygon)
        : m_pGeosPolygon(NULL)
    {
        // Don't log function entry/exit
        
        // first coordinate needs to be added to both the start and end of the sequence
        // therefore, we need num_coordinates+1
        GEOSCoordSequence* geosCoordSequence = GEOSCoordSeq_create(polygon.num_coordinates+1, 2);
        if (!geosCoordSequence)
        {
            LOG_ERROR("Exception when creating GEOS Coordinate Sequence");
            throw;
        }
        for (uint i = 0; i < polygon.num_coordinates+1; i++)
        {
            if (!GEOSCoordSeq_setX(geosCoordSequence, i, 
                    double(polygon.coordinates[(i)%polygon.num_coordinates].x)) or   
                !GEOSCoordSeq_setY(geosCoordSequence, i, 
                    double(polygon.coordinates[(i)%polygon.num_coordinates].y))) 
            {
                LOG_ERROR("Exception when setting GEOS Coordinate Sequence");
                throw;
            }
        }
        
        GEOSGeometry* outerRing = GEOSGeom_createLinearRing(geosCoordSequence);
        if (!outerRing)
        {
            LOG_ERROR("Exception when creating GEOS outer ring");
            throw;
        }

        m_pGeosPolygon = GEOSGeom_createPolygon(outerRing, NULL, 0);
        if (!m_pGeosPolygon)
        {
            LOG_ERROR("Exception when creating GEOS Polygon");
            throw;
        }
    }

    GeosPolygon::GeosPolygon(const NvOSD_RectParams& rectangle)
        : m_pGeosPolygon(NULL)
    {
        // Don't log function entry/exit
        
        GEOSCoordSequence* geosCoordSequence = GEOSCoordSeq_create(5, 2);
        if (!geosCoordSequence)
        {
            LOG_ERROR("Exception when creating GEOS Coordinate Sequence");
            throw;
        }
        if (!GEOSCoordSeq_setX(geosCoordSequence, 0, double(rectangle.left)) or   
            !GEOSCoordSeq_setY(geosCoordSequence, 0, double(rectangle.top)) or
            !GEOSCoordSeq_setX(geosCoordSequence, 1, double(rectangle.left + rectangle.width)) or   
            !GEOSCoordSeq_setY(geosCoordSequence, 1, double(rectangle.top)) or
            !GEOSCoordSeq_setX(geosCoordSequence, 2, double(rectangle.left + rectangle.width)) or   
            !GEOSCoordSeq_setY(geosCoordSequence, 2, double(rectangle.top + rectangle.height)) or
            !GEOSCoordSeq_setX(geosCoordSequence, 3, double(rectangle.left)) or   
            !GEOSCoordSeq_setY(geosCoordSequence, 3, double(rectangle.top + rectangle.height)) or
            !GEOSCoordSeq_setX(geosCoordSequence, 4, double(rectangle.left)) or   
            !GEOSCoordSeq_setY(geosCoordSequence, 4, double(rectangle.top))) 
        {
            LOG_ERROR("Exception when setting GEOS Coordinate Sequence");
            throw;
        }
        
        GEOSGeometry* outerRing = GEOSGeom_createLinearRing(geosCoordSequence);
        if (!outerRing)
        {
            LOG_ERROR("Exception when creating GEOS outer ring");
            throw;
        }

        m_pGeosPolygon = GEOSGeom_createPolygon(outerRing, NULL, 0);
        if (!m_pGeosPolygon)
        {
            LOG_ERROR("Exception when creating GEOS Polygon");
            throw;
        }
    }
    
    GeosPolygon::~GeosPolygon()
    {
        // Don't log function entry/exit
        
        if (m_pGeosPolygon)
        {
            GEOSGeom_destroy(m_pGeosPolygon);
        }
    }

    bool GeosPolygon::Overlaps(const GeosPolygon& testPolygon)
    {
        // Don't log function entry/exit

        char result = GEOSOverlaps(m_pGeosPolygon, testPolygon.m_pGeosPolygon);
        if (result == 2)
        {
            LOG_ERROR("Exception when testing if GEOS Polygons intersect");
            throw;
        }
        return bool(result);
        
    }

    bool GeosPolygon::Contains(const GeosPolygon& testPolygon)
    {
        // Don't log function entry/exit

        char result = GEOSContains(m_pGeosPolygon, testPolygon.m_pGeosPolygon);
        if (result == 2)
        {
            LOG_ERROR("Exception when testing if GEOS Polygons intersect");
            throw;
        }
        return bool(result);
        
    }

    bool GeosPolygon::Contains(const GeosPoint& testPoint)
    {
        // Don't log function entry/exit
        
        char result = GEOSContains(m_pGeosPolygon, testPoint.m_pGeosPoint);
        
        if (result == 2)
        {
            LOG_ERROR("Exception when testing if GEOS Polygons intersect");
            throw;
        }
        return bool(result);
    }

}
