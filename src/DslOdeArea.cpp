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

#include "Dsl.h"
#include "DslOdeArea.h"

namespace DSL
{

    OdeArea::OdeArea(const char* name, 
        DSL_DISPLAY_TYPE_PTR pDisplayType, bool show, uint bboxTestPoint)
        : Base(name)
        , m_pDisplayType(pDisplayType)
        , m_show(show)
        , m_bboxTestPoint(bboxTestPoint)
    {
        LOG_FUNC();
    }
    
    OdeArea::~OdeArea()
    {
        LOG_FUNC();
    }
        
    void OdeArea::AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData,  
        NvDsFrameMeta* pFrameMeta)
    {
        LOG_FUNC();
        
        if (!m_show)
        {
            return;
        }
        
        // If this is the first time seeing a frame for the reported Source Id.
        if (m_frameNumPerSource.find(pFrameMeta->source_id) == m_frameNumPerSource.end())
        {
            // Initial the frame number for the new source
            m_frameNumPerSource[pFrameMeta->source_id] = 0;
        }

        // If the last frame number for the reported source is different from the current frame
        // This can be either greater than or less than depending on play direction.
        if (m_frameNumPerSource[pFrameMeta->source_id] != pFrameMeta->frame_num)
        {
            // Update the frame number so we only add the rectangle once
            m_frameNumPerSource[pFrameMeta->source_id] = pFrameMeta->frame_num;

            m_pDisplayType->AddMeta(displayMetaData, pFrameMeta);
        }
    }

    void OdeArea::getCoordinate(const NvOSD_RectParams& bbox, 
        dsl_coordinate& coordinate)
    {
        switch (m_bboxTestPoint)
        {
        case DSL_BBOX_POINT_CENTER :
            coordinate.x = round(bbox.left + bbox.width/2);
            coordinate.y = round(bbox.top + bbox.height/2);
            break;
        case DSL_BBOX_POINT_NORTH_WEST :
            coordinate.x = round(bbox.left);
            coordinate.y = round(bbox.top);
            break;
        case DSL_BBOX_POINT_NORTH :
            coordinate.x = round(bbox.left + bbox.width/2);
            coordinate.y = round(bbox.top);
            break;
        case DSL_BBOX_POINT_NORTH_EAST :
            coordinate.x = round(bbox.left + bbox.width);
            coordinate.y = round(bbox.top);
            break;
        case DSL_BBOX_POINT_EAST :
            coordinate.x = round(bbox.left + bbox.width);
            coordinate.y = round(bbox.top + bbox.height/2);
            break;
        case DSL_BBOX_POINT_SOUTH_EAST :
            coordinate.x = round(bbox.left + bbox.width);
            coordinate.y = round(bbox.top + bbox.height);
            break;
        case DSL_BBOX_POINT_SOUTH :
            coordinate.x = round(bbox.left + bbox.width/2);
            coordinate.y = round(bbox.top + bbox.height);
            break;
        case DSL_BBOX_POINT_SOUTH_WEST :
            coordinate.x = round(bbox.left);
            coordinate.y = round(bbox.top + bbox.height);
            break;
        case DSL_BBOX_POINT_WEST :
            coordinate.x = round(bbox.left);
            coordinate.y = round(bbox.top + bbox.height/2);
            break;
        default:
            LOG_ERROR("Invalid DSL_BBOX_POINT = '" << m_bboxTestPoint 
                << "' for Tracked Object ");
            throw;
        }          
    }
    
    // *****************************************************************************

    OdePolygonArea::OdePolygonArea(const char* name, 
        DSL_RGBA_POLYGON_PTR pPolygon, bool show, uint bboxTestPoint)
        : OdeArea(name, pPolygon, show, bboxTestPoint)
        , m_pPolygon(pPolygon)
    {
        LOG_FUNC();
    }
    
    OdePolygonArea::~OdePolygonArea()
    {
        LOG_FUNC();
    }
    
    bool OdePolygonArea::IsBboxInside(const NvOSD_RectParams& bbox)
    {
        // Do not log function entry
        
        GeosPolygon testPolygon(bbox);
        
        if (m_bboxTestPoint == DSL_BBOX_POINT_ANY)
        {
            return (((GeosPolygon)*m_pPolygon).Overlaps(testPolygon) or
                ((GeosPolygon)*m_pPolygon).Contains(testPolygon) or
                testPolygon.Contains((GeosPolygon)*m_pPolygon));
        }        
        dsl_coordinate coordinate;
        getCoordinate(bbox, coordinate);
        
        GeosPoint testPoint(coordinate.x, coordinate.y);
        return ((GeosPolygon)*m_pPolygon).Contains(testPoint);
    }

    bool OdePolygonArea::IsPointInside(const dsl_coordinate& coordinate)
    {
        // Do not log function entry

        GeosPoint testPoint(coordinate.x, coordinate.y);

        for (uint i = 0; i < m_pPolygon->num_coordinates-1; i++)
        {
            GeosLine lineSegment(
                m_pPolygon->coordinates[i].x, 
                m_pPolygon->coordinates[i].y, 
                m_pPolygon->coordinates[(i+1)].x, 
                m_pPolygon->coordinates[(i+1)].y);
            
            if (lineSegment.Distance(testPoint) <= 
                (m_pPolygon->border_width/2))
            {
                return false;
            }
            return ((GeosPolygon)*m_pPolygon).Contains(testPoint);          
        }
    }
    
    uint OdePolygonArea::GetPointLocation(const dsl_coordinate& coordinate)
    {
        // Do not log function entry
        
        GeosPoint testPoint(coordinate.x, coordinate.y);

        for (uint i = 0; i < m_pPolygon->num_coordinates-1; i++)
        {
            GeosLine lineSegment(
                m_pPolygon->coordinates[i].x, 
                m_pPolygon->coordinates[i].y, 
                m_pPolygon->coordinates[(i+1)].x, 
                m_pPolygon->coordinates[(i+1)].y);
            
            if (lineSegment.Distance(testPoint) <= 
                (m_pPolygon->border_width/2))
            {
                return DSL_AREA_POINT_LOCATION_ON_LINE;
            }
        }
        return ((GeosPolygon)*m_pPolygon).Contains(testPoint)
            ? DSL_AREA_POINT_LOCATION_INSIDE
            : DSL_AREA_POINT_LOCATION_OUTSIDE;
    }
    
    bool OdePolygonArea::IsPointOnLine(const dsl_coordinate& coordinate)
    {
        // Do not log function entry

        GeosPoint point(coordinate.x, coordinate.y);
        
        for (uint i = 0; i < m_pPolygon->num_coordinates-1; i++)
        {
            GeosLine lineSegment(
                m_pPolygon->coordinates[i].x, 
                m_pPolygon->coordinates[i].y, 
                m_pPolygon->coordinates[(i+1)].x, 
                m_pPolygon->coordinates[(i+1)].y);
            
            if (lineSegment.Distance(point) <= 
                (m_pPolygon->border_width/2))
            {
                return true;
            }
        }
        return false;
    }
    
    bool OdePolygonArea::DoesTraceCrossLine(dsl_coordinate* coordinates, 
        uint numCoordinates, uint& direction)
    {
        // Do not log function entry
        
        // covert the trace vector to line-parameters for testing
        dsl_multi_line_params lineParms = {coordinates, numCoordinates};
        
        direction = DSL_AREA_CROSS_DIRECTION_NONE;

        // create a Geos object from the line-parameters to check 
        // for cross with this Area's line.
        GeosMultiLine multiLine(lineParms);
        
        if (!multiLine.Crosses(*m_pPolygon))
        { 
            return false;
        }
        
        // use the Area's line width and trace-endpoint to determine if the cross
        // is sufficient to report, i.e. the line width is used as hysteresis.
        GeosPoint endPoint(
            coordinates[numCoordinates-1].x, 
            coordinates[numCoordinates-1].y);
        
        bool crossed(((GeosPolygon)*m_pPolygon).Distance(endPoint) > 
            (m_pPolygon->border_width/2));

        if (crossed)
        {
            // in case the object's trace crosses the line more than once.
            if (GetPointLocation(coordinates[numCoordinates-1]) == 
                GetPointLocation(coordinates[0]))
            {
                return false;
            }
            direction = GetPointLocation(coordinates[numCoordinates-1]);
        }
        return crossed;
    }
    
    // *****************************************************************************
    
    OdeInclusionArea::OdeInclusionArea(const char* name, 
        DSL_RGBA_POLYGON_PTR pPolygon, bool show, uint bboxTestPoint)
        : OdePolygonArea(name, pPolygon, show, bboxTestPoint)
    {
        LOG_FUNC();
    }
    
    OdeInclusionArea::~OdeInclusionArea()
    {
        LOG_FUNC();
    }
    
    // *****************************************************************************
    
    OdeExclusionArea::OdeExclusionArea(const char* name, 
        DSL_RGBA_POLYGON_PTR pPolygon, bool show, uint bboxTestPoint)
        : OdePolygonArea(name, pPolygon, show, bboxTestPoint)
    {
        LOG_FUNC();
    }
    
    OdeExclusionArea::~OdeExclusionArea()
    {
        LOG_FUNC();
    }

    // *****************************************************************************
    
    OdeLineArea::OdeLineArea(const char* name, 
        DSL_RGBA_LINE_PTR pLine, bool show, uint bboxTestPoint)
        : OdeArea(name, pLine, show, bboxTestPoint)
        , m_pLine(pLine)
    {
        LOG_FUNC();
    }
    
    OdeLineArea::~OdeLineArea()
    {
        LOG_FUNC();
    }
    
    bool OdeLineArea::IsBboxInside(const NvOSD_RectParams& bbox)
    {
        // Do not log function entry

        dsl_coordinate coordinate;
        getCoordinate(bbox, coordinate);

        int dvalue = 
            (coordinate.x - m_pLine->x1) * (m_pLine->y2 - m_pLine->y1) -
            (coordinate.y - m_pLine->y1) * (m_pLine->x2 - m_pLine->x1);
            
        return (dvalue <= 0) ;
    }
    
    bool OdeLineArea::IsPointInside(const dsl_coordinate& coordinate)
    {
        // Do not log function entry

        int dvalue = 
            (coordinate.x - m_pLine->x1) * (m_pLine->y2 - m_pLine->y1) -
            (coordinate.y - m_pLine->y1) * (m_pLine->x2 - m_pLine->x1);
            
        return (dvalue <= 0) ;
    }

    uint OdeLineArea::GetPointLocation(const dsl_coordinate& coordinate)
    {
        // Do not log function entry

        GeosPoint point(coordinate.x, coordinate.y);
        
        if (((GeosLine)*m_pLine).Distance(point) <= 
            (m_pLine->line_width/2))
        {
            return DSL_AREA_POINT_LOCATION_ON_LINE;
        }
        int dvalue = 
            (coordinate.x - m_pLine->x1) * (m_pLine->y2 - m_pLine->y1) -
            (coordinate.y - m_pLine->y1) * (m_pLine->x2 - m_pLine->x1);

        return (dvalue > 0) 
            ? DSL_AREA_POINT_LOCATION_OUTSIDE
            : DSL_AREA_POINT_LOCATION_INSIDE;
    }
    
    bool OdeLineArea::IsPointOnLine(const dsl_coordinate& coordinate)
    {
        // Do not log function entry

        GeosPoint point(coordinate.x, coordinate.y);
        
        return (((GeosLine)*m_pLine).Distance(point) <= 
            (m_pLine->line_width/2));
    }
    
    bool OdeLineArea::DoesTraceCrossLine(dsl_coordinate* coordinates,
        uint numCoordinates, uint& direction)
    {
        // Do not log function entry
        
        // covert the trace vector to line-parameters for testing
        dsl_multi_line_params lineParms = {coordinates, 
            numCoordinates};

        direction = DSL_AREA_CROSS_DIRECTION_NONE;
        
        // create a Geos object from the line-parameters to check 
        // for cross with this Area's line.
        GeosMultiLine multiLine(lineParms);
        
        if (!multiLine.Crosses(*m_pLine))
        { 
            return false;
        }

        // use the Area's line width and trace-endpoint to determine if the cross
        // is sufficient to report, i.e. the line width is used as hysteresis.
        GeosPoint endPoint(
            coordinates[numCoordinates-1].x, 
            coordinates[numCoordinates-1].y);
        
        bool crossed(((GeosLine)*m_pLine).Distance(endPoint) > 
            (m_pLine->line_width/2));
            
        if (crossed)
        {
            // in case the object's trace crosses the line more than once.
            if (GetPointLocation(coordinates[numCoordinates-1]) == 
                GetPointLocation(coordinates[0]))
            {
                return false;
            }
            direction = GetPointLocation(coordinates[numCoordinates-1]);
        }
        return crossed;    
    }
    
    // *****************************************************************************
    
    OdeMultiLineArea::OdeMultiLineArea(const char* name, 
        DSL_RGBA_MULTI_LINE_PTR pMultiLine, bool show, uint bboxTestPoint)
        : OdeArea(name, pMultiLine, show, bboxTestPoint)
        , m_pMultiLine(pMultiLine)
    {
        LOG_FUNC();
    }
    
    OdeMultiLineArea::~OdeMultiLineArea()
    {
        LOG_FUNC();
    }
    
    bool OdeMultiLineArea::IsBboxInside(const NvOSD_RectParams& bbox)
    {
        // Do not log function entry

        uint inside(0), outside(0);
        dsl_coordinate coordinate;
        getCoordinate(bbox, coordinate);

        for (uint i = 0; i < m_pMultiLine->num_coordinates-1; i++)
        {
            int dvalue = 
                (coordinate.x - m_pMultiLine->coordinates[i].x) * 
                    (m_pMultiLine->coordinates[(i+1)].y - m_pMultiLine->coordinates[i].y) -
                (coordinate.y - m_pMultiLine->coordinates[i].y) * 
                    (m_pMultiLine->coordinates[(i+1)].x - m_pMultiLine->coordinates[i].x);
            
            if (dvalue <= 0)
            {
                inside++;
            }
            else
            {
                outside++;
            }
        }
        return inside > outside;
    }
    
    bool OdeMultiLineArea::IsPointInside(const dsl_coordinate& coordinate)
    {
        // Do not log function entry

        uint inside(0), outside(0);
        GeosPoint point(coordinate.x, coordinate.y);

        if (((GeosMultiLine)*m_pMultiLine).Distance(point) <= 
            (m_pMultiLine->line_width/2))
        {
            return false;
        }
        for (uint i = 0; i < m_pMultiLine->num_coordinates-1; i++)
        {
            int dvalue = 
                (coordinate.x - m_pMultiLine->coordinates[i].x) * 
                    (m_pMultiLine->coordinates[(i+1)].y - m_pMultiLine->coordinates[i].y) -
                (coordinate.y - m_pMultiLine->coordinates[i].y) * 
                    (m_pMultiLine->coordinates[(i+1)].x - m_pMultiLine->coordinates[i].x);
            
            if (dvalue <= 0)
            {
                inside++;
            }
            else
            {
                outside++;
            }
        }
        return inside > outside;
    }

    uint OdeMultiLineArea::GetPointLocation(const dsl_coordinate& coordinate)
    {
        // Do not log function entry

        uint inside(0), outside(0);
        GeosPoint point(coordinate.x, coordinate.y);

        if (((GeosMultiLine)*m_pMultiLine).Distance(point) <= 
            (m_pMultiLine->line_width/2))
        {
            return DSL_AREA_POINT_LOCATION_ON_LINE;
        }
        for (uint i = 0; i < m_pMultiLine->num_coordinates-1; i++)
        {
            int dvalue = 
                (coordinate.x - m_pMultiLine->coordinates[i].x) * 
                    (m_pMultiLine->coordinates[(i+1)].y - m_pMultiLine->coordinates[i].y) -
                (coordinate.y - m_pMultiLine->coordinates[i].y) * 
                    (m_pMultiLine->coordinates[(i+1)].x - m_pMultiLine->coordinates[i].x);
            
            if (dvalue <= 0)
            {
                inside++;
            }
            else
            {
                outside++;
            }
        }
        return (inside > outside)
            ? DSL_AREA_POINT_LOCATION_INSIDE
            : DSL_AREA_POINT_LOCATION_OUTSIDE;
    }
    
    bool OdeMultiLineArea::IsPointOnLine(const dsl_coordinate& coordinate)
    {
        GeosPoint point(coordinate.x, coordinate.y);
        
        return (((GeosMultiLine)*m_pMultiLine).Distance(point) <= 
            (m_pMultiLine->line_width/2));
    }
    
    bool OdeMultiLineArea::DoesTraceCrossLine(dsl_coordinate* coordinates, 
        uint numCoordinates, uint& direction)
    {
        // Do not log function entry
        
        // covert the trace vector to line-parameters for testing
        dsl_multi_line_params lineParms = {coordinates, 
            numCoordinates};

        direction = DSL_AREA_CROSS_DIRECTION_NONE;
        
        // create a Geos object from the line-parameters to check 
        // for cross with this Area's line.
        GeosMultiLine multiLine(lineParms);
        
        if (!multiLine.Crosses(*m_pMultiLine))
        { 
            return false;
        }
        
        // use the Area's line width and trace-endpoint to determine if the cross
        // is sufficient to report, i.e. the line width is used as hysteresis.
        GeosPoint endPoint(
            coordinates[numCoordinates-1].x, 
            coordinates[numCoordinates-1].y);
        
        bool crossed(((GeosMultiLine)*m_pMultiLine).Distance(endPoint) > 
            (m_pMultiLine->line_width/2));
            
        if (crossed)
        {
            // in case the object's trace crosses the line more than once.
            if (GetPointLocation(coordinates[numCoordinates-1]) == 
                GetPointLocation(coordinates[0]))
            {
                return false;
            }
            direction = GetPointLocation(coordinates[numCoordinates-1]);
        }
        return crossed;
    }
}