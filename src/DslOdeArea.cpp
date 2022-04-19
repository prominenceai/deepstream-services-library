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
    
    bool OdePolygonArea::CheckForWithin(const NvOSD_RectParams& bbox)
    {
        // Do not log function entry
        
        uint x(0), y(0) ;
        GeosPolygon testPolygon(bbox);
        
        switch (m_bboxTestPoint)
        {
        case DSL_BBOX_POINT_ANY :
            return (((GeosPolygon)*m_pPolygon).Overlaps(testPolygon) or
                ((GeosPolygon)*m_pPolygon).Contains(testPolygon) or
                testPolygon.Contains((GeosPolygon)*m_pPolygon));
                
        case DSL_BBOX_POINT_CENTER :
            x = round(bbox.left + bbox.width/2);
            y = round(bbox.top + bbox.height/2);
            break;
        case DSL_BBOX_POINT_NORTH_WEST :
            x = round(bbox.left);
            y = round(bbox.top);
            break;
        case DSL_BBOX_POINT_NORTH :
            x = round(bbox.left + bbox.width/2);
            y = round(bbox.top);
            break;
        case DSL_BBOX_POINT_NORTH_EAST :
            x = round(bbox.left + bbox.width);
            y = round(bbox.top);
            break;
        case DSL_BBOX_POINT_EAST :
            x = round(bbox.left + bbox.width);
            y = round(bbox.top + bbox.height/2);
            break;
        case DSL_BBOX_POINT_SOUTH_EAST :
            x = round(bbox.left + bbox.width);
            y = round(bbox.top + bbox.height);
            break;
        case DSL_BBOX_POINT_SOUTH :
            x = round(bbox.left + bbox.width/2);
            y = round(bbox.top + bbox.height);
            break;
        case DSL_BBOX_POINT_SOUTH_WEST :
            x = round(bbox.left);
            y = round(bbox.top + bbox.height);
            break;
        case DSL_BBOX_POINT_WEST :
            x = round(bbox.left);
            y = round(bbox.top + bbox.height/2);
            break;
        default:
            LOG_ERROR("Invalid DSL_BBOX_POINT = '" << m_bboxTestPoint 
                << "' for OdePolygonArea '" << GetName() << "'");
            throw;
        }
        
        GeosPoint testPoint(x,y);
        return ((GeosPolygon)*m_pPolygon).Contains(testPoint);
    }

    uint OdePolygonArea::GetPointDirection(dsl_coordinate& coordinate)
    {
        return 0;
    }
    
    bool OdePolygonArea::IsPointOnLine(dsl_coordinate& coordinate)
    {
        return false;
    }
    
    bool OdePolygonArea::CheckForCross(
        const std::shared_ptr<std::vector<dsl_coordinate>> pTrace,
        uint& distance)
    {
        // Do not log function entry
        
        return false;
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
    
    bool OdeLineArea::CheckForWithin(const NvOSD_RectParams& bbox)
    {
        // Do not log function entry
        
//        NvOSD_LineParams testEdge{0};
//        
//        switch (m_bboxTestPoint)
//        {
//        case DSL_BBOX_EDGE_TOP :
//            testEdge.x1 = bbox.left;
//            testEdge.y1 = bbox.top;
//            testEdge.x2 = bbox.left + bbox.width;
//            testEdge.y2 = bbox.top;
//            break;
//        case DSL_BBOX_EDGE_BOTTOM :
//            testEdge.x1 = bbox.left;
//            testEdge.y1 = bbox.top + bbox.height;
//            testEdge.x2 = bbox.left + bbox.width;
//            testEdge.y2 = bbox.top + bbox.height;
//            break;
//        case DSL_BBOX_EDGE_LEFT :
//            testEdge.x1 = bbox.left;
//            testEdge.y1 = bbox.top;
//            testEdge.x2 = bbox.left;
//            testEdge.y2 = bbox.top + bbox.height;
//            break;
//        case DSL_BBOX_EDGE_RIGHT :
//            testEdge.x1 = bbox.left + bbox.width;
//            testEdge.y1 = bbox.top;
//            testEdge.x2 = bbox.left + bbox.width;
//            testEdge.y2 = bbox.top + bbox.height;
//            break;
//        default:
//            LOG_ERROR("Invalid DSL_BBOX_EDGE = '" << m_bboxTestEdge 
//                << "' for OdeLineArea '" << GetName() << "'");
//            throw;
//        }
//        
//        GeosLine testGeosLine(testEdge);
//
//        return m_pGeosLine.Intersects(testGeosLine);
        return false;
    }

    uint OdeLineArea::GetPointDirection(dsl_coordinate& coordinate)
    {
        int dvalue = 
            (coordinate.x - m_pLine->x1) * (m_pLine->y2 - m_pLine->y1) -
            (coordinate.y - m_pLine->y1) * (m_pLine->x2 - m_pLine->x1);
            
        return (dvalue > 0) 
            ? DSL_AREA_CROSS_DIRECTION_OUT
            : DSL_AREA_CROSS_DIRECTION_IN;
    }
    
    bool OdeLineArea::IsPointOnLine(dsl_coordinate& coordinate)
    {
        GeosPoint point(coordinate.x, coordinate.y);
        
        return (((GeosLine)*m_pLine).Distance(point) <= 
            (m_pLine->line_width/2));
    }
    
    bool OdeLineArea::CheckForCross(
        const std::shared_ptr<std::vector<dsl_coordinate>> pTrace,
        uint& direction)
    {
        // Do not log function entry
        
        // covert the trace vector to line-parameters for testing
        dsl_multi_line_params lineParms = {pTrace->data(), 
            (uint)pTrace->size()};
        
        // create a Geos object from the line-parameters to check 
        // for cross with this Area's line.
        GeosMultiLine multiLine(lineParms);
        
        if (!multiLine.Crosses(*m_pLine))
        { 
            return false;
        }

        direction = GetPointDirection(pTrace->back());
        
        // use the Area's line width and trace-endpoint to determine if the cross
        // is sufficient to report, i.e. the line width is used as hysteresis.
        GeosPoint endPoint(pTrace->back().x, pTrace->back().y);
        
        return (((GeosLine)*m_pLine).Distance(endPoint) > 
            (m_pLine->line_width/2));
    }
    
}