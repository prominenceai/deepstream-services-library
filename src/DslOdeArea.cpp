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
        DSL_DISPLAY_TYPE_PTR pDisplayType, bool show)
        : Base(name)
        , m_pDisplayType(pDisplayType)
        , m_show(show)
    {
        LOG_FUNC();
        
    }
    
    OdeArea::~OdeArea()
    {
        LOG_FUNC();
    }
        
    void OdeArea::AddMeta(NvDsDisplayMeta* pDisplayMeta,  NvDsFrameMeta* pFrameMeta)
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

            m_pDisplayType->AddMeta(pDisplayMeta, pFrameMeta);
        }
    }
    
    // *****************************************************************************

    OdePolygonArea::OdePolygonArea(const char* name, 
        DSL_RGBA_POLYGON_PTR pPolygon, bool show, uint bboxTestPoint)
        : OdeArea(name, pPolygon, show)
        , m_pGeosPolygon(*pPolygon)
        , m_bboxTestPoint(bboxTestPoint)
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
            return (m_pGeosPolygon.Overlaps(testPolygon) or
                m_pGeosPolygon.Contains(testPolygon) or
                testPolygon.Contains(m_pGeosPolygon));
                
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
                << "' for OdeLineArea '" << GetName() << "'");
            throw;
        }
        
        GeosPoint testPoint(x,y);
        return m_pGeosPolygon.Contains(testPoint);
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
        DSL_RGBA_LINE_PTR pLine, bool show, uint bboxTestEdge)
        : OdeArea(name, pLine, show)
        , m_pGeosLine(*pLine)
        , m_bboxTestEdge(bboxTestEdge)
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
        
        NvOSD_LineParams testEdge{0};
        
        switch (m_bboxTestEdge)
        {
        case DSL_BBOX_EDGE_TOP :
            testEdge.x1 = bbox.left;
            testEdge.y1 = bbox.top;
            testEdge.x2 = bbox.left + bbox.width;
            testEdge.y2 = bbox.top;
            break;
        case DSL_BBOX_EDGE_BOTTOM :
            testEdge.x1 = bbox.left;
            testEdge.y1 = bbox.top + bbox.height;
            testEdge.x2 = bbox.left + bbox.width;
            testEdge.y2 = bbox.top + bbox.height;
            break;
        case DSL_BBOX_EDGE_LEFT :
            testEdge.x1 = bbox.left;
            testEdge.y1 = bbox.top;
            testEdge.x2 = bbox.left;
            testEdge.y2 = bbox.top + bbox.height;
            break;
        case DSL_BBOX_EDGE_RIGHT :
            testEdge.x1 = bbox.left + bbox.width;
            testEdge.y1 = bbox.top;
            testEdge.x2 = bbox.left + bbox.width;
            testEdge.y2 = bbox.top + bbox.height;
            break;
        default:
            LOG_ERROR("Invalid DSL_BBOX_EDGE = '" << m_bboxTestEdge 
                << "' for OdeLineArea '" << GetName() << "'");
            throw;
        }
        
        GeosLine testGeosLine(testEdge);

        return m_pGeosLine.Intersects(testGeosLine);
    }
    
}