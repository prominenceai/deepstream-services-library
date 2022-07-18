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

#include "DslOdeHeatMapper.h"

#define DATE_BUFF_LENGTH 40

namespace DSL
{
    OdeHeatMapper::OdeHeatMapper(const char* name, uint cols, uint rows,
        uint bboxTestPoint, DSL_RGBA_COLOR_PALETTE_PTR pColorPalette)
        : OdeBase(name)
        , m_cols(cols)
        , m_rows(rows)
        , m_gridRectWidth(0)
        , m_gridRectHeight(0)
        , m_bboxTestPoint(bboxTestPoint)
        , m_pColorPalette(pColorPalette)
        , m_heatMap(rows, std::vector<uint64_t> (cols, 0))
        , m_mostOccurrences(0)
        , m_legendEnabled(false)
        , m_legendLocation(0)
        , m_legendLeft(0)
        , m_legendTop(0)
        , m_legendWidth(0)
        , m_legendHeight(0)
    {
        LOG_FUNC();
        
        m_outBuffer = std::unique_ptr<uint64_t[]>(new uint64_t[cols*rows]);
    }

    OdeHeatMapper::~OdeHeatMapper()
    {
        LOG_FUNC();
    }
    
    DSL_RGBA_COLOR_PALETTE_PTR OdeHeatMapper::GetColorPalette()
    {
        LOG_FUNC();
        
        return m_pColorPalette;
    }
    
    bool OdeHeatMapper::SetColorPalette(DSL_RGBA_COLOR_PALETTE_PTR pColorPalette)
    {
        LOG_FUNC();
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

            m_pColorPalette = pColorPalette;
        }
        // need to recalculated legend settings.
        return SetLegendSettings(m_legendEnabled, m_legendLocation,
            m_legendWidth, m_legendHeight);
    }
    
    void OdeHeatMapper::GetLegendSettings(bool* enabled, uint* location, 
        uint* width, uint* height)
    {
        LOG_FUNC();
        
        *enabled = m_legendEnabled;
        *location = m_legendLocation;
        *width = m_legendWidth;
        *height = m_legendHeight;
    }            

    bool OdeHeatMapper::SetLegendSettings(bool enabled, uint location, 
        uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        // disable untill all params are checked.
        m_legendEnabled = false;
        
        // If client is disabling - done
        if (!enabled)
        {
            return true;
        }
         
        if (location == DSL_HEAT_MAP_LEGEND_LOCATION_TOP or
            location == DSL_HEAT_MAP_LEGEND_LOCATION_BOTTOM)
        {
            // Ensure we have enough horizontal space 
            if ((width*m_pColorPalette->GetSize() + 2) > m_cols)
            {
                LOG_ERROR("Insufficient columns = " << m_cols 
                   << "to display legend for Heat-Mapper '" << GetName());
                return false;
            }
            m_legendLeft = m_cols / 2 - m_pColorPalette->GetSize()/2*width;
            m_legendTop = (location == DSL_HEAT_MAP_LEGEND_LOCATION_TOP)
                ? 1
                : m_rows - 1 - height;
        }         
        if (location == DSL_HEAT_MAP_LEGEND_LOCATION_LEFT or
            location == DSL_HEAT_MAP_LEGEND_LOCATION_RIGHT)
        {
            if (height*m_pColorPalette->GetSize() + 2 > m_rows)
            {
                LOG_ERROR("Insufficient rows = " << m_rows 
                   << "to display legend for Heat-Mapper '" << GetName());
                return false;
            }
            m_legendLeft = (location == DSL_HEAT_MAP_LEGEND_LOCATION_LEFT)
                ? 1
                : m_cols - 1 - width;
            m_legendTop = m_rows / 2 - m_pColorPalette->GetSize()*height;
        }         

        m_legendEnabled = true;
        m_legendLocation = location;
        m_legendWidth = width;
        m_legendHeight = height;

        return true;
    }            

    void OdeHeatMapper::HandleOccurrence(NvDsFrameMeta* pFrameMeta, 
        NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        // one-time initialization of the grid rectangle dimensions
        if (!m_gridRectWidth)
        {
            m_gridRectWidth = pFrameMeta->source_frame_width/m_cols;
            m_gridRectHeight = pFrameMeta->source_frame_height/m_rows;
        }
        
        // get the x,y map coordinates based on the bbox and test-point.
        dsl_coordinate mapCoordinate;
        getCoordinate(pObjectMeta, mapCoordinate);

        // determine the column and row that maps to the x, y coordinates
        // coordinates are 1-based, so subtract 1 pixel to keep within map.
        uint colPosition((mapCoordinate.x-1)/m_gridRectWidth);
        uint rowPosition((mapCoordinate.y-1)/m_gridRectHeight);

        // increment the running count of occurrences at this poisition
        m_heatMap[rowPosition][colPosition] += 1;
        
        // if the new total for this position is now the greatest  
        if (m_heatMap[rowPosition][colPosition] > m_mostOccurrences)
        {
            m_mostOccurrences = m_heatMap[rowPosition][colPosition];
        }
    }
  
    void OdeHeatMapper::AddDisplayMeta(std::vector<NvDsDisplayMeta*>& displayMetaData)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        // Add legend first, just in case we run out of display-meta
        if (m_legendEnabled)
        {
            // If the legend is added to a vertical axis
            if (m_legendLocation == DSL_HEAT_MAP_LEGEND_LOCATION_TOP or
                m_legendLocation == DSL_HEAT_MAP_LEGEND_LOCATION_BOTTOM)
            {
                for (uint j=0; j < m_pColorPalette->GetSize(); j++)
                {
                    m_pColorPalette->SetIndex(j);

                    DSL_RGBA_RECTANGLE_PTR pRectangle = DSL_RGBA_RECTANGLE_NEW("", 
                        m_legendLeft*m_gridRectWidth + j*m_gridRectWidth*m_legendWidth, 
                        m_legendTop*m_gridRectHeight, 
                        m_gridRectWidth*m_legendWidth, 
                        m_gridRectHeight*m_legendHeight, 
                        false, m_pColorPalette, true, m_pColorPalette);
                        
                    pRectangle->AddMeta(displayMetaData, NULL);
                }
            }
            // Else the legend is added to a horizontal axis
            else
            {
                for (uint i=0; i < m_pColorPalette->GetSize(); i++)
                {
                    m_pColorPalette->SetIndex(i);

                    DSL_RGBA_RECTANGLE_PTR pRectangle = DSL_RGBA_RECTANGLE_NEW("", 
                        m_legendLeft*m_gridRectWidth, 
                        m_legendTop*m_gridRectHeight + i*m_gridRectHeight*m_legendHeight, 
                        m_gridRectWidth*m_legendWidth, 
                        m_gridRectHeight*m_legendHeight, 
                        false, m_pColorPalette, true, m_pColorPalette);
                        
                    pRectangle->AddMeta(displayMetaData, NULL);
                }
            }    
        }
        // Iterate through all rows
        for (uint i=0; i < m_rows; i++)
        {
            // and for each row, iterate through all columns.
            for (uint j=0; j < m_cols; j++)
            {
                // if we have at least one occurrence at the current iteration
                if (m_heatMap[i][j])
                {
                    // Callculate the index into the color palette of size 10 as 
                    // a ratio of occurrences for the current position vs. the 
                    // position with the most occurrences. 
                    
                    // multiply the occurrence for the current position by 10 and 
                    // divide by the most occurrences rouded up or down.
                    m_pColorPalette->SetIndex(
                        std::round((double)m_heatMap[i][j]*(m_pColorPalette->GetSize()-1) / 
                            (double)(m_mostOccurrences)));
                    
                    DSL_RGBA_RECTANGLE_PTR pRectangle = DSL_RGBA_RECTANGLE_NEW(
                        "", j*m_gridRectWidth, i*m_gridRectHeight, m_gridRectWidth, 
                        m_gridRectHeight, false, m_pColorPalette, true, m_pColorPalette);
                        
                    pRectangle->AddMeta(displayMetaData, NULL);
                }
            }
        }
    }

    void OdeHeatMapper::ClearMetrics()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        // Iterate through all rows
        for (uint i=0; i < m_rows; i++)
        {
            // and for each row, iterate through all columns.
            for (uint j=0; j < m_cols; j++)
            {
                // clear data by resetting to 0
                m_heatMap[i][j] = 0;
            }
        }
        m_mostOccurrences = 0;
    }

    void OdeHeatMapper::GetMetrics(const uint64_t** buffer, uint* size)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        int i=0;
        for (auto const& ivec: m_heatMap)
        {
            for (auto const& jvec: ivec)
            {
                m_outBuffer[i++] = jvec;
            }
        }
        *buffer = m_outBuffer.get();
        *size = m_cols * m_rows;
    }

    void OdeHeatMapper::PrintMetrics()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        uint charwidth = (m_mostOccurrences)
            ? floor(log10(m_mostOccurrences)) + 2
            : 2;
        
        for (auto const& ivec: m_heatMap)
        {
            std::stringstream ss;
            for (auto const& jvec: ivec)
            {
                ss << std::setw(charwidth) << std::setfill(' ') << jvec;
            }
            std::cout << ss.str();
            std::cout << std::endl;
        }
    }

    void OdeHeatMapper::LogMetrics()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        uint charwidth = (m_mostOccurrences)
            ? floor(log10(m_mostOccurrences)) + 2
            : 2;

        for (auto const& ivec: m_heatMap)
        {
            std::stringstream ss;
            for (auto const& jvec: ivec)
            {
                ss << std::setw(charwidth) << std::setfill(' ') << jvec;
            }
            LOG_INFO(ss.str());
        }
    }

    bool OdeHeatMapper::FileMetrics(const char* filePath, uint mode, uint format)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        std::fstream ostream;

        try
        {
            if (mode == DSL_WRITE_MODE_APPEND)
            {
                ostream.open(filePath, std::fstream::out | std::fstream::app);
            }
            else
            {
                ostream.open(filePath, std::fstream::out | std::fstream::trunc);
            }
        }
        catch(...) 
        {
            LOG_ERROR("OdeHeatMapper '" << GetName() 
                << "' failed to open file to write metrics");
            return false;
        }

        uint charwidth = (m_mostOccurrences)
            ? floor(log10(m_mostOccurrences)) + 2
            : 2;
    
        if ( format == DSL_EVENT_FILE_FORMAT_TEXT)
        {
            char dateTime[DATE_BUFF_LENGTH] = {0};
            time_t seconds = time(NULL);
            struct tm currentTm;
            localtime_r(&seconds, &currentTm);

            strftime(dateTime, DATE_BUFF_LENGTH, "%a, %d %b %Y %H:%M:%S %z", 
                &currentTm);
            std::string dateTimeStr(dateTime);
            
            ostream << "-------------------------------------------------------------------" << "\n";
            ostream << " File opened: " << dateTimeStr.c_str() << "\n";
            ostream << "-------------------------------------------------------------------" << "\n";
            
            for (auto const& ivec: m_heatMap)
            {
                for (auto const& jvec: ivec)
                {
                    ostream << std::setw(charwidth) << std::setfill(' ') << jvec;
                }
                ostream << std::endl;
            }
        }
        else
        {
            for (auto const& ivec: m_heatMap)
            {
                for (auto const& jvec: ivec)
                {
                    ostream << jvec << ",";
                }
                ostream << std::endl;
            }
        }
        ostream.close();
        
        return true;
    }

    void OdeHeatMapper::getCoordinate(NvDsObjectMeta* pObjectMeta, 
        dsl_coordinate& mapCoordinate)
    {
        switch (m_bboxTestPoint)
        {
        case DSL_BBOX_POINT_CENTER :
            mapCoordinate.x = round(pObjectMeta->rect_params.left + 
                pObjectMeta->rect_params.width/2);
            mapCoordinate.y = round(pObjectMeta->rect_params.top + 
                pObjectMeta->rect_params.height/2);
            break;
        case DSL_BBOX_POINT_NORTH_WEST :
            mapCoordinate.x = round(pObjectMeta->rect_params.left);
            mapCoordinate.y = round(pObjectMeta->rect_params.top);
            break;
        case DSL_BBOX_POINT_NORTH :
            mapCoordinate.x = round(pObjectMeta->rect_params.left + 
                pObjectMeta->rect_params.width/2);
            mapCoordinate.y = round(pObjectMeta->rect_params.top);
            break;
        case DSL_BBOX_POINT_NORTH_EAST :
            mapCoordinate.x = round(pObjectMeta->rect_params.left + 
                pObjectMeta->rect_params.width);
            mapCoordinate.y = round(pObjectMeta->rect_params.top);
            break;
        case DSL_BBOX_POINT_EAST :
            mapCoordinate.x = round(pObjectMeta->rect_params.left + 
                pObjectMeta->rect_params.width);
            mapCoordinate.y = round(pObjectMeta->rect_params.top + 
                pObjectMeta->rect_params.height/2);
            break;
        case DSL_BBOX_POINT_SOUTH_EAST :
            mapCoordinate.x = round(pObjectMeta->rect_params.left + 
                pObjectMeta->rect_params.width);
            mapCoordinate.y = round(pObjectMeta->rect_params.top + 
                pObjectMeta->rect_params.height);
            break;
        case DSL_BBOX_POINT_SOUTH :
            mapCoordinate.x = round(pObjectMeta->rect_params.left + 
                pObjectMeta->rect_params.width/2);
            mapCoordinate.y = round(pObjectMeta->rect_params.top + 
                pObjectMeta->rect_params.height);
            break;
        case DSL_BBOX_POINT_SOUTH_WEST :
            mapCoordinate.x = round(pObjectMeta->rect_params.left);
            mapCoordinate.y = round(pObjectMeta->rect_params.top + 
                pObjectMeta->rect_params.height);
            break;
        case DSL_BBOX_POINT_WEST :
            mapCoordinate.x = round(pObjectMeta->rect_params.left);
            mapCoordinate.y = round(pObjectMeta->rect_params.top + 
                pObjectMeta->rect_params.height/2);
            break;
        default:
            LOG_ERROR("Invalid DSL_BBOX_POINT = '" << m_bboxTestPoint 
                << "' for Heat-Mapper");
            throw;
        }          
    }
    
}