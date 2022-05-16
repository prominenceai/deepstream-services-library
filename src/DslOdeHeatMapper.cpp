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
        DSL_RGBA_COLOR_PALETTE_PTR pColorPalette)
        : OdeBase(name)
        , m_cols(cols)
        , m_rows(rows)
        , m_pColorPalette(pColorPalette)
        , m_heatMap(rows, std::vector<uint64_t> (cols, 0))
        , m_totalPoints(0)
        , m_mostPoints(0)
    {
        LOG_FUNC();
    }

    OdeHeatMapper::~OdeHeatMapper()
    {
        LOG_FUNC();
    }

    void OdeHeatMapper::HandleOccurrence(NvDsFrameMeta* pFrameMeta, 
        NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        uint rectWidth(pFrameMeta->source_frame_width/m_cols);
        uint rectHeight(pFrameMeta->source_frame_height/m_rows);
        
        uint cx(pObjectMeta->rect_params.left + pObjectMeta->rect_params.width/2);
        uint cy(pObjectMeta->rect_params.top + pObjectMeta->rect_params.height/2);
        
        uint colPosition(cx/rectWidth);
        uint rowPosition(cy/rectHeight);
        
        m_heatMap[rowPosition][colPosition] += 1;
        if (m_heatMap[rowPosition][colPosition] > m_mostPoints)
        {
            m_mostPoints = m_heatMap[rowPosition][colPosition];
        }
    }
  
    void OdeHeatMapper::AddDisplayMeta(std::vector<NvDsDisplayMeta*>& displayMetaData)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
    }
    
    void OdeHeatMapper::Reset()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        for (uint j=0; j < m_rows; j++)
        {
            for (uint i=0; i < m_cols; i++)
            {
                m_heatMap[j][i] = 0;
            }
        }
    }
    
    void OdeHeatMapper::Dump()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        for (auto const& ivec: m_heatMap)
        {
            for (auto const& jvec: ivec)
            {
                std::stringstream ss;
                ss << std::setw(7) << std::setfill(' ') << jvec;
                std::cout << ss.str();
            }
            std::cout << std::endl;
        }
    }
    
}