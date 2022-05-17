/*
The MIT License

Copyright (c) 2019-2022, Prominence AI, Inc.

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

#ifndef _DSL_ODE_HEAT_MAPPER_H
#define _DSL_ODE_HEAT_MAPPER_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslOdeBase.h"
#include "DslDisplayTypes.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_ODE_HEAT_MAPPER_PTR std::shared_ptr<OdeHeatMapper>
    #define DSL_ODE_HEAT_MAPPER_NEW(name, rows, cols, bboxTestPoint, pColorPalette) \
        std::shared_ptr<OdeHeatMapper>(new OdeHeatMapper(name, \
            rows, cols, bboxTestPoint, pColorPalette))
    
    // ********************************************************************

    class OdeHeatMapper : public OdeBase
    {
    public: 
    
        /**
         * @brief ctor for the ODE Heat Mapper class
         * @param[in] name unique name for the ODE Heat Mapper
         * @param[in] cols number of columns along the horizontal axis
         * screen width / cols = width of heat map square.
         * @param[in] rows number of rows along the vertical axis
         * screen height / rows = height of heat map square.
         * @param[in]
         */
        OdeHeatMapper(const char* name, uint cols, uint rows,
            uint bboxTestPoint, DSL_RGBA_COLOR_PALETTE_PTR pColorPalette);

        /**
         * @brief dtor for the ODE virtual base class
         */
        ~OdeHeatMapper();
        
        /**
         * @brief Handles the ODE occurrence by updating the heat-map with new 
         * the bounding box center point provided by pObjectMeta,  
         * @param[in] pFrameMeta pointer to the Frame Meta data for the current frame
         * @param[in] pObjectMeta pointer to Object Meta that that triggered the event
         */
        void HandleOccurrence(NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief and adds the heat-map's display-metadata to displayMetaData for
         * downstream display.
         * @param[in] displayMetaData Vector of metadata structures to add the 
         * heat-map's display-metadata to.
         */
        void AddDisplayMeta(std::vector<NvDsDisplayMeta*>& displayMetaData);
        
        /**
         * @brief Resets the OdeHeatMapper which clears the 2D m_heatMap vector.
         */
        void Reset();
        
        /**
         * @brief Dumps the 2D m_heatMap vector to the console.
         */
        void Dump(); 
        
    private:
    
        /**
         * @brief returs x,y coordinates from an Object's bbox coordinates
         * and size as determined by the bboxTextPoint
         * @param[in] pObjectMeta object metadata with bbox data
         * @param[out] mapCoordinate x,y coordinate structure
         */
        void getCoordinate(NvDsObjectMeta* pObjectMeta, 
            dsl_coordinate& mapCoordinate);
    
        /**
         * @brief number of columns along the horizontal axis
         * screen width / cols = width of heat map square.
         */
        uint m_cols;

        /**
         * @brief number of rows along the vertical axis
         * screen height / rows = height of heat map square.
         */
        uint m_rows;
        
        /**
         * @brief width of the grid rectangles in pixels.
         */
        uint m_gridRectWidth;

        /**
         * @brief height of the grid rectangles in pixels.
         */
        uint m_gridRectHeight;
        
        /**
         * @brief one of DSL_BBOX_POINT values defining which point of a
         * object's bounding box to use as map coordinates.
         */
        uint m_bboxTestPoint;

        /**
         * @brief shared pointer to a RGBA Color Palette to color
         * the heat map.
         */
        DSL_RGBA_COLOR_PALETTE_PTR m_pColorPalette;
        
        /**
         * @brief two dimensional vector sized cols x rows
         */
        std::vector<std::vector<uint64_t>> m_heatMap;
        
        /**
         * @brief running count of total occurrence added to the heatmap.
         */
        uint64_t m_totalOccurrences;
        
        /**
         * @brief the most occurrences in any one map location..
         */
        uint64_t m_mostOccurrences;
    };
}

#endif // _DSL_ODE_ACTION_H
    