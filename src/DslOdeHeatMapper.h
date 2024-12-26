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

#ifndef _DSL_ODE_HEAT_MAPPER_H
#define _DSL_ODE_HEAT_MAPPER_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslDeBase.h"
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

    class OdeHeatMapper : public DeBase
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
         * @brief Gets the current Color Palette in use
         * @return shared pointer to the RGBA Color Palette in use
         */
        DSL_RGBA_COLOR_PALETTE_PTR GetColorPalette();
        
        /**
         * @brief Sets the Color Palette to use for the OdeHeatMapper.
         * @return true if the color Palette could be updated
         */
        bool SetColorPalette(DSL_RGBA_COLOR_PALETTE_PTR pColorPalette);
        
        /**
         * @brief Gets the current Legend Settings for the OdeHeatMapper.
         * @param[in] enabled true if display is enabled, false otherwise.
         * @param[out] location one of the DSL_HEAT_MAP_LEGEND_LOCATION_* contants.
         * @param[out] width width of each legend entry in units of grid rectangles.
         * @param[out] height height of each legend entry in units of grid rectangles.
         */
        void GetLegendSettings(bool* enabled, uint* location, 
            uint* width, uint* height);

        /**
         * @brief Gets the current Legend Settings for the OdeHeatMapper.
         * @param[in] enabled set to true to display, false to disable.
         * @param[in] location one of the DSL_HEAT_MAP_LEGEND_LOCATION_* contants.
         * @param[in] width width of each legend entry in units of grid rectangles.
         * @param[in] height height of each legend entry in units of grid rectangles.
         * @return true on successful update, false otherwise
         */
        bool SetLegendSettings(bool enabled, uint location, 
            uint width, uint height);
        
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
        void ClearMetrics();
        
        /**
         * @brief Gets the 2D m_heatMap vector as a linear buffer.
         * @param[out] buffer pointer to the returned buffer
         * @param[out] size of the return buffer m_cols*m_rows
         */
        void GetMetrics(const uint64_t** buffer, uint* size); 

        /**
         * @brief Prints the 2D m_heatMap vector to the console.
         */
        void PrintMetrics(); 
        
        /**
         * @brief Logs the 2D m_heatMap vector at level = INFO.
         */
        void LogMetrics(); 
        
        /**
         * @brief Writes the 2D m_heatMap vector to a file.
         * @param[in] relative or absolute path to the file to write to.
         * @param[in] mode file open/write mode, one of DSL_EVENT_FILE_MODE_* options
         * @param[in] format one of the DSL_EVENT_FILE_FORMAT_* options
         */
        bool FileMetrics(const char* filePath, uint mode, uint format); 
        
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
         * @brief a linear array of heat-map metrics updated on
         * on call to get metrics and returned to the caller.
         */
        std::unique_ptr<uint64_t[]> m_outBuffer;
        
        /**
         * @brief the most occurrences in any one map location..
         */
        uint64_t m_mostOccurrences;
        
        /**
         * @brief true if Legend display is enabled, false otherwise.
         */
        bool m_legendEnabled;
        
        /**
         * @brief one of the DSL_HEAT_MAP_LEGEND_LOCATION_* contants.
         */
        uint m_legendLocation;
        
        /**
         * @brief left position of the legend in units of grid rectangles.
         */
        uint m_legendLeft;
        
        /**
         * @brief top position of the legend in units of grid rectangles.
         */
        uint m_legendTop;
        
        /**
         * @brief width of each legend entry in units of grid rectangles.
         */
        uint m_legendWidth; 

        /**
         * @brief height of each legend entry in units of grid rectangles.
         */
        uint m_legendHeight;
    };
}

#endif // _DSL_ODE_ACTION_H
    