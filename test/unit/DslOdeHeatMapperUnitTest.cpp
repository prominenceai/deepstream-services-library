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

#include "catch.hpp"
#include "DslOdeHeatMapper.h"
#include "DslServices.h"

using namespace DSL;

static std::vector<NvDsDisplayMeta*> displayMetaData;

SCENARIO( "A new OdeHeatMapper is created correctly", "[OdeHeatMapper]" )
{
    GIVEN( "Attributes for a new HeatMapper" ) 
    {
        std::string colorPaletteName("color-palette");
        std::string odeHeatMapperName("accumulator");
        uint cols(16), rows(9);
        double alpha(0.2);
        
        std::shared_ptr<std::vector<DSL_RGBA_COLOR_PTR>> pColorPalette = 
            std::shared_ptr<std::vector<DSL_RGBA_COLOR_PTR>>{
                new std::vector<DSL_RGBA_COLOR_PTR>};
        
        for (auto const& ivec: RgbaPredefinedColor::s_predefinedColorPalettes[
            DSL_COLOR_PREDEFINED_PALETTE_SPECTRAL])
        {
            DSL_RGBA_COLOR_PTR pColor = std::shared_ptr<RgbaColor>
                (new RgbaColor("", ivec));
            pColor->alpha = alpha;
            
            pColorPalette->push_back(pColor);
        }
        
        DSL_RGBA_COLOR_PALETTE_PTR pPredefinedColorPalette = 
            DSL_RGBA_COLOR_PALETTE_NEW(colorPaletteName.c_str(), pColorPalette);

        WHEN( "A new HeatMapper is created" )
        {
            DSL_ODE_HEAT_MAPPER_PTR pOdeHeatMapper = 
                DSL_ODE_HEAT_MAPPER_NEW(odeHeatMapperName.c_str(), 
                    rows, cols, DSL_BBOX_POINT_SOUTH, pPredefinedColorPalette);

            THEN( "The OdeAccumulator's memebers are setup and returned correctly" )
            {
                REQUIRE( pOdeHeatMapper->GetName() == odeHeatMapperName );
            }
        }
    }
}

SCENARIO( "A new OdeHeatMapper can HandleOccurrence correctly", "[OdeHeatMapper]" )
{
    GIVEN( "A new HeatMapper in memory" ) 
    {
        std::string colorPaletteName("color-palette");
        std::string odeHeatMapperName("accumulator");
        uint cols(16), rows(9);
        double alpha(0.2);
        
        std::shared_ptr<std::vector<DSL_RGBA_COLOR_PTR>> pColorPalette = 
            std::shared_ptr<std::vector<DSL_RGBA_COLOR_PTR>>{
                new std::vector<DSL_RGBA_COLOR_PTR>};
        
        for (auto const& ivec: RgbaPredefinedColor::s_predefinedColorPalettes[
            DSL_COLOR_PREDEFINED_PALETTE_SPECTRAL])
        {
            DSL_RGBA_COLOR_PTR pColor = std::shared_ptr<RgbaColor>
                (new RgbaColor("", ivec));
            pColor->alpha = alpha;
            
            pColorPalette->push_back(pColor);
        }
        
        DSL_RGBA_COLOR_PALETTE_PTR pPredefinedColorPalette = 
            DSL_RGBA_COLOR_PALETTE_NEW(colorPaletteName.c_str(), pColorPalette);

        DSL_ODE_HEAT_MAPPER_PTR pOdeHeatMapper = 
            DSL_ODE_HEAT_MAPPER_NEW(odeHeatMapperName.c_str(), 
                cols, rows, DSL_BBOX_POINT_SOUTH, pPredefinedColorPalette);

        WHEN( "The OdeHeatMapper is called to HandleOccurrence" )
        {
            NvDsFrameMeta frameMeta =  {0};
            frameMeta.source_frame_width = DSL_STREAMMUX_DEFAULT_WIDTH;
            frameMeta.source_frame_height = DSL_STREAMMUX_DEFAULT_HEIGHT;

            NvDsObjectMeta objectMeta = {0};
            objectMeta.rect_params.left = 10;
            objectMeta.rect_params.top = 10;
            objectMeta.rect_params.width = 20;
            objectMeta.rect_params.height = 20;
            
            pOdeHeatMapper->HandleOccurrence(&frameMeta, &objectMeta);

            objectMeta.rect_params.left = DSL_STREAMMUX_DEFAULT_WIDTH-30;
            objectMeta.rect_params.top = DSL_STREAMMUX_DEFAULT_HEIGHT-30;
            
            pOdeHeatMapper->HandleOccurrence(&frameMeta, &objectMeta);

            THEN( "The OdeHeatMaper's heat-map is udated correctly" )
            {
                pOdeHeatMapper->PrintMetrics();
            }
        }
    }
}

SCENARIO( "A new OdeHeatMapper can clear metrics correctly", "[OdeHeatMapper]" )
{
    GIVEN( "A new HeatMapper in memory" ) 
    {
        std::string colorPaletteName("color-palette");
        std::string odeHeatMapperName("accumulator");
        uint cols(16), rows(9);
        double alpha(0.2);
        
        std::shared_ptr<std::vector<DSL_RGBA_COLOR_PTR>> pColorPalette = 
            std::shared_ptr<std::vector<DSL_RGBA_COLOR_PTR>>{
                new std::vector<DSL_RGBA_COLOR_PTR>};
        
        for (auto const& ivec: RgbaPredefinedColor::s_predefinedColorPalettes[
            DSL_COLOR_PREDEFINED_PALETTE_SPECTRAL])
        {
            DSL_RGBA_COLOR_PTR pColor = std::shared_ptr<RgbaColor>
                (new RgbaColor("", ivec));
            pColor->alpha = alpha;
            
            pColorPalette->push_back(pColor);
        }
        
        DSL_RGBA_COLOR_PALETTE_PTR pPredefinedColorPalette = 
            DSL_RGBA_COLOR_PALETTE_NEW(colorPaletteName.c_str(), pColorPalette);

        DSL_ODE_HEAT_MAPPER_PTR pOdeHeatMapper = 
            DSL_ODE_HEAT_MAPPER_NEW(odeHeatMapperName.c_str(), 
                cols, rows, DSL_BBOX_POINT_SOUTH, pPredefinedColorPalette);

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.source_frame_width = DSL_STREAMMUX_DEFAULT_WIDTH;
        frameMeta.source_frame_height = DSL_STREAMMUX_DEFAULT_HEIGHT;

        NvDsObjectMeta objectMeta = {0};
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 20;
        objectMeta.rect_params.height = 20;
        
        pOdeHeatMapper->HandleOccurrence(&frameMeta, &objectMeta);

        WHEN( "The OdeHeatMapper is called to Clear metrics" )
        {
            pOdeHeatMapper->PrintMetrics();
            pOdeHeatMapper->ClearMetrics();

            THEN( "The OdeHeatMaper's heat-map is cleared" )
            {
                pOdeHeatMapper->PrintMetrics();
            }
        }
    }
}

SCENARIO( "A new OdeHeatMapper can Get metrics correctly", "[OdeHeatMapper]" )
{
    GIVEN( "A new HeatMapper in memory" ) 
    {
        std::string colorPaletteName("color-palette");
        std::string odeHeatMapperName("accumulator");
        uint cols(16), rows(9);
        double alpha(0.2);
        
        std::shared_ptr<std::vector<DSL_RGBA_COLOR_PTR>> pColorPalette = 
            std::shared_ptr<std::vector<DSL_RGBA_COLOR_PTR>>{
                new std::vector<DSL_RGBA_COLOR_PTR>};
        
        for (auto const& ivec: RgbaPredefinedColor::s_predefinedColorPalettes[
            DSL_COLOR_PREDEFINED_PALETTE_SPECTRAL])
        {
            DSL_RGBA_COLOR_PTR pColor = std::shared_ptr<RgbaColor>
                (new RgbaColor("", ivec));
            pColor->alpha = alpha;
            
            pColorPalette->push_back(pColor);
        }
        
        DSL_RGBA_COLOR_PALETTE_PTR pPredefinedColorPalette = 
            DSL_RGBA_COLOR_PALETTE_NEW(colorPaletteName.c_str(), pColorPalette);

        DSL_ODE_HEAT_MAPPER_PTR pOdeHeatMapper = 
            DSL_ODE_HEAT_MAPPER_NEW(odeHeatMapperName.c_str(), 
                cols, rows, DSL_BBOX_POINT_SOUTH, pPredefinedColorPalette);

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.source_frame_width = DSL_STREAMMUX_DEFAULT_WIDTH;
        frameMeta.source_frame_height = DSL_STREAMMUX_DEFAULT_HEIGHT;

        NvDsObjectMeta objectMeta = {0};
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 20;
        objectMeta.rect_params.height = 20;
        
        pOdeHeatMapper->HandleOccurrence(&frameMeta, &objectMeta);

        objectMeta.rect_params.left = DSL_STREAMMUX_DEFAULT_WIDTH - 30;
        objectMeta.rect_params.top = DSL_STREAMMUX_DEFAULT_HEIGHT - 30;
        objectMeta.rect_params.width = 20;
        objectMeta.rect_params.height = 20;
        
        pOdeHeatMapper->HandleOccurrence(&frameMeta, &objectMeta);

        WHEN( "The OdeHeatMapper is called to Get metrics" )
        {
            const uint64_t* outBuffer;
            uint size;
            
            pOdeHeatMapper->GetMetrics(&outBuffer, &size);

            THEN( "The returned buffer values are correct" )
            {
                REQUIRE( size == cols*rows );
                for (int i=0; i<size;i++)
                {
                    std::cout << outBuffer[i];
                }
                std::cout << std::endl;
                REQUIRE( outBuffer[0] == 1 );
                REQUIRE( outBuffer[1] == 0 );
                REQUIRE( outBuffer[size-2] == 0 );
                REQUIRE( outBuffer[size-1] == 1 );
            }
        }
    }
}

SCENARIO( "A new OdeHeatMapper can File metrics correctly", "[OdeHeatMapper]" )
{
    GIVEN( "A new HeatMapper in memory" ) 
    {
        std::string colorPaletteName("color-palette");
        std::string odeHeatMapperName("accumulator");
        uint cols(16), rows(9);
        double alpha(0.2);
        
        std::string textFileName("./heat-map.txt");
        std::string csvFileName("./heat-map.csv");
        
        std::shared_ptr<std::vector<DSL_RGBA_COLOR_PTR>> pColorPalette = 
            std::shared_ptr<std::vector<DSL_RGBA_COLOR_PTR>>{
                new std::vector<DSL_RGBA_COLOR_PTR>};
        
        for (auto const& ivec: RgbaPredefinedColor::s_predefinedColorPalettes[
            DSL_COLOR_PREDEFINED_PALETTE_SPECTRAL])
        {
            DSL_RGBA_COLOR_PTR pColor = std::shared_ptr<RgbaColor>
                (new RgbaColor("", ivec));
            pColor->alpha = alpha;
            
            pColorPalette->push_back(pColor);
        }
        
        DSL_RGBA_COLOR_PALETTE_PTR pPredefinedColorPalette = 
            DSL_RGBA_COLOR_PALETTE_NEW(colorPaletteName.c_str(), pColorPalette);

        DSL_ODE_HEAT_MAPPER_PTR pOdeHeatMapper = 
            DSL_ODE_HEAT_MAPPER_NEW(odeHeatMapperName.c_str(), 
                cols, rows, DSL_BBOX_POINT_SOUTH, pPredefinedColorPalette);

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.source_frame_width = DSL_STREAMMUX_DEFAULT_WIDTH;
        frameMeta.source_frame_height = DSL_STREAMMUX_DEFAULT_HEIGHT;

        NvDsObjectMeta objectMeta = {0};
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 20;
        objectMeta.rect_params.height = 20;
        
        WHEN( "The OdeHeatMapper heat-map is updated" )
        {
            pOdeHeatMapper->HandleOccurrence(&frameMeta, &objectMeta);

            THEN( "The OdeHeatMaper's heat-map can be written to a text file cleared" )
            {
                REQUIRE( pOdeHeatMapper->FileMetrics(textFileName.c_str(), 
                    DSL_WRITE_MODE_APPEND, DSL_EVENT_FILE_FORMAT_TEXT) == true );
            }
        }
        WHEN( "The OdeHeatMapper heat-map is updated" )
        {
            pOdeHeatMapper->HandleOccurrence(&frameMeta, &objectMeta);
            pOdeHeatMapper->HandleOccurrence(&frameMeta, &objectMeta);
            pOdeHeatMapper->HandleOccurrence(&frameMeta, &objectMeta);

            THEN( "New heat-map data can be appended to the same file" )
            {
                REQUIRE( pOdeHeatMapper->FileMetrics(textFileName.c_str(), 
                    DSL_WRITE_MODE_APPEND, DSL_EVENT_FILE_FORMAT_TEXT) == true );
            }
        }
        WHEN( "The OdeHeatMapper heat-map is updated" )
        {
            pOdeHeatMapper->HandleOccurrence(&frameMeta, &objectMeta);

            THEN( "The OdeHeatMaper's heat-map can be written to a text file cleared" )
            {
                REQUIRE( pOdeHeatMapper->FileMetrics(csvFileName.c_str(), 
                    DSL_WRITE_MODE_TRUNCATE, DSL_EVENT_FILE_FORMAT_CSV) == true );
            }
        }
    }
}

