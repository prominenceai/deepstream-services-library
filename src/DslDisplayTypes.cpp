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

#include "DslDisplayTypes.h"
#include "DslServices.h"

namespace DSL
{

//            name='Black', red=0, green= 0, blue=0, 
//            name='Gray-50%', red=127, green= 127, blue=127, 
//            name='Dark-red', red=136, green= 0, blue=21, 
//            name='Red', red=237, green= 28, blue=36, 
//            name='Orange', red=255, green= 127, blue=39, 
//            name='Yellow', red=255, green=242, blue=0, 
//            name='Green', red=34, green=177, blue=76, 
//            name='Turquoise', red=0, green=162, blue=232, 
//            name='Indigo', red=63, green=72, blue=204, 
//            name='Purple', red=163, green=73, blue=164, 
//            name='White', red=255, green=255, blue=255, 
//            name='Gray-25', red=195, green=195, blue=195, 
//            name='Brown', red=185, green=122, blue=87, 
//            name='Rose', red=255, green=174, blue=201, 
//            name='Gold', red=255, green=201, blue=14, 
//            name='Light yellow', red=239, green=228, blue=176, 
//            name='Lime', red=181, green= 230, blue=29, 
//            name='Light turquoise', red=153, green= 217, blue=234, 
//            name='Blue-gray', red=112, green=146, blue=190, 
//            name='Lavender', red=200, green=191, blue=231, 

    std::map<uint, NvOSD_ColorParams> RgbaPredefinedColor::s_predefinedColors = {
        {DSL_COLOR_PREDEFINED_BLACK,            {0.0,   0.0,   0.0,   0.0}},
        {DSL_COLOR_PREDEFINED_GRAY_50,          {0.498, 0.498, 0.498, 0.0}},
        {DSL_COLOR_PREDEFINED_DARK_RED,         {0.533, 0.0,   0.082, 0.0}},
        {DSL_COLOR_PREDEFINED_RED,              {0.929, 0.010, 0.141, 0.0}},
        {DSL_COLOR_PREDEFINED_ORANGE,           {1.0,   0.498, 0.153, 0.0}},
        {DSL_COLOR_PREDEFINED_YELLOW,           {1.0,   0.949, 0.0,   0.0}},
        {DSL_COLOR_PREDEFINED_GREEN,            {0.133, 0.694, 0.298, 0.0}},
        {DSL_COLOR_PREDEFINED_TURQUOISE,        {0.0,   0.635, 0.910, 0.0}},
        {DSL_COLOR_PREDEFINED_INDIGO,           {0.247, 0.282, 0.8,   0.0}},
        {DSL_COLOR_PREDEFINED_PURPLE,           {0.639, 0.286, 0.643, 0.0}},
        {DSL_COLOR_PREDEFINED_WHITE,            {1.0,   1.0,   1.0,   0.0}},
        {DSL_COLOR_PREDEFINED_GRAY_25,          {0.765, 0.765, 0.765, 0.0}},
        {DSL_COLOR_PREDEFINED_BROWN,            {0.725, 0.478, 0.341, 0.0}},
        {DSL_COLOR_PREDEFINED_ROSE,             {1.0,   0.682, 0.682, 0.0}},
        {DSL_COLOR_PREDEFINED_GOLD,             {1.0,   0.788, 0.055, 0.0}},
        {DSL_COLOR_PREDEFINED_LIGHT_YELLOW,     {0.937, 0.894, 0.690, 0.0}},
        {DSL_COLOR_PREDEFINED_LIME,             {0.710, 0.902, 0.114, 0.0}},
        {DSL_COLOR_PREDEFINED_LIGHT_TURQUOISE,  {0.6,   0.851, 0.918, 0.0}},
        {DSL_COLOR_PREDEFINED_BLUE_GRAY,        {0.439, 0.573, 0.745, 0.0}},
        {DSL_COLOR_PREDEFINED_LAVENDER,         {0.784, 0.749, 0.906, 0.0}}
    };

    std::map<uint, std::vector<NvOSD_ColorParams>> 
        RgbaPredefinedColor::s_predefinedColorPalettes = {
            {DSL_COLOR_PREDEFINED_PALETTE_SPECTRAL, 
                {
                    {0.369, 0.310, 0.635, 0.0},
                    {0.196, 0.533, 0.741, 0.0},
                    {0.400, 0.761, 0.647, 0.0},
                    {0.671, 0.867, 0.643, 0.0},
                    {0.902, 0.961, 0.596, 0.0},
                    {0.945, 0.953, 0.655, 0.0},
                    {0.996, 0.878, 0.565, 0.0},
                    {0.992, 0.682, 0.380, 0.0},
                    {0.957, 0.427, 0.263, 0.0},
                    {0.835, 0.243, 0.310, 0.0},
                    {0.620, 0.004, 0.259, 0.0}
                }
            },
            {DSL_COLOR_PREDEFINED_PALETTE_RED,
                {
                    {0.404, 0.000, 0.051, 0.0},
                    {0.647, 0.059, 0.082, 0.0},
                    {0.796, 0.094, 0.114, 0.0},
                    {0.937, 0.231, 0.173, 0.0},
                    {0.984, 0.416, 0.290, 0.0},
                    {0.988, 0.573, 0.447, 0.0},
                    {0.988, 0.733, 0.631, 0.0},
                    {0.996, 0.878, 0.824, 0.0},
                    {1.000, 0.961, 0.941, 0.0}
                }
            },
            {DSL_COLOR_PREDEFINED_PALETTE_GREEN,
                {
                    {0.000, 0.267, 0.098, 0.0},
                    {0.000, 0.427, 0.173, 0.0},
                    {0.137, 0.545, 0.271, 0.0},
                    {0.267, 0.670, 0.365, 0.0},
                    {0.455, 0.769, 0.463, 0.0},
                    {0.631, 0.851, 0.608, 0.0},
                    {0.780, 0.914, 0.753, 0.0},
                    {0.898, 0.961, 0.878, 0.0},
                    {0.969, 0.988, 0.961, 0.0}
                }
            },
            {DSL_COLOR_PREDEFINED_PALETTE_BLUE, 
                {
                    {0.032, 0.188, 0.420, 0.0},
                    {0.032, 0.318, 0.612, 0.0},
                    {0.129, 0.522, 0.710, 0.0},
                    {0.259, 0.573, 0.776, 0.0},
                    {0.420, 0.682, 0.839, 0.0},
                    {0.620, 0.792, 1.000, 0.0},
                    {0.776, 0.859, 0.937, 0.0},
                    {0.871, 0.921, 0.969, 0.0},
                    {0.969, 0.985, 1.000, 0.0}
                }
            },
            {DSL_COLOR_PREDEFINED_PALETTE_GREY, 
                {
                    {0.100, 0.100, 0.100, 0.0},
                    {0.200, 0.200, 0.200, 0.0},
                    {0.300, 0.300, 0.300, 0.0},
                    {0.400, 0.400, 0.400, 0.0},
                    {0.500, 0.500, 0.500, 0.0},
                    {0.600, 0.600, 0.600, 0.0},
                    {0.700, 0.700, 0.700, 0.0},
                    {0.800, 0.800, 0.800, 0.0},
                    {0.900, 0.900, 0.900, 0.0},
                    {1.000, 1.000, 1.000, 0.0}
                }
            }
        };

 
   // ********************************************************************

    
    DisplayType::DisplayType(const char* name)
        : Base(name)
    {
        LOG_FUNC();
        
        g_mutex_init(&m_propertyMutex);
    }

    DisplayType::~DisplayType()
    {
        LOG_FUNC();

        g_mutex_clear(&m_propertyMutex);
    }

    inline void DisplayType::Lock()
    {
        g_mutex_lock(&m_propertyMutex);        
    }

    inline void DisplayType::Unlock()
    {
        g_mutex_unlock(&m_propertyMutex);        
    }
    
    void DisplayType::AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta) 
    {
        LOG_FUNC();
        LOG_ERROR("Base Display Type can not be overlaid");
    }
    
    // ********************************************************************

    RgbaColor::RgbaColor(const char* name, 
        double red, double green, double blue, double alpha)
        : DisplayType(name)
        , NvOSD_ColorParams{red, green, blue, alpha}
    {
        LOG_FUNC();
    }
    
    RgbaColor::RgbaColor(const RgbaColor& color)
        : DisplayType("")
        , NvOSD_ColorParams{color.red, color.green, color.blue, color.alpha}
    {
        LOG_FUNC();
    }

    RgbaColor::RgbaColor()
        : DisplayType("")
        , NvOSD_ColorParams{0.0, 0.0, 0.0, 0.0}
    {
        LOG_FUNC();
    }

    RgbaColor::RgbaColor(const char* name, const NvOSD_ColorParams& color)
        : DisplayType(name)
        , NvOSD_ColorParams{color.red, color.green, color.blue, color.alpha}
    {
        LOG_FUNC();
    }

    RgbaColor::~RgbaColor()
    {
        LOG_FUNC();
    }
    
    // ********************************************************************

    RgbaPredefinedColor::RgbaPredefinedColor(const char* name,
        uint colorId, double alpha)
        : RgbaColor(name, s_predefinedColors[colorId])
        , m_colorId(colorId)
    {
        LOG_FUNC();
        
        this->alpha = alpha;
    }

    RgbaPredefinedColor::~RgbaPredefinedColor()
    {
        LOG_FUNC();
    }
    
    // ********************************************************************

    RgbaColorPalette::RgbaColorPalette(const char* name, 
        std::shared_ptr<std::vector<DSL_RGBA_COLOR_PTR>> pColorPalette)
        : RgbaColor(name, 0.0, 0.0, 0.0, 0.0)
        , m_pColorPalette(pColorPalette)
        , m_currentColorIndex(0)
    {
        LOG_FUNC();

        SetNext();
    }

    RgbaColorPalette::~RgbaColorPalette()
    {
        LOG_FUNC();
    }
    
    void RgbaColorPalette::SetNext()
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        // don't log function entry/exit
        red = m_pColorPalette->at(m_currentColorIndex)->red;
        green = m_pColorPalette->at(m_currentColorIndex)->green;
        blue = m_pColorPalette->at(m_currentColorIndex)->blue;
        alpha = m_pColorPalette->at(m_currentColorIndex)->alpha;
        m_currentColorIndex = (m_currentColorIndex+1)%m_pColorPalette->size();
    }

    uint RgbaColorPalette::GetIndex()
    {
        LOG_FUNC();
        
        return m_currentColorIndex;
    }
    
    bool RgbaColorPalette::SetIndex(uint index)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (index >= m_pColorPalette->size())
        {
            LOG_ERROR("Invalid index = " << index 
                << " for RGBA Color Palette " << GetName() 
                << " with palette size = " << m_pColorPalette->size());
            return false;
        }
        m_currentColorIndex = index;
        red = m_pColorPalette->at(m_currentColorIndex)->red;
        green = m_pColorPalette->at(m_currentColorIndex)->green;
        blue = m_pColorPalette->at(m_currentColorIndex)->blue;
        alpha = m_pColorPalette->at(m_currentColorIndex)->alpha;
        return true;
    }
    
    uint RgbaColorPalette::GetSize()
    {
        return m_pColorPalette->size();
    }
    
    // ********************************************************************

    RgbaRandomColor::RgbaRandomColor(const char* name,
        uint hue, uint luminosity, double alpha, uint seed)
        : RgbaColor(name, 0.0, 0.0, 0.0, alpha)
        , m_hue((RandomColor::Color)hue)
        , m_luminosity((RandomColor::Luminosity)luminosity)
    {
        LOG_FUNC();
        
        m_randomColor.setSeed(seed);
        SetNext();
    }

    RgbaRandomColor::~RgbaRandomColor()
    {
        LOG_FUNC();
    }
    
    void RgbaRandomColor::SetNext()
    {
        // don't log function entry/exit
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        int color(m_randomColor.generate(m_hue, m_luminosity));

        red = ((color >> 16) & 0xFF) / 255.0;
        green = ((color >> 8) & 0xFF) / 255.0;
        blue = ((color) & 0xFF) / 255.0;
    }

    
    // ********************************************************************

    RgbaOnDemandColor::RgbaOnDemandColor(const char* name, 
        dsl_display_type_rgba_color_provider_cb provider, void* clientData)
        : RgbaColor(name, 0.0, 0.0, 0.0, 0.0)
        , m_provider(provider)
        , m_clientData(clientData)
    {
        LOG_FUNC();
        
        SetNext();
    }

    RgbaOnDemandColor::~RgbaOnDemandColor()
    {
        LOG_FUNC();
    }
    
    void RgbaOnDemandColor::SetNext()
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (m_provider)
        {
            try
            {
                m_provider(&red, &green, &blue, &alpha, m_clientData);
            }
            catch(...)
            {
                LOG_ERROR("RGBA Color On-Demand '" << GetName() 
                    << "' threw exception calling client color provider callback");
                    
                // Disable client provider to avoid repetative error logs.
                m_provider = NULL;
            }
        }
    }
    
    // ********************************************************************

    RgbaFont::RgbaFont(const char* name, 
        const char* font, uint size, DSL_RGBA_COLOR_PTR color)
        : DisplayType(name)
        , m_fontName(font)
        , m_pColor(color)
        , NvOSD_FontParams{NULL, size, *color}
    {
        LOG_FUNC();
    }

    RgbaFont::~RgbaFont()
    {
        LOG_FUNC();
    }

    inline void RgbaFont::Lock()
    {
        m_pColor->Lock();
        font_color = *m_pColor;
    }

    inline void RgbaFont::Unlock()
    {
        m_pColor->Unlock();
    }
    
    // ********************************************************************

    RgbaText::RgbaText(const char* name, 
        const char* text, uint x_offset, uint y_offset, 
        DSL_RGBA_FONT_PTR pFont, bool hasBgColor, DSL_RGBA_COLOR_PTR pBgColor)
        : DisplayType(name)
        , m_text(text)
        , m_pFont(pFont)
        , m_pBgColor(pBgColor)
        , NvOSD_TextParams{NULL, x_offset, y_offset, 
            *pFont, hasBgColor, *pBgColor}
        , m_shadowEnabled(false)
        , m_shadowXOffset(0)
        , m_shadowYOffset(0)
    {
        LOG_FUNC();
    }

    RgbaText::~RgbaText()
    {
        LOG_FUNC();
    }
    
    inline void RgbaText::Lock()
    {
        m_pFont->Lock();
        font_params = *m_pFont;
        m_pBgColor->Lock();
        text_bg_clr = *m_pBgColor;
    }

    inline void RgbaText::Unlock()
    {
        m_pFont->Unlock();
        m_pBgColor->Unlock();
    }
    
    bool RgbaText::AddShadow(uint xOffset, uint yOffset, DSL_RGBA_COLOR_PTR pColor)
    {
        LOG_FUNC();
        
        if (m_shadowEnabled)
        {
            LOG_ERROR("Shadow is already enabled for RgbaText '" << 
                GetName() << "'");
        }
        m_shadowEnabled = true;
        m_shadowXOffset = xOffset;
        m_shadowYOffset = yOffset;
        m_pShadowColor = pColor;
        
        m_pShadowFont = DSL_RGBA_FONT_NEW("", 
            m_pFont->m_fontName.c_str(), m_pFont->font_size, m_pShadowColor);
    }
    
    void RgbaText::AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta) 
    {
//        LOG_FUNC();

        // check to see if we're adding meta data - client can disable
        // by setting the PPH ODE display meta alloc size to 0.
        // and ensure we have available space in the vector of meta structs.
        NvDsDisplayMeta* pDisplayMeta(NULL);
        for (const auto& ivec: displayMetaData)
        {
            if (ivec->num_labels < MAX_ELEMENTS_IN_DISPLAY_META)
            {
                pDisplayMeta = ivec;
                break;
            }
        }
        if (!pDisplayMeta)
        {
            return;
        }
        if (m_shadowEnabled)
        {
            NvOSD_TextParams *pTextParams = &pDisplayMeta->
                text_params[pDisplayMeta->num_labels++];
            
            
            pTextParams->x_offset = x_offset + m_shadowXOffset;
            pTextParams->y_offset = y_offset + m_shadowYOffset;
            
            pTextParams->set_bg_clr = true;
            pTextParams->text_bg_clr = *m_pShadowColor;

            // need to allocate storage for actual text, then copy.
            pTextParams->display_text = (gchar*) g_malloc0(MAX_DISPLAY_LEN);
            m_text.copy(pTextParams->display_text, MAX_DISPLAY_LEN, 0);

            pTextParams->font_params = *m_pShadowFont;
            // Font, font-size, font-color
            pTextParams->font_params.font_name = (gchar*) g_malloc0(MAX_DISPLAY_LEN);
            m_pShadowFont->m_fontName.copy(pTextParams->font_params.font_name, 
                MAX_DISPLAY_LEN, 0);
                
        }
        NvOSD_TextParams *pTextParams = &pDisplayMeta->
            text_params[pDisplayMeta->num_labels++];
        
        Lock();
        // copy over our text params, display_text currently == NULL
        *pTextParams = *this;
        Unlock();
        
        // need to allocate storage for actual text, then copy.
        pTextParams->display_text = (gchar*) g_malloc0(MAX_DISPLAY_LEN);
        m_text.copy(pTextParams->display_text, MAX_DISPLAY_LEN, 0);
        
        // Font, font-size, font-color
        pTextParams->font_params.font_name = (gchar*) g_malloc0(MAX_DISPLAY_LEN);
        m_pFont->m_fontName.copy(pTextParams->font_params.font_name, 
            MAX_DISPLAY_LEN, 0);
    }
        
    // ********************************************************************

    RgbaLine::RgbaLine(const char* name, uint x1, uint y1, uint x2, uint y2, 
        uint width, DSL_RGBA_COLOR_PTR pColor)
        : DisplayType(name)
        , m_pColor(pColor)
        , NvOSD_LineParams{x1, y1, x2, y2, width, *pColor}
    {
        LOG_FUNC();
    }

    RgbaLine::~RgbaLine()
    {
        LOG_FUNC();
    }

    void RgbaLine::AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta) 
    {
//        LOG_FUNC();

        // check to see if we're adding meta data - client can disable
        // by setting the PPH ODE display meta alloc size to 0.
        // and ensure we have available space in the vector of meta structs.
        NvDsDisplayMeta* pDisplayMeta(NULL);
        for (const auto& ivec: displayMetaData)
        {
            if (ivec->num_lines < MAX_ELEMENTS_IN_DISPLAY_META)
            {
                pDisplayMeta = ivec;
                break;
            }
        }
        if (!pDisplayMeta)
        {
            return;
        }
        m_pColor->Lock();
        line_color = *m_pColor;
        m_pColor->Unlock();

        pDisplayMeta->line_params[pDisplayMeta->num_lines++] = *this;
    }
    
    // ********************************************************************

    RgbaArrow::RgbaArrow(const char* name, uint x1, uint y1, uint x2, uint y2, 
        uint width, uint head, DSL_RGBA_COLOR_PTR pColor)
        : DisplayType(name)
        , m_pColor(pColor)
        , NvOSD_ArrowParams{x1, y1, x2, y2, width, 
            (NvOSD_Arrow_Head_Direction)head, *pColor}
    {
        LOG_FUNC();
    }

    RgbaArrow::~RgbaArrow()
    {
        LOG_FUNC();
    }

    void RgbaArrow::AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta) 
    {
//        LOG_FUNC();

        // check to see if we're adding meta data - client can disable
        // by setting the PPH ODE display meta alloc size to 0.
        // and ensure we have available space in the vector of meta structs.
        NvDsDisplayMeta* pDisplayMeta(NULL);
        for (const auto& ivec: displayMetaData)
        {
            if (ivec->num_arrows < MAX_ELEMENTS_IN_DISPLAY_META)
            {
                pDisplayMeta = ivec;
                break;
            }
        }
        if (!pDisplayMeta)
        {
            return;
        }
        m_pColor->Lock();
        arrow_color = *m_pColor;
        m_pColor->Unlock();
        pDisplayMeta->arrow_params[pDisplayMeta->num_arrows++] = *this;
    }

    // ********************************************************************

    RgbaRectangle::RgbaRectangle(const char* name, 
        uint left, uint top, uint width, uint height, 
        uint borderWidth, DSL_RGBA_COLOR_PTR pColor,
        bool hasBgColor, DSL_RGBA_COLOR_PTR pBgColor)
        : DisplayType(name)
        , m_pColor(pColor)
        , m_pBgColor(pBgColor)
        , NvOSD_RectParams{(float)left, (float)top, (float)width, (float)height, 
            borderWidth, *pColor, hasBgColor, 0, *pBgColor}
    {
        LOG_FUNC();
    }

    RgbaRectangle::~RgbaRectangle()
    {
        LOG_FUNC();
    }

    void RgbaRectangle::AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta) 
    {
//        LOG_FUNC();

        // check to see if we're adding meta data - client can disable
        // by setting the PPH ODE display meta alloc size to 0.
        // and ensure we have available space in the vector of meta structs.
        NvDsDisplayMeta* pDisplayMeta(NULL);
        for (const auto& ivec: displayMetaData)
        {
            if (ivec->num_rects < MAX_ELEMENTS_IN_DISPLAY_META)
            {
                pDisplayMeta = ivec;
                break;
            }
        }
        if (!pDisplayMeta)
        {
            return;
        }
        m_pColor->Lock();
        border_color = *m_pColor;
        m_pColor->Unlock();

        m_pBgColor->Lock();
        bg_color = *m_pBgColor;
        m_pBgColor->Unlock();
        
        pDisplayMeta->rect_params[pDisplayMeta->num_rects++] = *this;
    }
    
    // ********************************************************************

    RgbaPolygon::RgbaPolygon(const char* name, const dsl_coordinate* coordinates, 
        uint numCoordinates, uint lineWidth, DSL_RGBA_COLOR_PTR pColor)
        : DisplayType(name)
        , m_pColor(pColor)
        , dsl_polygon_params{NULL, numCoordinates, lineWidth, *pColor}
    {
        LOG_FUNC();
        
        // allocate data for coordinates and copy over values
        this->coordinates = 
            (dsl_coordinate*) g_malloc0(numCoordinates*sizeof(dsl_coordinate));
        memcpy(this->coordinates, coordinates, numCoordinates*sizeof(dsl_coordinate));
    }

    RgbaPolygon::~RgbaPolygon()
    {
        LOG_FUNC();
        
        g_free(coordinates);
    }

    void RgbaPolygon::AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta) 
    {
//        LOG_FUNC();

        m_pColor->Lock();
        color = *m_pColor;
        m_pColor->Unlock();
        
        for (uint i = 0; i < num_coordinates; i++)
        {
            // check to see if we're adding meta data - client can disable
            // by setting the PPH ODE display meta alloc size to 0.
            // and ensure we have available space in the vector of meta structs.
            NvDsDisplayMeta* pDisplayMeta(NULL);
            for (const auto& ivec: displayMetaData)
            {
                if (ivec->num_lines < MAX_ELEMENTS_IN_DISPLAY_META)
                {
                    pDisplayMeta = ivec;
                    break;
                }
            }
            if (!pDisplayMeta)
            {
                return;
            }
            
            NvOSD_LineParams line = {
                coordinates[i].x, 
                coordinates[i].y, 
                coordinates[(i+1)%num_coordinates].x, 
                coordinates[(i+1)%num_coordinates].y, 
                border_width, 
                color};
                
            pDisplayMeta->line_params[pDisplayMeta->num_lines++] = line;
        }
    }

    // ********************************************************************

    RgbaMultiLine::RgbaMultiLine(const char* name, 
        const dsl_coordinate* coordinates, 
        uint numCoordinates, uint lineWidth, DSL_RGBA_COLOR_PTR pColor)
        : DisplayType(name)
        , dsl_multi_line_params{NULL, numCoordinates, lineWidth, *pColor}
        , m_pColor(pColor)
    {
        LOG_FUNC();
        
        // allocate data for coordinates and copy over values
        this->coordinates = 
            (dsl_coordinate*) g_malloc0(numCoordinates*sizeof(dsl_coordinate));
        memcpy(this->coordinates, coordinates, numCoordinates*sizeof(dsl_coordinate));
    }

    RgbaMultiLine::~RgbaMultiLine()
    {
        LOG_FUNC();
        
        g_free(coordinates);
    }

    void RgbaMultiLine::AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta) 
    {
//        LOG_FUNC();

        m_pColor->Lock();
        color = *m_pColor;
        m_pColor->Unlock();

        for (uint i = 0; i < num_coordinates-1; i++)
        {
            // check to see if we're adding meta data - client can disable
            // by setting the PPH ODE display meta alloc size to 0.
            // and ensure we have available space in the vector of meta structs.
            NvDsDisplayMeta* pDisplayMeta(NULL);
            for (const auto& ivec: displayMetaData)
            {
                if (ivec->num_lines < MAX_ELEMENTS_IN_DISPLAY_META)
                {
                    pDisplayMeta = ivec;
                    break;
                }
            }
            if (!pDisplayMeta)
            {
                return;
            }
            NvOSD_LineParams line = {
                coordinates[i].x, 
                coordinates[i].y, 
                coordinates[(i+1)].x, 
                coordinates[(i+1)].y, 
                line_width, 
                color};
                
            pDisplayMeta->line_params[pDisplayMeta->num_lines++] = line;
        }
    }
    // ********************************************************************

    RgbaCircle::RgbaCircle(const char* name, uint x_center, uint y_center, uint radius,
        DSL_RGBA_COLOR_PTR pColor, bool hasBgColor, DSL_RGBA_COLOR_PTR pBgColor)
        : DisplayType(name)
        , m_pColor(pColor)
        , m_pBgColor(pBgColor)
        , NvOSD_CircleParams{x_center, y_center, radius, *pColor, hasBgColor, *pBgColor}
    {
        LOG_FUNC();
    }

    RgbaCircle::~RgbaCircle()
    {
        LOG_FUNC();
    }

    void RgbaCircle::AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta) 
    {
//        LOG_FUNC();

        // check to see if we're adding meta data - client can disable
        // by setting the PPH ODE display meta alloc size to 0.
        if (displayMetaData.empty())
        {
            return;
        }
        m_pColor->Lock();
        circle_color = *m_pColor;
        m_pColor->Unlock();
        m_pBgColor->Lock();
        bg_color = *m_pBgColor;
        m_pBgColor->Unlock();
        
        displayMetaData.at(0)->circle_params[
            displayMetaData.at(0)->num_circles++] = *this;
    }

    // ********************************************************************

    SourceDimensions::SourceDimensions(const char* name, 
        uint x_offset, uint y_offset, 
        DSL_RGBA_FONT_PTR pFont, bool hasBgColor, DSL_RGBA_COLOR_PTR pBgColor)
        : RgbaText(name, "", x_offset, y_offset, pFont, hasBgColor, pBgColor)
    {
        LOG_FUNC();
    }

    SourceDimensions::~SourceDimensions()
    {
        LOG_FUNC();
    }

    void SourceDimensions::AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta) 
    {
//        LOG_FUNC();
        m_text = std::to_string(pFrameMeta->source_frame_width) + " x " + 
            std::to_string(pFrameMeta->source_frame_height);

        RgbaText::AddMeta(displayMetaData, pFrameMeta);
    }

    // ********************************************************************

    SourceFrameRate::SourceFrameRate(const char* name, 
        uint x_offset, uint y_offset, DSL_RGBA_FONT_PTR pFont, 
        bool hasBgColor, DSL_RGBA_COLOR_PTR pBgColor)
        : RgbaText(name, "", x_offset, y_offset, pFont, hasBgColor, pBgColor)
    {
        LOG_FUNC();
    }

    SourceFrameRate::~SourceFrameRate()
    {
        LOG_FUNC();
    }

    void SourceFrameRate::AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta) 
    {
//        LOG_FUNC();

//        m_text = std::to_string(pFrameMeta->source_frame_width) + " x " + 
//            std::to_string(pFrameMeta->source_frame_height);

        RgbaText::AddMeta(displayMetaData, pFrameMeta);
    }
    
    // ********************************************************************

    SourceNumber::SourceNumber(const char* name, 
        uint x_offset, uint y_offset, DSL_RGBA_FONT_PTR pFont, 
        bool hasBgColor, DSL_RGBA_COLOR_PTR pBgColor)
        : RgbaText(name, "", x_offset, y_offset, pFont, hasBgColor, pBgColor)
    {
        LOG_FUNC();
    }

    SourceNumber::~SourceNumber()
    {
        LOG_FUNC();
    }

    void SourceNumber::AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta) 
    {
//        LOG_FUNC();

        m_text = std::to_string(pFrameMeta->source_id);

        RgbaText::AddMeta(displayMetaData, pFrameMeta);
    }

    // ********************************************************************

    SourceName::SourceName(const char* name, uint x_offset, uint y_offset, 
        DSL_RGBA_FONT_PTR pFont, bool hasBgColor, DSL_RGBA_COLOR_PTR pBgColor)
        : RgbaText(name, "", x_offset, y_offset, pFont, hasBgColor, pBgColor)
    {
        LOG_FUNC();
    }

    SourceName::~SourceName()
    {
        LOG_FUNC();
    }

    void SourceName::AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta) 
    {
//        LOG_FUNC();
        
        const char* name;
        
        if (Services::GetServices()->SourceNameGet(pFrameMeta->source_id, &name) == 
            DSL_RESULT_SUCCESS)
        {
            m_text.assign(name);

            RgbaText::AddMeta(displayMetaData, pFrameMeta);
        }
        
    }
}
    
