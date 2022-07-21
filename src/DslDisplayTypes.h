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

#ifndef _DSL_DISPLAY_TYPES_H
#define _DSL_DISPLAY_TYPES_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslBase.h"
#include "randomcolor.h"

#define MAX_DISPLAY_LEN 64

namespace DSL
{
    #define DSL_DISPLAY_TYPE_PTR std::shared_ptr<DisplayType>

    #define DSL_RGBA_COLOR_PTR std::shared_ptr<RgbaColor>
    #define DSL_RGBA_COLOR_NEW(name, red, green, blue, alpha) \
        std::shared_ptr<RgbaColor>(new RgbaColor(name, \
            red, green, blue, alpha))

    #define DSL_RGBA_RANDOM_COLOR_PTR std::shared_ptr<RgbaRandomColor>
    #define DSL_RGBA_RANDOM_COLOR_NEW(name, hue, luminosity, alpha, seed) \
        std::shared_ptr<RgbaRandomColor>(new RgbaRandomColor(name, \
            hue, luminosity, alpha, seed))

    #define DSL_RGBA_PREDEFINED_COLOR_PTR std::shared_ptr<RgbaPredefinedColor>
    #define DSL_RGBA_PREDEFINED_COLOR_NEW(name, colorId, alpha) \
        std::shared_ptr<RgbaPredefinedColor>(new RgbaPredefinedColor(name, \
            colorId, alpha))

    #define DSL_RGBA_COLOR_PALETTE_PTR std::shared_ptr<RgbaColorPalette>
    #define DSL_RGBA_COLOR_PALETTE_NEW(name, pColorPalette) \
        std::shared_ptr<RgbaColorPalette>(new RgbaColorPalette(name, \
            pColorPalette))

    #define DSL_RGBA_ON_DEMAND_COLOR_PTR std::shared_ptr<RgbaOnDemandColor>
    #define DSL_RGBA_ON_DEMAND_COLOR_NEW(name, provider, clientData) \
        std::shared_ptr<RgbaOnDemandColor>(new RgbaOnDemandColor(name, \
            provider, clientData))

    #define DSL_RGBA_FONT_PTR std::shared_ptr<RgbaFont>
    #define DSL_RGBA_FONT_NEW(name, font, size, pColor) \
        std::shared_ptr<RgbaFont>(new RgbaFont(name, font, size, pColor))

    #define DSL_RGBA_TEXT_PTR std::shared_ptr<RgbaText>
    #define DSL_RGBA_TEXT_NEW(name, \
        text, x_offset, y_offset, font, hasBgColor, pBgColor) \
        std::shared_ptr<RgbaText>(new RgbaText(name, \
            text, x_offset, y_offset, font, hasBgColor, pBgColor))
        
    #define DSL_RGBA_LINE_PTR std::shared_ptr<RgbaLine>
    #define DSL_RGBA_LINE_NEW(name, \
        x1, y1, x2, y2, width, pColor) \
        std::shared_ptr<RgbaLine>(new RgbaLine(name, \
            x1, y1, x2, y2, width, pColor))

    #define DSL_RGBA_ARROW_PTR std::shared_ptr<RgbaArrow>
    #define DSL_RGBA_ARROW_NEW(name, \
        x1, y1, x2, y2, width, head, pColor) \
        std::shared_ptr<RgbaArrow>(new RgbaArrow(name, \
            x1, y1, x2, y2, width, head, pColor))

    #define DSL_RGBA_RECTANGLE_PTR std::shared_ptr<RgbaRectangle>
    #define DSL_RGBA_RECTANGLE_NEW(name, \
        left, top, width, height, borderWidth, pColor, hasBgColor, pBgColor) \
        std::shared_ptr<RgbaRectangle>(new RgbaRectangle(name, \
            left, top, width, height, borderWidth, pColor, hasBgColor, pBgColor))

    #define DSL_RGBA_POLYGON_PTR std::shared_ptr<RgbaPolygon>
    #define DSL_RGBA_POLYGON_NEW(name, \
        coordinates, numCoordinates, borderWidth, pColor) \
        std::shared_ptr<RgbaPolygon>(new RgbaPolygon(name, \
            coordinates, numCoordinates, borderWidth, pColor))
    
    #define DSL_RGBA_MULTI_LINE_PTR std::shared_ptr<RgbaMultiLine>
    #define DSL_RGBA_MULTI_LINE_NEW(name, \
        coordinates, numCoordinates, borderWidth, pColor) \
        std::shared_ptr<RgbaMultiLine>(new RgbaMultiLine(name, \
            coordinates, numCoordinates, borderWidth, pColor))
    
    #define DSL_RGBA_CIRCLE_PTR std::shared_ptr<RgbaCircle>
    #define DSL_RGBA_CIRCLE_NEW(name, \
        x_center, y_center, radius, pColor, hasBgColor, pBgColor) \
        std::shared_ptr<RgbaCircle>(new RgbaCircle(name, \
            x_center, y_center, radius, pColor, hasBgColor, pBgColor))

    #define DSL_SOURCE_DIMENSIONS_PTR std::shared_ptr<SourceDimensions>
    #define DSL_SOURCE_DIMENSIONS_NEW(name, \
        x_offset, y_offset, font, hasBgColor, pBgColor) \
        std::shared_ptr<SourceDimensions>(new SourceDimensions(name, \
            x_offset, y_offset, font, hasBgColor, pBgColor))

    #define DSL_SOURCE_FRAME_RATE_PTR std::shared_ptr<SourceFrameRate>
    #define DSL_SOURCE_FRAME_RATE_NEW(name, \
        x_offset, y_offset, font, hasBgColor, pBgColor) \
        std::shared_ptr<SourceFrameRate>(new SourceFrameRate(name, \
            x_offset, y_offset, font, hasBgColor, pBgColor))

    #define DSL_SOURCE_NUMBER_PTR std::shared_ptr<SourceNumber>
    #define DSL_SOURCE_NUMBER_NEW(name, \
        x_offset, y_offset, font, hasBgColor, pBgColor) \
        std::shared_ptr<SourceNumber>(new SourceNumber(name, \
            x_offset, y_offset, font, hasBgColor, pBgColor))

    #define DSL_SOURCE_NAME_PTR std::shared_ptr<SourceName>
    #define DSL_SOURCE_NAME_NEW(name, \
        x_offset, y_offset, font, hasBgColor, pBgColor) \
        std::shared_ptr<SourceName>(new SourceName(name, \
            x_offset, y_offset, font, hasBgColor, pBgColor))

    // ********************************************************************

    class DisplayType : public Base
    {
    public: 
    
        /**
         * @brief ctor for the virtual DisplayType
         * @param[in] name unique name for the RGBA Color
         */
        DisplayType(const char* name);

        ~DisplayType();

        /**
         * @brief Locks the DisplayType's property mutex.
         */
        virtual inline void Lock();
        
        /**
         * @brief Unlocks the DisplayType's property mutex
         */
        virtual inline void Unlock();
        
        /**
         * @brief Adds the Display Type's meta to the provided displayMetaData
         * @param displayMetaData vector of allocated Display metadata to add 
         * the meta to
         * @param pFrameMeta frame meta for the frame the display meta 
         * will be added to.
         */
        virtual void AddMeta(std::vector<NvDsDisplayMeta*>& 
            displayMetaData, NvDsFrameMeta* pFrameMeta);
            
    protected:
        
        /**
         * @brief Mutex to ensure mutual exlusion for propery read/writes
         */
        GMutex m_propertyMutex;
    };
    
    // ********************************************************************

    class RgbaColor : public DisplayType, public NvOSD_ColorParams
    {
    public: 
    
        /**
         * @brief ctor for RGBA Color
         * @param[in] name unique name for the RGBA Color
         * @param[in] red red level for the RGB color [0..1]
         * @param[in] blue blue level for the RGB color [0..1]
         * @param[in] green green level for the RGB color [0..1]
         * @param[in] alpha alpha level for the RGB color [0..1]
         */
        RgbaColor(const char* name, 
            double red, double green, double blue, double alpha);

        /**
         * @breif ctor for the RBGA Color
         * @param[in] name unique name for the RGBA Color
         * @param[in] color NVIDIA color params to copy from.
         */
        RgbaColor(const char* name, const NvOSD_ColorParams& color);

        /**
         * @breif copy ctor for the RBGA Color
         * @param[in] color source RgbaColor to copy from
         */
        RgbaColor(const RgbaColor& color);
        
        /**
         * @breif ctor for the no-color RBGA Color
         */
        RgbaColor();
        
        /**
         * @breif dtor for the RBGA Color
         */
        ~RgbaColor();
        
        /**
         * @brief noop SetNext for static color.
         */
        virtual void SetNext(){};
    };

    // ********************************************************************

    class RgbaPredefinedColor : public RgbaColor
    {
    public: 
    
        /**
         * @brief ctor for the RGBA Predefined Color class
         * @param[in] name unique name for the RGBA Predefined color
         * @param[in] colorId predefined color identifier. 
         * @param[in] alpha alpha level for the RGB color [0..1]
         */
        RgbaPredefinedColor(const char* name, uint colorId, double alpha);

        /**
         * @brief dtor for RGBA Predefined Color
         */
        ~RgbaPredefinedColor();

        /**
         * @brief
         */
        static std::map<uint, 
            std::vector<NvOSD_ColorParams>> s_predefinedColorPalettes;        
        
    private:

        /**
         * @brief
         */
        static std::map<uint, NvOSD_ColorParams> s_predefinedColors;

        /**
         * @brief predefined color identifier. 
         */
        uint m_colorId;
        
    };
    
    // ********************************************************************

    class RgbaColorPalette : public RgbaColor
    {
    public: 
    
        /**
         * @brief ctor for the RGBA RgbaColorPalette Color class
         * @param[in] name unique name for the RGBA RgbaColorPalette color
         * @param[in] pColorPalette a shared pointer to a vector of
         * RGBA Colors.
         */
        RgbaColorPalette(const char* name, 
            std::shared_ptr<std::vector<DSL_RGBA_COLOR_PTR>> pColorPalette);

        /**
         * @brief dtor for RGBA RgbaColorPalette Color
         */
        ~RgbaColorPalette();
        
        /**
         * @brief Set the RGB values to the next color in the Palette.
         */
        void SetNext();
        
        /**
         * @brief Gets the palette index.
         * @return current palette index values.
         */
        uint GetIndex();
        
        /**
         * @brief Sets the palette index and color
         * @param index new palette index
         * @return true if successful, false otherwise.
         */
        bool SetIndex(uint index);
        
        /**
         * @brief Gets the current size of the color-palette
         * @return size of the color palette
         */
        uint GetSize();

        
    private:
    
        /**
         * @brief index of the current color in the color palette vector
         */
        uint m_currentColorIndex;
        /**
         * @brief a shared pointer to a vector of
         * RGBA Colors. 
         */
        std::shared_ptr<std::vector<DSL_RGBA_COLOR_PTR>> m_pColorPalette;
    };
    
    // ********************************************************************

    class RgbaRandomColor : public RgbaColor
    {
    public: 
    
        /**
         * @brief ctor for RGBA Random Color
         * @param[in] name unique name for the RGBA Random olor
         * @param[in] hue color hue to use for random color generation.
         * @param[in] luminosity luminocity level to use for random color generation. 
         * @param[in] alpha alpha level for the RGB color [0..1]
         * @param[in] seed value to seed the random generator 
         */
        RgbaRandomColor(const char* name, 
            uint hue, uint luminosity, double alpha, uint seed);

        /**
         * @brief dtor for RGBA Random Color
         */
        ~RgbaRandomColor();

        /**
         * @brief Set the RGB values to the next random color.
         */
        void SetNext();
        
    private:
    
        /**
         * @brief color hue to use for random color generation. 
         */
        RandomColor::Color m_hue;
        
        /**
         * @brief luminocity level to use for random color generation. 
         */
        RandomColor::Luminosity m_luminosity;
        
        /**
         * @brief random color generator
         */
        RandomColor m_randomColor;
    };
    
    // ********************************************************************

    class RgbaOnDemandColor : public RgbaColor
    {
    public: 
    
        /**
         * @brief ctor for RGBA On Demand Color
         * @param[in] name unique name for the RGBA On-Demand Color
         * @param[in] provider callback function to be called on SetNext()
         * @param[in] clientData opaque pointer to client's user data.
         */
        RgbaOnDemandColor(const char* name, 
            dsl_display_type_rgba_color_provider_cb provider, void* clientData);

        /**
         * @brief dtor for RGBA On Demand Color
         */
        ~RgbaOnDemandColor();

        /**
         * @brief Calls the client's call back to get the next RGB values.
         */
        void SetNext();
        
    private:
    
        /**
         * @brief Client callback to call on SetNext(). 
         */
        dsl_display_type_rgba_color_provider_cb m_provider;
        
        /**
         * @brief opaque pointer to client's user data based back to m_provider.
         */
        void* m_clientData;
        
    };
    
    // ********************************************************************

    class RgbaFont : public DisplayType, public NvOSD_FontParams
    {
    public:
    
        /**
         * @brief ctor for RGBA Font
         * @param[in] name unique name for the RgbaFont DisplayType
         * @param[in] font standard string name of the actual tty font type
         * @param[in] size size of the font
         * @param[in] color RGBA Color for the RGBA font
         */
        RgbaFont(const char* name, 
            const char* font, uint size, DSL_RGBA_COLOR_PTR color);

        /**
         * @brief ctor for RGBA Font
         */
        ~RgbaFont();

        /**
         * @brief Locks the DisplayType's property mutex.
         */
        inline void Lock();
        
        /**
         * @brief Unlocks the DisplayType's property mutex
         */
        inline void Unlock();
        
        /**
         * @breif actual tty font name
         */
        std::string m_fontName;

    private:

        /**
         * @breif shared pointer to a RGBA Color Type for this RGBA Font
         */
        DSL_RGBA_COLOR_PTR m_pColor;
    };
    
    // ********************************************************************

    class RgbaText : public DisplayType, public NvOSD_TextParams
    {
    public:

        /**
         * @brief ctor for RGBA Text
         * @param[in] name unique name of the RGBA Text
         * @param[in] text text string to display
         * @param[in] x_offset starting x positional offset
         * @param[in] y_offset starting y positional offset
         * @param[in] font RGBA font to use for the display dext
         * @param[in] hasBgColor set to true to enable bacground color, false otherwise
         * @param[in] pBgColor RGBA Color for the Text background if set
         */
        RgbaText(const char* name, const char* text, uint x_offset, uint y_offset, 
            DSL_RGBA_FONT_PTR pFont, bool hasBgColor, DSL_RGBA_COLOR_PTR pBgColor);

        ~RgbaText();

        /**
         * @brief Locks the DisplayType's property mutex.
         */
        inline void Lock();
        
        /**
         * @brief Unlocks the DisplayType's property mutex
         */
        inline void Unlock();
        
        /**
         * @brief Enables addition of a dropped shadow for the text background.
         * @param xOffset[in] x-offset for the shadow in units of pixels.
         * @param yOffset[in] y-offset for the shadow in units of pixels.
         * @param pColor[in] RGBA Color for the Text background shadow.
         */
        bool AddShadow(uint xOffset, uint yOffset, DSL_RGBA_COLOR_PTR pColor);

        /**
         * @brief Adds the Display Type's meta to the provided displayMetaData
         * @param displayMetaData vector of allocated Display metadata to add 
         * the meta to
         * @param pFrameMeta frame meta for the frame the display meta 
         * will be added to.
         */
        void AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta);
        
        std::string m_text;
        
    private:
    
        /**
         * @breif shared pointer to a RGBA Font Type for this RGBA Text
         */
        DSL_RGBA_FONT_PTR m_pFont;

        /**
         * @breif shared pointer to a RGBA Color Type for this RGBA Text
         */
        DSL_RGBA_COLOR_PTR m_pBgColor;
        
        /**
         * @brief true if the text has a dropped shadow, false (default) otherwise.
         */
        bool m_shadowEnabled;

        /**
         * @brief x-offset for the text shadow if enabled. 
         */
        uint m_shadowXOffset;
    
        /**
         * @brief y-offset for the text shadow if enabled. 
         */
        uint m_shadowYOffset;

        /**
         * @breif shared pointer to a RGBA Font Type for the shadow if enabled.
         */
        DSL_RGBA_FONT_PTR m_pShadowFont;
    
        /**
         * @breif shared pointer to a RGBA Color Type for the shadow if enabled.
         */
        DSL_RGBA_COLOR_PTR m_pShadowColor;
    };
    
    // ********************************************************************

    class RgbaLine : public DisplayType, public NvOSD_LineParams
    {
    public:

        /**
         * @brief ctor for RGBA Line
         * @param[in] name unique name for the RGBA LIne
         * @param[in] x1 starting x positional offest
         * @param[in] y1 starting y positional offest
         * @param[in] x2 ending x positional offest
         * @param[in] y2 ending y positional offest
         * @param[in] width width of the line in pixels
         * @param[in] color RGBA Color for thIS RGBA Line
         */
        RgbaLine(const char* name, uint x1, uint y1, uint x2, uint y2, 
            uint width, DSL_RGBA_COLOR_PTR pColor);

        ~RgbaLine();

        /**
         * @brief Adds the Display Type's meta to the provided displayMetaData
         * @param displayMetaData vector of allocated Display metadata to add 
         * the meta to
         * @param pFrameMeta frame meta for the frame the display meta 
         * will be added to.
         */
        void AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta);
            
    private:
    
        /**
         * @breif shared pointer to a RGBA Color Type for this RGBA Line
         */
        DSL_RGBA_COLOR_PTR m_pColor;
    };
    
    // ********************************************************************

    class RgbaArrow : public DisplayType, public NvOSD_ArrowParams
    {
    public:

        /**
         * @brief ctor for RGBA Line
         * @param[in] name unique name for the RGBA Arrow
         * @param[in] x1 starting x positional offest
         * @param[in] y1 starting y positional offest
         * @param[in] x2 ending x positional offest
         * @param[in] y2 ending y positional offest
         * @param[in] width width of the line in pixels
         * @param[in] head position of arrow head START_HEAD, END_HEAD, BOTH_HEAD
         * @param[in] pColor RGBA Color for thIS RGBA Line
         */
        RgbaArrow(const char* name, uint x1, uint y1, uint x2, uint y2, 
            uint width, uint head, DSL_RGBA_COLOR_PTR pColor);

        ~RgbaArrow();

        /**
         * @brief Adds the Display Type's meta to the provided displayMetaData
         * @param displayMetaData vector of allocated Display metadata to add 
         * the meta to
         * @param pFrameMeta frame meta for the frame the display meta 
         * will be added to.
         */
        void AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta);
            
    private:
    
        /**
         * @breif shared pointer to a RGBA Color Type for this RGBA Arrow
         */
        DSL_RGBA_COLOR_PTR m_pColor;
    
    };

    // ********************************************************************

    class RgbaRectangle : public DisplayType, public NvOSD_RectParams
    {
    public:

        /**
         * @brief ctor for RGBA Rectangle
         * @param[in] name unique name for the RGBA Rectangle
         * @param[in] left left positional offest
         * @param[in] top positional offest
         * @param[in] width width of the rectangle in Pixels
         * @param[in] height height of the rectangle in Pixels
         * @param[in] width width of the line in pixels
         * @param[in] pColor RGBA Color for thIS RGBA Line
         * @param[in] hasBgColor set to true to enable bacground color, false otherwise
         * @param[in] pBgColor RGBA Color for the Circle background if set
         */
        RgbaRectangle(const char* name, uint left, uint top, uint width, uint height, 
            uint borderWidth, DSL_RGBA_COLOR_PTR pColor, bool hasBgColor, 
            DSL_RGBA_COLOR_PTR pBgColor);

        ~RgbaRectangle();

        /**
         * @brief Adds the Display Type's meta to the provided displayMetaData
         * @param displayMetaData vector of allocated Display metadata to add 
         * the meta to
         * @param pFrameMeta frame meta for the frame the display meta 
         * will be added to.
         */
        void AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta);
            
    private:
    
        /**
         * @breif shared pointer to a RGBA Color Type for this RGBA Font
         */
        DSL_RGBA_COLOR_PTR m_pColor;

        /**
         * @breif shared pointer to a RGBA Color Type for this RGBA Font
         */
        DSL_RGBA_COLOR_PTR m_pBgColor;
    };
    
    // ********************************************************************

    /**
     * @struct dsl_polygon
     * @brief the following is an interim solution until Nvidia provides 
     * a version (with bg_color) - as promised for the next release. 
     */
    typedef struct _dsl_polygon_params
    {
        /**
         * @brief an array coordinates defining the polygon
         * The last point provided will be connected to the first
         */
        dsl_coordinate* coordinates;
        
        /**
         * @brief the number of coordinates in the polygon
         */
        uint num_coordinates;
         
        /**
         * @brief width of the polygon lines in pixels
         */
        uint border_width;    

        /**
         * @brief RGBA color of the polygon lines in pixels
         */
        NvOSD_ColorParams color;
         
    } dsl_polygon_params;

    
    class RgbaPolygon : public DisplayType, public dsl_polygon_params
    {
    public:

        /**
         * @brief ctor for RGBA Polygon
         * @param[in] name unique name for the RGBA Rectangle
         * @param[in] coordinates an array of xy dsl_coordinates defining the Polygon
         * @param[in] numPoints number of xy coordinates in the array.
         * @param[in] lineWidth width of the Polygon line in pixels
         * @param[in] pColor RGBA Color for this RGBA Polygon
         */
        RgbaPolygon(const char* name, const dsl_coordinate* coordinates, uint numCoordinates,
            uint lineWidth, DSL_RGBA_COLOR_PTR pColor);

        ~RgbaPolygon();

        /**
         * @brief Adds the Display Type's meta to the provided displayMetaData
         * @param displayMetaData vector of allocated Display metadata to add 
         * the meta to
         * @param pFrameMeta frame meta for the frame the display meta 
         * will be added to.
         */
        void AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta);

    private:
    
        /**
         * @breif shared pointer to a RGBA Color Type for this RGBA Polygon
         */
        DSL_RGBA_COLOR_PTR m_pColor;
    };

    // ********************************************************************

    /**
     * @struct dsl_multi_line
     * @brief 
     */
    typedef struct _dsl_multi_line_parms
    {
        /**
         * @brief an array coordinates defining the multi-line
         * The last point provided will be connected to the first
         */
        dsl_coordinate* coordinates;
        
        /**
         * @brief the number of coordinates in the multi-line
         */
        uint num_coordinates;
         
        /**
         * @brief width of the multiple lines in pixels
         */
        uint line_width;    

        /**
         * @brief RGBA color of the multi-line lines in pixels
         */
        NvOSD_ColorParams color;
         
    } dsl_multi_line_params;

    class RgbaMultiLine : public DisplayType, public dsl_multi_line_params
    {
    public:

        /**
         * @brief ctor for RGBA Multi-Line class
         * @param[in] name unique name for the RGBA Multi-Line
         * @param[in] coordinates an array of xy dsl_coordinates defining the Multi-Line
         * @param[in] numPoints number of xy coordinates in the array.
         * @param[in] lineWidth width of the Multi-Line in pixels
         * @param[in] pColor RGBA Color for this RGBA Polygon
         */
        RgbaMultiLine(const char* name, 
            const dsl_coordinate* coordinates, uint numCoordinates,
            uint lineWidth, DSL_RGBA_COLOR_PTR pColor);

        ~RgbaMultiLine();

        /**
         * @brief Adds the Display Type's meta to the provided displayMetaData
         * @param displayMetaData vector of allocated Display metadata to add 
         * the meta to
         * @param pFrameMeta frame meta for the frame the display meta 
         * will be added to.
         */
        void AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta);

    private:
    
        /**
         * @breif shared pointer to a RGBA Color Type for this RGBA Multi-Line
         */
        DSL_RGBA_COLOR_PTR m_pColor;
    };

    
    // ********************************************************************

    class RgbaCircle : public DisplayType, public NvOSD_CircleParams
    {
    public:

        /**
         * @brief ctor for RGBA Circle
         * @param[in] name unique name for the RGBA Circle
         * @param[in] x_center X positional offset to center of Circle
         * @param[in] y_center y positional offset to center of Circle
         * @param[in] radius radius of the RGBA Circle in pixels 
         * @param[in] pColor RGBA Color for the RGBA Circle
         * @param[in] hasBgColor set to true to enable bacground color, false otherwise
         * @param[in] pBgColor RGBA Color for the Circle background if set
         */
        RgbaCircle(const char* name, uint x_center, uint y_center, uint radius,
            DSL_RGBA_COLOR_PTR pColor, bool hasBgColor, DSL_RGBA_COLOR_PTR pBgColor);

        ~RgbaCircle();

        /**
         * @brief Adds the Display Type's meta to the provided displayMetaData
         * @param displayMetaData vector of allocated Display metadata to add 
         * the meta to
         * @param pFrameMeta frame meta for the frame the display meta 
         * will be added to.
         */
        void AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta);

    private:
    
        /**
         * @breif shared pointer to a RGBA Color Type for this RGBA Circle
         */
        DSL_RGBA_COLOR_PTR m_pColor;

        /**
         * @breif shared pointer to a RGBA Color Type for this RGBA Circle
         */
        DSL_RGBA_COLOR_PTR m_pBgColor;
    };

    // ********************************************************************

    class SourceDimensions : public RgbaText
    {
    public:

        /**
         * @brief ctor for RGBA Text
         * @param[in] name unique name of the SourceDimensions Display
         * @param[in] x_offset starting x positional offset
         * @param[in] y_offset starting y positional offset
         * @param[in] font RGBA font to use for the display dext
         * @param[in] hasBgColor set to true to enable bacground color, false otherwise
         * @param[in] pBgColor RGBA Color for the Text background if set
         */
        SourceDimensions(const char* name, uint x_offset, uint y_offset, 
            DSL_RGBA_FONT_PTR pFont, bool hasBgColor, DSL_RGBA_COLOR_PTR pBgColor);

        ~SourceDimensions();

        /**
         * @brief Adds the Display Type's meta to the provided displayMetaData.
         * @param displayMetaData vector of allocated Display metadata to add
         * the meta to.
         * @param pFrameMeta frame meta for the frame the display meta
         * will be added to.
         */
        void AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta);
        
    private:
    
        /**
         * @breif shared pointer to a RGBA Font Type for this RGBA SourceDiemnsions.
         */
        DSL_RGBA_FONT_PTR m_pFont;

        /**
         * @breif shared pointer to a RGBA Color Type for this RGBA SourceDiemnsions.
         */
        DSL_RGBA_COLOR_PTR m_pBgColor;
    };
 
    // ********************************************************************

    class SourceFrameRate : public RgbaText
    {
    public:

        /**
         * @brief ctor for RGBA Text
         * @param[in] name unique name of the SourceFrameRate Display
         * @param[in] x_offset starting x positional offset
         * @param[in] y_offset starting y positional offset
         * @param[in] font RGBA font to use for the display dext
         * @param[in] hasBgColor set to true to enable bacground color, false otherwise
         * @param[in] pBgColor RGBA Color for the Text background if set
         */
        SourceFrameRate(const char* name, uint x_offset, uint y_offset, 
            DSL_RGBA_FONT_PTR pFont, bool hasBgColor, DSL_RGBA_COLOR_PTR pBgColor);

        ~SourceFrameRate();

        /**
         * @brief Adds the Display Type's meta to the provided displayMetaData
         * @param displayMetaData vector of allocated Display metadata to add 
         * the meta to
         * @param pFrameMeta frame meta for the frame the display meta 
         * will be added to.
         */
        void AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta);
        
    private:
    
        /**
         * @breif shared pointer to a RGBA Font Type for this RGBA SourceFrameRate.
         */
        DSL_RGBA_FONT_PTR m_pFont;

        /**
         * @breif shared pointer to a RGBA Color Type for this RGBA SourceFrameRate.
         */
        DSL_RGBA_COLOR_PTR m_pBgColor;
    };

    // ********************************************************************

    class SourceNumber : public RgbaText
    {
    public:

        /**
         * @brief ctor for Source Number Display Type
         * @param[in] name unique name of the SourceNumber Display Type
         * @param[in] x_offset starting x positional offset
         * @param[in] y_offset starting y positional offset
         * @param[in] font RGBA font to use for the display dext
         * @param[in] hasBgColor set to true to enable bacground color, false otherwise
         * @param[in] pBgColor RGBA Color for the Text background if set
         */
        SourceNumber(const char* name, uint x_offset, uint y_offset, 
            DSL_RGBA_FONT_PTR pFont, bool hasBgColor, DSL_RGBA_COLOR_PTR pBgColor);

        ~SourceNumber();

        /**
         * @brief Adds the Display Type's meta to the provided displayMetaData
         * @param displayMetaData vector of allocated Display metadata to add 
         * the meta to
         * @param pFrameMeta frame meta for the frame the display meta 
         * will be added to.
         */
        void AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta);
        
    private:
    
        /**
         * @breif shared pointer to a RGBA Font Type for this RGBA SourceNumber.
         */
        DSL_RGBA_FONT_PTR m_pFont;

        /**
         * @breif shared pointer to a RGBA Color Type for this RGBA SourceNumber.
         */
        DSL_RGBA_COLOR_PTR m_pBgColor;
    };

    // ********************************************************************

    class SourceName : public RgbaText
    {
    public:

        /**
         * @brief ctor for the Source Name
         * @param[in] name unique name of the Source Name Display Type
         * @param[in] x_offset starting x positional offset
         * @param[in] y_offset starting y positional offset
         * @param[in] font RGBA font to use for the display dext
         * @param[in] hasBgColor set to true to enable bacground color, false otherwise
         * @param[in] pBgColor RGBA Color for the Text background if set
         */
        SourceName(const char* name, uint x_offset, uint y_offset, 
            DSL_RGBA_FONT_PTR pFont, bool hasBgColor, DSL_RGBA_COLOR_PTR pBgColor);

        ~SourceName();

        /**
         * @brief Adds the Display Type's meta to the provided displayMetaData
         * @param displayMetaData vector of allocated Display metadata to add 
         * the meta to
         * @param pFrameMeta frame meta for the frame the display meta 
         * will be added to.
         */
        void AddMeta(std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta);
        
    private:
    
        /**
         * @breif shared pointer to a RGBA Font Type for this RGBA SourceName.
         */
        DSL_RGBA_FONT_PTR m_pFont;

        /**
         * @breif shared pointer to a RGBA Color Type for this RGBA SourceName.
         */
        DSL_RGBA_COLOR_PTR m_pBgColor;
    };

}
#endif // _DSL_DISPLAY_TYPES_H
    
