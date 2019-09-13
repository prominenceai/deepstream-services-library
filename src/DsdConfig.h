/*
The MIT License

Copyright (c) 2019-Present, ROBERT HOWELL

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

#ifndef _DSD_CONFIG_H
#define _DSD_CONFIG_H

#include "Dsd.h"

namespace DSD 
{

    enum GroupNames
    { 
        evApplication, 
        evTiledDisplay,
        evTracker,
        evSource0, 
        evSink0,
        evSink1,
        evSink2,
        evOsd,
        evStreamMux,
        evPrimaryGie,
        evTests
    };

    enum AppCfgItems
    {
        evEnablePerfMeasurement,
        evPerfMeasurementInteralSec,
        evGieKittiOutputDir, 
        evKittiTrackOutputDir
    };

    struct Sink
    {
        /** 
         * 
         */
        NvDsSinkSubBinConfig config;

        /** 
         * 
         */
        Window window;
    };

    /**
     * @class Config
     * @brief 
     */
    class Config
    {
    public:
    
        /** 
         * 
         */
        Config(Display* pDisplay);

        ~Config();
        
        /**
        * 
        */
        bool LoadFile(const std::string& cfgFileSpec);
        
        /**
        * 
        */
        bool IsTiledDisplayEnabled();

        /**
        * 
        */
        bool IsPerfMetricEnabled();

        /**
        * 
        */
        bool SetPerfMetricEnabled(bool newValue);
        
        /**
        * 
        */
        gint GetMetricInterval();
        
        /**
        * 
        */
        gint SetMetricInterval(gint newValue);
        
        bool ConfigureTiledDisplay();
        
        void ConfigureStreamMux();
        
        /**
         * @brief Creates a new Tiled-Display bin if enabled
         * @return newly created bin if enabled, NULL if not.
         */
        GstElement* CreateTiledDisplayBin();
        
        /**
         * @brief Creates a new bin for all Sources
         * @return newly created bin
         */
        GstElement* CreateSourcesBin();

        /**
         * @brief Creates a new bin for all Sinkds
         * @return newly created bin
         */
        GstElement* CreateSinksBin();
        
        /**
         * @brief Creates a new OSD bin if enabled
         * @return newly created bin if enabled, NULL if not.
         */
        GstElement* CreateOsdBin();
       
        /**
         * @brief Creates a new Tracker bin if enabled
         * @return newly created bin if enabled, NULL if not.
         */
        GstElement* CreateTrackerBin();
       
        /**
         * @brief Creates a new Primary GIE bin if enabled
         * @return newly created bin if enabled, NULL if not.
         * @throws general exeception on failure
         */
        GstElement* CreatePrimaryGieBin();

        /**
         * @brief Creates a Secondary GIES bin if the Primary  
         * GIE binis enabled.
         * @return newly created bin if enabled, NULL if not.
         * @throws general exeception on failure
         */
        GstElement* CreateSecondaryGiesBin();
    private:

        /**
         * @brief handle to a common display used by all pipelines
         */
        Display* m_pDisplay;

        /**
         * @brief mutex to prevent config reentry
        */
        GMutex m_configMutex;
        
        /**
        * 
        */       
        std::string m_cfgFileSpec;
        
        /**
        * 
        */
        GKeyFile* m_pCfgKeyFile;
        
        /**
        * 
        */
        std::map<std::string, GroupNames> m_mapGroupNames;
        
        /**
        * 
        */
        std::map<std::string, AppCfgItems> m_mapAppCfgItems;
        
        /**
        * 
        */
        bool m_isPerfMetricEnabled;

        /**
        * 
        */
        guint m_perfMetricInterval;

        /**
        * 
        */
        std::vector<NvDsSourceConfig> m_sourcesConfig; 
            
        NvDsSrcParentBin m_sourcesBintr;

        /**
        * 
        */
        struct
        {
            NvDsGieConfig config;
            NvDsPrimaryGieBin bintr;
        } m_primaryGie;
        
        
        /**
        * 
        */
        std::vector<NvDsGieConfig> m_secondaryGieConfigs;
        
        NvDsSecondaryGieBin m_secondaryGiesbintr;
        

        /**
        * 
        */
        std::vector<NvDsSinkSubBinConfig> m_sinksConfig;  
        
        /**
        * 
        */
        NvDsSinkBin m_sinkBintr;

        /**
        * 
        */
        struct
        {
            NvDsStreammuxConfig config;
            NvDsSrcParentBin bin;
        } m_streamMux;     

        /**
        * 
        */
        struct
        {
            NvDsOSDConfig config;
            NvDsOSDBin bintr;
        } m_osd;
  
        /**
        * 
        */
        struct
        {
            NvDsTrackerConfig config;
            NvDsTrackerBin bintr;
        } m_tracker;

        /**
        * 
        */
        struct
        {
            NvDsTiledDisplayConfig config;
            NvDsTiledDisplayBin bintr;            
        } m_tiledDisplay;

        /**
        * 
        */
        NvDsDsExampleConfig m_dsExampleConfig;
        
        /**
        * 
        */
        std::string m_bBoxDir;

        /**
        * 
        */
        std::string m_kittiTrackDir;
        
        /**
        * 
        */
        gint m_fileLoop;
    
        /**
         * @struct m_secondaryGie
         * @brief 
         */
        struct
        {
            NvDsGieConfig config;
            NvDsSecondaryGieBin bin;
        } m_secondaryGie;
                
        /**
         * @brief 
         * @param group
         * @return true if "application" group was parsed successfully.
         */
        bool _parseApplicationGroup();
        
        /**
         * @brief 
         * @param group
         * @return true if a "source<n>" group was parsed successfully.
         */
        bool _parseSourceGroup(gchar* group);
        
        /**
         * @brief 
         * @param group
         * @return true if a "sink<n>" group was parsed successfully.
         */
        bool _parseSinkGroup(gchar* group);
        
        /**
         * @brief 
         * @return true if an "osd" group was parsed successfully.
         */
        bool _parseOSD();

        /**
         * @brief 
         * @return true if a "primary-gie" group was parsed successfully.
         */
        bool _parsePrimaryGieGroup(gchar* group);

        /**
         * @brief 
         * @return true if a "secondary-gie" group was parsed successfully.
         */
        bool _parseSecondaryGieGroup(gchar* group);

        /**
         * @brief 
         * @return true if the "streammux" group was parsed successfully.
         */
        bool _parseStreamMuxGroup();
        
        /**
         * @brief 
         * @return true if the "titled-display" group was parsed successfully.
         */
        bool _parseTiledDisplayGroup();

        /**
         * @brief 
         * @return true if the "tracker" group was parsed successfully.
         */
        bool _parseTrackerGroup();
    };
} // DSD

#endif // _DSD_CONFIG_H
