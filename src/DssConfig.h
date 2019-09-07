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

#ifndef _DSS_CONFIG_H
#define _DSS_CONFIG_H

#include "Dss.h"

namespace DSS 
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

    struct PrimaryGie
    {
        NvDsGieConfig config;
        NvDsPrimaryGieBin bin;
    };
    
    struct SecondaryGie
    {
        NvDsGieConfig config;
        NvDsSecondaryGieBin bin;
    };
    
    struct Source
    {
        NvDsSourceConfig config;
        NvDsSrcBin bin;
    };
    
    /**
     * @class Config
     * @file  DssConfig.h
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
        
        /**
         * @brief 
         * @return true if the tiled display was configured successfully.
         */
        bool ConfigureTiledDisplay();
            
    private:

        /**
         * @brief handle to a common display used by all pipelines
         */
        Display* m_pDisplay;

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
        std::vector<Source> m_vSources;          

        /**
        * 
        */
        std::vector<PrimaryGie> m_vPrimaryGies;
        
        /**
        * 
        */
        std::vector<SecondaryGie> m_vSecondaryGies;          

        /**
        * 
        */
        std::vector<Sink> m_vSinks;          

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
        GstElement *pInstanceBin;

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
            NvDsOSDBin bin;
        } m_osd;
  
        /**
        * 
        */
        struct
        {
            NvDsTrackerConfig config;
            NvDsTrackerBin bin;
        } m_tracker;

        /**
        * 
        */
        struct
        {
            NvDsTiledDisplayConfig config;
            NvDsTiledDisplayBin bin;            
        } m_tiledDisplay;

        /**
        * 
        */
        NvDsDsExampleConfig m_dsExampleConfig;
        
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
        bool ParseApplicationGroup();
        
        /**
         * @brief 
         * @param group
         * @return true if a "source<n>" group was parsed successfully.
         */
        bool ParseSourceGroup(gchar* group);
        
        /**
         * @brief 
         * @param group
         * @return true if a "sink<n>" group was parsed successfully.
         */
        bool ParseSinkGroup(gchar* group);
        
        /**
         * @brief 
         * @return true if an "osd" group was parsed successfully.
         */
        bool ParseOSD();

        /**
         * @brief 
         * @return true if an "primary-gie" group was parsed successfully.
         */
        bool ParsePrimaryGieGroup(gchar* group);

        /**
         * @brief 
         * @return true if the "streammux" group was parsed successfully.
         */
        bool ParseStreamMuxGroup();
        
        /**
         * @brief 
         * @return true if the "titled-display" group was parsed successfully.
         */
        bool ParseTiledDisplayGroup();

        /**
         * @brief 
         * @return true if the "tracker" group was parsed successfully.
         */
        bool ParseTrackerGroup();
    };
} // DSS

#endif // _DSS_CONFIG_H
