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
    
    struct SyncSubBin
    {
        /** 
         * 
         */
        NvDsSinkSubBinConfig config;

        /** 
         * 
         */
        NvDsSinkBin sinkSubBin;
    };
    
    struct Source
    {
        /** 
         * 
         */
        NvDsSourceConfig config;
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
        Config();

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
        * 
        */
        void ConfigureNewWindows(Display* display);
        
    
    private:

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
        std::vector<NvDsSourceConfig> m_sourceSubBinConfigs;          

        /**
        * 
        */
        std::vector<NvDsGieConfig> m_secondaryGieSubBinsConfigs;          

        /**
        * 
        */
        std::vector<SyncSubBin> m_sinkSubBins;          

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
        NvDsStreammuxConfig m_streammuxConfig;

        /**
        * 
        */
        NvDsOSDConfig m_osdConfig;

        /**
        * 
        */
        NvDsGieConfig m_primaryGieConfig;

        /**
        * 
        */
        NvDsTrackerConfig m_trackerConfig;

        /**
        * 
        */
        NvDsTiledDisplayConfig m_tiledDisplayConfig;

        /**
        * 
        */
        NvDsDsExampleConfig m_dsExampleConfig;
        
        /**
        * 
        */
        gint m_fileLoop;

        /**
         * 
         */
        NvDsSinkBin m_sinkBin;
        
        /**
        * 
        */
        bool ParseApplicationGroup();
        
        /**
        * 
        */
        bool ParseSourceGroup(gchar* group);
        
        /**
        * 
        */
        bool ParseSinkGroup(gchar* group);
        
        
    };
} // DSS

#endif // _DSS_CONFIG_H
