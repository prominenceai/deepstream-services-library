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
        evNotDefined, 
        evApplication, 
        evTiledDisplay, 
        evSource0, 
        evSink0,
        evSink1,
        evSink2,
        evOsd,
        evStreamMux,
        evPrimaryGie,
        evTests,
        evEnd
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
        
        bool LoadFile(const std::string& cfgFilePathSpec);
    
    private:

        /**
        * 
        */
        
        GKeyFile* m_pCfgFile;
        
        std::map<std::string, GroupNames> m_mapGroupNames;
        
        bool m_isPerfMetricEnabled;
        guint m_perfMetricInterval;
        gint m_fileLoop;


        // guint m_sourceSubBinsCount;
        // NvDsSourceConfig m_sourceConfigs[MAX_SOURCE_BINS];
        std::vector<NvDsSourceConfig> m_sourceConfigs;          

//        guint m_secondaryGieSubBinsCount;
//        NvDsGieConfig m_secondaryGieSubBinsConfigs[MAX_SECONDARY_GIE_BINS];
        std::vector<NvDsGieConfig> m_secondaryGieSubBinsConfigs;          

//        guint m_sinkSubBinsCount;
//        NvDsSinkSubBinConfig m_sinkSubBinsConfigs[MAX_SINK_BINS];
        std::vector<NvDsSinkSubBinConfig> m_sinkSubBinsConfigs;          

        gchar *m_pBBoxDir;
        gchar *m_pKittiTrackDir;

        NvDsStreammuxConfig m_streammuxConfig;

        NvDsOSDConfig m_osdConfig;

        NvDsGieConfig m_primaryGieConfig;
        NvDsTrackerConfig m_trackerConfig;
        NvDsTiledDisplayConfig m_tiledDisplayConfig;
        NvDsDsExampleConfig m_dsExampleConfig;

    /**
    * 
    */


    };
} // DSS

#endif // _DSS_CONFIG_H
