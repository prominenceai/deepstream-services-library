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

#include "Dss.h"
#include "DssConfig.h"

namespace DSS
{
         
    Config::Config()
//        : m_cfgFileSpec()
        : m_pCfgKeyFile(g_key_file_new())
        , m_isPerfMetricEnabled(false)
        , m_perfMetricInterval(0)
        , m_fileLoop(0)
        , m_pBBoxDir(NULL)
        , m_pKittiTrackDir(NULL)
        , m_streammuxConfig{0}
        , m_osdConfig{0}
        , m_primaryGieConfig{0}
        , m_trackerConfig{0}
        , m_tiledDisplayConfig{0}
        , m_dsExampleConfig{0}

    {
        LOG_FUNC();
        
        m_mapGroupNames["application"] = evApplication;
        m_mapGroupNames["tiled-display"] = evTiledDisplay;
        m_mapGroupNames["tracker"] = evTracker;
        m_mapGroupNames["source0"] = evSource0;
        m_mapGroupNames["sink0"] = evSink0;
        m_mapGroupNames["sink1"] = evSink1;
        m_mapGroupNames["sink2"] = evSink2;
        m_mapGroupNames["osd"] = evOsd;
        m_mapGroupNames["streammux"] = evStreamMux;
        m_mapGroupNames["primary-gie"] = evPrimaryGie;
        m_mapGroupNames["tests"] = evTests;

    }

    Config::~Config()
    {
        LOG_FUNC();
        
        if (m_pCfgKeyFile)
        {
            LOG_INFO("Releasing the Configuration Key File");
            g_key_file_free(m_pCfgKeyFile);
        }
    }

    bool Config::LoadFile(const std::string& cfgFileSpec) 
    {
        LOG_FUNC();
        
        m_cfgFileSpec.assign(cfgFileSpec);
        
        LOG_INFO("loading config file:: " << m_cfgFileSpec);
        
        GError *error = NULL;
        
        if (!g_key_file_load_from_file(m_pCfgKeyFile, 
            m_cfgFileSpec.c_str(), G_KEY_FILE_NONE, &error)) 
        {
            LOG_ERROR("Failed to load config file:: " << error->message);
            return false;
        }
        
        gchar** groups = g_key_file_get_groups(m_pCfgKeyFile, NULL);

        for (gchar** group = groups; *group; group++) 
        {
            LOG_INFO("Parsing group:: " << *group);
            
            switch (m_mapGroupNames[*group])
            {
                case evApplication: 
                    break;
                case evTiledDisplay: 
                    parse_tiled_display(&m_tiledDisplayConfig, m_pCfgKeyFile);
                    break;
                case evTracker: 
                    parse_tracker(&m_trackerConfig, m_pCfgKeyFile, (gchar*)(m_cfgFileSpec.c_str()));
                    break;
                case evSource0: 
                    break;
                case evSink0: 
                    break;
                case evSink1: 
                    break;
                case evSink2: 
                    break;
                case evOsd: 
                    parse_osd(&m_osdConfig, m_pCfgKeyFile);
                    break;
                case evStreamMux: 
                    parse_streammux(&m_streammuxConfig, m_pCfgKeyFile);
                    break;
                case evPrimaryGie: 
                    parse_osd(&m_osdConfig, m_pCfgKeyFile);
                    break;
                case evTests: 
                    break;
                default:
                    LOG_ERROR("Unknown group:: " << *group);
            }
                    
        }

        return true;
    }
    
//    bool Config::ParseApplicationGroup(const gchar** group)
//    {
//        LOG_FUNC();
//        
//    }
}