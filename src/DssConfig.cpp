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
        : m_pCfgKeyFile(g_key_file_new())
        , m_isPerfMetricEnabled(false)
        , m_perfMetricInterval(0)
        , m_fileLoop(0)
        , m_streammuxConfig{0}
        , m_osdConfig{0}
        , m_primaryGieConfig{0}
        , m_trackerConfig{0}
        , m_tiledDisplayConfig{0}
        , m_dsExampleConfig{0}

    {
        LOG_FUNC();
        
        // Map the CFG group names
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
        
        // Map the Application specific CFG items
        m_mapAppCfgItems["enable-perf-measurement"] = evEnablePerfMeasurement;
        m_mapAppCfgItems["perf-measurement-interval-sec"] = evPerfMeasurementInteralSec;
        m_mapAppCfgItems["gie-kitti-output-dir"] = evGieKittiOutputDir;
        m_mapAppCfgItems["kitti-track-output-dir"] = evKittiTrackOutputDir;

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

        // itereate through the Groups of config options
        gchar** groups = g_key_file_get_groups(m_pCfgKeyFile, NULL);

        for (gchar** group = groups; *group; group++) 
        {
            LOG_INFO("Parsing group:: " << *group);
            bool result = true;
            
            switch (m_mapGroupNames[*group])
            {
                case evApplication: 
                    result = ParseApplicationGroup();
                    break;

                case evTiledDisplay: 
                    result = parse_tiled_display(
                        &m_tiledDisplayConfig, m_pCfgKeyFile);
                    break;

                case evTracker: 
                    result = parse_tracker(
                        &m_trackerConfig, m_pCfgKeyFile, (gchar*)m_cfgFileSpec.c_str());
                    break;

                case evSource0: 
                    result = ParseSourceGroup(*group);
                    break;

                case evSink0: 
                case evSink1: 
                case evSink2: 
                    result = ParseSinkGroup(*group);
                    break;
                    
                case evOsd: 
                    result = parse_osd(&m_osdConfig, m_pCfgKeyFile);
                    break;
                    
                case evStreamMux: 
                    result = parse_streammux(&m_streammuxConfig, m_pCfgKeyFile);
                    break;
                    
                case evPrimaryGie: 
                    result = parse_gie(
                        &m_primaryGieConfig, m_pCfgKeyFile, *group, (gchar*)m_cfgFileSpec.c_str());
                    break;
                    
                case evTests: 
                    break;
                default:
                    LOG_ERROR("Unknown group:: " << *group);
                    result = false;
            }
            if(!result)
            {
                LOG_ERROR("Failure parsing group:: " << *group);
                return false;
            }
        }

        return true;
    }

    bool Config::IsTiledDisplayEnabled()
    {
        LOG_FUNC();
        LOG_INFO("Tiled display enabled:: " << m_tiledDisplayConfig.enable);
        
        return m_tiledDisplayConfig.enable;
    }

    bool Config::IsPerfMetricEnabled()
    {
        LOG_FUNC();
        LOG_INFO("Performance metric is:: " << m_isPerfMetricEnabled);
        
        return m_isPerfMetricEnabled;
    }

    bool Config::SetPerfMetricEnabled(bool newValue)
    {
        LOG_FUNC();
        
        bool prevValue = m_isPerfMetricEnabled;
        m_isPerfMetricEnabled = newValue;
    
        LOG_INFO("Performance metric enabled:: " << m_isPerfMetricEnabled);
        
        return prevValue;
    }

    gint Config::GetMetricInterval()
    {
        LOG_FUNC();
        LOG_INFO("Performance metric interval:: " << m_perfMetricInterval << "s");
        
        return m_perfMetricInterval;
    }

    gint Config::SetMetricInterval(gint newValue)
    {
        LOG_FUNC();
        
        gint prevValue = m_perfMetricInterval;
        m_perfMetricInterval = newValue;
    
        LOG_INFO("Performance metric interval:: " << m_perfMetricInterval << "s");
        
        return prevValue;
    }
    
    void Config::ConfigureNewXWindows()
    {
        LOG_FUNC();
        
        for (auto i = m_sinkSubBinsConfigs.begin(); i != m_sinkSubBinsConfigs.end(); ++i)
        {
            uint width, height = 0;
            if (i.render_config.width)
            {
                width = i.render_config.width
                LOG_INFO("Using render_config.width:: " << width);
            }
            else
            {
                width = i.tiled_display_config.width;
                LOG_INFO("Using 'tiled_display_config.width':: " << width);
            }
            if (i.render_config.height)
            {
                height = i.render_config.height
                LOG_INFO("Using render_config.height:: " << height);
            }
            else
            {
                height = i.tiled_display_config.height;
                LOG_INFO("Using 'tiled_display_config.height':: " << height);
            }
            
            m_windows.push_back(
                XCreateSimpleWindow(display, 
                    RootWindow(display, DefaultScreen(display)), 
                    0, 0, width, height, 2, 0, 0);            
        }
    }
    

    bool Config::ParseApplicationGroup()
    {
        LOG_FUNC();
        
        GError *error = NULL;

        // itereate through the Groups of config options
        gchar** keys = g_key_file_get_keys(m_pCfgKeyFile, 
            "application", NULL, &error);
            
        for (gchar** key = keys; *key; key++) 
        {
            LOG_INFO("Parsing key:: " << *key);
            bool result = true;

            switch (m_mapAppCfgItems[*key])
            {
                case evEnablePerfMeasurement: 
                    m_isPerfMetricEnabled = g_key_file_get_integer(
                        m_pCfgKeyFile, "application", *key, &error);
                    result = !error;
                    break;
                    
                case evPerfMeasurementInteralSec: 
                    m_perfMetricInterval = g_key_file_get_integer(
                        m_pCfgKeyFile, "application", *key, &error);
                    result = !error;
                    break;
                    
                case evGieKittiOutputDir: 
                    break;
                case evKittiTrackOutputDir: 
                    break;
                default:
                    LOG_ERROR("Unknown key:: " << *key);
                    result = false;
            }
            if(!result)
            {
                LOG_ERROR("Failure parsing key:: " << *key);
                return false;
            }
        }
        return true;
    }
    bool Config::ParseSourceGroup(gchar* group)
    {
        if (m_sourceConfigs.size() >= MAX_SOURCE_BINS)
        {
            LOG_ERROR("Exceeded MAX_SOURCE_BINS:: " << MAX_SOURCE_BINS);
            return false;
        }

        NvDsSourceConfig sourceConfig;
        
        if (!(parse_source(
            &sourceConfig, m_pCfgKeyFile, group, (gchar*)m_cfgFileSpec.c_str())))
        {
            LOG_ERROR("Failure parsing source");
            return false;
        }
        m_sourceConfigs.push_back(sourceConfig);
        LOG_INFO("Source Configs count:: " << m_sourceConfigs.size());
        return true;
    }

    bool Config::ParseSinkGroup(gchar* group)
    {
        if (m_sinkSubBinsConfigs.size() >= MAX_SINK_BINS)
        {
            LOG_ERROR("Exceeded MAX_SINK_BINS:: " << MAX_SINK_BINS);
            return false;
        }

        NvDsSinkSubBinConfig sinkSubBinConfig;
        
        if (!(parse_sink(&sinkSubBinConfig, m_pCfgKeyFile, group)))
        {
            LOG_ERROR("Failure parsing sink::" << group);
            return false;
        }
        m_sinkSubBinsConfigs.push_back(sinkSubBinConfig);
        LOG_INFO("Sink Sub-Bin Configs count:: " << m_sinkSubBinsConfigs.size());
        return true;
    }
}