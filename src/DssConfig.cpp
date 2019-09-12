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

#include <math.h>

#include "Dss.h"
#include "DssConfig.h"

namespace DSS
{
         
    Config::Config(Display* pDisplay)
        : m_pDisplay(pDisplay)
        , m_pCfgKeyFile(g_key_file_new())
        , m_isPerfMetricEnabled(false)
        , m_perfMetricInterval(0)
        , m_fileLoop(0)
        , m_streamMux{0}
        , m_osd{0}
        , m_tracker{0}
        , m_tiledDisplay{0}
        , m_dsExampleConfig{0} 
        , m_sourcesBintr{0}
        , m_secondaryGiesbintr{0}
    {
        LOG_FUNC();

            
        g_mutex_init(&m_configMutex);

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
        g_mutex_clear(&m_configMutex);
    }

    bool Config::LoadFile(const std::string& cfgFileSpec) 
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_configMutex);
        
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
                    result = _parseApplicationGroup();
                    break;

                case evTiledDisplay: 
                    result = _parseTiledDisplayGroup();
                    break;

                case evTracker: 
                    result = _parseTrackerGroup();
                    break;

                case evSource0: 
                    result = _parseSourceGroup(*group);
                    break;

                case evSink0: 
                case evSink1: 
                case evSink2: 
                    result = _parseSinkGroup(*group);
                    break;
                    
                case evOsd: 
                    result = _parseOSD();
                    break;
                    
                case evStreamMux: 
                    result = _parseStreamMuxGroup();
                    break;
                    
                case evPrimaryGie: 
                    result = _parsePrimaryGieGroup(*group);
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
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_configMutex);

        LOG_INFO("Tiled display enabled:: " << m_tiledDisplay.config.enable);

        return m_tiledDisplay.config.enable;
    }

    bool Config::IsPerfMetricEnabled()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_configMutex);

        LOG_INFO("Performance metric is:: " << m_isPerfMetricEnabled);
        
        return m_isPerfMetricEnabled;
    }

    bool Config::SetPerfMetricEnabled(bool newValue)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_configMutex);
        
        bool prevValue = m_isPerfMetricEnabled;
        m_isPerfMetricEnabled = newValue;
    
        LOG_INFO("Performance metric enabled:: " << m_isPerfMetricEnabled);
        
        return prevValue;
    }

    gint Config::GetMetricInterval()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_configMutex);

        LOG_INFO("Performance metric interval:: " << m_perfMetricInterval << "s");
        
        return m_perfMetricInterval;
    }

    gint Config::SetMetricInterval(gint newValue)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_configMutex);
        
        gint prevValue = m_perfMetricInterval;
        m_perfMetricInterval = newValue;
    
        LOG_INFO("Performance metric interval:: " << m_perfMetricInterval << "s");
        
        return prevValue;
    }

    void Config::ConfigureStreamMux()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_configMutex);

        if (m_streamMux.config.is_parsed)
        {
            LOG_INFO("Setting streammux properties");
            
            set_streammux_properties(&m_streamMux.config,
                m_sourcesBintr.streammux);
        }
    }        
       
    bool Config::ConfigureTiledDisplay()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_configMutex);

        
        for (auto itSink = m_sinksConfig.begin(); itSink != m_sinksConfig.end(); itSink++)
        {
            gint width, height = 0;
            
            if (itSink->render_config.width)
            {
                width = itSink->render_config.width;
                LOG_INFO("Using render_config.width:: " << width);
            }
            else
            {
                width = m_tiledDisplay.config.width;
                LOG_INFO("Using 'tiled_display_config.width':: " << width);
            }
            if (itSink->render_config.height)
            {
                height = itSink->render_config.height;
                LOG_INFO("Using render_config.height:: " << height);
            }
            else
            {
                height = m_tiledDisplay.config.height;
                LOG_INFO("Using 'tiled_display_config.height':: " << height);
            }

            Window window = XCreateSimpleWindow(m_pDisplay, 
                RootWindow(m_pDisplay, DefaultScreen(m_pDisplay)), 
                0, 0, width, height, 2, 0x00000000, 0x00000000);            

            XSetWindowAttributes attr = { 0 };
            
            if ((m_tiledDisplay.config.enable &&
                m_tiledDisplay.config.rows * m_tiledDisplay.config.columns == 1) ||
                (!m_tiledDisplay.config.enable && m_sourcesConfig.size() == 1))
            {
                attr.event_mask = KeyPress;
            } 
            else
            {
                attr.event_mask = ButtonPress | KeyRelease;
            }
            XChangeWindowAttributes(m_pDisplay, window, CWEventMask, &attr);

            Atom wmDeleteMessage = XInternAtom(m_pDisplay, "WM_DELETE_WINDOW", False);
            if (wmDeleteMessage != None)
            {
                XSetWMProtocols(m_pDisplay, window, &wmDeleteMessage, 1);
            }
            XMapRaised(m_pDisplay, window);
            XSync(m_pDisplay, 1);       
        }     
        return true;
    }

    GstElement* Config::CreateTiledDisplayBin()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_configMutex);
        
        if (!m_tiledDisplay.config.enable)
        {
            LOG_INFO("Tiled Display is disabled");
            return NULL;
        }
        
        if (m_tiledDisplay.config.columns *
            m_tiledDisplay.config.rows < m_sourcesConfig.size()) 
        {
            // TODO - check with test... looks suspicious
            if (m_tiledDisplay.config.columns == 0) 
            {
                m_tiledDisplay.config.columns = 
                    (guint)(sqrt(m_sourcesConfig.size()) + 0.5);
            }
            m_tiledDisplay.config.rows = 
                (guint)ceil(1.0 * m_sourcesConfig.size() /
                    m_tiledDisplay.config.columns);
            LOG_WARN("Adjusting display:: " << m_tiledDisplay.config.rows << " rows, " <<
                m_tiledDisplay.config.columns << " columns");
        }
        if (!create_tiled_display_bin(&m_tiledDisplay.config, &m_tiledDisplay.bintr))
        {
            LOG_ERROR("Failed to create tiled display bin");
            throw;
        }
            
        return m_tiledDisplay.bintr.bin;
    }
    
    GstElement* Config::CreateSourcesBin()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_configMutex);
        
        if (!create_multi_source_bin(m_sourcesConfig.size(),
              &m_sourcesConfig[0], &m_sourcesBintr))
        {
            LOG_ERROR("Failed to create multi source bin");
            throw;
        }
        return m_sourcesBintr.bin;
    }

//        if (m_streamMux.config.is_parsed)
//        {
//            set_streammux_properties(&m_streamMux.config,
//                m_sourcesBintr.streammux);
//        }
        

    GstElement* Config::CreateOsdBin()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_configMutex);
        
        if (!m_osd.config.enable)
        {
            LOG_INFO("OSD is disabled");
            return NULL;
        }
        if(!create_osd_bin(&m_osd.config, &m_osd.bintr))
        {
            LOG_ERROR("Failed to create OSD bin");
            throw;
        }
        return m_osd.bintr.bin;
    }    

    GstElement* Config::CreateSinksBin()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_configMutex);

        if (!create_sink_bin(m_sinksConfig.size(), &m_sinksConfig[0], &m_sinkBintr, 0))
        {
            LOG_ERROR("Failed to create sink bin");
            throw;
        }
        return m_sinkBintr.bin;
    }
    
    GstElement* Config::CreateTrackerBin()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_configMutex);

        if (!m_tracker.config.enable)
        {
            LOG_WARN("Tracker is disabled");
            return NULL;
        }
        if(!create_tracking_bin(&m_tracker.config, &m_tracker.bintr))
        {
            LOG_ERROR("Failed to create Tracker bin");
            throw;
        }
        return m_tracker.bintr.bin;
    }

    GstElement* Config::CreatePrimaryGieBin()
    {
        
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_configMutex);

        if (!m_primaryGie.config.enable)
        {
            LOG_INFO("Primary GIE is disabled");
            return NULL;
        }
        if (!create_primary_gie_bin(&m_primaryGie.config, &m_primaryGie.bintr)) 
        {
            LOG_ERROR("Failed to create Primary GIE bin ");
            throw;
        }
        return m_primaryGie.bintr.bin;
    }
    
    GstElement* Config::CreateSecondaryGiesBin()
    {
        
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_configMutex);

        if (!m_primaryGie.config.enable)
        {
            LOG_INFO("Primary GIE is disabled");
            return NULL;
        }
        if (!create_secondary_gie_bin(m_secondaryGieConfigs.size(),
            m_secondaryGieConfigs.size(), &m_secondaryGieConfigs[0],
            &m_secondaryGiesbintr))
        {
            LOG_ERROR("Failed to create Seconday GIEs bin");
            throw;
        }
        return m_secondaryGiesbintr.bin;
    }
//        GstPad *gstpad = gst_element_get_static_pad(prevBin, "sink"); \
//        if (!gstpad)
//        {
//            LOG_ERROR("Failed to create a static pad");
//            return false;
//        } 
//        gst_element_add_pad(prevBin, gst_ghost_pad_new("sink", gstpad)); \
//        gst_object_unref(gstpad);

    bool Config::_parseApplicationGroup()
    {
        LOG_FUNC();
        
        GError *error = NULL;

        // itereate through the Groups of config options
        gchar** keys = g_key_file_get_keys(
            m_pCfgKeyFile, "application", NULL, &error);
            
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
    
    bool Config::_parseSourceGroup(gchar* group)
    {
        LOG_FUNC();
        
        if (m_sourcesConfig.size() >= MAX_SOURCE_BINS)
        {
            LOG_ERROR("Exceeded MAX_SOURCE_BINS:: " << MAX_SOURCE_BINS);
            return false;
        }

        // TODO: how to initialize?
        NvDsSourceConfig config;
        
        LOG_INFO((gchar*)m_cfgFileSpec.c_str());
        
        if (!(parse_source(
            &config, m_pCfgKeyFile, group, (gchar*)m_cfgFileSpec.c_str())))
        {
            LOG_ERROR("Failure parsing source");
            return false;
        }
        m_sourcesConfig.push_back(config);
        LOG_INFO("Source Configs count:: " << m_sourcesConfig.size());
        
        return true;
    }

    bool Config::_parseSinkGroup(gchar* group)
    {
        LOG_FUNC();
        // TODO: fix - should be collective size of all Applications
        if (m_sinksConfig.size() >= MAX_SINK_BINS)
        {
            LOG_ERROR("Exceeded MAX_SINK_BINS:: " << MAX_SINK_BINS);
            return false;
        }

        NvDsSinkSubBinConfig config = {0};
        
        if (!(parse_sink(&config, m_pCfgKeyFile, group)))
        {
            LOG_ERROR("Failure parsing sink::" << group);
            return false;
        }
        switch (config.type)
        {
        case NV_DS_SINK_FAKE:
        case NV_DS_SINK_RENDER_EGL:
        case NV_DS_SINK_RENDER_OVERLAY:
            if (!config.render_config.qos_value_specified)
            {
                // Force QoS events to be generated by sink if soures
                // are live or from synchronous non-live playback
                config.render_config.qos = 
                    m_streamMux.config.live_source || 
                    config.render_config.sync;
            }
        default:
            break;
        }

        m_sinksConfig.push_back(config);
        LOG_INFO("Sink Sub-Bin count:: " << m_sinksConfig.size());
        
        return true;
    }
    
    bool Config::_parseOSD()
    {
        LOG_FUNC();
        
        if (!parse_osd(&m_osd.config, m_pCfgKeyFile))
        {
            LOG_ERROR("Failed to parse_osd");
            return false;
        }
        
        // TODO: Why must buffers be 8 or greater.
        if (m_osd.config.num_out_buffers < 8)
        {
            m_osd.config.num_out_buffers = 8;
        }
        return true;
    }
 
    bool Config::_parsePrimaryGieGroup(gchar* group)
    {
        LOG_FUNC();
        
        if (!parse_gie(&m_primaryGie.config, m_pCfgKeyFile, 
            group, (gchar*)m_cfgFileSpec.c_str()))
        {
            LOG_ERROR("Failed to parse primary-gie");
            return false;
        }
        
        return true;
    }
            
    bool Config::_parseSecondaryGieGroup(gchar* group)
    {
        LOG_FUNC();
        
        NvDsGieConfig config;
        
        LOG_INFO((gchar*)m_cfgFileSpec.c_str());
        
        if (!(parse_gie(&config, m_pCfgKeyFile, 
            group, (gchar*)m_cfgFileSpec.c_str())))
        {
            LOG_ERROR("Failure parsing source");
            return false;
        }
        m_secondaryGieConfigs.push_back(config);
        LOG_INFO("Secondary GIE Configs count:: " << m_secondaryGieConfigs.size());
        
        return true;
    }

    bool Config::_parseStreamMuxGroup()
    {
        LOG_FUNC();
        
        if(!parse_streammux(&m_streamMux.config, m_pCfgKeyFile))
        {
            LOG_ERROR("Failed to parse_streammux");
            return false;
        }
        
        return true;
    }

    bool Config::_parseTiledDisplayGroup()
    {
        LOG_FUNC();
        
        if(!parse_tiled_display(&m_tiledDisplay.config, m_pCfgKeyFile))
        {
            LOG_ERROR("Failed to parse_tiled_display");
            return false;
        }
        return true;
    }

    bool Config::_parseTrackerGroup()
    {
        LOG_FUNC();

        if (!parse_tracker(&m_tracker.config, m_pCfgKeyFile, 
            (gchar*)m_cfgFileSpec.c_str()))
        {
            LOG_ERROR("Failed to parse tracker");
            return false;
        }
        return true;
    }
        
} // namespace DSS  
