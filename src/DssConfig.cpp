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
                    result = ParseTiledDisplayGroup();
                    break;

                case evTracker: 
                    result = ParseTrackerGroup();
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
                    result = ParseOSD();
                    break;
                    
                case evStreamMux: 
                    result = ParseStreamMuxGroup();
                    break;
                    
                case evPrimaryGie: 
                    result = ParsePrimaryGieGroup(*group);
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
        LOG_INFO("Tiled display enabled:: " << m_tiledDisplay.config.enable);
        
        return m_tiledDisplay.config.enable;
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
    

    bool Config::ParseApplicationGroup()
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
    
    bool Config::ParseSourceGroup(gchar* group)
    {
        LOG_FUNC();
        
        if (m_vSources.size() >= MAX_SOURCE_BINS)
        {
            LOG_ERROR("Exceeded MAX_SOURCE_BINS:: " << MAX_SOURCE_BINS);
            return false;
        }

//        Source source = {0};
        Source source;
        
        if (!(parse_source(
            &source.config, m_pCfgKeyFile, group, (gchar*)m_cfgFileSpec.c_str())))
        {
            LOG_ERROR("Failure parsing source");
            return false;
        }
        if (!create_source_bin(&source.config, &source.bin)) 
        {
            LOG_ERROR("Failed to create source bin ");
            return false;
        }
        m_vSources.push_back(source);
        LOG_INFO("Source Configs count:: " << m_vSources.size());
        
        return true;
    }

    bool Config::ParseSinkGroup(gchar* group)
    {
        LOG_FUNC();
        // TODO: fix - should be collective size of all Applications
        if (m_vSinks.size() >= MAX_SINK_BINS)
        {
            LOG_ERROR("Exceeded MAX_SINK_BINS:: " << MAX_SINK_BINS);
            return false;
        }

        Sink sink = { 0 };
        
        if (!(parse_sink(&sink.config, m_pCfgKeyFile, group)))
        {
            LOG_ERROR("Failure parsing sink::" << group);
            return false;
        }
        switch (sink.config.type)
        {
        case NV_DS_SINK_FAKE:
        case NV_DS_SINK_RENDER_EGL:
        case NV_DS_SINK_RENDER_OVERLAY:
            if (!sink.config.render_config.qos_value_specified)
            {
                // Force QoS events to be generated by sink if soures
                // are live or from synchronous non-live playback
                sink.config.render_config.qos = 
                    m_streamMux.config.live_source || 
                    sink.config.render_config.sync;
            }
        default:
            break;
        }

//              if (!create_sink_bin(m_sinkSubBins.size(),
//                    i->config, &i->sinkSubBin.sink, index)) {
//                goto done;
//              }
            
//            if (!GST_IS_VIDEO_OVERLAY(i->sinkSubBin.sink)){
//                LOG_INFO("!GST_IS_VIDEO_OVERLAY");
//                continue;
//            }

        m_vSinks.push_back(sink);
        LOG_INFO("Sink Sub-Bin count:: " << m_vSinks.size());
        
        return true;
    }
    
    bool Config::ParseOSD()
    {
        LOG_FUNC();
        
        if (!parse_osd(&m_osd.config, m_pCfgKeyFile))
        {
            LOG_ERROR("Failed to parse_osd");
        }
        if (!create_osd_bin(&m_osd.config, &m_osd.bin)) 
        {
            LOG_ERROR("Failed to create tracker bin ");
            return false;
        }
    }
 
    bool Config::ParsePrimaryGieGroup(gchar* group)
    {
        LOG_FUNC();
        
        PrimaryGie primaryGie;
        
        if (!parse_gie(&primaryGie.config, m_pCfgKeyFile, 
            group, (gchar*)m_cfgFileSpec.c_str()))
        {
            LOG_ERROR("Failed to parse parse");
            return false;
        }
        if (!create_primary_gie_bin(&primaryGie.config, &primaryGie.bin)) 
        {
            LOG_ERROR("Failed to create tracker bin ");
            return false;
        }
        m_vPrimaryGies.push_back(primaryGie);
        
        return true;
    }
            
    bool Config::ParseStreamMuxGroup()
    {
        LOG_FUNC();
        
        if(!parse_streammux(&m_streamMux.config, m_pCfgKeyFile))
        {
            LOG_ERROR("Failed to parse_streammux");
            return false;
        }
        NVGSTDS_LINK_ELEMENT (pipeline->multi_src_bin.bin, last_elem);

        set_streammux_properties(&m_streamMux.config, m_streamMux.bin.streammux);
        
        return true;
    }

    bool Config::ParseTiledDisplayGroup()
    {
        LOG_FUNC();
        
        if(!parse_tiled_display(&m_tiledDisplay.config, m_pCfgKeyFile))
        {
            LOG_ERROR("Failed to parse_tiled_display");
            return false;
        }
        return true;
    }

    bool Config::ParseTrackerGroup()
    {
        LOG_FUNC();

        if (!parse_tracker(&m_tracker.config, m_pCfgKeyFile, 
            (gchar*)m_cfgFileSpec.c_str()))
        {
            LOG_ERROR("Failed to parse tracker");
            return false;
        }    
        if (m_tracker.config.enable) 
        {
            if (!create_tracking_bin(&m_tracker.config, &m_tracker.bin)) 
            {
                LOG_ERROR("Failed to create tracker bin ");
                return false;
            }
        }
        

    }
        
    bool Config::ConfigureTiledDisplay()
    {
        LOG_FUNC();

        if (!m_tiledDisplay.config.enable)
        {
            LOG_WARN("Tiled Display is disabled");
            return false;
        }
        
        if (m_tiledDisplay.config.columns *
            m_tiledDisplay.config.rows < m_vSources.size()) 
        {
            // TODO - check with test... looks suspicious
            if (m_tiledDisplay.config.columns == 0) 
            {
                m_tiledDisplay.config.columns = 
                    (guint)(sqrt(m_vSources.size()) + 0.5);
            }
            m_tiledDisplay.config.rows = 
                (guint)ceil(1.0 * m_vSources.size() /
                    m_tiledDisplay.config.columns);
            LOG_WARN("Adjusting display:: " << m_tiledDisplay.config.rows << " rows, " <<
                m_tiledDisplay.config.columns << " columns");
        }
        
        for (std::vector<Sink>::iterator itSink = m_vSinks.begin(); itSink != m_vSinks.end(); itSink++)
        {
            gint width, height = 0;
            
            if (itSink->config.render_config.width)
            {
                width = itSink->config.render_config.width;
                LOG_INFO("Using render_config.width:: " << width);
            }
            else
            {
                width = m_tiledDisplay.config.width;
                LOG_INFO("Using 'tiled_display_config.width':: " << width);
            }
            if (itSink->config.render_config.height)
            {
                height = itSink->config.render_config.height;
                LOG_INFO("Using render_config.height:: " << height);
            }
            else
            {
                height = m_tiledDisplay.config.height;
                LOG_INFO("Using 'tiled_display_config.height':: " << height);
            }

            itSink->window = XCreateSimpleWindow(m_pDisplay, 
                RootWindow(m_pDisplay, DefaultScreen(m_pDisplay)), 
                0, 0, width, height, 2, 0x00000000, 0x00000000);            

            XSetWindowAttributes attr = { 0 };
            
            if ((m_tiledDisplay.config.enable &&
                m_tiledDisplay.config.rows * m_tiledDisplay.config.columns == 1) ||
                (!m_tiledDisplay.config.enable && m_vSources.size() == 1))
            {
                attr.event_mask = KeyPress;
            } 
            else
            {
                attr.event_mask = ButtonPress | KeyRelease;
            }
            XChangeWindowAttributes(m_pDisplay, itSink->window, CWEventMask, &attr);

            Atom wmDeleteMessage = XInternAtom(m_pDisplay, "WM_DELETE_WINDOW", False);
            if (wmDeleteMessage != None)
            {
                XSetWMProtocols(m_pDisplay, itSink->window, &wmDeleteMessage, 1);
            }
            XMapRaised(m_pDisplay, itSink->window);
            XSync(m_pDisplay, 1);       

            
            if (!create_tiled_display_bin(&m_tiledDisplay.config, &m_tiledDisplay.bin))
            {
                LOG_ERROR("Failed to create tiled display bin");
                return false;
            }
        }     
        return true;
    }
   
