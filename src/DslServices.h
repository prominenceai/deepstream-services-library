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

#ifndef _DSL_DRIVER_H
#define _DSL_DRIVER_H

#include "DslPipelineBintr.h"

namespace DSL {
    
    /**
     * @class Services
     * @brief Implements a singlton instance 
     */
    class Services
    {
    public:
    
        /** 
         * @brief Returns a pointer to this singleton
         * 
         * @return instance pointer to Services
         */
        static Services* GetServices();
        
        /***************************************************************
         **** all Services defined below are documented in DslApi.h ****
         ***************************************************************/
        DslReturnType SourceCsiNew(const char* name, 
            uint width, uint height, uint fps_n, uint fps_d);
        
        DslReturnType SourceUriNew(const char* name, const char* uri, 
            boolean isLive, uint cudadecMemType, uint intraDecode, uint dropFrameInterval);
            
        DslReturnType SourceRtspNew(const char* name, const char* uri, 
            uint protocol, uint cudadecMemType, uint intraDecode, uint dropFrameInterval);
            
        DslReturnType SourceDimensionsGet(const char* name, uint* width, uint* height);
        
        DslReturnType SourceFrameRateGet(const char* name, uint* fps_n, uint* fps_d);
            
        DslReturnType SourcePause(const char* name);

        DslReturnType SourceResume(const char* name);

        boolean SourceIsLive(const char* name);
        
        uint GetNumSourceInUse();
        
        uint GetNumSourceInUseMax();
        
        void SetNumSourceInUseMax(uint max);
        
        DslReturnType PrimaryGieNew(const char* name, const char* inferConfigFile,
            const char* modelEngineFile, uint interval);
        
        DslReturnType PrimaryGieBatchMetaHandlerAdd(const char* name, uint pad, dsl_batch_meta_handler_cb handler, void* user_data);

        DslReturnType PrimaryGieBatchMetaHandlerRemove(const char* name, uint pad);

        DslReturnType SecondaryGieNew(const char* name, const char* inferConfigFile,
            const char* modelEngineFile, const char* inferOnGieName);
            
        DslReturnType TrackerKtlNew(const char* name, uint width, uint height);
        
        DslReturnType TrackerIouNew(const char* name, const char* configFile, uint width, uint height);
        
        DslReturnType TrackerMaxDimensionsGet(const char* name, uint* width, uint* height);
        
        DslReturnType TrackerMaxDimensionsSet(const char* name, uint width, uint height);
        
        DslReturnType TrackerBatchMetaHandlerAdd(const char* name, uint pad, dsl_batch_meta_handler_cb handler, void* user_data);

        DslReturnType TrackerBatchMetaHandlerRemove(const char* name, uint pad);
        
        DslReturnType TilerNew(const char* name, uint width, uint height);
        
        DslReturnType TilerDimensionsGet(const char* name, uint* width, uint* height);

        DslReturnType TilerDimensionsSet(const char* name, uint width, uint height);

        DslReturnType TilerTilesGet(const char* name, uint* cols, uint* rows);

        DslReturnType TilerTilesSet(const char* name, uint cols, uint rows);

        DslReturnType TilerBatchMetaHandlerAdd(const char* name, uint pad, dsl_batch_meta_handler_cb handler, void* user_data);

        DslReturnType TilerBatchMetaHandlerRemove(const char* name, uint pad);
        
        DslReturnType OsdNew(const char* name, boolean isClockEnabled);

        DslReturnType OsdBatchMetaHandlerAdd(const char* name, uint pad, dsl_batch_meta_handler_cb handler, void* user_data);

        DslReturnType OsdBatchMetaHandlerRemove(const char* name, uint pad);

        DslReturnType SinkOverlayNew(const char* name, 
            uint offsetX, uint offsetY, uint width, uint height);
                
        DslReturnType SinkWindowNew(const char* name, 
            uint offsetX, uint offsetY, uint width, uint height);
                
// TODO        
//        boolean ComponentIsInUse(const char* component);
        
        DslReturnType ComponentDelete(const char* component);

        DslReturnType ComponentDeleteAll();
        
        uint ComponentListSize();
        
        DslReturnType PipelineNew(const char* pipeline);
        
        DslReturnType PipelineDelete(const char* pipeline);
        
        DslReturnType PipelineDeleteAll();

        uint PipelineListSize();
        
        DslReturnType PipelineComponentAdd(const char* pipeline, const char* component);

        DslReturnType PipelineComponentRemove(const char* pipeline, const char* component);

        DslReturnType PipelineStreamMuxBatchPropertiesGet(const char* pipeline,
            uint* batchSize, uint* batchTimeout);

        DslReturnType PipelineStreamMuxBatchPropertiesSet(const char* pipeline,
            uint batchSize, uint batchTimeout);

        DslReturnType PipelineStreamMuxDimensionsGet(const char* pipeline,
            uint* width, uint* height);

        DslReturnType PipelineStreamMuxDimensionsSet(const char* pipeline,
            uint width, uint height);
            
        DslReturnType PipelineStreamMuxPaddingGet(const char* pipeline, boolean* enabled);

        DslReturnType PipelineStreamMuxPaddingSet(const char* pipeline, boolean enabled);

        DslReturnType PipelineXWindowDimensionsGet(const char* pipeline,
            uint* width, uint* height);

        DslReturnType PipelineXWindowDimensionsSet(const char* pipeline,
            uint width, uint height);
            
        DslReturnType PipelinePause(const char* pipeline);
        
        DslReturnType PipelinePlay(const char* pipeline);
        
        DslReturnType PipelineStop(const char* pipeline);
        
        DslReturnType PipelineGetState(const char* pipeline);
        
        DslReturnType PipelineDumpToDot(const char* pipeline, char* filename);
        
        DslReturnType PipelineDumpToDotWithTs(const char* pipeline, char* filename);
        
        DslReturnType PipelineStateChangeListenerAdd(const char* pipeline, 
            dsl_state_change_listener_cb listener, void* userdata);
        
        DslReturnType PipelineStateChangeListenerRemove(const char* pipeline, 
            dsl_state_change_listener_cb listener);
                        
        DslReturnType PipelineEosListenerAdd(const char* pipeline, 
            dsl_eos_listener_cb listener, void* userdata);
        
        DslReturnType PipelineEosListenerRemove(const char* pipeline, 
            dsl_eos_listener_cb listener);
                        
        DslReturnType PipelineXWindowKeyEventHandlerAdd(const char* pipeline, 
            dsl_xwindow_key_event_handler_cb handler, void* userdata);

        DslReturnType PipelineXWindowKeyEventHandlerRemove(const char* pipeline, 
            dsl_xwindow_key_event_handler_cb handler);

        DslReturnType PipelineXWindowButtonEventHandlerAdd(const char* pipeline, 
            dsl_xwindow_button_event_handler_cb handler, void* userdata);

        DslReturnType PipelineXWindowButtonEventHandlerRemove(const char* pipeline, 
            dsl_xwindow_button_event_handler_cb handler);
        
        DslReturnType PipelineXWindowDeleteEventHandlerAdd(const char* pipeline, 
            dsl_xwindow_delete_event_handler_cb handler, void* userdata);

        DslReturnType PipelineXWindowDeleteEventHandlerRemove(const char* pipeline, 
            dsl_xwindow_delete_event_handler_cb handler);
        
        GMainLoop* GetMainLoopHandle()
        {
            LOG_FUNC();
            LOG_INFO("Returning Handle to MainLoop");
            
            return m_pMainLoop;
        }
                        
        /** 
         * @brief Handles all pending events
         * 
         * @return true if all events were handled succesfully
         */
        bool HandleXWindowEvents(); 

    private:
    
        /**
         * @brief called during construction to intialize all const-to-string maps
         */
        void _initMaps();

        /**
         * @brief private ctor for this singleton class
         * @parm[in] doGstDeinit, if true, the Services will call gst_deinit on dtor.
         */
        Services(bool doGstDeinit);

        /**
         * @brief private dtor for this singleton class
         */
        ~Services();
        
        /**
         * @brief instance pointer for this singleton class
         */
        static Services* m_pInstatnce;
        
        /**
         * @breif flag set during construction, determines if the Services should call
         * gst_deinit() on dtor, or to let the client. 
         */
        bool m_doGstDeinit;
        
        /**
         * @brief handle to the single main loop
        */
        GMainLoop* m_pMainLoop;
            
        /**
         * @brief mutex to prevent Services reentry
        */
        GMutex m_servicesMutex;

        /**
         * @brief maximum number of sources that can be in use at one time
         * Set to the default in service contructor, the value can be read
         * and updated as the first call to DSL.
         */
        uint m_numSourceInUseMax;
        
        /**
         * @brief map of all pipelines creaated by the client, key=name
         */
        std::map <std::string, std::shared_ptr<PipelineBintr>> m_pipelines;
        
        /**
         * @brief map of all pipeline components creaated by the client, key=name
         */
        std::map <std::string, std::shared_ptr<Bintr>> m_components;
        
        std::map <uint, std::string> m_mapParserTypes;
    };  

    static gboolean MainLoopThread(gpointer arg);
}


#endif // _DSL_DRIVER_H