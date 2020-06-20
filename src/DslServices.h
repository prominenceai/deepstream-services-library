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

#include "Dsl.h"
#include "DslApi.h"
#include "DslBase.h"
#include "DslOdeAction.h"
#include "DslOdeArea.h"
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
        DslReturnType OdeActionCallbackNew(const char* name,
            dsl_ode_handle_occurrence_cb clientHandler, void* clientData);

        DslReturnType OdeActionCaptureFrameNew(const char* name, const char* outdir);
        
        DslReturnType OdeActionCaptureObjectNew(const char* name, const char* outdir);
        
        DslReturnType OdeActionDisplayNew(const char* name,
            uint offsetX, uint offsetY, bool offsetY_with_classId);
        
        DslReturnType OdeActionLogNew(const char* name);
        
        DslReturnType OdeActionFillNew(const char* name,
            double red, double green, double blue, double alpha);

        DslReturnType OdeActionHandlerDisableNew(const char* name, const char* handler);

        DslReturnType OdeActionHideNew(const char* name, boolean text, boolean border);
        
        DslReturnType OdeActionPauseNew(const char* name, const char* pipeline);

        DslReturnType OdeActionPrintNew(const char* name);
        
        DslReturnType OdeActionRedactNew(const char* name);

        DslReturnType OdeActionSinkAddNew(const char* name, 
            const char* pipeline, const char* sink);

        DslReturnType OdeActionSinkRemoveNew(const char* name, 
            const char* pipeline, const char* sink);

        DslReturnType OdeActionSourceAddNew(const char* name, 
            const char* pipeline, const char* source);

        DslReturnType OdeActionSourceRemoveNew(const char* name, 
            const char* pipeline, const char* source);

        DslReturnType OdeActionActionAddNew(const char* name, 
            const char* trigger, const char* odeAction);

        DslReturnType OdeActionActionDisableNew(const char* name, const char* action);

        DslReturnType OdeActionActionEnableNew(const char* name, const char* action);

        DslReturnType OdeActionActionRemoveNew(const char* name, 
            const char* trigger, const char* action);

        DslReturnType OdeActionAreaAddNew(const char* name, 
            const char* trigger, const char* area);
        
        DslReturnType OdeActionAreaRemoveNew(const char* name, 
            const char* trigger, const char* area);
        
        DslReturnType OdeActionTriggerAddNew(const char* name, 
            const char* odeHandler, const char* trigger);

        DslReturnType OdeActionTriggerDisableNew(const char* name, const char* trigger);

        DslReturnType OdeActionTriggerEnableNew(const char* name, const char* trigger);

        DslReturnType OdeActionTriggerRemoveNew(const char* name, 
            const char* odeHandler, const char* trigger);
        
        DslReturnType OdeActionEnabledGet(const char* name, boolean* enabled);

        DslReturnType OdeActionEnabledSet(const char* name, boolean enabled);

        DslReturnType OdeActionDelete(const char* name);
        
        DslReturnType OdeActionDeleteAll();
        
        uint OdeActionListSize();

        DslReturnType OdeAreaNew(const char* name, 
            uint left, uint top, uint width, uint height, boolean display);

        DslReturnType OdeAreaGet(const char* name, 
            uint* left, uint* top, uint* width, uint* height, boolean* display);

        DslReturnType OdeAreaSet(const char* name, 
            uint left, uint top, uint width, uint height, boolean display);

        DslReturnType OdeAreaColorGet(const char* name, 
            double* red, double* green, double* blue, double* alpha);

        DslReturnType OdeAreaColorSet(const char* name, 
            double red, double green, double blue, double alpha);

        DslReturnType OdeAreaDelete(const char* name);
        
        DslReturnType OdeAreaDeleteAll();
        
        uint OdeAreaListSize();
        
        DslReturnType OdeTriggerOccurrenceNew(const char* name, uint classId, uint limit);
        
        DslReturnType OdeTriggerAbsenceNew(const char* name, uint classId, uint limit);
        
        DslReturnType OdeTriggerIntersectionNew(const char* name, uint classId, uint limit);

        DslReturnType OdeTriggerSummationNew(const char* name, uint classId, uint limit);

        DslReturnType OdeTriggerCustomNew(const char* name, 
            uint classId, uint limit,  dsl_ode_check_for_occurrence_cb client_checker, void* client_data);

        DslReturnType OdeTriggerMinimumNew(const char* name, uint classId, uint limit, uint minimum);
        
        DslReturnType OdeTriggerMaximumNew(const char* name, uint classId, uint limit, uint maximum);
        
        DslReturnType OdeTriggerEnabledGet(const char* name, boolean* enabled);

        DslReturnType OdeTriggerEnabledSet(const char* name, boolean enabled);

        DslReturnType OdeTriggerClassIdGet(const char* name, uint* classId);
        
        DslReturnType OdeTriggerClassIdSet(const char* name, uint classId);
        
        DslReturnType OdeTriggerSourceIdGet(const char* name, uint* sourceId);
        
        DslReturnType OdeTriggerSourceIdSet(const char* name, uint sourceId);
        
        DslReturnType OdeTriggerDimensionsMinGet(const char* name, uint* min_width, uint* min_height);
        
        DslReturnType OdeTriggerDimensionsMinSet(const char* name, uint min_width, uint min_height);

        DslReturnType OdeTriggerFrameCountMinGet(const char* name, uint* min_count_n, uint* min_count_d);

        DslReturnType OdeTriggerFrameCountMinSet(const char* name, uint min_count_n, uint min_count_d);
        
        DslReturnType OdeTriggerActionAdd(const char* name, const char* action);

        DslReturnType OdeTriggerActionRemove(const char* name, const char* action);

        DslReturnType OdeTriggerActionRemoveAll(const char* name);

        DslReturnType OdeTriggerAreaAdd(const char* name, const char* area);

        DslReturnType OdeTriggerAreaRemove(const char* name, const char* area);

        DslReturnType OdeTriggerAreaRemoveAll(const char* name);

        DslReturnType OdeTriggerDelete(const char* name);
        
        DslReturnType OdeTriggerDeleteAll();
        
        uint OdeTriggerListSize();
        
        DslReturnType SourceCsiNew(const char* name, 
            uint width, uint height, uint fps_n, uint fps_d);
        
        DslReturnType SourceUsbNew(const char* name, 
            uint width, uint height, uint fps_n, uint fps_d);
        
        DslReturnType SourceUriNew(const char* name, const char* uri, 
            boolean isLive, uint cudadecMemType, uint intraDecode, uint dropFrameInterval);
            
        DslReturnType SourceRtspNew(const char* name, const char* uri, 
            uint protocol, uint cudadecMemType, uint intraDecode, uint dropFrameInterval);
            
        DslReturnType SourceDimensionsGet(const char* name, uint* width, uint* height);
        
        DslReturnType SourceFrameRateGet(const char* name, uint* fps_n, uint* fps_d);

        DslReturnType SourceDecodeUriGet(const char* name, const char** uri);

        DslReturnType SourceDecodeUriSet(const char* name, const char* uri);
    
        DslReturnType SourceDecodeDewarperAdd(const char* name, const char* dewarper);
    
        DslReturnType SourceDecodeDewarperRemove(const char* name);
    
        DslReturnType SourcePause(const char* name);

        DslReturnType SourceResume(const char* name);

        boolean SourceIsLive(const char* name);
        
        uint SourceNumInUseGet();
        
        uint SourceNumInUseMaxGet();
        
        boolean SourceNumInUseMaxSet(uint max);
        
        DslReturnType DewarperNew(const char* name, const char* configFile);

        DslReturnType PrimaryGieNew(const char* name, const char* inferConfigFile,
            const char* modelEngineFile, uint interval);

        DslReturnType PrimaryGieKittiOutputEnabledSet(const char* name, boolean enabled, const char* file);
        
        DslReturnType PrimaryGieBatchMetaHandlerAdd(const char* name, uint pad, dsl_batch_meta_handler_cb handler, void* userData);

        DslReturnType PrimaryGieBatchMetaHandlerRemove(const char* name, uint pad, dsl_batch_meta_handler_cb handler);

        DslReturnType SecondaryGieNew(const char* name, const char* inferConfigFile,
            const char* modelEngineFile, const char* inferOnGieName, uint interval);

        DslReturnType GieInferConfigFileGet(const char* name, const char** inferConfigFile);

        DslReturnType GieInferConfigFileSet(const char* name, const char* inferConfigFile);
            
        DslReturnType GieModelEngineFileGet(const char* name, const char** modelEngineFile);

        DslReturnType GieModelEngineFileSet(const char* name, const char* modelEngineFile);
            
        DslReturnType GieRawOutputEnabledSet(const char* name, boolean enabled,
            const char* path);
            
        DslReturnType GieIntervalGet(const char* name, uint* interval);

        DslReturnType GieIntervalSet(const char* name, uint interval);

        DslReturnType TrackerKtlNew(const char* name, uint width, uint height);
        
        DslReturnType TrackerIouNew(const char* name, const char* configFile, uint width, uint height);
        
        DslReturnType TrackerMaxDimensionsGet(const char* name, uint* width, uint* height);
        
        DslReturnType TrackerMaxDimensionsSet(const char* name, uint width, uint height);
        
        DslReturnType TrackerBatchMetaHandlerAdd(const char* name, uint pad, dsl_batch_meta_handler_cb handler, void* userData);

        DslReturnType TrackerBatchMetaHandlerRemove(const char* name, uint pad, dsl_batch_meta_handler_cb handler);
        
        DslReturnType TrackerKittiOutputEnabledSet(const char* name, boolean enabled, const char* file);

        DslReturnType TeeDemuxerNew(const char* name);
        
        DslReturnType TeeSplitterNew(const char* name);
        
        DslReturnType TeeBranchAdd(const char* demuer, const char* branch);

        DslReturnType TeeBranchRemove(const char* demuxer, const char* branch);
        
        DslReturnType TeeBranchRemoveAll(const char* demuxer);

        DslReturnType TeeBranchCountGet(const char* demuxer, uint* count);

        DslReturnType TeeBatchMetaHandlerAdd(const char* name, dsl_batch_meta_handler_cb handler, void* userData);

        DslReturnType TeeBatchMetaHandlerRemove(const char* name, dsl_batch_meta_handler_cb handler);
        
        DslReturnType TilerNew(const char* name, uint width, uint height);
        
        DslReturnType TilerDimensionsGet(const char* name, uint* width, uint* height);

        DslReturnType TilerDimensionsSet(const char* name, uint width, uint height);

        DslReturnType TilerTilesGet(const char* name, uint* cols, uint* rows);

        DslReturnType TilerTilesSet(const char* name, uint cols, uint rows);

        DslReturnType TilerBatchMetaHandlerAdd(const char* name, uint pad, dsl_batch_meta_handler_cb handler, void* userData);

        DslReturnType TilerBatchMetaHandlerRemove(const char* name, uint pad, dsl_batch_meta_handler_cb handler);
        
        DslReturnType OdeHandlerNew(const char* name);

        DslReturnType OdeHandlerEnabledGet(const char* name, boolean* enabled);
        
        DslReturnType OdeHandlerEnabledSet(const char* name, boolean enabled);
        
        DslReturnType OdeHandlerTriggerAdd(const char* odeHandler, const char* trigger);

        DslReturnType OdeHandlerTriggerRemove(const char* odeHandler, const char* trigger);

        DslReturnType OdeHandlerTriggerRemoveAll(const char* odeHandler);

        DslReturnType OfvNew(const char* name);

        DslReturnType OsdNew(const char* name, boolean clockEnabled);
        
        DslReturnType OsdClockEnabledGet(const char* name, boolean* enabled);

        DslReturnType OsdClockEnabledSet(const char* name, boolean enabled);

        DslReturnType OsdClockOffsetsGet(const char* name, uint* offsetX, uint* offsetY);

        DslReturnType OsdClockOffsetsSet(const char* name, uint offsetX, uint offsetY);

        DslReturnType OsdClockFontGet(const char* name, const char** font, uint* size);

        DslReturnType OsdClockFontSet(const char* name, const char* font, uint size);

        DslReturnType OsdClockColorGet(const char* name, double* red, double* green, double* blue, double* alpha);

        DslReturnType OsdClockColorSet(const char* name, double red, double green, double blue, double alpha);

        DslReturnType OsdCropSettingsGet(const char* name, uint* left, uint* top, uint* width, uint* height);

        DslReturnType OsdCropSettingsSet(const char* name, uint left, uint top, uint width, uint height);

        DslReturnType OsdRedactionEnabledGet(const char* name, boolean* enabled);

        DslReturnType OsdRedactionEnabledSet(const char* name, boolean enabled);

        DslReturnType OsdRedactionClassAdd(const char* name, int classId, double red, double blue, double green, double alpha);

        DslReturnType OsdRedactionClassRemove(const char* name, int classId);

        DslReturnType OsdBatchMetaHandlerAdd(const char* name, uint pad, dsl_batch_meta_handler_cb handler, void* userData);

        DslReturnType OsdBatchMetaHandlerRemove(const char* name, uint pad, dsl_batch_meta_handler_cb handler);

        DslReturnType OsdKittiOutputEnabledSet(const char* name, boolean enabled, const char* file);
        
        DslReturnType SinkFakeNew(const char* name);

        DslReturnType SinkOverlayNew(const char* name, uint overlay_id, uint display_id,
            uint depth, uint offsetX, uint offsetY, uint width, uint height);
                
        DslReturnType SinkWindowNew(const char* name, 
            uint offsetX, uint offsetY, uint width, uint height);
                
        DslReturnType SinkFileNew(const char* name, const char* filepath, 
            uint codec, uint muxer, uint bit_rate, uint interval);
            
        DslReturnType SinkFileVideoFormatsGet(const char* name, uint* codec, uint* container);

        DslReturnType SinkFileEncoderSettingsGet(const char* name, uint* bitrate, uint* interval);

        DslReturnType SinkFileEncoderSettingsSet(const char* name, uint bitrate, uint interval);

        DslReturnType SinkRtspNew(const char* name, const char* host, 
            uint updPort, uint rtspPort, uint codec, uint bit_rate, uint interval);
            
        DslReturnType SinkRtspServerSettingsGet(const char* name, uint* updPort, uint* rtspPort, uint* codec);

        DslReturnType SinkRtspEncoderSettingsGet(const char* name, uint* bitrate, uint* interval);

        DslReturnType SinkRtspEncoderSettingsSet(const char* name, uint bitrate, uint interval);

        DslReturnType SinkImageNew(const char* name, const char* outdir);

        DslReturnType SinkImageOutdirGet(const char* name, const char** outdir);

        DslReturnType SinkImageOutdirSet(const char* name, const char* outdir);

        DslReturnType SinkImageFrameCaptureIntervalGet(const char* name, uint* interval);

        DslReturnType SinkImageFrameCaptureIntervalSet(const char* name, uint interval);
            
        DslReturnType SinkImageFrameCaptureEnabledGet(const char* name, boolean* enabled);

        DslReturnType SinkImageFrameCaptureEnabledSet(const char* name, boolean enabled);
            
        DslReturnType SinkImageObjectCaptureEnabledGet(const char* name, boolean* enabled);

        DslReturnType SinkImageObjectCaptureEnabledSet(const char* name, boolean enabled);

        DslReturnType SinkImageObjectCaptureClassAdd(const char* name, uint classId, boolean fullFrame, uint captureLimit);

        DslReturnType SinkImageObjectCaptureClassRemove(const char* name, uint classId);

        uint SinkNumInUseGet();
        
        uint SinkNumInUseMaxGet();
        
        boolean SinkNumInUseMaxSet(uint max);

        // TODO        
        // boolean ComponentIsInUse(const char* component);
        
        DslReturnType ComponentDelete(const char* component);

        DslReturnType ComponentDeleteAll();
        
        uint ComponentListSize();

        DslReturnType ComponentGpuIdGet(const char* component, uint* gpuid);
        
        DslReturnType ComponentGpuIdSet(const char* component, uint gpuid);
        
        DslReturnType BranchNew(const char* name);
        
        DslReturnType BranchComponentAdd(const char* branch, const char* component);

        DslReturnType BranchComponentRemove(const char* branch, const char* component);

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

        DslReturnType PipelineXWindowClear(const char* pipeline);
        
        DslReturnType PipelineXWindowDimensionsGet(const char* pipeline,
            uint* width, uint* height);

        DslReturnType PipelineXWindowDimensionsSet(const char* pipeline,
            uint width, uint height);
            
        DslReturnType PipelinePause(const char* pipeline);
        
        DslReturnType PipelinePlay(const char* pipeline);
        
        DslReturnType PipelineStop(const char* pipeline);
        
        DslReturnType PipelineStateGet(const char* pipeline, uint* state);
        
        DslReturnType PipelineIsLive(const char* pipeline, boolean* isLive);
        
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
        
        const wchar_t* ReturnValueToString(uint result);
        
        const wchar_t* StateValueToString(uint state);

        const wchar_t* VersionGet();
                        
        /** 
         * @brief Handles all pending events
         * 
         * @return true if all events were handled succesfully
         */
        bool HandleXWindowEvents(); 

    private:

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
         * @brief private helper function to test component for Source Type identity
         * @param[in] unique component name check
         * @returns true if component is a sink. 
         */
        bool IsSourceComponent(const char* component);
    
        /**
         * @brief private helper function to collect Sources in use stats from all Pipelines
         * @returns the current total number of all sinks in use
         */
        uint GetNumSourcesInUse();
        
        /**
         * @brief private helper function to test component for Sink Type identity
         * @param[in] unique component name to check
         * @returns true if component is a sink. 
         */
        bool IsSinkComponent(const char* component);
    
        /**
         * @brief private helper function to collect Sinks in use stats from all Pipelines
         * @returns the current total number of all sinks in use
         */
        uint GetNumSinksInUse();
        
        /**
         * @brief called during construction to intialize all const-to-string maps
         */
        void InitToStringMaps();
        
        std::map <uint, std::wstring> m_returnValueToString;
        
        std::map <uint, std::wstring> m_stateValueToString;
        
        std::map <uint, std::string> m_mapParserTypes;
        
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
        uint m_sourceNumInUseMax;
        
        /**
         * @brief maximum number of sinks that can be in use at one time
         * Set to the default in service contructor, the value can be read
         * and updated as the first call to DSL.
         */
        uint m_sinkNumInUseMax;
        
        /**
         * @brief map of all ODE Actions created by the client, key=name
         */
        std::map <std::string, DSL_ODE_ACTION_PTR> m_odeActions;
        
        /**
         * @brief map of all ODE Areas created by the client, key=name
         */
        std::map <std::string, DSL_ODE_AREA_PTR> m_odeAreas;
        
        /**
         * @brief map of all ODE Types created by the client, key=name
         */
        std::map <std::string, DSL_ODE_TRIGGER_PTR> m_odeTriggers;
        
        /**
         * @brief map of all pipelines creaated by the client, key=name
         */
        std::map <std::string, std::shared_ptr<PipelineBintr>> m_pipelines;
        
        /**
         * @brief map of all pipeline components creaated by the client, key=name
         */
        std::map <std::string, std::shared_ptr<Bintr>> m_components;
    };  

    static gboolean MainLoopThread(gpointer arg);
}


#endif // _DSL_DRIVER_H