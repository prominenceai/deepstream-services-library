/*
The MIT License

Copyright (c) 2019-2021, Prominence AI, Inc.

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
#include "DslOdeTrigger.h"
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
        DslReturnType DisplayTypeRgbaColorNew(const char* name, 
            double red, double green, double blue, double alpha);

        DslReturnType DisplayTypeRgbaFontNew(const char* name, const char* font,
            uint size, const char* color);
            
        DslReturnType DisplayTypeRgbaTextNew(const char* name, const char* text, 
            uint xOffset, uint yOffset, const char* font, boolean hasBgColor, const char* bgColor);

        DslReturnType DisplayTypeRgbaLineNew(const char* name, 
            uint x1, uint y1, uint x2, uint y2, uint width, const char* color);
            
        DslReturnType DisplayTypeRgbaArrowNew(const char* name, 
            uint x1, uint y1, uint x2, uint y2, uint width, uint head, const char* color);
            
        DslReturnType DisplayTypeRgbaRectangleNew(const char* name, uint left, uint top, 
            uint width, uint height, uint borderWidth, const char* color, 
            bool hasBgColor, const char* bgColor);
    
        DslReturnType DisplayTypeRgbaPolygonNew(const char* name, 
            const dsl_coordinate* coordinates, uint numCoordinates, 
            uint borderWidth, const char* color);

        DslReturnType DisplayTypeRgbaCircleNew(const char* name, uint xCenter, uint yCenter, uint radius,
            const char* color, bool hasBgColor, const char* bgColor);
    
        DslReturnType DisplayTypeSourceNumberNew(const char* name, 
            uint xOffset, uint yOffset, const char* font, boolean hasBgColor, const char* bgColor);

        DslReturnType DisplayTypeSourceNameNew(const char* name, 
            uint xOffset, uint yOffset, const char* font, boolean hasBgColor, const char* bgColor);

        DslReturnType DisplayTypeSourceDimensionsNew(const char* name, 
            uint xOffset, uint yOffset, const char* font, boolean hasBgColor, const char* bgColor);

        DslReturnType DisplayTypeSourceFrameRateNew(const char* name, 
            uint xOffset, uint yOffset, const char* font, boolean hasBgColor, const char* bgColor);

        DslReturnType DisplayTypeMetaAdd(const char* name, void* pDisplayMeta, void* pFrameMeta);
        
        DslReturnType DisplayTypeDelete(const char* name);
        
        DslReturnType DisplayTypeDeleteAll();
        
        uint DisplayTypeListSize();
         
        DslReturnType OdeActionCustomNew(const char* name,
            dsl_ode_handle_occurrence_cb clientHandler, void* clientData);
            
        DslReturnType OdeActionCaptureFrameNew(const char* name, const char* outdir, boolean annotate);
        
        DslReturnType OdeActionCaptureObjectNew(const char* name, const char* outdir);

        DslReturnType OdeActionCaptureCompleteListenerAdd(const char* name, 
            dsl_capture_complete_listener_cb listener, void* clientData);
        
        DslReturnType OdeActionCaptureCompleteListenerRemove(const char* name, 
            dsl_capture_complete_listener_cb listener);
            
        DslReturnType OdeActionCaptureImagePlayerAdd(const char* name,
            const char* player);
        
        DslReturnType OdeActionCaptureImagePlayerRemove(const char* name,
            const char* player);
        
        DslReturnType OdeActionCaptureMailerAdd(const char* name,
            const char* mailer, const char* subject, boolean attach);
        
        DslReturnType OdeActionCaptureMailerRemove(const char* name,
            const char* mailer);

        DslReturnType OdeActionCustomizeLabelNew(const char* name, 
            const uint* contentTypes, uint size, uint mode);
        
        DslReturnType OdeActionDisplayNew(const char* name, 
            const char* formatString, uint offsetX, uint offsetY, 
            const char* font, boolean hasBgColor, const char* bgColor);
            
        DslReturnType OdeActionFormatBBoxNew(const char* name,
            uint borderWidth, const char* borderColor, boolean hasBgColor, const char* bgColor);

        DslReturnType OdeActionFormatLabelNew(const char* name,
            const char* font, boolean hasBgColor, const char* bgColor);
        
        DslReturnType OdeActionLogNew(const char* name);

        DslReturnType OdeActionEmailNew(const char* name, 
            const char* mailer, const char* subject);
        
        DslReturnType OdeActionFileNew(const char* name, 
            const char* filePath, uint mode, uint format, boolean forceFlush);
        
        DslReturnType OdeActionFillSurroundingsNew(const char* name, const char* color);
        
        DslReturnType OdeActionFillFrameNew(const char* name, const char* color);

        DslReturnType OdeActionHandlerDisableNew(const char* name, const char* handler);

        DslReturnType OdeActionDisplayMetaAddNew(const char* name, const char* displayType);
        
        DslReturnType OdeActionDisplayMetaAddDisplayType(const char* name, const char* displayType);

        DslReturnType OdeActionPauseNew(const char* name, const char* pipeline);

        DslReturnType OdeActionPrintNew(const char* name, boolean forceFlush);
        
        DslReturnType OdeActionRedactNew(const char* name);

        DslReturnType OdeActionSinkAddNew(const char* name, 
            const char* pipeline, const char* sink);

        DslReturnType OdeActionSinkRemoveNew(const char* name, 
            const char* pipeline, const char* sink);

        DslReturnType OdeActionSinkRecordStartNew(const char* name,
            const char* recordSink, uint start, uint duration, void* clientData);

        DslReturnType OdeActionSinkRecordStopNew(const char* name,
            const char* recordSink);

        DslReturnType OdeActionSourceAddNew(const char* name, 
            const char* pipeline, const char* source);

        DslReturnType OdeActionSourceRemoveNew(const char* name, 
            const char* pipeline, const char* source);

        DslReturnType OdeActionTapRecordStartNew(const char* name,
            const char* recordTap, uint start, uint duration, void* clientData);

        DslReturnType OdeActionTapRecordStopNew(const char* name,
            const char* recordTap);

        DslReturnType OdeActionActionDisableNew(const char* name, const char* action);

        DslReturnType OdeActionActionEnableNew(const char* name, const char* action);
        
        DslReturnType OdeActionTilerShowSourceNew(const char* name, 
            const char* tiler, uint timeout, bool hasPrecedence);

        DslReturnType OdeActionAreaAddNew(const char* name, 
            const char* trigger, const char* area);
        
        DslReturnType OdeActionAreaRemoveNew(const char* name, 
            const char* trigger, const char* area);
        
        DslReturnType OdeActionTriggerDisableNew(const char* name, const char* trigger);

        DslReturnType OdeActionTriggerEnableNew(const char* name, const char* trigger);

        DslReturnType OdeActionTriggerResetNew(const char* name, const char* trigger);

        DslReturnType OdeActionEnabledGet(const char* name, boolean* enabled);

        DslReturnType OdeActionEnabledSet(const char* name, boolean enabled);

        DslReturnType OdeActionDelete(const char* name);
        
        DslReturnType OdeActionDeleteAll();
        
        uint OdeActionListSize();

        DslReturnType OdeAreaInclusionNew(const char* name, 
            const char* polygon, boolean display, uint bboxTestPoint);

        DslReturnType OdeAreaExclusionNew(const char* name, 
            const char* polygon, boolean display, uint bboxTestPoint);

        DslReturnType OdeAreaLineNew(const char* name, 
            const char* line, boolean display, uint bboxTestEdge);

        DslReturnType OdeAreaDelete(const char* name);
        
        DslReturnType OdeAreaDeleteAll();
        
        uint OdeAreaListSize();
        
        DslReturnType OdeTriggerAlwaysNew(const char* name, 
            const char* source, uint when);
        
        DslReturnType OdeTriggerOccurrenceNew(const char* name, 
            const char* source, uint classId, uint limit);
        
        DslReturnType OdeTriggerAbsenceNew(const char* name, 
            const char* source, uint classId, uint limit);

        DslReturnType OdeTriggerAccumulationNew(const char* name, 
            const char* source, uint classId, uint limit);

        DslReturnType OdeTriggerInstanceNew(const char* name, 
            const char* source, uint classId, uint limit);
        
        DslReturnType OdeTriggerIntersectionNew(const char* name, 
            const char* source, uint classIdA, uint classIdB, uint limit);

        DslReturnType OdeTriggerSummationNew(const char* name, 
            const char* source, uint classId, uint limit);

        DslReturnType OdeTriggerCustomNew(const char* name, const char* source, 
            uint classId, uint limit,  dsl_ode_check_for_occurrence_cb client_checker, 
            dsl_ode_post_process_frame_cb client_post_processor, void* client_data);

        DslReturnType OdeTriggerPersistenceNew(const char* name, const char* source,
            uint classId, uint limit, uint minimum, uint maximum);

        DslReturnType OdeTriggerPersistenceRangeGet(const char* name, 
            uint* minimum, uint* maximum);
        
        DslReturnType OdeTriggerPersistenceRangeSet(const char* name, 
            uint minimum, uint maximum);

        DslReturnType OdeTriggerCountNew(const char* name, const char* source, 
            uint classId, uint limit, uint minimum, uint maximum);

        DslReturnType OdeTriggerCountRangeGet(const char* name, 
            uint* minimum, uint* maximum);
        
        DslReturnType OdeTriggerCountRangeSet(const char* name, 
            uint minimum, uint maximum);
        
        DslReturnType OdeTriggerDistanceNew(const char* name, const char* source, 
            uint classIdA, uint classIdB, uint limit, uint minimum, uint maximum, 
            uint testPoint, uint testMethod);

        DslReturnType OdeTriggerDistanceRangeGet(const char* name, 
            uint* minimum, uint* maximum);
        
        DslReturnType OdeTriggerDistanceRangeSet(const char* name, 
            uint minimum, uint maximum);

        DslReturnType OdeTriggerDistanceTestParamsGet(const char* name, 
            uint* testPoint, uint* testMethod);
        
        DslReturnType OdeTriggerDistanceTestParamsSet(const char* name, 
            uint testPoint, uint testMethod);
        
        DslReturnType OdeTriggerSmallestNew(const char* name, 
            const char* source, uint classId, uint limit);

        DslReturnType OdeTriggerLargestNew(const char* name, 
            const char* source, uint classId, uint limit);

        DslReturnType OdeTriggerLatestNew(const char* name, 
            const char* source, uint classId, uint limit);

        DslReturnType OdeTriggerEarliestNew(const char* name, 
            const char* source, uint classId, uint limit);

        DslReturnType OdeTriggerNewHighNew(const char* name, 
            const char* source, uint classId, uint limit, uint preset);

        DslReturnType OdeTriggerNewLowNew(const char* name, 
            const char* source, uint classId, uint limit, uint preset);

        DslReturnType OdeTriggerReset(const char* name);

        DslReturnType OdeTriggerResetTimeoutGet(const char* name, uint* timeout);

        DslReturnType OdeTriggerResetTimeoutSet(const char* name, uint timeout);

        DslReturnType OdeTriggerEnabledGet(const char* name, boolean* enabled);

        DslReturnType OdeTriggerEnabledSet(const char* name, boolean enabled);

        DslReturnType OdeTriggerSourceGet(const char* name, const char** source);
        
        DslReturnType OdeTriggerSourceSet(const char* name, const char* source);
        
        DslReturnType OdeTriggerClassIdGet(const char* name, uint* classId);
        
        DslReturnType OdeTriggerClassIdSet(const char* name, uint classId);
        
        DslReturnType OdeTriggerClassIdABGet(const char* name, uint* classIdA, uint* classIdB);
        
        DslReturnType OdeTriggerClassIdABSet(const char* name, uint classIdA, uint classIdB);
        
        DslReturnType OdeTriggerLimitGet(const char* name, uint* limit);
        
        DslReturnType OdeTriggerLimitSet(const char* name, uint limit);
        
        DslReturnType OdeTriggerConfidenceMinGet(const char* name, float* minConfidence);
        
        DslReturnType OdeTriggerConfidenceMinSet(const char* name, float minConfidence);
        
        DslReturnType OdeTriggerDimensionsMinGet(const char* name, float* min_width, float* min_height);
        
        DslReturnType OdeTriggerDimensionsMinSet(const char* name, float min_width, float min_height);

        DslReturnType OdeTriggerDimensionsMaxGet(const char* name, float* max_width, float* max_height);
        
        DslReturnType OdeTriggerDimensionsMaxSet(const char* name, float max_width, float max_height);

        DslReturnType OdeTriggerFrameCountMinGet(const char* name, uint* min_count_n, uint* min_count_d);

        DslReturnType OdeTriggerFrameCountMinSet(const char* name, uint min_count_n, uint min_count_d);
        
        DslReturnType OdeTriggerInferDoneOnlyGet(const char* name, boolean* inferDoneOnly);
        
        DslReturnType OdeTriggerInferDoneOnlySet(const char* name, boolean inferDoneOnly);
        
        DslReturnType OdeTriggerIntervalGet(const char* name, uint* interval);
        
        DslReturnType OdeTriggerIntervalSet(const char* name, uint interval);
        
        DslReturnType OdeTriggerActionAdd(const char* name, const char* action);

        DslReturnType OdeTriggerActionRemove(const char* name, const char* action);

        DslReturnType OdeTriggerActionRemoveAll(const char* name);

        DslReturnType OdeTriggerAreaAdd(const char* name, const char* area);

        DslReturnType OdeTriggerAreaRemove(const char* name, const char* area);

        DslReturnType OdeTriggerAreaRemoveAll(const char* name);

        DslReturnType OdeTriggerDelete(const char* name);
        
        DslReturnType OdeTriggerDeleteAll();
        
        uint OdeTriggerListSize();

        DslReturnType PphCustomNew(const char* name,
            dsl_pph_custom_client_handler_cb clientHandler, void* clientData);

        DslReturnType PphMeterNew(const char* name, uint interval, 
            dsl_pph_meter_client_handler_cb clientHandler, void* clientData);
            
        DslReturnType PphMeterIntervalGet(const char* name, uint* interval);
        
        DslReturnType PphMeterIntervalSet(const char* name, uint interval);
        
        DslReturnType PphOdeNew(const char* name);

        DslReturnType PphOdeTriggerAdd(const char* name, const char* trigger);

        DslReturnType PphOdeTriggerRemove(const char* name, const char* trigger);

        DslReturnType PphOdeTriggerRemoveAll(const char* name);

        DslReturnType PphEnabledGet(const char* name, boolean* enabled);
        
        DslReturnType PphEnabledSet(const char* name, boolean enabled);

        DslReturnType PphDelete(const char* name);
        
        DslReturnType PphDeleteAll();
        
        uint PphListSize();
        
        DslReturnType SourceCsiNew(const char* name, 
            uint width, uint height, uint fpsN, uint fpsD);
        
        DslReturnType SourceUsbNew(const char* name, 
            uint width, uint height, uint fpsN, uint fpsD);
        
        DslReturnType SourceUriNew(const char* name, const char* uri, 
            boolean isLive, uint cudadecMemType, uint intraDecode, uint dropFrameInterval);
            
        DslReturnType SourceFileNew(const char* name, const char* filePath, 
            boolean repeatEnabled);

        DslReturnType SourceFilePathGet(const char* name, const char** filePath);

        DslReturnType SourceFilePathSet(const char* name, const char* filePath);

        DslReturnType SourceFileRepeatEnabledGet(const char* name, boolean* enabled);
    
        DslReturnType SourceFileRepeatEnabledSet(const char* name, boolean enabled);
            
        DslReturnType SourceImageNew(const char* name, const char* filePath, 
            boolean isLive, uint fpsN, uint fpsD, uint timeout);

        DslReturnType SourceImagePathGet(const char* name, const char** filePath);

        DslReturnType SourceImagePathSet(const char* name, const char* filePath);

        DslReturnType SourceImageTimeoutGet(const char* name, uint* timeout);
    
        DslReturnType SourceImageTimeoutSet(const char* name, uint timeout);
            
        DslReturnType SourceRtspNew(const char* name, const char* uri, uint protocol, 
            uint cudadecMemType, uint intraDecode, uint dropFrameInterval, uint latency, uint timeout);
            
        DslReturnType SourceDimensionsGet(const char* name, uint* width, uint* height);
        
        DslReturnType SourceFrameRateGet(const char* name, uint* fpsN, uint* fpsD);

        DslReturnType SourceDecodeUriGet(const char* name, const char** uri);

        DslReturnType SourceDecodeUriSet(const char* name, const char* uri);
    
        DslReturnType SourceDecodeDewarperAdd(const char* name, const char* dewarper);
    
        DslReturnType SourceDecodeDewarperRemove(const char* name);
        
        DslReturnType SourceRtspTimeoutGet(const char* name, uint* timeout);

        DslReturnType SourceRtspTimeoutSet(const char* name, uint timeout);
        
        DslReturnType SourceRtspReconnectionParamsGet(const char* name, uint* sleep, uint* timeout);

        DslReturnType SourceRtspReconnectionParamsSet(const char* name, uint sleep, uint timeout);
        
        DslReturnType SourceRtspConnectionDataGet(const char* name, dsl_rtsp_connection_data* data);
        
        DslReturnType SourceRtspConnectionStatsClear(const char* name);
        
        DslReturnType SourceRtspStateChangeListenerAdd(const char* name, 
            dsl_state_change_listener_cb listener, void* clientData);
        
        DslReturnType SourceRtspStateChangeListenerRemove(const char* name, 
            dsl_state_change_listener_cb listener);
        
        DslReturnType SourceRtspTapAdd(const char* name, const char* tap);
    
        DslReturnType SourceRtspTapRemove(const char* name);
        
        DslReturnType SourceNameGet(int sourceId, const char** name);

        DslReturnType SourceIdGet(const char* name, int* sourceId);
    
        DslReturnType _sourceNameSet(uint sourceId, const char* name);
    
        DslReturnType _sourceNameErase(uint sourceId);
    
        DslReturnType SourcePause(const char* name);

        DslReturnType SourceResume(const char* name);

        boolean SourceIsLive(const char* name);
        
        uint SourceNumInUseGet();
        
        uint SourceNumInUseMaxGet();
        
        boolean SourceNumInUseMaxSet(uint max);
        
        DslReturnType DewarperNew(const char* name, const char* configFile);
        
        DslReturnType TapRecordNew(const char* name, const char* outdir, 
            uint container, dsl_record_client_listener_cb clientListener);
            
        DslReturnType TapRecordSessionStart(const char* name, 
            uint start, uint duration, void* clientData);

        DslReturnType TapRecordSessionStop(const char* name);

        DslReturnType TapRecordOutdirGet(const char* name, const char** outdir);
            
        DslReturnType TapRecordOutdirSet(const char* name, const char* outdir);
        
        DslReturnType TapRecordContainerGet(const char* name, uint* container);
            
        DslReturnType TapRecordContainerSet(const char* name, uint container);
        
        DslReturnType TapRecordCacheSizeGet(const char* name, uint* cacheSize);
            
        DslReturnType TapRecordCacheSizeSet(const char* name, uint cacheSize);
        
        DslReturnType TapRecordDimensionsGet(const char* name, uint* width, uint* height);

        DslReturnType TapRecordDimensionsSet(const char* name, uint width, uint height);

        DslReturnType TapRecordIsOnGet(const char* name, boolean* isOn);

        DslReturnType TapRecordResetDoneGet(const char* name, boolean* resetDone);
        
        DslReturnType TapRecordVideoPlayerAdd(const char* name,
            const char* player);
        
        DslReturnType TapRecordVideoPlayerRemove(const char* name,
            const char* player);

        DslReturnType TapRecordMailerAdd(const char* name,
            const char* mailer, const char* subject);
        
        DslReturnType TapRecordMailerRemove(const char* name,
            const char* mailer);

        DslReturnType SegVisualNew(const char* name, uint width, uint height);
        
        DslReturnType SegVisualDimensionsGet(const char* name, uint* width, uint* height);

        DslReturnType SegVisualDimensionsSet(const char* name, uint width, uint height);

        DslReturnType SegVisualPphAdd(const char* name, const char* handler);

        DslReturnType SegVisualPphRemove(const char* name, const char* handler);

        DslReturnType PrimaryGieNew(const char* name, const char* inferConfigFile,
            const char* modelEngineFile, uint interval);

        DslReturnType PrimaryTisNew(const char* name, 
            const char* inferConfigFile, uint interval);

        DslReturnType SecondaryGieNew(const char* name, const char* inferConfigFile,
            const char* modelEngineFile, const char* inferOnGieName, uint interval);

        DslReturnType SecondaryTisNew(const char* name, const char* inferConfigFile,
            const char* inferOnGieName, uint interval);

        DslReturnType PrimaryInferPphAdd(const char* name, const char* handler, uint pad);

        DslReturnType PrimaryInferPphRemove(const char* name, const char* handler, uint pad);

        DslReturnType GieModelEngineFileGet(const char* name, const char** modelEngineFile);

        DslReturnType GieModelEngineFileSet(const char* name, const char* modelEngineFile);

        DslReturnType InferConfigFileGet(const char* name, const char** inferConfigFile);

        DslReturnType InferConfigFileSet(const char* name, const char* inferConfigFile);
            
        DslReturnType InferRawOutputEnabledSet(const char* name, boolean enabled,
            const char* path);
            
        DslReturnType InferIntervalGet(const char* name, uint* interval);

        DslReturnType InferIntervalSet(const char* name, uint interval);

        DslReturnType TrackerDcfNew(const char* name, 
            const char* configFile, uint width, uint height,
            boolean batch_processing_enabled, boolean past_frame_reporting_enabled);
        
        DslReturnType TrackerKtlNew(const char* name, uint width, uint height);
        
        DslReturnType TrackerIouNew(const char* name, 
            const char* configFile, uint width, uint height);

        DslReturnType TrackerConfigFileGet(const char* name, const char** inferConfigFile);

        DslReturnType TrackerConfigFileSet(const char* name, const char* inferConfigFile);
        
        DslReturnType TrackerDimensionsGet(const char* name, uint* width, uint* height);
        
        DslReturnType TrackerDimensionsSet(const char* name, uint width, uint height);
        
        DslReturnType TrackerDcfBatchProcessingEnabledGet(const char* name, 
            boolean* enabled);
        
        DslReturnType TrackerDcfBatchProcessingEnabledSet(const char* name, 
            boolean enabled);

        DslReturnType TrackerDcfPastFrameReportingEnabledGet(const char* name, 
            boolean* enabled);

        DslReturnType TrackerDcfPastFrameReportingEnabledSet(const char* name, 
            boolean enabled);
        
        DslReturnType TrackerPphAdd(const char* name, const char* handler, uint pad);

        DslReturnType TrackerPphRemove(const char* name, const char* handler, uint pad);
        
        DslReturnType TeeDemuxerNew(const char* name);
        
        DslReturnType TeeSplitterNew(const char* name);
        
        DslReturnType TeeBranchAdd(const char* demuer, const char* branch);

        DslReturnType TeeBranchRemove(const char* demuxer, const char* branch);
        
        DslReturnType TeeBranchRemoveAll(const char* demuxer);

        DslReturnType TeeBranchCountGet(const char* demuxer, uint* count);

        DslReturnType TeePphAdd(const char* name, const char* handler);

        DslReturnType TeePphRemove(const char* name, const char* handler);
        
        DslReturnType TilerNew(const char* name, uint width, uint height);
        
        DslReturnType TilerDimensionsGet(const char* name, uint* width, uint* height);

        DslReturnType TilerDimensionsSet(const char* name, uint width, uint height);

        DslReturnType TilerTilesGet(const char* name, uint* columns, uint* rows);

        DslReturnType TilerTilesSet(const char* name, uint columns, uint rows);

        DslReturnType TilerSourceShowGet(const char* name, const char** source, uint* timeout);

        DslReturnType TilerSourceShowSet(const char* name, const char* source, uint timeout, bool hasPrecedence);

        // called by the Show Source Action only. 
        DslReturnType TilerSourceShowSet(const char* name, uint sourceId, uint timeout, bool hasPrecedence);

        DslReturnType TilerSourceShowSelect(const char* name, 
            int xPos, int yPos, uint windowWidth, uint windowHeight, uint timeout);

        DslReturnType TilerSourceShowAll(const char* name);

        DslReturnType TilerSourceShowCycle(const char* name, uint timeout);

        DslReturnType TilerPphAdd(const char* name, const char* handler, uint pad);

        DslReturnType TilerPphRemove(const char* name, const char* handler, uint pad);

        DslReturnType OfvNew(const char* name);

        DslReturnType OsdNew(const char* name, boolean textEnabled, boolean clockEnabled);
        
        DslReturnType OsdTextEnabledGet(const char* name, boolean* enabled);

        DslReturnType OsdTextEnabledSet(const char* name, boolean enabled);

        DslReturnType OsdClockEnabledGet(const char* name, boolean* enabled);

        DslReturnType OsdClockEnabledSet(const char* name, boolean enabled);

        DslReturnType OsdClockOffsetsGet(const char* name, uint* offsetX, uint* offsetY);

        DslReturnType OsdClockOffsetsSet(const char* name, uint offsetX, uint offsetY);

        DslReturnType OsdClockFontGet(const char* name, const char** font, uint* size);

        DslReturnType OsdClockFontSet(const char* name, const char* font, uint size);

        DslReturnType OsdClockColorGet(const char* name, double* red, double* green, double* blue, double* alpha);

        DslReturnType OsdClockColorSet(const char* name, double red, double green, double blue, double alpha);

        DslReturnType OsdPphAdd(const char* name, const char* handler, uint pad);

        DslReturnType OsdPphRemove(const char* name, const char* handler, uint pad);

        DslReturnType SinkFakeNew(const char* name);

        DslReturnType SinkOverlayNew(const char* name, uint display_id,
            uint depth, uint offsetX, uint offsetY, uint width, uint height);
                
        DslReturnType SinkWindowNew(const char* name, 
            uint offsetX, uint offsetY, uint width, uint height);
            
        DslReturnType SinkWindowForceAspectRatioGet(const char* name, 
            boolean* force);

        DslReturnType SinkWindowForceAspectRatioSet(const char* name, 
            boolean force);
            
        DslReturnType SinkRenderOffsetsGet(const char* name, uint* offsetX, uint* offsetY);

        DslReturnType SinkRenderOffsetsSet(const char* name, uint offsetX, uint offsetY);
        
        DslReturnType SinkRenderDimensionsGet(const char* name, uint* width, uint* height);

        DslReturnType SinkRenderDimensionsSet(const char* name, uint width, uint height);
        
        DslReturnType SinkRenderReset(const char* name);

        DslReturnType SinkFileNew(const char* name, const char* filepath, 
            uint codec, uint muxer, uint bit_rate, uint interval);
            
        DslReturnType SinkRecordNew(const char* name, const char* outdir, 
            uint codec, uint container, uint bitrate, uint interval, dsl_record_client_listener_cb clientListener);
            
        DslReturnType SinkRecordSessionStart(const char* name, 
            uint start, uint duration, void* clientData);

        DslReturnType SinkRecordSessionStop(const char* name);

        DslReturnType SinkRecordOutdirGet(const char* name, const char** outdir);
            
        DslReturnType SinkRecordOutdirSet(const char* name, const char* outdir);
        
        DslReturnType SinkRecordContainerGet(const char* name, uint* container);
            
        DslReturnType SinkRecordContainerSet(const char* name, uint container);

        DslReturnType SinkRecordCacheSizeGet(const char* name, uint* cacheSize);
            
        DslReturnType SinkRecordCacheSizeSet(const char* name, uint cacheSize);
        
        DslReturnType SinkRecordDimensionsGet(const char* name, uint* width, uint* height);

        DslReturnType SinkRecordDimensionsSet(const char* name, uint width, uint height);

        DslReturnType SinkRecordIsOnGet(const char* name, boolean* isOn);

        DslReturnType SinkRecordResetDoneGet(const char* name, boolean* resetDone);

        DslReturnType SinkRecordVideoPlayerAdd(const char* name,
            const char* player);
        
        DslReturnType SinkRecordVideoPlayerRemove(const char* name,
            const char* player);

        DslReturnType SinkRecordMailerAdd(const char* name,
            const char* mailer, const char* subject);
        
        DslReturnType SinkRecordMailerRemove(const char* name,
            const char* mailer);

        DslReturnType SinkEncodeVideoFormatsGet(const char* name, uint* codec, uint* container);

        DslReturnType SinkEncodeSettingsGet(const char* name, uint* bitrate, uint* interval);

        DslReturnType SinkEncodeSettingsSet(const char* name, uint bitrate, uint interval);

        DslReturnType SinkRtspNew(const char* name, const char* host, 
            uint updPort, uint rtspPort, uint codec, uint bit_rate, uint interval);
            
        DslReturnType SinkRtspServerSettingsGet(const char* name, uint* updPort, uint* rtspPort, uint* codec);

        DslReturnType SinkRtspEncoderSettingsGet(const char* name, uint* bitrate, uint* interval);

        DslReturnType SinkRtspEncoderSettingsSet(const char* name, uint bitrate, uint interval);

        DslReturnType SinkPphAdd(const char* name, const char* handler);

        DslReturnType SinkPphRemove(const char* name, const char* handler);

        DslReturnType SinkSyncSettingsGet(const char* name,  boolean* sync, boolean* async);

        DslReturnType SinkSyncSettingsSet(const char* name,  boolean sync, boolean async);

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

        DslReturnType PipelineStreamMuxNumSurfacesPerFrameGet(const char* pipeline, uint* num);

        DslReturnType PipelineStreamMuxNumSurfacesPerFrameSet(const char* pipeline, uint num);

        DslReturnType PipelineXWindowHandleGet(const char* pipeline, uint64_t* xwindow);

        DslReturnType PipelineXWindowHandleSet(const char* pipeline, uint64_t xwindow);
		
        DslReturnType PipelineXWindowClear(const char* pipeline);
        
        DslReturnType PipelineXWindowDestroy(const char* pipeline);
        
        DslReturnType PipelineXWindowOffsetsGet(const char* pipeline,
            uint* xOffset, uint* yOffset);
            
        DslReturnType PipelineXWindowDimensionsGet(const char* pipeline,
            uint* width, uint* height);
            
        DslReturnType PipelineXWindowFullScreenEnabledGet(const char* pipeline, boolean* enabled);
        
        DslReturnType PipelineXWindowFullScreenEnabledSet(const char* pipeline, boolean enabled);
        
        DslReturnType PipelinePause(const char* pipeline);
        
        DslReturnType PipelinePlay(const char* pipeline);
        
        DslReturnType PipelineStop(const char* pipeline);
        
        DslReturnType PipelineStateGet(const char* pipeline, uint* state);
        
        DslReturnType PipelineIsLive(const char* pipeline, boolean* isLive);
        
        DslReturnType PipelineDumpToDot(const char* pipeline, char* filename);
        
        DslReturnType PipelineDumpToDotWithTs(const char* pipeline, char* filename);
        
        DslReturnType PipelineStateChangeListenerAdd(const char* pipeline, 
            dsl_state_change_listener_cb listener, void* clientData);
        
        DslReturnType PipelineStateChangeListenerRemove(const char* pipeline, 
            dsl_state_change_listener_cb listener);
                        
        DslReturnType PipelineEosListenerAdd(const char* pipeline, 
            dsl_eos_listener_cb listener, void* clientData);
        
        DslReturnType PipelineEosListenerRemove(const char* pipeline, 
            dsl_eos_listener_cb listener);

        DslReturnType PipelineErrorMessageHandlerAdd(const char* pipeline, 
            dsl_error_message_handler_cb handler, void* clientData);

        DslReturnType PipelineErrorMessageHandlerRemove(const char* pipeline, 
            dsl_error_message_handler_cb handler);
            
        DslReturnType PipelineErrorMessageLastGet(const char* pipeline,
            std::wstring& source, std::wstring& message);
                        
        DslReturnType PipelineXWindowKeyEventHandlerAdd(const char* pipeline, 
            dsl_xwindow_key_event_handler_cb handler, void* clientData);

        DslReturnType PipelineXWindowKeyEventHandlerRemove(const char* pipeline, 
            dsl_xwindow_key_event_handler_cb handler);

        DslReturnType PipelineXWindowButtonEventHandlerAdd(const char* pipeline, 
            dsl_xwindow_button_event_handler_cb handler, void* clientData);

        DslReturnType PipelineXWindowButtonEventHandlerRemove(const char* pipeline, 
            dsl_xwindow_button_event_handler_cb handler);
        
        DslReturnType PipelineXWindowDeleteEventHandlerAdd(const char* pipeline, 
            dsl_xwindow_delete_event_handler_cb handler, void* clientData);

        DslReturnType PipelineXWindowDeleteEventHandlerRemove(const char* pipeline, 
            dsl_xwindow_delete_event_handler_cb handler);

        DslReturnType PlayerNew(const char* name, const char* source, const char* sink);

        DslReturnType PlayerRenderVideoNew(const char* name, const char* filePath,
            uint renderType, uint offsetX, uint offsetY, uint zoom, boolean repeatEnabled);

        DslReturnType PlayerRenderImageNew(const char* name, const char* filePath,
            uint renderType, uint offsetX, uint offsetY, uint zoom, uint timeout);
            
        DslReturnType PlayerRenderFilePathGet(const char* name, const char** filePath);

        DslReturnType PlayerRenderFilePathSet(const char* name, const char* filePath);
            
        DslReturnType PlayerRenderFilePathQueue(const char* name, const char* filePath);

        DslReturnType PlayerRenderOffsetsGet(const char* name, uint* offsetX, uint* offsetY);

        DslReturnType PlayerRenderOffsetsSet(const char* name, uint offsetX, uint offsetY);

        DslReturnType PlayerRenderZoomGet(const char* name, uint* zoom);

        DslReturnType PlayerRenderZoomSet(const char* name, uint zoom);

        DslReturnType PlayerRenderReset(const char* name);

        DslReturnType PlayerRenderImageTimeoutGet(const char* name, uint* timeout);

        DslReturnType PlayerRenderImageTimeoutSet(const char* name, uint timeout);
        
        DslReturnType PlayerRenderVideoRepeatEnabledGet(const char* name, 
            boolean* repeatEnabled);

        DslReturnType PlayerRenderVideoRepeatEnabledSet(const char* name, 
            boolean repeatEnabled);

        DslReturnType PlayerTerminationEventListenerAdd(const char* name,
            dsl_player_termination_event_listener_cb listener, void* clientData);
        
        DslReturnType PlayerTerminationEventListenerRemove(const char* name,
            dsl_player_termination_event_listener_cb listener);

        DslReturnType PlayerXWindowHandleGet(const char* name, uint64_t* xwindow);

        DslReturnType PlayerXWindowHandleSet(const char* name, uint64_t xwindow);

        DslReturnType PlayerXWindowKeyEventHandlerAdd(const char* name, 
            dsl_xwindow_key_event_handler_cb handler, void* clientData);

        DslReturnType PlayerXWindowKeyEventHandlerRemove(const char* name, 
            dsl_xwindow_key_event_handler_cb handler);
        
        DslReturnType PlayerPause(const char* name);
        
        DslReturnType PlayerPlay(const char* name);
        
        DslReturnType PlayerStop(const char* name);

        DslReturnType PlayerRenderNext(const char* name);

        DslReturnType PlayerStateGet(const char* name, uint* state);
        
        boolean PlayerExists(const char* name);
        
        DslReturnType PlayerDelete(const char* name);
        
        DslReturnType PlayerDeleteAll(bool checkInUse=true);

        uint PlayerListSize();
        
        DslReturnType MailerNew(const char* name);

        DslReturnType MailerEnabledGet(const char* name, boolean* enabled);
        
        DslReturnType MailerEnabledSet(const char* name, boolean enabled);   
            
        DslReturnType MailerCredentialsSet(const char* name, 
            const char* username, const char* password);
        
        DslReturnType MailerServerUrlGet(const char* name, const char** serverUrl);
        
        DslReturnType MailerServerUrlSet(const char* name, const char* serverUrl);

        DslReturnType MailerFromAddressGet(const char* name, 
            const char** displayName, const char** address);

        DslReturnType MailerFromAddressSet(const char* name, 
            const char* displayName, const char* address);
        
        DslReturnType MailerSslEnabledGet(const char* name, boolean* enabled);
        
        DslReturnType MailerSslEnabledSet(const char* name, boolean enabled);
        
        DslReturnType MailerToAddressAdd(const char* name, 
            const char* displayName, const char* address);
        
        DslReturnType MailerToAddressesRemoveAll(const char* name);
        
        DslReturnType MailerCcAddressAdd(const char* name, 
            const char* displayName, const char* address);

        DslReturnType MailerCcAddressesRemoveAll(const char* name);
        
        DslReturnType MailerSendTestMessage(const char* name);

        DslReturnType MailerExists(const char* name);
        
        DslReturnType MailerDelete(const char* name);
        
        DslReturnType MailerDeleteAll();
        
        uint MailerListSize();
        
        void DeleteAll();
        
        DslReturnType StdOutRedirect(const char* filepath);
        
        void StdOutRestore();

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

        /**
         * @brief called during construction to intialize the NO type Display Types.
         */
        void DisplayTypeCreateIntrinsicTypes();
        
        std::map <uint, std::wstring> m_returnValueToString;
        
        std::map <uint, std::wstring> m_stateValueToString;
        
        std::map <uint, std::string> m_mapParserTypes;
        
        /**
         * @brief instance pointer for this singleton class
         */
        static Services* m_pInstance;
        
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
         * @brief map of all default intrinsic RGBA Display Types
         */
        std::map<std::string, DSL_BASE_PTR> m_intrinsicDisplayTypes;
        
        /**
         * @brief map of all client created RGBA Display Types
         */
        std::map<std::string, DSL_BASE_PTR> m_displayTypes;
        
        /**
         * @brief map of all ODE Actions created by the client, key=name
         */
        std::map <std::string, DSL_ODE_ACTION_PTR> m_odeActions;
        
        /**
         * @brief map of all ODE Areas created by the client, key=name
         */
        std::map <std::string, DSL_ODE_AREA_PTR> m_odeAreas;
        
        /**
         * @brief map of all ODE Triggers created by the client, key=name
         */
        std::map <std::string, DSL_ODE_TRIGGER_PTR> m_odeTriggers;
        
        /**
         * @brief map of all ODE Handlers created by the client, key=name
         */
        std::map <std::string, DSL_PPH_PTR> m_padProbeHandlers;

        /**
         * @brief map of all pipelines creaated by the client, key=name
         */
        std::map <std::string, std::shared_ptr<PipelineBintr>> m_pipelines;
        
        /**
         * @brief map of all players creaated by the client, key=name
         */
        std::map <std::string, std::shared_ptr<PlayerBintr>> m_players;
        
        /**
         * @brief map of all pipeline components creaated by the client, key=name
         */
        std::map <std::string, std::shared_ptr<Bintr>> m_components;
        
        /**
         * @brief map of all source ids to source names
         */
        std::map <uint, std::string> m_sourceNames;

        /**
         * @brief map of all source names to source ids
         */
        std::map <std::string, uint> m_sourceIds;
        
        /**
         * @brief DSL Comms object for libcurl services
         */
        std::map <std::string, std::shared_ptr<Mailer>> m_mailers;
        
        std::fstream m_stdOutRedirectFile;
        
        std::streambuf* m_stdOutRdBufBackup;
        
    };  

    /**
     * @brief Intrinsic Display Types created on DSL instantiation.
     */
    static const std::string DISPLAY_TYPE_NO_COLOR("no-color");
    static const std::string DISPLAY_TYPE_NO_FONT("no-font");
    

    static gboolean MainLoopThread(gpointer arg);
}


#endif // _DSL_DRIVER_H