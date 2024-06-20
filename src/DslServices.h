/*
The MIT License

Copyright (c) 2019-2024, Prominence AI, Inc.

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
#include "DslOdeAccumulator.h"
#include "DslOdeHeatMapper.h"
#include "DslOdeTrigger.h"
#include "DslPipelineBintr.h"
#include "DslMessageBroker.h"
#if !defined(BUILD_WEBRTC)
    #error "BUILD_WEBRTC must be defined"
#elif BUILD_WEBRTC == true
    #include "DslSinkWebRtcBintr.h"
#endif
#include "spdlog/spdlog.h"


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
         * @return instance pointer to singleton services object.
         */
        static Services* GetServices();
        
        /** 
         * @brief Returns the state of the USE_NEW_NVSTREAMMUX env var.
         * @return true if USE_NEW_NVSTREAMMUX=yes, false otherwise.
         */
        bool UseNewStreammuxGet(){return m_useNewStreammux;};

        /***************************************************************
         **** all Services defined below are documented in DslApi.h ****
         ***************************************************************/ 
        DslReturnType DisplayTypeRgbaColorNew(const char* name, 
            double red, double green, double blue, double alpha);

        DslReturnType DisplayTypeRgbaColorPredefinedNew(const char* name, 
            uint colorId, double alpha);

        DslReturnType DisplayTypeRgbaColorRandomNew(const char* name, 
            uint hue, uint luminosity, double alpha, uint seed);

        DslReturnType DisplayTypeRgbaColorOnDemandNew(const char* name, 
            dsl_display_type_rgba_color_provider_cb provider, void* clientData);

        DslReturnType DisplayTypeRgbaColorPaletteNew(const char* name, 
            const char** colors, uint num_colors);

        DslReturnType DisplayTypeRgbaColorPalettePredefinedNew(const char* name, 
            uint paletteId, double alpha);

        DslReturnType DisplayTypeRgbaColorPaletteRandomNew(const char* name, 
            uint size, uint hue, uint luminosity, double alpha, uint seed);

        DslReturnType DisplayTypeRgbaColorPaletteIndexGet(const char* name, 
            uint* index);

        DslReturnType DisplayTypeRgbaColorPaletteIndexSet(const char* name, 
            uint index);

        DslReturnType DisplayTypeRgbaColorNextSet(const char* name);
            
        DslReturnType DisplayTypeRgbaFontNew(const char* name, const char* font,
            uint size, const char* color);
            
        DslReturnType DisplayTypeRgbaTextNew(const char* name, const char* text, 
            uint xOffset, uint yOffset, const char* font, boolean hasBgColor, 
            const char* bgColor);

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

        DslReturnType DisplayTypeRgbaLineMultiNew(const char* name, 
            const dsl_coordinate* coordinates, uint numCoordinates, 
            uint borderWidth, const char* color);

        DslReturnType DisplayTypeRgbaCircleNew(const char* name, 
            uint xCenter, uint yCenter, uint radius,
            const char* color, bool hasBgColor, const char* bgColor);
    
        DslReturnType DisplayTypeSourceUniqueIdNew(const char* name, 
            uint xOffset, uint yOffset, const char* font, 
            boolean hasBgColor, const char* bgColor);

        DslReturnType DisplayTypeSourceStreamIdNew(const char* name, 
            uint xOffset, uint yOffset, const char* font, 
            boolean hasBgColor, const char* bgColor);

        DslReturnType DisplayTypeSourceNameNew(const char* name, 
            uint xOffset, uint yOffset, const char* font, 
            boolean hasBgColor, const char* bgColor);

        DslReturnType DisplayTypeSourceDimensionsNew(const char* name, 
            uint xOffset, uint yOffset, const char* font, 
            boolean hasBgColor, const char* bgColor);

        DslReturnType DisplayTypeSourceFrameRateNew(const char* name, 
            uint xOffset, uint yOffset, const char* font, 
            boolean hasBgColor, const char* bgColor);

        DslReturnType DisplayRgbaTextShadowAdd(const char* name, 
            uint xOffset, uint yOffset, const char* color);
            
        DslReturnType DisplayTypeMetaAdd(const char* name, 
        void* pDisplayMeta, void* pFrameMeta);
        
        DslReturnType DisplayTypeDelete(const char* name);
        
        DslReturnType DisplayTypeDeleteAll();
        
        uint DisplayTypeListSize();
         
        DslReturnType OdeActionCustomNew(const char* name,
            dsl_ode_handle_occurrence_cb clientHandler, void* clientData);
            
        DslReturnType OdeActionCaptureFrameNew(const char* name, 
            const char* outdir);
        
        DslReturnType OdeActionCaptureObjectNew(const char* name, 
            const char* outdir);

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

        DslReturnType OdeActionDisplayNew(const char* name, 
            const char* formatString, uint offsetX, uint offsetY, 
            const char* font, boolean hasBgColor, const char* bgColor);
            
        DslReturnType OdeActionBBoxFormatNew(const char* name,
            uint borderWidth, const char* borderColor, boolean hasBgColor, const char* bgColor);
            
        DslReturnType OdeActionBBoxScaleNew(const char* name, uint scale);

        DslReturnType OdeActionBBoxStyleCornersNew(const char* name, 
            const char* color, uint length, uint maxLength,
            dsl_threshold_value* thicknessValues, uint numValues);
            
        DslReturnType OdeActionBBoxStyleCrosshairNew(const char* name, 
            const char* color, uint radius, uint maxRadius, uint innerRadius,
            dsl_threshold_value* thicknessValues, uint numValues);
            
        DslReturnType OdeActionLabelCustomizeNew(const char* name, 
            const uint* contentTypes, uint size);

        DslReturnType OdeActionLabelCustomizeGet(const char* name, 
            uint* contentTypes, uint* size);
        
        DslReturnType OdeActionLabelCustomizeSet(const char* name, 
            const uint* contentTypes, uint size);
        
        DslReturnType OdeActionLabelOffsetNew(const char* name, 
            int offset_x, int offset_y);
        
        DslReturnType OdeActionLabelSnapToGridNew(const char* name, 
            uint moduleWidth, uint moduleHeight);
        
        DslReturnType OdeActionLabelConnectToBBoxNew(const char* name, 
            const char* lineColor, uint lineWidth, uint bboxPoint);
        
        DslReturnType OdeActionLabelFormatNew(const char* name,
            const char* font, boolean hasBgColor, const char* bgColor);
        
        DslReturnType OdeActionLogNew(const char* name);

        DslReturnType OdeActionMessageMetaAddNew(const char* name);
        
        DslReturnType OdeActionMessageMetaTypeGet(const char* name,
            uint* metaType);

        DslReturnType OdeActionMessageMetaTypeSet(const char* name,
            uint metaType);
            
        DslReturnType OdeActionMonitorNew(const char* name,
            dsl_ode_monitor_occurrence_cb clientMonitor, void* clientData);
            
        DslReturnType OdeActionObjectRemoveNew(const char* name);

        DslReturnType OdeActionEmailNew(const char* name, 
            const char* mailer, const char* subject);
        
        DslReturnType OdeActionFileNew(const char* name, 
            const char* filePath, uint mode, uint format, boolean forceFlush);
        
        DslReturnType OdeActionFillSurroundingsNew(const char* name, const char* color);
        
        DslReturnType OdeActionFillFrameNew(const char* name, const char* color);

        DslReturnType OdeActionHandlerDisableNew(const char* name, const char* handler);

        DslReturnType OdeActionDisplayMetaAddNew(const char* name, const char* displayType);
        
        DslReturnType OdeActionDisplayMetaAddDisplayType(const char* name, const char* displayType);

        DslReturnType OdeActionPipelinePauseNew(const char* name, 
            const char* pipeline);

        DslReturnType OdeActionPipelinePlayNew(const char* name, 
            const char* pipeline);

        DslReturnType OdeActionPipelineStopNew(const char* name, 
            const char* pipeline);

        DslReturnType OdeActionPlayerPauseNew(const char* name, 
            const char* player);

        DslReturnType OdeActionPlayerPlayNew(const char* name, 
            const char* player);

        DslReturnType OdeActionPlayerStopNew(const char* name, 
            const char* player);

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

        DslReturnType OdeActionBranchAddNew(const char* name, 
            const char* tee, const char* branch);

        DslReturnType OdeActionBranchAddToNew(const char* name, 
            const char* demuxer, const char* branch);

        DslReturnType OdeActionBranchMoveToNew(const char* name, 
            const char* demuxer, const char* branch);

        DslReturnType OdeActionBranchRemoveNew(const char* name, 
            const char* tee, const char* branch);

        DslReturnType OdeActionEnabledGet(const char* name, boolean* enabled);

        DslReturnType OdeActionEnabledSet(const char* name, boolean enabled);

        DslReturnType OdeActionEnabledStateChangeListenerAdd(const char* name,
            dsl_ode_enabled_state_change_listener_cb listener, void* clientData);

        DslReturnType OdeActionEnabledStateChangeListenerRemove(const char* name,
            dsl_ode_enabled_state_change_listener_cb listener);

        DslReturnType OdeActionDelete(const char* name);
        
        DslReturnType OdeActionDeleteAll();
        
        uint OdeActionListSize();

        DslReturnType OdeAreaInclusionNew(const char* name, 
            const char* polygon, boolean display, uint bboxTestPoint);

        DslReturnType OdeAreaExclusionNew(const char* name, 
            const char* polygon, boolean display, uint bboxTestPoint);

        DslReturnType OdeAreaLineNew(const char* name, 
            const char* line, boolean display, uint bboxTestPoint);

        DslReturnType OdeAreaLineMultiNew(const char* name, 
            const char* multiLine, boolean display, uint bboxTestPoint);

        DslReturnType OdeAreaDelete(const char* name);
        
        DslReturnType OdeAreaDeleteAll();
        
        uint OdeAreaListSize();
        
        DslReturnType OdeTriggerAlwaysNew(const char* name, 
            const char* source, uint when);
        
        DslReturnType OdeTriggerOccurrenceNew(const char* name, 
            const char* source, uint classId, uint limit);
        
        DslReturnType OdeTriggerAbsenceNew(const char* name, 
            const char* source, uint classId, uint limit);

        DslReturnType OdeTriggerInstanceNew(const char* name, 
            const char* source, uint classId, uint limit);
            
        DslReturnType OdeTriggerInstanceCountSettingsGet(const char* name,
            uint* instanceCount, uint* suppressionCount);
        
        DslReturnType OdeTriggerInstanceCountSettingsSet(const char* name,
            uint instanceCount, uint suppressionCount);
        
        DslReturnType OdeTriggerIntersectionNew(const char* name, 
            const char* source, uint classIdA, uint classIdB, uint limit);

        DslReturnType OdeTriggerSummationNew(const char* name, 
            const char* source, uint classId, uint limit);

        DslReturnType OdeTriggerCustomNew(const char* name, const char* source, 
            uint classId, uint limit,  dsl_ode_check_for_occurrence_cb client_checker, 
            dsl_ode_post_process_frame_cb client_post_processor, void* client_data);

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

        DslReturnType OdeTriggerNewLowNew(const char* name, 
            const char* source, uint classId, uint limit, uint preset);

        DslReturnType OdeTriggerNewHighNew(const char* name, 
            const char* source, uint classId, uint limit, uint preset);

        DslReturnType OdeTriggerCrossNew(const char* name, 
            const char* source, uint classId, uint limit, 
            uint minFrameCount, uint maxFrameCount, uint testMethod);
            
        DslReturnType OdeTriggerPersistenceNew(const char* name, 
            const char* source, uint classId, uint limit, uint minimum, uint maximum);

        DslReturnType OdeTriggerPersistenceRangeGet(const char* name, 
            uint* minimum, uint* maximum);
        
        DslReturnType OdeTriggerPersistenceRangeSet(const char* name, 
            uint minimum, uint maximum);

        DslReturnType OdeTriggerEarliestNew(const char* name, 
            const char* source, uint classId, uint limit);
            
        DslReturnType OdeTriggerLatestNew(const char* name, 
            const char* source, uint classId, uint limit);
            
        DslReturnType OdeTriggerCrossTestSettingsGet(const char* name, 
            uint* minFrameCount, uint* maxFrameCount, uint* testMethod);
            
        DslReturnType OdeTriggerCrossTestSettingsSet(const char* name, 
            uint minFrameCount, uint maxFrameCount, uint testMethod);
            
        DslReturnType OdeTriggerCrossViewSettingsGet(const char* name, 
            boolean* enabled, const char** color, uint* lineWidth);
            
        DslReturnType OdeTriggerCrossViewSettingsSet(const char* name, 
            boolean enabled, const char* color, uint lineWidth);
        
        DslReturnType OdeTriggerReset(const char* name);

        DslReturnType OdeTriggerResetTimeoutGet(const char* name, uint* timeout);

        DslReturnType OdeTriggerResetTimeoutSet(const char* name, uint timeout);
        
        DslReturnType OdeTriggerLimitStateChangeListenerAdd(const char* name,
            dsl_ode_trigger_limit_state_change_listener_cb listener, void* clientData);

        DslReturnType OdeTriggerLimitStateChangeListenerRemove(const char* name,
            dsl_ode_trigger_limit_state_change_listener_cb listener);

        DslReturnType OdeTriggerEnabledGet(const char* name, boolean* enabled);

        DslReturnType OdeTriggerEnabledSet(const char* name, boolean enabled);

        DslReturnType OdeTriggerEnabledStateChangeListenerAdd(const char* name,
            dsl_ode_enabled_state_change_listener_cb listener, void* clientData);

        DslReturnType OdeTriggerEnabledStateChangeListenerRemove(const char* name,
            dsl_ode_enabled_state_change_listener_cb listener);

        DslReturnType OdeTriggerSourceGet(const char* name, const char** source);
        
        DslReturnType OdeTriggerSourceSet(const char* name, const char* source);
        
        DslReturnType OdeTriggerInferGet(const char* name, const char** infer);
        
        DslReturnType OdeTriggerInferSet(const char* name, const char* infer);
        
        DslReturnType OdeTriggerClassIdGet(const char* name, uint* classId);
        
        DslReturnType OdeTriggerClassIdSet(const char* name, uint classId);
        
        DslReturnType OdeTriggerClassIdABGet(const char* name, 
            uint* classIdA, uint* classIdB);
        
        DslReturnType OdeTriggerClassIdABSet(const char* name, 
            uint classIdA, uint classIdB);
        
        DslReturnType OdeTriggerLimitEventGet(const char* name, uint* limit);
        
        DslReturnType OdeTriggerLimitEventSet(const char* name, uint limit);
        
        DslReturnType OdeTriggerLimitFrameGet(const char* name, uint* limit);
        
        DslReturnType OdeTriggerLimitFrameSet(const char* name, uint limit);
        
        DslReturnType OdeTriggerConfidenceMinGet(const char* name, 
            float* minConfidence);
        
        DslReturnType OdeTriggerConfidenceMinSet(const char* name, 
            float minConfidence);
        
        DslReturnType OdeTriggerConfidenceMaxGet(const char* name, 
            float* maxConfidence);
        
        DslReturnType OdeTriggerConfidenceMaxSet(const char* name, 
            float maxConfidence);
        
        DslReturnType OdeTriggerTrackerConfidenceMinGet(const char* name, 
            float* minConfidence);
        
        DslReturnType OdeTriggerTrackerConfidenceMinSet(const char* name, 
            float minConfidence);
        
        DslReturnType OdeTriggerTrackerConfidenceMaxGet(const char* name, 
            float* maxConfidence);
        
        DslReturnType OdeTriggerTrackerConfidenceMaxSet(const char* name, 
            float maxConfidence);
        
        DslReturnType OdeTriggerDimensionsMinGet(const char* name, 
            float* min_width, float* min_height);
        
        DslReturnType OdeTriggerDimensionsMinSet(const char* name, 
            float min_width, float min_height);

        DslReturnType OdeTriggerDimensionsMaxGet(const char* name, 
            float* max_width, float* max_height);
        
        DslReturnType OdeTriggerDimensionsMaxSet(const char* name, 
            float max_width, float max_height);

        DslReturnType OdeTriggerFrameCountMinGet(const char* name, 
            uint* min_count_n, uint* min_count_d);

        DslReturnType OdeTriggerFrameCountMinSet(const char* name, 
            uint min_count_n, uint min_count_d);
        
        DslReturnType OdeTriggerInferDoneOnlyGet(const char* name, 
            boolean* inferDoneOnly);
        
        DslReturnType OdeTriggerInferDoneOnlySet(const char* name, 
            boolean inferDoneOnly);
        
        DslReturnType OdeTriggerIntervalGet(const char* name, uint* interval);
        
        DslReturnType OdeTriggerIntervalSet(const char* name, uint interval);
        
        DslReturnType OdeTriggerActionAdd(const char* name, const char* action);

        DslReturnType OdeTriggerActionRemove(const char* name, const char* action);

        DslReturnType OdeTriggerActionRemoveAll(const char* name);

        DslReturnType OdeTriggerAreaAdd(const char* name, const char* area);

        DslReturnType OdeTriggerAreaRemove(const char* name, const char* area);

        DslReturnType OdeTriggerAreaRemoveAll(const char* name);

        DslReturnType OdeTriggerAccumulatorAdd(const char* name, 
            const char* accumulator);

        DslReturnType OdeTriggerAccumulatorRemove(const char* name);

        DslReturnType OdeTriggerHeatMapperAdd(const char* name, 
            const char* heatMapper);

        DslReturnType OdeTriggerHeatMapperRemove(const char* name);

        DslReturnType OdeTriggerDelete(const char* name);
        
        DslReturnType OdeTriggerDeleteAll();
        
        uint OdeTriggerListSize();

        DslReturnType OdeAccumulatorNew(const char* name);

        DslReturnType OdeAccumulatorActionAdd(const char* name, const char* action);

        DslReturnType OdeAccumulatorActionRemove(const char* name, const char* action);

        DslReturnType OdeAccumulatorActionRemoveAll(const char* name);

        DslReturnType OdeAccumulatorDelete(const char* name);
        
        DslReturnType OdeAccumulatorDeleteAll();
        
        uint OdeAccumulatorListSize();

        DslReturnType OdeHeatMapperNew(const char* name,
            uint cols, uint rows, uint bboxTestPoint, const char* colorPalette);
            
        DslReturnType OdeHeatMapperColorPaletteGet(const char* name,
            const char** colorPalette);
        
        DslReturnType OdeHeatMapperColorPaletteSet(const char* name,
            const char* colorPalette);
        
        DslReturnType OdeHeatMapperLegendSettingsGet(const char* name,
            boolean* enabled, uint* location, uint* width, uint* height);

        DslReturnType OdeHeatMapperLegendSettingsSet(const char* name,
            boolean enabled, uint location, uint width, uint height);

        DslReturnType OdeHeatMapperMetricsClear(const char* name);

        DslReturnType OdeHeatMapperMetricsGet(const char* name,
            const uint64_t** buffer, uint* size);

        DslReturnType OdeHeatMapperMetricsPrint(const char* name);

        DslReturnType OdeHeatMapperMetricsLog(const char* name);

        DslReturnType OdeHeatMapperMetricsFile(const char* name,
            const char* filePath, uint mode, uint format);

        DslReturnType OdeHeatMapperDelete(const char* name);
        
        DslReturnType OdeHeatMapperDeleteAll();
        
        uint OdeHeatMapperListSize();

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
        
        DslReturnType PphOdeDisplayMetaAllocSizeGet(const char* name, uint* size);

        DslReturnType PphOdeDisplayMetaAllocSizeSet(const char* name, uint size);

        DslReturnType PphNmpNew(const char* name, const char* labelFile,
            uint processMethod, uint matchMethod, float matchThreshold);
            
        DslReturnType PphNmpLabelFileGet(const char* name, const char** labelFile);
        
        DslReturnType PphNmpLabelFileSet(const char* name, const char* labelFile);

        DslReturnType PphNmpProcessMethodGet(const char* name, uint* processMethod);
        
        DslReturnType PphNmpProcessMethodSet(const char* name, uint processMethod);
        
        DslReturnType PphNmpMatchSettingsGet(const char* name, 
            uint* matchMethod, float* matchThreshold);
        
        DslReturnType PphNmpMatchSettingsSet(const char* name, 
            uint matchMethod, float matchThreshold);
        
        DslReturnType PphBufferTimeoutNew(const char* name,
            uint timeout, dsl_pph_buffer_timeout_handler_cb handler, void* clientData);
    
        DslReturnType PphEosNew(const char* name,
            dsl_pph_eos_handler_cb handler, void* clientData);
    
        DslReturnType PphStreamEventNew(const char* name,
            dsl_pph_stream_event_handler_cb handler, void* clientData);
    
        DslReturnType PphEnabledGet(const char* name, boolean* enabled);
        
        DslReturnType PphEnabledSet(const char* name, boolean enabled);

        DslReturnType PphDelete(const char* name);
        
        DslReturnType PphDeleteAll();
        
        uint PphListSize();
        
        DslReturnType GstElementNew(const char* name, const char* factoryName);
        
        DslReturnType GstElementDelete(const char* name);
        
        DslReturnType GstElementDeleteAll();
        
        uint GstElementListSize();
        
        DslReturnType GstElementGet(const char* name, void** element);

        DslReturnType GstElementPropertyBooleanGet(const char* name, 
            const char* property, boolean* value);
        
        DslReturnType GstElementPropertyBooleanSet(const char* name, 
            const char* property, boolean value);
        
        DslReturnType GstElementPropertyFloatGet(const char* name, 
            const char* property, float* value);
        
        DslReturnType GstElementPropertyFloatSet(const char* name, 
            const char* property, float value);
        
        DslReturnType GstElementPropertyUintGet(const char* name, 
            const char* property, uint* value);
        
        DslReturnType GstElementPropertyUintSet(const char* name, 
            const char* property, uint value);
        
        DslReturnType GstElementPropertyIntGet(const char* name, 
            const char* property, int* value);
        
        DslReturnType GstElementPropertyIntSet(const char* name, 
            const char* property, int value);
        
        DslReturnType GstElementPropertyUint64Get(const char* name, 
            const char* property, uint64_t* value);
        
        DslReturnType GstElementPropertyUint64Set(const char* name, 
            const char* property, uint64_t value);
        
        DslReturnType GstElementPropertyInt64Get(const char* name, 
            const char* property, int64_t* value);
        
        DslReturnType GstElementPropertyInt64Set(const char* name, 
            const char* property, int64_t value);
        
        DslReturnType GstElementPropertyStringGet(const char* name, 
            const char* property, const char** value);
        
        DslReturnType GstElementPropertyStringSet(const char* name, 
            const char* property, const char* value);
   
        DslReturnType GstElementPphAdd(const char* name, 
            const char* handler, uint pad);

        DslReturnType GstElementPphRemove(const char* name, 
            const char* handler, uint pad);
            
        DslReturnType GstBinNew(const char* name);
        
        DslReturnType GstBinElementAdd(const char* name, const char* element);

        DslReturnType GstBinElementRemove(const char* name, const char* element);
        
        DslReturnType SourceAppNew(const char* name, boolean isLive, 
            const char* bufferInFormat, uint width, uint height, 
            uint fpsN, uint fpsD);

        DslReturnType SourceCustomNew(const char* name, const char* elementName, 
            const char* factory, void** element);

        DslReturnType SourceAppDataHandlersAdd(const char* name,
            dsl_source_app_need_data_handler_cb needDataHandler, 
            dsl_source_app_enough_data_handler_cb enoughDataHandler, 
            void* clientData);

        DslReturnType SourceAppDataHandlersRemove(const char* name);
            
        DslReturnType SourceAppBufferPush(const char* name, void* buffer);

        DslReturnType SourceAppSamplePush(const char* name, void* sample);

        DslReturnType SourceAppEos(const char* name);
        
        DslReturnType SourceAppStreamFormatGet(const char* name,
            uint* StreamFormat);
        
        DslReturnType SourceAppStreamFormatSet(const char* name,
            uint bufferFormat);
        
        DslReturnType SourceAppDoTimestampGet(const char* name, boolean* doTimestamp);
            
        DslReturnType SourceAppDoTimestampSet(const char* name, boolean doTimestamp);
            
        DslReturnType SourceAppBlockEnabledGet(const char* name,
            boolean* enabled);
        
        DslReturnType SourceAppBlockEnabledSet(const char* name,
            boolean enabled);
        
        DslReturnType SourceAppCurrentLevelBytesGet(const char* name,
            uint64_t* level);
        
        DslReturnType SourceAppMaxLevelBytesGet(const char* name,
            uint64_t* level);
        
        DslReturnType SourceAppMaxLevelBytesSet(const char* name,
            uint64_t level);
        
//        DslReturnType SourceAppLeakyTypeGet(const char* name,
//            uint* leakyType);
//        
//        DslReturnType SourceAppLeakyTypeSet(const char* name,
//            uint leakyType);

        DslReturnType SourceCsiNew(const char* name, 
            uint width, uint height, uint fpsN, uint fpsD);
            
        DslReturnType SourceCsiSensorIdGet(const char* name, 
            uint* sensorId);
        
        DslReturnType SourceCsiSensorIdSet(const char* name, 
            uint sensorId);
        
        DslReturnType SourceV4l2New(const char* name, 
            const char* deviceLocation);

        DslReturnType SourceV4l2DeviceLocationGet(const char* name, 
            const char** deviceLocation);
        
        DslReturnType SourceV4l2DeviceLocationSet(const char* name, 
            const char* deviceLocation);
        
        DslReturnType SourceV4l2DimensionsSet(const char* name, 
            uint width, uint height);

        DslReturnType SourceV4l2FrameRateSet(const char* name, 
            uint fps_n, uint fps_d);

        DslReturnType SourceV4l2DeviceNameGet(const char* name, 
            const char** deviceName);

        DslReturnType SourceV4l2DeviceFdGet(const char* name, 
            int* deviceFd);

        DslReturnType SourceV4l2DeviceFlagsGet(const char* name, 
            uint* deviceFlags);

        DslReturnType SourceV4l2PictureSettingsGet(const char* name, 
            int* brightness, int* contrast, int* hue);

        DslReturnType SourceV4l2PictureSettingsSet(const char* name, 
            int brightness, int contrast, int hue);

        DslReturnType SourceUriNew(const char* name, const char* uri, 
            boolean isLive, uint skipFrames, uint dropFrameInterval);
            
        DslReturnType SourceFileNew(const char* name, const char* filePath, 
            boolean repeatEnabled);

        DslReturnType SourceFileFilePathGet(const char* name, const char** filePath);

        DslReturnType SourceFileFilePathSet(const char* name, const char* filePath);

        DslReturnType SourceFileRepeatEnabledGet(const char* name, boolean* enabled);
    
        DslReturnType SourceFileRepeatEnabledSet(const char* name, boolean enabled);
            
        DslReturnType SourceImageNew(const char* name, 
            const char* filePath);

        DslReturnType SourceImageMultiNew(const char* name, 
            const char* filePath, uint fpsN, uint fpsD);
            
        DslReturnType SourceImageMultiLoopEnabledGet(const char* name,
            boolean* enabled);
        
        DslReturnType SourceImageMultiLoopEnabledSet(const char* name,
            boolean enabled);
        
        DslReturnType SourceImageMultiIndicesGet(const char* name,
            int* startIndex, int* stopIndex);
        
        DslReturnType SourceImageMultiIndicesSet(const char* name,
            int startIndex, int stopIndex);

        DslReturnType SourceImageStreamNew(const char* name, const char* filePath, 
            boolean isLive, uint fpsN, uint fpsD, uint timeout);

        DslReturnType SourceImageStreamTimeoutGet(const char* name, uint* timeout);
    
        DslReturnType SourceImageStreamTimeoutSet(const char* name, uint timeout);
            
        DslReturnType SourceImageFilePathGet(const char* name, const char** filePath);

        DslReturnType SourceImageFilePathSet(const char* name, const char* filePath);

        DslReturnType SourceInterpipeNew(const char* name, const char* listenTo,
            boolean isLive, boolean acceptEos, boolean acceptEvents);
            
        DslReturnType SourceInterpipeListenToGet(const char* name, const char** listenTo);
            
        DslReturnType SourceInterpipeListenToSet(const char* name, const char* listenTo);
        
        DslReturnType SourceInterpipeAcceptSettingsGet(const char* name,
            boolean* acceptEos, boolean* acceptEvents);
            
        DslReturnType SourceInterpipeAcceptSettingsSet(const char* name,
            boolean acceptEos, boolean acceptEvents);
            
        DslReturnType SourceRtspNew(const char* name, const char* uri, uint protocol, 
            uint skipFrames, uint dropFrameInterval, uint latency, uint timeout);

        DslReturnType SourceDuplicateNew(const char* name, const char* original);

        DslReturnType SourceDuplicateOriginalGet(const char* name, 
            const char** original);

        DslReturnType SourceDuplicateOriginalSet(const char* name, 
            const char* original);

        DslReturnType SourcePphAdd(const char* name, const char* handler);

        DslReturnType SourcePphRemove(const char* name, const char* handler);

        DslReturnType SourceMediaTypeGet(const char* name, 
            const char** mediaType);

        DslReturnType SourceVideoBufferOutFormatGet(const char* name, 
            const char** format);

        DslReturnType SourceVideoBufferOutFormatSet(const char* name, 
            const char* format);
            
        DslReturnType SourceVideoBufferOutDimensionsGet(const char* name, 
            uint* width, uint* height);

        DslReturnType SourceVideoBufferOutDimensionsSet(const char* name, 
            uint width, uint height);

        DslReturnType SourceVideoBufferOutFrameRateGet(const char* name, 
            uint* fps_n, uint* fps_d);

        DslReturnType SourceVideoBufferOutFrameRateSet(const char* name, 
            uint fps_n, uint fps_d);

        DslReturnType SourceVideoBufferOutCropRectangleGet(const char* name, 
            uint cropAt, uint* left, uint* top, uint* width, uint* height);

        DslReturnType SourceVideoBufferOutCropRectangleSet(const char* name, 
            uint cropAt, uint left, uint top, uint width, uint height);

        DslReturnType SourceVideoBufferOutOrientationGet(const char* name, 
            uint* orientation);

        DslReturnType SourceVideoBufferOutOrientationSet(const char* name, 
            uint orientation);

        DslReturnType SourceVideoDimensionsGet(const char* name, uint* width, uint* height);
        
        DslReturnType SourceFrameRateGet(const char* name, uint* fpsN, uint* fpsD);

        DslReturnType SourceVideoDewarperAdd(const char* name, const char* dewarper);
    
        DslReturnType SourceVideoDewarperRemove(const char* name);

        DslReturnType SourceUriUriGet(const char* name, const char** uri);

        DslReturnType SourceUriUriSet(const char* name, const char* uri);
    
        DslReturnType SourceRtspUriGet(const char* name, const char** uri);

        DslReturnType SourceRtspUriSet(const char* name, const char* uri);
    
        DslReturnType SourceRtspTimeoutGet(const char* name, uint* timeout);

        DslReturnType SourceRtspTimeoutSet(const char* name, uint timeout);
        
        DslReturnType SourceRtspConnectionParamsGet(const char* name, 
            uint* sleep, uint* timeout);

        DslReturnType SourceRtspConnectionParamsSet(const char* name, 
            uint sleep, uint timeout);
        
        DslReturnType SourceRtspConnectionDataGet(const char* name, 
            dsl_rtsp_connection_data* data);
        
        DslReturnType SourceRtspConnectionStatsClear(const char* name);

        DslReturnType SourceRtspLatencyGet(const char* name, 
            uint* latency);

        DslReturnType SourceRtspLatencySet(const char* name, 
            uint latency);
        
        DslReturnType SourceRtspDropOnLatencyEnabledGet(const char* name, 
            boolean* enabled);

        DslReturnType SourceRtspDropOnLatencyEnabledSet(const char* name, 
            boolean enabled);
        
        DslReturnType SourceRtspTlsValidationFlagsGet(const char* name, 
            uint* flags);

        DslReturnType SourceRtspTlsValidationFlagsSet(const char* name, 
            uint flags);

        DslReturnType SourceRtspStateChangeListenerAdd(const char* name, 
            dsl_state_change_listener_cb listener, void* clientData);
        
        DslReturnType SourceRtspStateChangeListenerRemove(const char* name, 
            dsl_state_change_listener_cb listener);
        
        DslReturnType SourceRtspTapAdd(const char* name, const char* tap);
    
        DslReturnType SourceRtspTapRemove(const char* name);
        
        DslReturnType SourceUniqueIdGet(const char* name, int* uniqueId);
    
        DslReturnType SourceStreamIdGet(const char* name, int* streamId);
    
        DslReturnType SourceNameGet(int uniqueId, const char** name);

        void _sourceNameSet(const char* name, uint uniqueId);
    
        bool _sourceNameErase(const char* name);
    
        DslReturnType SourcePause(const char* name);

        DslReturnType SourceResume(const char* name);

        boolean SourceIsLive(const char* name);
        
        DslReturnType DewarperNew(const char* name, 
            const char* configFile, uint sourceId);
        
        DslReturnType DewarperConfigFileGet(const char* name, 
            const char** configFile);
            
        DslReturnType DewarperConfigFileSet(const char* name, 
            const char* configFile);
            
        DslReturnType DewarperCameraIdGet(const char* name, uint* cameraId);

        DslReturnType DewarperCameraIdSet(const char* name, uint cameraId);

        DslReturnType DewarperNumBatchBuffersGet(const char* name, uint* num);

        DslReturnType DewarperNumBatchBuffersSet(const char* name, uint num);

        DslReturnType TapRecordNew(const char* name, const char* outdir, 
            uint container, dsl_record_client_listener_cb clientListener);
            
        DslReturnType TapRecordSessionStart(const char* name, 
            uint start, uint duration, void* clientData);

        DslReturnType TapRecordSessionStop(const char* name, boolean sync);

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

        DslReturnType PreprocNew(const char* name, const char* configFile);
        
        DslReturnType PreprocConfigFileGet(const char* name, 
            const char** configFile);
            
        DslReturnType PreprocConfigFileSet(const char* name, 
            const char* configFile);
            
        DslReturnType PreprocEnabledGet(const char* name, 
            boolean* enabled);
            
        DslReturnType PreprocEnabledSet(const char* name, 
            boolean enabled);
            
        DslReturnType PreprocUniqueIdGet(const char* name, 
            uint* uniqueId);

        DslReturnType PreprocPphAdd(const char* name, 
            const char* handler, uint pad);

        DslReturnType PreprocPphRemove(const char* name, 
            const char* handler, uint pad);


        DslReturnType SegVisualNew(const char* name, uint width, uint height);
        
        DslReturnType SegVisualDimensionsGet(const char* name, uint* width, uint* height);

        DslReturnType SegVisualDimensionsSet(const char* name, uint width, uint height);

        DslReturnType SegVisualPphAdd(const char* name, const char* handler);

        DslReturnType SegVisualPphRemove(const char* name, const char* handler);

        DslReturnType InferPrimaryGieNew(const char* name, const char* inferConfigFile,
            const char* modelEngineFile, uint interval);

        DslReturnType InferPrimaryTisNew(const char* name, 
            const char* inferConfigFile, uint interval);

        DslReturnType InferSecondaryGieNew(const char* name, const char* inferConfigFile,
            const char* modelEngineFile, const char* inferOnGieName, uint interval);

        DslReturnType InferSecondaryTisNew(const char* name, const char* inferConfigFile,
            const char* inferOnGieName, uint interval);

        DslReturnType InferBatchSizeGet(const char* name, uint* size);

        DslReturnType InferBatchSizeSet(const char* name, uint size);

        DslReturnType InferUniqueIdGet(const char* name, uint* id);

        DslReturnType InferPphAdd(const char* name, 
            const char* handler, uint pad);

        DslReturnType InferPphRemove(const char* name, 
            const char* handler, uint pad);

        DslReturnType InferGieModelEngineFileGet(const char* name, 
            const char** modelEngineFile);

        DslReturnType InferGieModelEngineFileSet(const char* name, 
            const char* modelEngineFile);

        DslReturnType InferConfigFileGet(const char* name, const char** inferConfigFile);

        DslReturnType InferConfigFileSet(const char* name, const char* inferConfigFile);
            
        DslReturnType InferRawOutputEnabledSet(const char* name, boolean enabled,
            const char* path);
            
        DslReturnType InferGieTensorMetaSettingsGet(const char* name, 
            boolean* inputEnabled, boolean* outputEnabled);
            
        DslReturnType InferGieTensorMetaSettingsSet(const char* name, 
            boolean inputEnabled, boolean outputEnabled);
            
        DslReturnType InferIntervalGet(const char* name, uint* interval);

        DslReturnType InferIntervalSet(const char* name, uint interval);
        
        DslReturnType InferNameGet(int inferId, const char** name);

        DslReturnType InferIdGet(const char* name, int* inferId);
    
        DslReturnType _inferAttributesSet(uint inferId, 
            const char* name, uint processMode);
    
        DslReturnType _inferAttributesErase(uint inferId);

        DslReturnType _inferAttributesGetByName(const char* name, 
            uint& inferId, uint& processMode);
    
        DslReturnType TrackerNew(const char* name, 
            const char* configFile, uint width, uint height);

        DslReturnType TrackerLibFileGet(const char* name, const char** libFile);

        DslReturnType TrackerLibFileSet(const char* name, const char* libFile);
        
        DslReturnType TrackerConfigFileGet(const char* name, const char** configFile);

        DslReturnType TrackerConfigFileSet(const char* name, const char* configFile);
        
        DslReturnType TrackerDimensionsGet(const char* name, uint* width, uint* height);
        
        DslReturnType TrackerDimensionsSet(const char* name, uint width, uint height);

        DslReturnType TrackerTensorMetaSettingsGet(const char* name, 
            boolean* inputEnabled, const char** trackOnGie);
        
        DslReturnType TrackerTensorMetaSettingsSet(const char* name, 
            boolean inputEnabled, const char* trackOnGie);
        
        DslReturnType TrackerIdDisplayEnabledGet(const char* name, 
            boolean* enabled);
        
        DslReturnType TrackerIdDisplayEnabledSet(const char* name, 
            boolean enabled);
        
        DslReturnType TrackerPphAdd(const char* name, const char* handler, uint pad);

        DslReturnType TrackerPphRemove(const char* name, const char* handler, uint pad);
        
        DslReturnType TeeDemuxerNew(const char* name, uint maxBranches);

        DslReturnType TeeDemuxerMaxBranchesGet(const char* name, uint* maxBranches);
        
        DslReturnType TeeDemuxerMaxBranchesSet(const char* name, uint maxBranches);
        
        DslReturnType TeeDemuxerBranchAddTo(const char* name, 
            const char* branch, uint stream_id);

        DslReturnType TeeDemuxerBranchMoveTo(const char* name, 
            const char* branch, uint stream_id);
            
        DslReturnType TeeSplitterNew(const char* name);
           
        DslReturnType TeeBranchAdd(const char* name, const char* branch);
        
        DslReturnType TeeBranchRemove(const char* name, const char* branch);
        
        DslReturnType TeeBranchRemoveAll(const char* name);

        DslReturnType TeeBranchCountGet(const char* name, uint* count);

        DslReturnType TeeBlockingTimeoutGet(const char* name, uint* timeout);
        
        DslReturnType TeeBlockingTimeoutSet(const char* name, uint timeout);
        
        DslReturnType TeePphAdd(const char* name, const char* handler);

        DslReturnType TeePphRemove(const char* name, const char* handler);

        DslReturnType RemuxerNew(const char* name);

        DslReturnType RemuxerBranchAddTo(const char* name, const char* branch,
            uint* streamIds, uint numStreamIds);

        DslReturnType RemuxerBranchAdd(const char* name, const char* branch);
        
        DslReturnType RemuxerBranchRemove(const char* name, const char* branch);
        
        DslReturnType RemuxerBranchRemoveAll(const char* name);

        DslReturnType RemuxerBranchCountGet(const char* name, uint* count);
        
        DslReturnType RemuxerBatchSizeGet(const char* name,
            uint* batchSize);

        DslReturnType RemuxerBatchSizeSet(const char* name,
            uint batchSize);

        DslReturnType RemuxerBranchConfigFileGet(const char* name,
            const char* branch, const char** configFile);

        DslReturnType RemuxerBranchConfigFileSet(const char* name,
            const char* branch, const char* configFile);

       DslReturnType RemuxerBatchPropertiesGet(const char* name,
            uint* batchSize, int* batchTimeout);

        DslReturnType RemuxerBatchPropertiesSet(const char* name,
            uint batchSize, int batchTimeout);

        DslReturnType RemuxerDimensionsGet(const char* name,
            uint* width, uint* height);

        DslReturnType RemuxerDimensionsSet(const char* name,
            uint width, uint height);
        
        DslReturnType RemuxerPphAdd(const char* name, 
            const char* handler, uint pad);

        DslReturnType RemuxerPphRemove(const char* name, 
            const char* handler, uint pad);

        DslReturnType TilerNew(const char* name, uint width, uint height);
        
        DslReturnType TilerDimensionsGet(const char* name, uint* width, uint* height);

        DslReturnType TilerDimensionsSet(const char* name, uint width, uint height);

        DslReturnType TilerTilesGet(const char* name, uint* columns, uint* rows);

        DslReturnType TilerTilesSet(const char* name, uint columns, uint rows);
        
        DslReturnType TilerFrameNumberingEnabledGet(const char* name,
            boolean* enabled);

        DslReturnType TilerFrameNumberingEnabledSet(const char* name,
            boolean enabled);
            
        DslReturnType TilerSourceShowGet(const char* name, 
            const char** source, uint* timeout);

        DslReturnType TilerSourceShowSet(const char* name, 
            const char* source, uint timeout, bool hasPrecedence);

        // called by the Show Source Action only. 
        DslReturnType TilerSourceShowSet(const char* name, 
            uint sourceId, uint timeout, bool hasPrecedence);

        DslReturnType TilerSourceShowSelect(const char* name, 
            int xPos, int yPos, uint windowWidth, uint windowHeight, uint timeout);

        DslReturnType TilerSourceShowAll(const char* name);

        DslReturnType TilerSourceShowCycle(const char* name, uint timeout);

        DslReturnType TilerPphAdd(const char* name, const char* handler, uint pad);

        DslReturnType TilerPphRemove(const char* name, const char* handler, uint pad);

        DslReturnType OfvNew(const char* name);

        DslReturnType OsdNew(const char* name, 
            boolean textEnabled, boolean clockEnabled,
            boolean bboxEnabled, boolean maskEnabled);
        
        DslReturnType OsdTextEnabledGet(const char* name, boolean* enabled);

        DslReturnType OsdTextEnabledSet(const char* name, boolean enabled);

        DslReturnType OsdClockEnabledGet(const char* name, boolean* enabled);

        DslReturnType OsdClockEnabledSet(const char* name, boolean enabled);

        DslReturnType OsdClockOffsetsGet(const char* name, uint* offsetX, uint* offsetY);

        DslReturnType OsdClockOffsetsSet(const char* name, uint offsetX, uint offsetY);

        DslReturnType OsdClockFontGet(const char* name, const char** font, uint* size);

        DslReturnType OsdClockFontSet(const char* name, const char* font, uint size);

        DslReturnType OsdClockColorGet(const char* name, 
            double* red, double* green, double* blue, double* alpha);

        DslReturnType OsdClockColorSet(const char* name, 
            double red, double green, double blue, double alpha);

        DslReturnType OsdBboxEnabledGet(const char* name, boolean* enabled);

        DslReturnType OsdBboxEnabledSet(const char* name, boolean enabled);

        DslReturnType OsdMaskEnabledGet(const char* name, boolean* enabled);

        DslReturnType OsdMaskEnabledSet(const char* name, boolean enabled);

        DslReturnType OsdProcessModeGet(const char* name, uint* mode);

        DslReturnType OsdProcessModeSet(const char* name, uint mode);

        DslReturnType OsdPphAdd(const char* name, const char* handler, uint pad);

        DslReturnType OsdPphRemove(const char* name, const char* handler, uint pad);

        DslReturnType SinkAppNew(const char* name, uint dataType,
            dsl_sink_app_new_data_handler_cb clientHandler, void* clientData);
            
        DslReturnType SinkAppDataTypeGet(const char* name, uint* dataType);

        DslReturnType SinkAppDataTypeSet(const char* name, uint dataType);

        DslReturnType SinkFakeNew(const char* name);

        // ---------------------------------------------------------------------------
        // The following three internal services provide access to the
        // database of active Window Sinks
        DslReturnType _sinkWindowRegister(DSL_BASE_PTR sink, GstObject* element);
        
        DslReturnType _sinkWindowUnregister(DSL_BASE_PTR sink);

        DSL_BASE_PTR _sinkWindowGet(GstObject* element);
        // ---------------------------------------------------------------------------
    
        DslReturnType SinkWindow3dNew(const char* name,
            uint offsetX, uint offsetY, uint width, uint height);
        
        DslReturnType SinkWindowEglNew(const char* name, 
            uint offsetX, uint offsetY, uint width, uint height);
            
        DslReturnType SinkWindowOffsetsGet(const char* name, 
            uint* offsetX, uint* offsetY);

        DslReturnType SinkWindowOffsetsSet(const char* name, 
            uint offsetX, uint offsetY);
        
        DslReturnType SinkWindowDimensionsGet(const char* name, 
            uint* width, uint* height);

        DslReturnType SinkWindowDimensionsSet(const char* name, 
            uint width, uint height);

        DslReturnType SinkWindowHandleGet(const char* name, uint64_t* handle);

        DslReturnType SinkWindowHandleSet(const char* name, uint64_t handle);
        
        DslReturnType SinkWindowClear(const char* name);
        
        DslReturnType SinkWindowFullScreenEnabledGet(const char* name, 
            boolean* enabled);
        
        DslReturnType SinkWindowFullScreenEnabledSet(const char* name, 
            boolean enabled);
        
        DslReturnType SinkWindowKeyEventHandlerAdd(const char* name, 
            dsl_sink_window_key_event_handler_cb handler, void* clientData);

        DslReturnType SinkWindowKeyEventHandlerRemove(const char* name, 
            dsl_sink_window_key_event_handler_cb handler);

        DslReturnType SinkWindowButtonEventHandlerAdd(const char* name, 
            dsl_sink_window_button_event_handler_cb handler, void* clientData);

        DslReturnType SinkWindowButtonEventHandlerRemove(const char* name, 
            dsl_sink_window_button_event_handler_cb handler);
        
        DslReturnType SinkWindowDeleteEventHandlerAdd(const char* name, 
            dsl_sink_window_delete_event_handler_cb handler, void* clientData);

        DslReturnType SinkWindowDeleteEventHandlerRemove(const char* name, 
            dsl_sink_window_delete_event_handler_cb handler);
        
        DslReturnType SinkWindowEglForceAspectRatioGet(const char* name, 
            boolean* force);

        DslReturnType SinkWindowEglForceAspectRatioSet(const char* name, 
            boolean force);
            
        DslReturnType SinkFileNew(const char* name, const char* filepath, 
            uint codec, uint container, uint bit_rate, uint interval);
            
        DslReturnType SinkRecordNew(const char* name, const char* outdir, 
            uint codec, uint container, uint bitrate, uint interval, 
            dsl_record_client_listener_cb clientListener);
            
        DslReturnType SinkRecordSessionStart(const char* name, 
            uint start, uint duration, void* clientData);

        DslReturnType SinkRecordSessionStop(const char* name, boolean sync);

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

        DslReturnType SinkEncodeDimensionsGet(const char* name, 
            uint* width, uint* height);

        DslReturnType SinkEncodeDimensionsSet(const char* name, 
            uint width, uint height);

        DslReturnType SinkEncodeSettingsGet(const char* name, 
            uint* codec, uint* bitrate, uint* interval);

        DslReturnType SinkEncodeSettingsSet(const char* name, 
            uint codec, uint bitrate, uint interval);

        DslReturnType SinkRtmpNew(const char* name, const char* uri, 
            uint bitrate, uint interval);

        DslReturnType SinkRtmpUriGet(const char* name, const char** uri);

        DslReturnType SinkRtmpUriSet(const char* name, const char* uri);
            
        DslReturnType SinkRtspServerNew(const char* name, const char* host, 
            uint updPort, uint rtspPort, uint codec, uint bitrate, uint interval);
            
        DslReturnType SinkRtspServerSettingsGet(const char* name, 
            uint* updPort, uint* rtspPort);
            
        DslReturnType SinkRtspClientNew(const char* name, const char* uri, 
            uint codec, uint bit_rate, uint interval);

        DslReturnType SinkRtspClientCredentialsSet(const char* name, 
            const char* userId, const char* userPw);

        DslReturnType SinkRtspClientLatencyGet(const char* name, 
            uint* latency);

        DslReturnType SinkRtspClientLatencySet(const char* name, 
            uint latency);
        
        DslReturnType SinkRtspClientProfilesGet(const char* name, 
            uint* profiles);

        DslReturnType SinkRtspClientProfilesSet(const char* name, 
            uint profiles);
            
        DslReturnType SinkRtspClientProtocolsGet(const char* name, 
            uint* protocols);

        DslReturnType SinkRtspClientProtocolsSet(const char* name, 
            uint protocols);
            
        DslReturnType SinkRtspClientTlsValidationFlagsGet(const char* name, 
            uint* flags);

        DslReturnType SinkRtspClientTlsValidationFlagsSet(const char* name, 
            uint flags);
            
        DslReturnType SinkInterpipeNew(const char* name,
            boolean forward_eos, boolean forward_events);

        DslReturnType SinkInterpipeForwardSettingsGet(const char* name,
            boolean* forward_eos, boolean* forward_events);

        DslReturnType SinkInterpipeForwardSettingsSet(const char* name,
            boolean forward_eos, boolean forward_events);

        DslReturnType SinkInterpipeNumListenersGet(const char* name,
            uint* numListeners);
            
        DslReturnType SinkImageMultiNew(const char* name, const char* filepath,
            uint width, uint height, uint fps_n, uint fps_d);

        DslReturnType SinkImageMultiFilePathGet(const char* name, 
            const char** filePath);

        DslReturnType SinkImageMultiFilePathSet(const char* name, 
            const char* filePath);

        DslReturnType SinkImageMultiDimensionsGet(const char* name, 
            uint* width, uint* height);

        DslReturnType SinkImageMultiDimensionsSet(const char* name, 
            uint width, uint height);
        
        DslReturnType SinkImageMultiFrameRateGet(const char* name, 
            uint* fpsN, uint* fpsD);

        DslReturnType SinkImageMultiFrameRateSet(const char* name, 
            uint fpsN, uint fpsD);
        
        DslReturnType SinkImageMultiFileMaxGet(const char* name, 
            uint* max);

        DslReturnType SinkImageMultiFileMaxSet(const char* name, 
            uint max);
        
        DslReturnType SinkFrameCaptureNew(const char* name,
            const char* frameCaptureAction);
            
        DslReturnType SinkFrameCaptureInitiate(const char* name);
            
        DslReturnType SinkFrameCaptureSchedule(const char* name,
            uint64_t frameNumber);
            
        DslReturnType SinkWebRtcNew(const char* name, const char* stunServer, 
            const char* turnServer, uint codec, uint bitrate, uint interval);

        DslReturnType SinkWebRtcConnectionClose(const char* name);

        DslReturnType SinkWebRtcServersGet(const char* name, const char** stunServer, 
            const char** turnServer);

        DslReturnType SinkWebRtcServersSet(const char* name, const char* stunServer, 
            const char* turnServer);

        DslReturnType SinkWebRtcClientListenerAdd(const char* name,
            dsl_sink_webrtc_client_listener_cb listener, void* clientData);

        DslReturnType SinkWebRtcClientListenerRemove(const char* name,
            dsl_sink_webrtc_client_listener_cb listener);

        DslReturnType SinkV4l2New(const char* name, const char* deviceLocation);

        DslReturnType SinkV4l2DeviceLocationGet(const char* name, 
            const char** deviceLocation);
        
        DslReturnType SinkV4l2DeviceLocationSet(const char* name, 
            const char* deviceLocation);

        DslReturnType SinkV4l2DeviceNameGet(const char* name, 
            const char** deviceName);

        DslReturnType SinkV4l2DeviceFdGet(const char* name, 
            int* deviceFd);

        DslReturnType SinkV4l2DeviceFlagsGet(const char* name, 
            uint* deviceFlags);

        DslReturnType SinkV4l2BufferInFormatGet(const char* name, 
            const char** format);

        DslReturnType SinkV4l2BufferInFormatSet(const char* name, 
            const char* format);

        DslReturnType SinkV4l2PictureSettingsGet(const char* name, 
            int* brightness, int* contrast, int* saturation);

        DslReturnType SinkV4l2PictureSettingsSet(const char* name, 
            int brightness, int contrast, int saturation);

        DslReturnType SinkSyncEnabledGet(const char* name, boolean* enabled);

        DslReturnType SinkSyncEnabledSet(const char* name, boolean enabled);

        DslReturnType SinkAsyncEnabledGet(const char* name, boolean* enabled);

        DslReturnType SinkAsyncEnabledSet(const char* name, boolean enabled);

        DslReturnType SinkMaxLatenessGet(const char* name, int64_t* maxLateness);

        DslReturnType SinkMaxLatenessSet(const char* name, int64_t maxLateness);

        DslReturnType SinkQosEnabledGet(const char* name, boolean* enabled);

        DslReturnType SinkQosEnabledSet(const char* name, boolean enabled);

        DslReturnType SinkPphAdd(const char* name, const char* handler);

        DslReturnType SinkPphRemove(const char* name, const char* handler);

        DslReturnType WebsocketServerPathAdd(const char* path);
        
        DslReturnType WebsocketServerListeningStart(uint portNumber);

        DslReturnType WebsocketServerListeningStop();

        DslReturnType WebsocketServerListeningStateGet(boolean* isListening, uint* portNumber);

        DslReturnType WebsocketServerClientListenerAdd(
            dsl_websocket_server_client_listener_cb listener, void* clientData);

        DslReturnType WebsocketServerClientListenerRemove(
            dsl_websocket_server_client_listener_cb listener);

        DslReturnType SinkMessageNew(const char* name, 
            const char* converterConfigFile, uint payloadType, 
            const char* brokerConfigFile, const char* protocolLib, 
            const char* connectionString, const char* topic);
            
        DslReturnType SinkMessageMetaTypeGet(const char* name,
            uint* metaType);
            
        DslReturnType SinkMessageMetaTypeSet(const char* name,
            uint metaType);
            
        DslReturnType SinkMessageConverterSettingsGet(const char* name, 
            const char** converterConfigFile, uint* payloadType);
            
        DslReturnType SinkMessageConverterSettingsSet(const char* name, 
            const char* converterConfigFile, uint payloadType);
            
        DslReturnType SinkMessageBrokerSettingsGet(const char* name, 
            const char** brokerConfigFile, const char** protocolLib,
            const char** connectionString, const char** topic);

        DslReturnType SinkMessageBrokerSettingsSet(const char* name, 
            const char* brokerConfigFile, const char* protocolLib,
            const char* connectionString, const char* topic);
        
        DslReturnType GetSinkMessagePayloadDebugDirGet(const char* name, 
            const char** debugDir);

        DslReturnType GetSinkMessagePayloadDebugDirSet(const char* name, 
            const char* debugDir);

        DslReturnType SinkWebRtcLiveKitNew(const char* name, 
            const char* url, const char*  apiKey, const char* secretKey, 
            const char* room, const char* identity, const char* participant);
            
        // TODO        
        // boolean ComponentIsInUse(const char* name);
        
        DslReturnType ComponentDelete(const char* name);

        DslReturnType ComponentDeleteAll();
        
        uint ComponentListSize();

        DslReturnType ComponentGpuIdGet(const char* name, uint* gpuid);
        
        DslReturnType ComponentGpuIdSet(const char* name, uint gpuid);
        
        DslReturnType ComponentNvbufMemTypeGet(const char* name, uint* type);
        
        DslReturnType ComponentNvbufMemTypeSet(const char* name, uint type);
        
        DslReturnType BranchNew(const char* name);
        
        DslReturnType BranchComponentAdd(const char* branch, const char* component);

        DslReturnType BranchComponentRemove(const char* branch, const char* component);

        DslReturnType PipelineNew(const char* name);
        
        DslReturnType PipelineDelete(const char* name);
        
        DslReturnType PipelineDeleteAll();

        uint PipelineListSize();
        
        DslReturnType PipelineComponentAdd(const char* name, const char* component);

        DslReturnType PipelineComponentRemove(const char* name, const char* component);

        //----------------------------------------------------------------------------
        // NEW STREAMMUX SERVICES - Start
        //----------------------------------------------------------------------------

        DslReturnType PipelineStreammuxConfigFileGet(const char* name, 
            const char** configFile);
            
        DslReturnType PipelineStreammuxConfigFileSet(const char* name, 
            const char* configFile);
            
        DslReturnType PipelineStreammuxBatchSizeGet(const char* name,
            uint* batchSize);

        DslReturnType PipelineStreammuxBatchSizeSet(const char* name,
            uint batchSize);

        //----------------------------------------------------------------------------
        // NEW STREAMMUX SERVICES - End
        //----------------------------------------------------------------------------

        DslReturnType PipelineStreammuxNumSurfacesPerFrameGet(const char* name, 
            uint* num);

        DslReturnType PipelineStreammuxNumSurfacesPerFrameSet(const char* name, 
            uint num);
        
        DslReturnType PipelineStreammuxAttachSysTsEnabledGet(const char* name, 
            boolean* enabled);

        DslReturnType PipelineStreammuxAttachSysTsEnabledSet(const char* name, 
            boolean enabled);

        DslReturnType PipelineStreammuxSyncInputsEnabledGet(const char* name, 
            boolean* enabled);

        DslReturnType PipelineStreammuxSyncInputsEnabledSet(const char* name, 
            boolean enabled);

        DslReturnType PipelineStreammuxMaxLatencyGet(const char* name, 
            uint* maxLatency);
        
        DslReturnType PipelineStreammuxMaxLatencySet(const char* name, 
            uint maxLatency);

        //----------------------------------------------------------------------------
        // OLD STREAMMUX SERVICES - Start
        //----------------------------------------------------------------------------

        DslReturnType PipelineStreammuxBatchPropertiesGet(const char* name,
            uint* batchSize, int* batchTimeout);

        DslReturnType PipelineStreammuxBatchPropertiesSet(const char* name,
            uint batchSize, int batchTimeout);

        DslReturnType PipelineStreammuxNvbufMemTypeGet(const char* name, 
            uint* type);

        DslReturnType PipelineStreammuxNvbufMemTypeSet(const char* name, 
            uint type);

        DslReturnType PipelineStreammuxGpuIdGet(const char* name, uint* gpuid);
        
        DslReturnType PipelineStreammuxGpuIdSet(const char* name, uint gpuid);

        DslReturnType PipelineStreammuxDimensionsGet(const char* name,
            uint* width, uint* height);

        DslReturnType PipelineStreammuxDimensionsSet(const char* name,
            uint width, uint height);
            
        DslReturnType PipelineStreammuxPaddingGet(const char* name, boolean* enabled);

        DslReturnType PipelineStreammuxPaddingSet(const char* name, boolean enabled);
        
        //----------------------------------------------------------------------------
        // OLD STREAMMUX SERVICES - End
        //----------------------------------------------------------------------------

        DslReturnType PipelineStreammuxTilerAdd(const char* name, const char* tiler);

        DslReturnType PipelineStreammuxTilerRemove(const char* name);

        DslReturnType PipelineStreammuxPphAdd(const char* name, 
            const char* handler);

        DslReturnType PipelineStreammuxPphRemove(const char* name, 
            const char* handler);
        
        DslReturnType PipelineLinkMethodGet(const char* name, uint* linkMethod);
        
        DslReturnType PipelineLinkMethodSet(const char* name, uint linkMethod);
        
        DslReturnType PipelinePause(const char* name);
        
        DslReturnType PipelinePlay(const char* name);
        
        DslReturnType PipelineStop(const char* name);
        
        DslReturnType PipelineStateGet(const char* name, uint* state);
        
        DslReturnType PipelineIsLive(const char* name, boolean* isLive);
        
        DslReturnType PipelineDumpToDot(const char* name, const char* filename);
        
        DslReturnType PipelineDumpToDotWithTs(const char* name, const char* filename);
        
        DslReturnType PipelineStateChangeListenerAdd(const char* name, 
            dsl_state_change_listener_cb listener, void* clientData);
        
        DslReturnType PipelineStateChangeListenerRemove(const char* name, 
            dsl_state_change_listener_cb listener);
                        
        DslReturnType PipelineEosListenerAdd(const char* name, 
            dsl_eos_listener_cb listener, void* clientData);
        
        DslReturnType PipelineEosListenerRemove(const char* name, 
            dsl_eos_listener_cb listener);

        DslReturnType PipelineErrorMessageHandlerAdd(const char* name, 
            dsl_error_message_handler_cb handler, void* clientData);

        DslReturnType PipelineErrorMessageHandlerRemove(const char* name, 
            dsl_error_message_handler_cb handler);
            
        DslReturnType PipelineErrorMessageLastGet(const char* name,
            std::wstring& source, std::wstring& message);
                        
        DslReturnType PipelineMainLoopNew(const char* name);

        DslReturnType PipelineMainLoopRun(const char* name);

        DslReturnType PipelineMainLoopQuit(const char* name);

        DslReturnType PipelineMainLoopDelete(const char* name);

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

        DslReturnType MessageBrokerNew(const char* name,
            const char* brokerConfigFile, const char* protocolLib, 
            const char* connectionString);
            
        DslReturnType MessageBrokerSettingsGet(const char* name, 
            const char** brokerConfigFile, const char** protocolLib, 
            const char** connectionString);
        
        DslReturnType MessageBrokerSettingsSet(const char* name, 
            const char* brokerConfigFile, const char* protocolLib,
            const char* connectionString);

        DslReturnType MessageBrokerConnect(const char* name);
        
        DslReturnType MessageBrokerDisconnect(const char* name);

        DslReturnType MessageBrokerIsConnected(const char* name,
            boolean* connected);

        DslReturnType MessageBrokerMessageSendAsync(const char* name,
            const char* topic, void* message, size_t size, 
            dsl_message_broker_send_result_listener_cb result_listener, void* clientData);
        
        DslReturnType MessageBrokerSubscriberAdd(const char* name,
            dsl_message_broker_subscriber_cb subscriber, const char** topics,
            uint numTopics, void* userData);
        
        DslReturnType MessageBrokerSubscriberRemove(const char* name,
            dsl_message_broker_subscriber_cb subscriber);
        
        DslReturnType MessageBrokerConnectionListenerAdd(const char* name,
            dsl_message_broker_connection_listener_cb handler, void* userData);
        
        DslReturnType MessageBrokerConnectionListenerRemove(const char* name,
            dsl_message_broker_connection_listener_cb handler);
        
        DslReturnType MessageBrokerDelete(const char* name);
        
        DslReturnType MessageBrokerDeleteAll();

        uint MessageBrokerListSize();
        
        void DeleteAll();
        
        DslReturnType InfoInitDebugSettings();
        
        DslReturnType InfoDeinitDebugSettings();
        
        DslReturnType InfoStdoutGet(const char** filePath);

        DslReturnType InfoStdoutRedirect(const char* filePath, uint mode);

        DslReturnType InfoStdoutRedirectWithTs(const char* filePath);

        DslReturnType InfoStdOutRestore();
        
        DslReturnType InfoLogLevelGet(const char** level);
        
        DslReturnType InfoLogLevelSet(const char* level);
        
        DslReturnType InfoLogFileGet(const char** filePath);
        
        DslReturnType InfoLogFileSet(const char* filePath, uint mode);
        
        DslReturnType InfoLogFileSetWithTs(const char* filePath);
        
        DslReturnType InfoLogFunctionRestore();
        
        FILE* InfoLogFileHandleGet();

        DslReturnType SetSpdLogger(spdlog::logger* logger);

        spdlog::logger* GetSpdLogger();

        GMainLoop* GetMainLoopHandle()
        {
            LOG_FUNC();
            
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

        /**
         * @brief GStreamer Debug environment variable name
         */
        static std::string GST_DEBUG;

        /**
         * @brief GStreamer Debug-File environment variable name
         */
        static std::string GST_DEBUG_FILE;
        
        /**
         *@brief Default stdout file_path value
         */
        static std::string CONSOLE;

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
         * @brief mutex to prevent Services re-entry
         */
        DslMutex m_servicesMutex;
        
        /**
         * @brief boolean flag to indicate if USE_NEW_NVSTREAMMUX=yes
         */
        bool m_useNewStreammux;
        
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
         * @brief map of all ODE Accumlators created by the client, key=name
         */
        std::map <std::string, DSL_ODE_ACCUMULATOR_PTR> m_odeAccumulators;
        
        /**
         * @brief map of all ODE Heat-Mappers created by the client, key=name
         */
        std::map <std::string, DSL_ODE_HEAT_MAPPER_PTR> m_odeHeatMappers;
        
        /**
         * @brief map of all ODE Triggers created by the client, key=name
         */
        std::map <std::string, DSL_ODE_TRIGGER_PTR> m_odeTriggers;
        
        /**
         * @brief map of all ODE Handlers created by the client, key=name
         */
        std::map <std::string, DSL_PPH_PTR> m_padProbeHandlers;
        
        /**
         * @brief map of all GST Elements created by the client, key=name
         */
        std::map <std::string, DSL_ELEMENT_PTR> m_gstElements;

        /**
         * @brief map of all pipelines creaated by the client, key=name
         */
        std::map <std::string, DSL_PIPELINE_PTR> m_pipelines;
        
        /**
         * @brief map of all players creaated by the client, key=name
         */
        std::map <std::string, std::shared_ptr<PlayerBintr>> m_players;
        
        /**
         * @brief map of all pipeline components creaated by the client, key=name
         */
        std::map <std::string, std::shared_ptr<Bintr>> m_components;

        /**
         * @brief map of all message borkers creaated by the client, key=name
         */
        std::map <std::string, std::shared_ptr<MessageBroker>> m_messageBrokers;
        
        /**
         * @brief container of all unique source Ids mapped by their unique name.
         */
        std::map <std::string, uint> m_sourceIdsByName;
        
        /**
         * @brief container of all unique source names mapped by their unique Id.
         */
        std::map <uint, std::string> m_sourceNamesById;
        
        /**
         * @brief map of all infer ids to infer names
         */
        std::map <uint, std::string> m_inferNames;

        /**
         * @brief map of all infer names to infer ids
         */
        std::map <std::string, uint> m_inferIds;
        
        /**
         * @brief map of all infer names to process-mode
         */
        std::map <std::string, uint> m_inferProcessModes;
        
        /**
         * @brief map of all Window-Sinks to their 3d/egl plugin object pointer.
         */
        std::map <DSL_BASE_PTR, GstObject*> m_windowSinkElements;

        /**
         * @brief mutex to prevent Window registry re-entry
         */
        DslMutex m_windowRegistryMutex;
        
        /**
         * @brief map of all mailer objects by name
         */
        std::map <std::string, std::shared_ptr<Mailer>> m_mailers;
        
        /**
         * @brief file-path of the redirected stdout if set.
         */
        std::string m_stdOutRedirectFilePath; 
        
        /**
         * @brief file-stream object for the redirected stdout.
         */
        std::fstream m_stdOutRedirectFile;
        
        /**
         * @brief back-up for the original stdout prior to redirection.
         */
        std::streambuf* m_stdOutRdBufBackup;
        
        /**
         * @brief Debug Log Level (threshold) to override the value of GST_DEBUG.
         */
        std::string m_gstDebugLogLevel;
        
        /**
         * @brief Debug Log File to override the value of GST_DEBUG_FILE.
         */
        std::string m_debugLogFilePath;
        
        /**
         * @brief File handle for the Debug Log File if open.
         */
        FILE* m_debugLogFileHandle;

        /**
        * @brief Shared pointer to the spdlog logger instance for logging.
        */
        spdlog::logger* m_spdLogger;

    };  

    /**
     * @brief Intrinsic Display Types created on DSL instantiation.
     */
    static const std::string DISPLAY_TYPE_NO_COLOR("no-color");
    static const std::string DISPLAY_TYPE_NO_FONT("no-font");
    

    static gboolean MainLoopThread(gpointer arg);
    
    static void gst_debug_log_override(GstDebugCategory * category, GstDebugLevel level,
        const gchar * file, const gchar * function, gint line,
        GObject * object, GstDebugMessage * message, gpointer unused);
}


#endif // _DSL_DRIVER_H