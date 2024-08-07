/*
The MIT License

Copyright (c) 2021-2024, Prominence AI, Inc.

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

#ifndef _DSL_SERVICES_VALIDATE_H
#define _DSL_SERVICES_VALIDATE_H

#include "DslApi.h"

#define DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(actions, name) do \
{ \
    if (actions.find(name) == actions.end()) \
    { \
        LOG_ERROR("ODE Action name '" << name << "' was not found"); \
        return DSL_RESULT_ODE_ACTION_NAME_NOT_FOUND; \
    } \
}while(0); 

#define DSL_RETURN_IF_ODE_AREA_NAME_NOT_FOUND(areas, name) do \
{ \
    if (areas.find(name) == areas.end()) \
    { \
        LOG_ERROR("ODE Area name '" << name << "' was not found"); \
        return DSL_RESULT_ODE_AREA_NAME_NOT_FOUND; \
    } \
}while(0); 

#define DSL_RETURN_IF_ODE_ACTION_IS_NOT_CORRECT_TYPE(actions, name, action) do \
{ \
    if (!actions[name]->IsType(typeid(action)))\
    { \
        LOG_ERROR("ODE Action '" << name << "' is not the correct type"); \
        return DSL_RESULT_ODE_ACTION_NOT_THE_CORRECT_TYPE; \
    } \
}while(0); 

#define DSL_RETURN_IF_ODE_ACTION_IS_NOT_CAPTURE_TYPE(actions, name) do \
{ \
    if (!actions[name]->IsType(typeid(CaptureFrameOdeAction)) and \
        !actions[name]->IsType(typeid(CaptureObjectOdeAction)))\
    { \
        LOG_ERROR("ODE Action '" << name << "' is not the correct type"); \
        return DSL_RESULT_ODE_ACTION_NOT_THE_CORRECT_TYPE; \
    } \
}while(0); 

#define DSL_RETURN_IF_ODE_ACCUMULATOR_NAME_NOT_FOUND(events, name) do \
{ \
    if (events.find(name) == events.end()) \
    { \
        LOG_ERROR("ODE Accumulator name '" << name << "' was not found"); \
        return DSL_RESULT_ODE_ACCUMULATOR_NAME_NOT_FOUND; \
    } \
}while(0); 

#define DSL_RETURN_IF_ODE_HEAT_MAPPER_NAME_NOT_FOUND(events, name) do \
{ \
    if (events.find(name) == events.end()) \
    { \
        LOG_ERROR("ODE Heat Mapper name '" << name << "' was not found"); \
        return DSL_RESULT_ODE_HEAT_MAPPER_NAME_NOT_FOUND; \
    } \
}while(0); 

#define DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(events, name) do \
{ \
    if (events.find(name) == events.end()) \
    { \
        LOG_ERROR("ODE Trigger name '" << name << "' was not found"); \
        return DSL_RESULT_ODE_TRIGGER_NAME_NOT_FOUND; \
    } \
}while(0); 

#define DSL_RETURN_IF_ODE_TRIGGER_IS_NOT_AB_TYPE(components, name) do \
{ \
    if (!components[name]->IsType(typeid(DistanceOdeTrigger)) and  \
        !components[name]->IsType(typeid(IntersectionOdeTrigger))) \
    { \
        LOG_ERROR("Component '" << name << "' is not an AB ODE Trigger"); \
        return DSL_RESULT_ODE_TRIGGER_IS_NOT_AB_TYPE; \
    } \
}while(0); 

#define DSL_RETURN_IF_BRANCH_NAME_NOT_FOUND(branches, name) do \
{ \
    if (branches.find(name) == branches.end()) \
    { \
        LOG_ERROR("Branch name '" << name << "' was not found"); \
        return DSL_RESULT_BRANCH_NAME_NOT_FOUND; \
    } \
}while(0); 
    
#define DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(pipelines, name) do \
{ \
    if (pipelines.find(name) == pipelines.end()) \
    { \
        LOG_ERROR("Pipeline name '" << name << "' was not found"); \
        return DSL_RESULT_PIPELINE_NAME_NOT_FOUND; \
    } \
}while(0); 
    
#define DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(players, name) do \
{ \
    if (players.find(name) == players.end()) \
    { \
        LOG_ERROR("Player name '" << name << "' was not found"); \
        return DSL_RESULT_PLAYER_NAME_NOT_FOUND; \
    } \
}while(0); 

#define DSL_RETURN_IF_PLAYER_IS_NOT_IMAGE_PLAYER(players, name) do \
{ \
    if (!players[name]->IsType(typeid(ImageRenderPlayerBintr))) \
    { \
        LOG_ERROR("Player '" << name << "' is not an Image Player"); \
        return DSL_RESULT_PLAYER_IS_NOT_IMAGE_PLAYER; \
    } \
}while(0); 

#define DSL_RETURN_IF_PLAYER_IS_NOT_VIDEO_PLAYER(players, name) do \
{ \
    if (!players[name]->IsType(typeid(VideoRenderPlayerBintr))) \
    { \
        LOG_ERROR("Player '" << name << "' is not an Video Player"); \
        return DSL_RESULT_PLAYER_IS_NOT_VIDEO_PLAYER; \
    } \
}while(0); 


#define DSL_RETURN_IF_PLAYER_IS_NOT_RENDER_PLAYER(players, name) do \
{ \
    if (!players[name]->IsType(typeid(ImageRenderPlayerBintr)) and  \
        !players[name]->IsType(typeid(VideoRenderPlayerBintr))) \
    { \
        LOG_ERROR("Player '" << name << "' is not a Render Player"); \
        return DSL_RESULT_PLAYER_IS_NOT_RENDER_PLAYER; \
    } \
}while(0); 
        
#define DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(components, name) do \
{ \
    if (components.find(name) == components.end()) \
    { \
        LOG_ERROR("Component name '" << name << "' was not found"); \
        return DSL_RESULT_COMPONENT_NAME_NOT_FOUND; \
    } \
}while(0); 

#define DSL_RETURN_IF_CAPS_NAME_NOT_FOUND(caps, name) do \
{ \
    if (caps.find(name) == caps.end()) \
    { \
        LOG_ERROR("Caps name '" << name << "' was not found"); \
        return DSL_RESULT_GST_CAPS_NAME_NOT_FOUND; \
    } \
}while(0); 

#define DSL_RETURN_IF_ELEMENT_NAME_NOT_FOUND(elements, name) do \
{ \
    if (elements.find(name) == elements.end()) \
    { \
        LOG_ERROR("Element name '" << name << "' was not found"); \
        return DSL_RESULT_GST_ELEMENT_NAME_NOT_FOUND; \
    } \
}while(0); 

#define DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(components, name, bintr) do \
{ \
    if (!components[name]->IsType(typeid(bintr)))\
    { \
        LOG_ERROR("Component '" << name << "' is not the correct type"); \
        return DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE; \
    } \
}while(0); 

#define DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(components, name) do \
{ \
    if (!components[name]->IsType(typeid(AppSourceBintr)) and  \
        !components[name]->IsType(typeid(CsiSourceBintr)) and  \
        !components[name]->IsType(typeid(V4l2SourceBintr)) and  \
        !components[name]->IsType(typeid(UriSourceBintr)) and  \
        !components[name]->IsType(typeid(FileSourceBintr)) and  \
        !components[name]->IsType(typeid(ImageSourceBintr)) and  \
        !components[name]->IsType(typeid(SingleImageSourceBintr)) and  \
        !components[name]->IsType(typeid(MultiImageSourceBintr)) and  \
        !components[name]->IsType(typeid(ImageStreamSourceBintr)) and  \
        !components[name]->IsType(typeid(InterpipeSourceBintr)) and  \
        !components[name]->IsType(typeid(RtspSourceBintr)) and \
        !components[name]->IsType(typeid(DuplicateSourceBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a Source"); \
        return DSL_RESULT_SOURCE_COMPONENT_IS_NOT_SOURCE; \
    } \
}while(0); 

#define DSL_RETURN_IF_COMPONENT_IS_NOT_IMAGE_SOURCE(components, name) do \
{ \
    if (!components[name]->IsType(typeid(SingleImageSourceBintr)) and  \
        !components[name]->IsType(typeid(MultiImageSourceBintr)) and  \
        !components[name]->IsType(typeid(ImageStreamSourceBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not an Image Source"); \
        return DSL_RESULT_SOURCE_COMPONENT_IS_NOT_FILE_SOURCE; \
    } \
}while(0); 


#if !defined(BUILD_WEBRTC)
    #error "BUILD_WEBRTC must be defined"
#elif BUILD_WEBRTC != true
#define DSL_RETURN_IF_COMPONENT_IS_NOT_ENCODE_SINK(components, name) do \
{ \
    if (!components[name]->IsType(typeid(FileSinkBintr)) and  \
        !components[name]->IsType(typeid(RecordSinkBintr)) and \
        !components[name]->IsType(typeid(RtmpSinkBintr)) and \
        !components[name]->IsType(typeid(RtspServerSinkBintr)) and \
        !components[name]->IsType(typeid(RtspClientSinkBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a Encode Sink"); \
        return DSL_RESULT_SINK_COMPONENT_IS_NOT_ENCODE_SINK; \
    } \
}while(0); 
#else
#define DSL_RETURN_IF_COMPONENT_IS_NOT_ENCODE_SINK(components, name) do \
{ \
    if (!components[name]->IsType(typeid(FileSinkBintr)) and  \
        !components[name]->IsType(typeid(RecordSinkBintr)) and \
        !components[name]->IsType(typeid(RtmpSinkBintr)) and \
        !components[name]->IsType(typeid(RtspServerSinkBintr)) and \
        !components[name]->IsType(typeid(RtspClientSinkBintr)) and \
        !components[name]->IsType(typeid(WebRtcSinkBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a Encode Sink"); \
        return DSL_RESULT_SINK_COMPONENT_IS_NOT_ENCODE_SINK; \
    } \
}while(0);
#endif

#define DSL_RETURN_IF_PREPROC_NAME_NOT_FOUND(events, name) do \
{ \
    if (events.find(name) == events.end()) \
    { \
        LOG_ERROR("Preprocessor name '" << name << "' was not found"); \
        return DSL_RESULT_PREPROC_NAME_NOT_FOUND; \
    } \
}while(0); 

#define DSL_RETURN_IF_COMPONENT_IS_NOT_GIE(components, name) do \
{ \
    if (!components[name]->IsType(typeid(PrimaryGieBintr)) and  \
        !components[name]->IsType(typeid(SecondaryGieBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a Primary or Secondary GIE"); \
        return DSL_RESULT_INFER_COMPONENT_IS_NOT_INFER; \
    } \
}while(0); 

#define DSL_RETURN_IF_COMPONENT_IS_NOT_INFER(components, name) do \
{ \
    if (!components[name]->IsType(typeid(PrimaryGieBintr)) and  \
        !components[name]->IsType(typeid(SecondaryGieBintr)) and \
        !components[name]->IsType(typeid(PrimaryTisBintr)) and \
        !components[name]->IsType(typeid(SecondaryTisBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a GIE or TIS"); \
        return DSL_RESULT_INFER_COMPONENT_IS_NOT_INFER; \
    } \
}while(0); 

#define DSL_RETURN_IF_COMPONENT_IS_NOT_PRIMARY_INFER_TYPE(components, name) do \
{ \
    if (!components[name]->IsType(typeid(PrimaryGieBintr)) and  \
        !components[name]->IsType(typeid(PrimaryTisBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a Primary GIE or TIS"); \
        return DSL_RESULT_INFER_COMPONENT_IS_NOT_INFER; \
    } \
}while(0); 

#define DSL_RETURN_IF_COMPONENT_IS_NOT_TEE(components, name) do \
{ \
    if (!components[name]->IsType(typeid(DemuxerBintr)) and  \
        !components[name]->IsType(typeid(SplitterBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a Tee"); \
        return DSL_RESULT_TEE_COMPONENT_IS_NOT_TEE; \
    } \
}while(0); 

#define DSL_RETURN_IF_COMPONENT_IS_NOT_WINDOW_SINK(components, name) do \
{ \
    if (!components[name]->IsType(typeid(EglSinkBintr)) and  \
        !components[name]->IsType(typeid(ThreeDSinkBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a Window Sink"); \
        return DSL_RESULT_SINK_COMPONENT_IS_NOT_WINDOW_SINK; \
    } \
}while(0); 

// All Bintr's that can be added as a "branch" to a "Tee"
#define DSL_RETURN_IF_COMPONENT_IS_NOT_BRANCH(components, name) do \
{ \
    if (!components[name]->IsType(typeid(AppSinkBintr)) and  \
        !components[name]->IsType(typeid(CustomSinkBintr)) and  \
        !components[name]->IsType(typeid(FrameCaptureSinkBintr)) and  \
        !components[name]->IsType(typeid(FakeSinkBintr)) and  \
        !components[name]->IsType(typeid(ThreeDSinkBintr)) and  \
        !components[name]->IsType(typeid(EglSinkBintr)) and  \
        !components[name]->IsType(typeid(FileSinkBintr)) and  \
        !components[name]->IsType(typeid(RecordSinkBintr)) and  \
        !components[name]->IsType(typeid(RtmpSinkBintr)) and \
        !components[name]->IsType(typeid(RtspClientSinkBintr)) and \
        !components[name]->IsType(typeid(RtspServerSinkBintr)) and \
        !components[name]->IsType(typeid(MessageSinkBintr)) and \
        !components[name]->IsType(typeid(InterpipeSinkBintr)) and \
        !components[name]->IsType(typeid(MultiImageSinkBintr)) and \
        !components[name]->IsType(typeid(V4l2SinkBintr)) and \
        !components[name]->IsType(typeid(DemuxerBintr)) and \
        !components[name]->IsType(typeid(SplitterBintr)) and \
        !components[name]->IsType(typeid(BranchBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a Branch type"); \
        return DSL_RESULT_TEE_BRANCH_IS_NOT_BRANCH; \
    } \
}while(0); 

// All Bintr's that can be added as a "branch" to a "Remuxer"
#define DSL_RETURN_IF_COMPONENT_IS_NOT_REMUXER_BRANCH(components, name) do \
{ \
    if (!components[name]->IsType(typeid(PrimaryGieBintr)) and  \
        !components[name]->IsType(typeid(PrimaryTisBintr)) and \
        !components[name]->IsType(typeid(BranchBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a Branch type"); \
        return DSL_RESULT_TEE_BRANCH_IS_NOT_BRANCH; \
    } \
}while(0); 

#if !defined(BUILD_WEBRTC)
    #error "BUILD_WEBRTC must be defined"
#elif BUILD_WEBRTC != true
#define DSL_RETURN_IF_COMPONENT_IS_NOT_SINK(components, name) do \
{ \
    if (!components[name]->IsType(typeid(AppSinkBintr)) and  \
        !components[name]->IsType(typeid(CustomSinkBintr)) and  \
        !components[name]->IsType(typeid(FrameCaptureSinkBintr)) and  \
        !components[name]->IsType(typeid(FakeSinkBintr)) and  \
        !components[name]->IsType(typeid(ThreeDSinkBintr)) and  \
        !components[name]->IsType(typeid(EglSinkBintr)) and  \
        !components[name]->IsType(typeid(FileSinkBintr)) and  \
        !components[name]->IsType(typeid(RecordSinkBintr)) and  \
        !components[name]->IsType(typeid(RtmpSinkBintr)) and \
        !components[name]->IsType(typeid(RtspClientSinkBintr)) and \
        !components[name]->IsType(typeid(RtspServerSinkBintr)) and \
        !components[name]->IsType(typeid(MessageSinkBintr)) and \
        !components[name]->IsType(typeid(V4l2SinkBintr)) and \
        !components[name]->IsType(typeid(InterpipeSinkBintr)) and \
        !components[name]->IsType(typeid(MultiImageSinkBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a Sink"); \
        return DSL_RESULT_SINK_COMPONENT_IS_NOT_SINK; \
    } \
}while(0);
#else
#define DSL_RETURN_IF_COMPONENT_IS_NOT_SINK(components, name) do \
{ \
    if (!components[name]->IsType(typeid(AppSinkBintr)) and  \
        !components[name]->IsType(typeid(CustomSinkBintr)) and  \
        !components[name]->IsType(typeid(FrameCaptureSinkBintr)) and  \
        !components[name]->IsType(typeid(FakeSinkBintr)) and  \
        !components[name]->IsType(typeid(ThreeDSinkBintr)) and  \
        !components[name]->IsType(typeid(EglSinkBintr)) and  \
        !components[name]->IsType(typeid(FileSinkBintr)) and  \
        !components[name]->IsType(typeid(RecordSinkBintr)) and  \
        !components[name]->IsType(typeid(RtmpSinkBintr)) and \
        !components[name]->IsType(typeid(RtspClientSinkBintr)) and \
        !components[name]->IsType(typeid(RtspServerSinkBintr)) and \
        !components[name]->IsType(typeid(MessageSinkBintr)) and \
        !components[name]->IsType(typeid(V4l2SinkBintr)) and \
        !components[name]->IsType(typeid(InterpipeSinkBintr)) and \
        !components[name]->IsType(typeid(MultiImageSinkBintr)) and \
        !components[name]->IsType(typeid(WebRtcSinkBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a Sink"); \
        return DSL_RESULT_SINK_COMPONENT_IS_NOT_SINK; \
    } \
}while(0);
#endif

#if !defined(BUILD_WEBRTC)
    #error "BUILD_WEBRTC must be defined"
#elif BUILD_WEBRTC != true
#define DSL_RETURN_IF_COMPONENT_IS_NOT_QBINTR(components, name) do \
{ \
    if (!components[name]->IsType(typeid(AppSourceBintr)) and  \
        !components[name]->IsType(typeid(CustomSourceBintr)) and  \
        !components[name]->IsType(typeid(CsiSourceBintr)) and  \
        !components[name]->IsType(typeid(V4l2SourceBintr)) and  \
        !components[name]->IsType(typeid(UriSourceBintr)) and  \
        !components[name]->IsType(typeid(FileSourceBintr)) and  \
        !components[name]->IsType(typeid(ImageSourceBintr)) and  \
        !components[name]->IsType(typeid(SingleImageSourceBintr)) and  \
        !components[name]->IsType(typeid(MultiImageSourceBintr)) and  \
        !components[name]->IsType(typeid(ImageStreamSourceBintr)) and  \
        !components[name]->IsType(typeid(InterpipeSourceBintr)) and  \
        !components[name]->IsType(typeid(RtspSourceBintr)) and \
        !components[name]->IsType(typeid(DuplicateSourceBintr)) and \
        !components[name]->IsType(typeid(RecordTapBintr)) and  \
        !components[name]->IsType(typeid(DewarperBintr)) and  \
        !components[name]->IsType(typeid(PreprocBintr)) and  \
        !components[name]->IsType(typeid(PrimaryGieBintr)) and  \
        !components[name]->IsType(typeid(PrimaryTisBintr)) and  \
        !components[name]->IsType(typeid(SecondaryGieBintr)) and  \
        !components[name]->IsType(typeid(SecondaryTisBintr)) and  \
        !components[name]->IsType(typeid(TrackerBintr)) and  \
        !components[name]->IsType(typeid(TilerBintr)) and  \
        !components[name]->IsType(typeid(OsdBintr)) and  \
        !components[name]->IsType(typeid(MultiSinksBintr)) and  \
        !components[name]->IsType(typeid(SplitterBintr)) and  \
        !components[name]->IsType(typeid(DemuxerBintr)) and  \
        !components[name]->IsType(typeid(AppSinkBintr)) and  \
        !components[name]->IsType(typeid(CustomSinkBintr)) and  \
        !components[name]->IsType(typeid(FrameCaptureSinkBintr)) and  \
        !components[name]->IsType(typeid(FakeSinkBintr)) and  \
        !components[name]->IsType(typeid(ThreeDSinkBintr)) and  \
        !components[name]->IsType(typeid(EglSinkBintr)) and  \
        !components[name]->IsType(typeid(FileSinkBintr)) and  \
        !components[name]->IsType(typeid(RecordSinkBintr)) and  \
        !components[name]->IsType(typeid(RtmpSinkBintr)) and \
        !components[name]->IsType(typeid(RtspClientSinkBintr)) and \
        !components[name]->IsType(typeid(RtspServerSinkBintr)) and \
        !components[name]->IsType(typeid(MessageSinkBintr)) and \
        !components[name]->IsType(typeid(V4l2SinkBintr)) and \
        !components[name]->IsType(typeid(InterpipeSinkBintr)) and \
        !components[name]->IsType(typeid(MultiImageSinkBintr)) and \
        !components[name]->IsType(typeid(CustomBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' does not have a queue element "); \
        return DSL_RESULT_SINK_COMPONENT_IS_NOT_SINK; \
    } \
}while(0);
#else
#define DSL_RETURN_IF_COMPONENT_IS_NOT_QBINTR(components, name) do \
{ \
    if (!components[name]->IsType(typeid(AppSourceBintr)) and  \
        !components[name]->IsType(typeid(CustomSourceBintr)) and  \
        !components[name]->IsType(typeid(CsiSourceBintr)) and  \
        !components[name]->IsType(typeid(V4l2SourceBintr)) and  \
        !components[name]->IsType(typeid(UriSourceBintr)) and  \
        !components[name]->IsType(typeid(FileSourceBintr)) and  \
        !components[name]->IsType(typeid(ImageSourceBintr)) and  \
        !components[name]->IsType(typeid(SingleImageSourceBintr)) and  \
        !components[name]->IsType(typeid(MultiImageSourceBintr)) and  \
        !components[name]->IsType(typeid(ImageStreamSourceBintr)) and  \
        !components[name]->IsType(typeid(InterpipeSourceBintr)) and  \
        !components[name]->IsType(typeid(RtspSourceBintr)) and \
        !components[name]->IsType(typeid(DuplicateSourceBintr)) and \
        !components[name]->IsType(typeid(RecordTapBintr)) and  \
        !components[name]->IsType(typeid(DewarperBintr)) and  \
        !components[name]->IsType(typeid(PreprocBintr)) and  \
        !components[name]->IsType(typeid(PrimaryGieBintr)) and  \
        !components[name]->IsType(typeid(PrimaryTisBintr)) and  \
        !components[name]->IsType(typeid(SecondaryGieBintr)) and  \
        !components[name]->IsType(typeid(SecondaryTisBintr)) and  \
        !components[name]->IsType(typeid(TrackerBintr)) and  \
        !components[name]->IsType(typeid(TilerBintr)) and  \
        !components[name]->IsType(typeid(OsdBintr)) and  \
        !components[name]->IsType(typeid(MultiSinksBintr)) and  \
        !components[name]->IsType(typeid(SplitterBintr)) and  \
        !components[name]->IsType(typeid(DemuxerBintr)) and  \
        !components[name]->IsType(typeid(AppSinkBintr)) and  \
        !components[name]->IsType(typeid(CustomSinkBintr)) and  \
        !components[name]->IsType(typeid(FrameCaptureSinkBintr)) and  \
        !components[name]->IsType(typeid(FakeSinkBintr)) and  \
        !components[name]->IsType(typeid(ThreeDSinkBintr)) and  \
        !components[name]->IsType(typeid(EglSinkBintr)) and  \
        !components[name]->IsType(typeid(FileSinkBintr)) and  \
        !components[name]->IsType(typeid(RecordSinkBintr)) and  \
        !components[name]->IsType(typeid(RtmpSinkBintr)) and \
        !components[name]->IsType(typeid(RtspClientSinkBintr)) and \
        !components[name]->IsType(typeid(RtspServerSinkBintr)) and \
        !components[name]->IsType(typeid(MessageSinkBintr)) and \
        !components[name]->IsType(typeid(V4l2SinkBintr)) and \
        !components[name]->IsType(typeid(InterpipeSinkBintr)) and \
        !components[name]->IsType(typeid(MultiImageSinkBintr)) and \
        !components[name]->IsType(typeid(WebRtcSinkBintr)) and \
        !components[name]->IsType(typeid(CustomBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' does not have a queue element "); \
        return DSL_RESULT_SINK_COMPONENT_IS_NOT_SINK; \
    } \
}while(0);
#endif

#define DSL_RETURN_IF_COMPONENT_IS_NOT_TAP(components, name) do \
{ \
    if (!components[name]->IsType(typeid(RecordTapBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a Tap"); \
        return DSL_RESULT_TAP_COMPONENT_IS_NOT_TAP; \
    } \
}while(0); 


#define DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(types, name) do \
{ \
    if (types.find(name) == types.end()) \
    { \
        LOG_ERROR("Display Type '" << name << "' was not found"); \
        return DSL_RESULT_DISPLAY_TYPE_NAME_NOT_FOUND; \
    } \
}while(0); 

#define DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(types, name, displayType) do \
{ \
    if (!types[name]->IsType(typeid(displayType))) \
    { \
        LOG_ERROR("Display Type '" << name << "' is not the correct type"); \
        return DSL_RESULT_DISPLAY_TYPE_NOT_THE_CORRECT_TYPE; \
    } \
}while(0); 

#define DSL_RETURN_IF_DISPLAY_TYPE_IS_BASE_TYPE(types, name) do \
{ \
    if (types[name]->IsType(typeid(RgbaColor)) or \
        types[name]->IsType(typeid(RgbaRandomColor))or \
        types[name]->IsType(typeid(RgbaFont))) \
    { \
        LOG_ERROR("Display Type '" << name << "' is base type and can not be displayed"); \
        return DSL_RESULT_DISPLAY_TYPE_IS_BASE_TYPE; \
    } \
}while(0); 

#define DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_COLOR(types, name) do \
{ \
    if (!types[name]->IsType(typeid(RgbaColor)) and \
        !types[name]->IsType(typeid(RgbaRandomColor)) and \
        !types[name]->IsType(typeid(RgbaPredefinedColor)) and \
        !types[name]->IsType(typeid(RgbaOnDemandColor)) and \
        !types[name]->IsType(typeid(RgbaOnDemandColor)) and \
        !types[name]->IsType(typeid(RgbaColorPalette))) \
    { \
        LOG_ERROR("Display Type '" << name << "' is not color type"); \
        return DSL_RESULT_DISPLAY_TYPE_NOT_THE_CORRECT_TYPE; \
    } \
}while(0); 

#define DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_TEXT(types, name) do \
{ \
    if (!types[name]->IsType(typeid(RgbaText)) and \
        !types[name]->IsType(typeid(SourceDimensions)) and \
        !types[name]->IsType(typeid(SourceUniqueId)) and \
        !types[name]->IsType(typeid(SourceStreamId)) and \
        !types[name]->IsType(typeid(SourceName))) \
    { \
        LOG_ERROR("Display Type '" << name << "' is not color type"); \
        return DSL_RESULT_DISPLAY_TYPE_NOT_THE_CORRECT_TYPE; \
    } \
}while(0); 

#define DSL_RETURN_IF_PPH_NAME_NOT_FOUND(handlers, name) do \
{ \
    if (handlers.find(name) == handlers.end()) \
    { \
        LOG_ERROR("Pad Probe Handler name '" << name << "' was not found"); \
        return DSL_RESULT_PPH_NAME_NOT_FOUND; \
    } \
}while(0); 

#define DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(mailers, name) do \
{ \
    if (mailers.find(name) == mailers.end()) \
    { \
        LOG_ERROR("Mailer name '" << name << "' was not found"); \
        return DSL_RESULT_MAILER_NAME_NOT_FOUND; \
    } \
}while(0); 

#define DSL_RETURN_IF_BROKER_NAME_NOT_FOUND(brokers, name) do \
{ \
    if (brokers.find(name) == brokers.end()) \
    { \
        LOG_ERROR("Message Broker name '" << name << "' was not found"); \
        return DSL_RESULT_BROKER_NAME_NOT_FOUND; \
    } \
}while(0); 

#endif // _DSL_SERVICES_VALIDATE_H

