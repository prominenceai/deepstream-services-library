/*
The MIT License

Copyright (c) 2019-2020, ROBERT HOWELL

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

#ifndef _DSL_API_H
#define _DSL_API_H

#ifdef __cplusplus
#define EXTERN_C_BEGIN extern "C" {
#define EXTERN_C_END   }
#else
#define EXTERN_C_BEGIN
#define EXTERN_C_END
#endif

#define DSL_FALSE                                                   0
#define DSL_TRUE                                                    1

#define DSL_RESULT_SUCCESS                                          0x00000000
#define DSL_RESULT_API_NOT_IMPLEMENTED                              0x00000001
#define DSL_RESULT_INVALID_INPUT_PARAM                              0x00000002
#define DSL_RESULT_INVALID_RESULT_CODE                              UINT32_MAX

/**
 * Component API Return Values
 */
#define DSL_RESULT_COMPONENT_RESULT                                 0x00010000
#define DSL_RESULT_COMPONENT_NAME_NOT_UNIQUE                        0x00010001
#define DSL_RESULT_COMPONENT_NAME_NOT_FOUND                         0x00010002
#define DSL_RESULT_COMPONENT_NAME_BAD_FORMAT                        0x00010003
#define DSL_RESULT_COMPONENT_IN_USE                                 0x00010004
#define DSL_RESULT_COMPONENT_NOT_USED_BY_PIPELINE                   0x00010005
#define DSL_RESULT_COMPONENT_NOT_USED_BY_BRANCH                     0x00010006
#define DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE                   0x00010007
#define DSL_RESULT_COMPONENT_SET_GPUID_FAILED                       0x00010008

/**
 * Source API Return Values
 */
#define DSL_RESULT_SOURCE_RESULT                                    0x00020000
#define DSL_RESULT_SOURCE_NAME_NOT_UNIQUE                           0x00020001
#define DSL_RESULT_SOURCE_NAME_NOT_FOUND                            0x00020002
#define DSL_RESULT_SOURCE_NAME_BAD_FORMAT                           0x00020003
#define DSL_RESULT_SOURCE_NOT_FOUND                                 0x00020004
#define DSL_RESULT_SOURCE_THREW_EXCEPTION                           0x00020005
#define DSL_RESULT_SOURCE_FILE_NOT_FOUND                            0x00020006
#define DSL_RESULT_SOURCE_NOT_IN_USE                                0x00020007
#define DSL_RESULT_SOURCE_NOT_IN_PLAY                               0x00020008
#define DSL_RESULT_SOURCE_NOT_IN_PAUSE                              0x00020009
#define DSL_RESULT_SOURCE_FAILED_TO_CHANGE_STATE                    0x0002000A
#define DSL_RESULT_SOURCE_CODEC_PARSER_INVALID                      0x0002000B
#define DSL_RESULT_SOURCE_DEWARPER_ADD_FAILED                       0x0002000C
#define DSL_RESULT_SOURCE_DEWARPER_REMOVE_FAILED                    0x0002000D
#define DSL_RESULT_SOURCE_TAP_ADD_FAILED                            0x0002000E
#define DSL_RESULT_SOURCE_TAP_REMOVE_FAILED                         0x0002000F
#define DSL_RESULT_SOURCE_COMPONENT_IS_NOT_SOURCE                   0x00020010
#define DSL_RESULT_SOURCE_CALLBACK_ADD_FAILED                       0x00020011
#define DSL_RESULT_SOURCE_CALLBACK_REMOVE_FAILED                    0x00020012
#define DSL_RESULT_SOURCE_SET_FAILED                                0x00020013


/**
 * Dewarper API Return Values
 */
#define DSL_RESULT_DEWARPER_RESULT                                  0x00090000
#define DSL_RESULT_DEWARPER_NAME_NOT_UNIQUE                         0x00090001
#define DSL_RESULT_DEWARPER_NAME_NOT_FOUND                          0x00090002
#define DSL_RESULT_DEWARPER_NAME_BAD_FORMAT                         0x00090003
#define DSL_RESULT_DEWARPER_THREW_EXCEPTION                         0x00090004
#define DSL_RESULT_DEWARPER_CONFIG_FILE_NOT_FOUND                   0x00090005

/**
 * Tracker API Return Values
 */
#define DSL_RESULT_TRACKER_RESULT                                   0x00030000
#define DSL_RESULT_TRACKER_NAME_NOT_UNIQUE                          0x00030001
#define DSL_RESULT_TRACKER_NAME_NOT_FOUND                           0x00030002
#define DSL_RESULT_TRACKER_NAME_BAD_FORMAT                          0x00030003
#define DSL_RESULT_TRACKER_THREW_EXCEPTION                          0x00030004
#define DSL_RESULT_TRACKER_CONFIG_FILE_NOT_FOUND                    0x00030005
#define DSL_RESULT_TRACKER_MAX_DIMENSIONS_INVALID                   0x00030006
#define DSL_RESULT_TRACKER_IS_IN_USE                                0x00030007
#define DSL_RESULT_TRACKER_SET_FAILED                               0x00030008
#define DSL_RESULT_TRACKER_HANDLER_ADD_FAILED                       0x00030009
#define DSL_RESULT_TRACKER_HANDLER_REMOVE_FAILED                    0x0003000A
#define DSL_RESULT_TRACKER_PAD_TYPE_INVALID                         0x0003000B
#define DSL_RESULT_TRACKER_COMPONENT_IS_NOT_TRACKER                 0x0003000C

/**
 * Sink API Return Values
 */
#define DSL_RESULT_SINK_RESULT                                      0x00040000
#define DSL_RESULT_SINK_NAME_NOT_UNIQUE                             0x00040001
#define DSL_RESULT_SINK_NAME_NOT_FOUND                              0x00040002
#define DSL_RESULT_SINK_NAME_BAD_FORMAT                             0x00040003
#define DSL_RESULT_SINK_THREW_EXCEPTION                             0x00040004
#define DSL_RESULT_SINK_FILE_PATH_NOT_FOUND                         0x00040005
#define DSL_RESULT_SINK_IS_IN_USE                                   0x00040007
#define DSL_RESULT_SINK_SET_FAILED                                  0x00040008
#define DSL_RESULT_SINK_CODEC_VALUE_INVALID                         0x00040009
#define DSL_RESULT_SINK_CONTAINER_VALUE_INVALID                     0x0004000A
#define DSL_RESULT_SINK_COMPONENT_IS_NOT_SINK                       0x0004000B
#define DSL_RESULT_SINK_COMPONENT_IS_NOT_ENCODE_SINK                0x0004000C
#define DSL_RESULT_SINK_OBJECT_CAPTURE_CLASS_ADD_FAILED             0x0004000D
#define DSL_RESULT_SINK_OBJECT_CAPTURE_CLASS_REMOVE_FAILED          0x0004000E
#define DSL_RESULT_SINK_HANDLER_ADD_FAILED                          0x0004000F
#define DSL_RESULT_SINK_HANDLER_REMOVE_FAILED                       0x00040010

/**
 * OSD API Return Values
 */
#define DSL_RESULT_OSD_RESULT                                       0x00050000
#define DSL_RESULT_OSD_NAME_NOT_UNIQUE                              0x00050001
#define DSL_RESULT_OSD_NAME_NOT_FOUND                               0x00050002
#define DSL_RESULT_OSD_NAME_BAD_FORMAT                              0x00050003
#define DSL_RESULT_OSD_THREW_EXCEPTION                              0x00050004
#define DSL_RESULT_OSD_MAX_DIMENSIONS_INVALID                       0x00050005
#define DSL_RESULT_OSD_IS_IN_USE                                    0x00050006
#define DSL_RESULT_OSD_SET_FAILED                                   0x00050007
#define DSL_RESULT_OSD_HANDLER_ADD_FAILED                           0x00050008
#define DSL_RESULT_OSD_HANDLER_REMOVE_FAILED                        0x00050009
#define DSL_RESULT_OSD_PAD_TYPE_INVALID                             0x0005000A
#define DSL_RESULT_OSD_COMPONENT_IS_NOT_OSD                         0x0005000B
#define DSL_RESULT_OSD_COLOR_PARAM_INVALID                          0x0005000C

/**
 * OFV API Return Values
 */
#define DSL_RESULT_OFV_RESULT                                       0x000C0000
#define DSL_RESULT_OFV_NAME_NOT_UNIQUE                              0x000C0001
#define DSL_RESULT_OFV_NAME_NOT_FOUND                               0x000C0002
#define DSL_RESULT_OFV_NAME_BAD_FORMAT                              0x000C0003
#define DSL_RESULT_OFV_THREW_EXCEPTION                              0x000C0004
#define DSL_RESULT_OFV_MAX_DIMENSIONS_INVALID                       0x000C0005
#define DSL_RESULT_OFV_IS_IN_USE                                    0x000C0006
#define DSL_RESULT_OFV_SET_FAILED                                   0x000C0007
#define DSL_RESULT_OFV_HANDLER_ADD_FAILED                           0x000C0008
#define DSL_RESULT_OFV_HANDLER_REMOVE_FAILED                        0x000C0009
#define DSL_RESULT_OFV_PAD_TYPE_INVALID                             0x000C000A
#define DSL_RESULT_OFV_COMPONENT_IS_NOT_OFV                         0x000C000B

/**
 * GIE API Return Values
 */
#define DSL_RESULT_GIE_RESULT                                       0x00060000
#define DSL_RESULT_GIE_NAME_NOT_UNIQUE                              0x00060001
#define DSL_RESULT_GIE_NAME_NOT_FOUND                               0x00060002
#define DSL_RESULT_GIE_NAME_BAD_FORMAT                              0x00060003
#define DSL_RESULT_GIE_CONFIG_FILE_NOT_FOUND                        0x00060004
#define DSL_RESULT_GIE_MODEL_FILE_NOT_FOUND                         0x00060005
#define DSL_RESULT_GIE_THREW_EXCEPTION                              0x00060006
#define DSL_RESULT_GIE_IS_IN_USE                                    0x00060007
#define DSL_RESULT_GIE_SET_FAILED                                   0x00060008
#define DSL_RESULT_GIE_HANDLER_ADD_FAILED                           0x00060009
#define DSL_RESULT_GIE_HANDLER_REMOVE_FAILED                        0x0006000A
#define DSL_RESULT_GIE_PAD_TYPE_INVALID                             0x0006000B
#define DSL_RESULT_GIE_COMPONENT_IS_NOT_GIE                         0x0006000C
#define DSL_RESULT_GIE_OUTPUT_DIR_DOES_NOT_EXIST                    0x0006000D

/**
 * Demuxer API Return Values
 */
#define DSL_RESULT_TEE_RESULT                                       0x000A0000
#define DSL_RESULT_TEE_NAME_NOT_UNIQUE                              0x000A0001
#define DSL_RESULT_TEE_NAME_NOT_FOUND                               0x000A0002
#define DSL_RESULT_TEE_NAME_BAD_FORMAT                              0x000A0003
#define DSL_RESULT_TEE_THREW_EXCEPTION                              0x000A0004
#define DSL_RESULT_TEE_BRANCH_IS_NOT_BRANCH                         0x000A0005
#define DSL_RESULT_TEE_BRANCH_IS_NOT_CHILD                          0x000A0006
#define DSL_RESULT_TEE_BRANCH_ADD_FAILED                            0x000A0007
#define DSL_RESULT_TEE_BRANCH_REMOVE_FAILED                         0x000A0008
#define DSL_RESULT_TEE_HANDLER_ADD_FAILED                           0x000A0009
#define DSL_RESULT_TEE_HANDLER_REMOVE_FAILED                        0x000A000A
#define DSL_RESULT_TEE_COMPONENT_IS_NOT_TEE                         0x000A000B

/**
 * Tile API Return Values
 */
#define DSL_RESULT_TILER_RESULT                                     0x00070000
#define DSL_RESULT_TILER_NAME_NOT_UNIQUE                            0x00070001
#define DSL_RESULT_TILER_NAME_NOT_FOUND                             0x00070002
#define DSL_RESULT_TILER_NAME_BAD_FORMAT                            0x00070003
#define DSL_RESULT_TILER_THREW_EXCEPTION                            0x00070004
#define DSL_RESULT_TILER_IS_IN_USE                                  0x00070005
#define DSL_RESULT_TILER_SET_FAILED                                 0x00070006
#define DSL_RESULT_TILER_HANDLER_ADD_FAILED                         0x00070007
#define DSL_RESULT_TILER_HANDLER_REMOVE_FAILED                      0x00070008
#define DSL_RESULT_TILER_PAD_TYPE_INVALID                           0x00070009
#define DSL_RESULT_TILER_COMPONENT_IS_NOT_TILER                     0x0007000A

/**
 * Pipeline API Return Values
 */
#define DSL_RESULT_PIPELINE_RESULT                                  0x00080000
#define DSL_RESULT_PIPELINE_NAME_NOT_UNIQUE                         0x00080001
#define DSL_RESULT_PIPELINE_NAME_NOT_FOUND                          0x00080002
#define DSL_RESULT_PIPELINE_NAME_BAD_FORMAT                         0x00080003
#define DSL_RESULT_PIPELINE_STATE_PAUSED                            0x00080004
#define DSL_RESULT_PIPELINE_STATE_RUNNING                           0x00080005
#define DSL_RESULT_PIPELINE_THREW_EXCEPTION                         0x00080006
#define DSL_RESULT_PIPELINE_COMPONENT_ADD_FAILED                    0x00080007
#define DSL_RESULT_PIPELINE_COMPONENT_REMOVE_FAILED                 0x00080008
#define DSL_RESULT_PIPELINE_STREAMMUX_GET_FAILED                    0x00080009
#define DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED                    0x0008000A
#define DSL_RESULT_PIPELINE_XWINDOW_GET_FAILED                      0x0008000B
#define DSL_RESULT_PIPELINE_XWINDOW_SET_FAILED                      0x0008000C
#define DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED                     0x0008000D
#define DSL_RESULT_PIPELINE_CALLBACK_REMOVE_FAILED                  0x0008000E
#define DSL_RESULT_PIPELINE_FAILED_TO_PLAY                          0x0008000F
#define DSL_RESULT_PIPELINE_FAILED_TO_PAUSE                         0x00080010
#define DSL_RESULT_PIPELINE_FAILED_TO_STOP                          0x00080011
#define DSL_RESULT_PIPELINE_SOURCE_MAX_IN_USE_REACHED               0x00080012
#define DSL_RESULT_PIPELINE_SINK_MAX_IN_USE_REACHED                 0x00080013

#define DSL_RESULT_BRANCH_RESULT                                    0x000B0000
#define DSL_RESULT_BRANCH_NAME_NOT_UNIQUE                           0x000B0001
#define DSL_RESULT_BRANCH_NAME_NOT_FOUND                            0x000B0002
#define DSL_RESULT_BRANCH_NAME_BAD_FORMAT                           0x000B0003
#define DSL_RESULT_BRANCH_THREW_EXCEPTION                           0x000B0004
#define DSL_RESULT_BRANCH_COMPONENT_ADD_FAILED                      0x000B0005
#define DSL_RESULT_BRANCH_COMPONENT_REMOVE_FAILED                   0x000B0006
#define DSL_RESULT_BRANCH_SOURCE_NOT_ALLOWED                        0x000B0007
#define DSL_RESULT_BRANCH_SINK_MAX_IN_USE_REACHED                   0x000B0008

/**
 * Pad Probe Handler API Return Values
 */
#define DSL_RESULT_PPH_RESULT                                       0x000D0000
#define DSL_RESULT_PPH_NAME_NOT_UNIQUE                              0x000D0001
#define DSL_RESULT_PPH_NAME_NOT_FOUND                               0x000D0002
#define DSL_RESULT_PPH_NAME_BAD_FORMAT                              0x000D0003
#define DSL_RESULT_PPH_THREW_EXCEPTION                              0x000D0004
#define DSL_RESULT_PPH_IS_IN_USE                                    0x000D0005
#define DSL_RESULT_PPH_SET_FAILED                                   0x000D0006
#define DSL_RESULT_PPH_ODE_TRIGGER_ADD_FAILED                       0x000D0007
#define DSL_RESULT_PPH_ODE_TRIGGER_REMOVE_FAILED                    0x000D0008
#define DSL_RESULT_PPH_ODE_TRIGGER_NOT_IN_USE                       0x000D0009
#define DSL_RESULT_PPH_METER_INVALID_INTERVAL                       0x0004000A
#define DSL_RESULT_PPH_PAD_TYPE_INVALID                             0x0004000B

/**
 * ODE Trigger API Return Values
 */
#define DSL_RESULT_ODE_TRIGGER_RESULT                               0x000E0000
#define DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE                      0x000E0001
#define DSL_RESULT_ODE_TRIGGER_NAME_NOT_FOUND                       0x000E0002
#define DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION                      0x000E0003
#define DSL_RESULT_ODE_TRIGGER_IN_USE                               0x000E0004
#define DSL_RESULT_ODE_TRIGGER_SET_FAILED                           0x000E0005
#define DSL_RESULT_ODE_TRIGGER_IS_NOT_ODE_TRIGGER                   0x000E0006
#define DSL_RESULT_ODE_TRIGGER_ACTION_ADD_FAILED                    0x000E0007
#define DSL_RESULT_ODE_TRIGGER_ACTION_REMOVE_FAILED                 0x000E0008
#define DSL_RESULT_ODE_TRIGGER_ACTION_NOT_IN_USE                    0x000E0009
#define DSL_RESULT_ODE_TRIGGER_AREA_ADD_FAILED                      0x000E000A
#define DSL_RESULT_ODE_TRIGGER_AREA_REMOVE_FAILED                   0x000E000B
#define DSL_RESULT_ODE_TRIGGER_AREA_NOT_IN_USE                      0x000E000C
#define DSL_RESULT_ODE_TRIGGER_CLIENT_CALLBACK_INVALID              0x000E000D
#define DSL_RESULT_ODE_TRIGGER_ALWAYS_WHEN_PARAMETER_INVALID        0x000E000E

/**
 * ODE Action API Return Values
 */
#define DSL_RESULT_ODE_ACTION_RESULT                                0x000F0000
#define DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE                       0x000F0001
#define DSL_RESULT_ODE_ACTION_NAME_NOT_FOUND                        0x000F0002
#define DSL_RESULT_ODE_ACTION_CAPTURE_TYPE_INVALID                  0x000F0003
#define DSL_RESULT_ODE_ACTION_THREW_EXCEPTION                       0x000F0004
#define DSL_RESULT_ODE_ACTION_IN_USE                                0x000F0005
#define DSL_RESULT_ODE_ACTION_SET_FAILED                            0x000F0006
#define DSL_RESULT_ODE_ACTION_IS_NOT_ACTION                         0x000F0007
#define DSL_RESULT_ODE_ACTION_FILE_PATH_NOT_FOUND                   0x000F0008
#define DSL_RESULT_ODE_ACTION_NOT_THE_CORRECT_TYPE                  0x000F0009

/**
 * ODE Area API Return Values
 */
#define DSL_RESULT_ODE_AREA_RESULT                                  0x00100000
#define DSL_RESULT_ODE_AREA_NAME_NOT_UNIQUE                         0x00100001
#define DSL_RESULT_ODE_AREA_NAME_NOT_FOUND                          0x00100002
#define DSL_RESULT_ODE_AREA_THREW_EXCEPTION                         0x00100003
#define DSL_RESULT_ODE_AREA_IN_USE                                  0x00100004
#define DSL_RESULT_ODE_AREA_SET_FAILED                              0x00100005

#define DSL_RESULT_DISPLAY_TYPE_RESULT                              0x00100000
#define DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE                     0x00100001
#define DSL_RESULT_DISPLAY_TYPE_NAME_NOT_FOUND                      0x00100002
#define DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION                     0x00100003
#define DSL_RESULT_DISPLAY_TYPE_IN_USE                              0x00100004
#define DSL_RESULT_DISPLAY_TYPE_NOT_THE_CORRECT_TYPE                0x00100005
#define DSL_RESULT_DISPLAY_TYPE_IS_BASE_TYPE                        0x00100006
#define DSL_RESULT_DISPLAY_RGBA_COLOR_NAME_NOT_UNIQUE               0x00100007
#define DSL_RESULT_DISPLAY_RGBA_FONT_NAME_NOT_UNIQUE                0x00100008
#define DSL_RESULT_DISPLAY_RGBA_TEXT_NAME_NOT_UNIQUE                0x00100009
#define DSL_RESULT_DISPLAY_RGBA_LINE_NAME_NOT_UNIQUE                0x0010000A
#define DSL_RESULT_DISPLAY_RGBA_ARROW_NAME_NOT_UNIQUE               0x0010000B
#define DSL_RESULT_DISPLAY_RGBA_ARROW_HEAD_INVALID                  0x0010000C
#define DSL_RESULT_DISPLAY_RGBA_RECTANGLE_NAME_NOT_UNIQUE           0x0010000D
#define DSL_RESULT_DISPLAY_RGBA_CIRCLE_NAME_NOT_UNIQUE              0x0010000E
#define DSL_RESULT_DISPLAY_SOURCE_NUMBER_NAME_NOT_UNIQUE            0x0010000F
#define DSL_RESULT_DISPLAY_SOURCE_NAME_NAME_NOT_UNIQUE              0x00100010
#define DSL_RESULT_DISPLAY_SOURCE_DIMENSIONS_NAME_NOT_UNIQUE        0x00100011
#define DSL_RESULT_DISPLAY_SOURCE_FRAMERATE_NAME_NOT_UNIQUE         0x00100012


/**
 * Tap API Return Values
 */
#define DSL_RESULT_TAP_RESULT                                       0x00200000
#define DSL_RESULT_TAP_NAME_NOT_UNIQUE                              0x00200001
#define DSL_RESULT_TAP_NAME_NOT_FOUND                               0x00200002
#define DSL_RESULT_TAP_THREW_EXCEPTION                              0x00200003
#define DSL_RESULT_TAP_IN_USE                                       0x00200004
#define DSL_RESULT_TAP_SET_FAILED                                   0x00200005
#define DSL_RESULT_TAP_COMPONENT_IS_NOT_TAP                         0x00200006
#define DSL_RESULT_TAP_FILE_PATH_NOT_FOUND                          0x00200007
#define DSL_RESULT_TAP_CONTAINER_VALUE_INVALID                      0x00200008

/**
 *
 */
#define DSL_CUDADEC_MEMTYPE_DEVICE                                  0
#define DSL_CUDADEC_MEMTYPE_PINNED                                  1
#define DSL_CUDADEC_MEMTYPE_UNIFIED                                 2

#define DSL_SOURCE_CODEC_PARSER_H264                                0
#define DSL_SOURCE_CODEC_PARSER_H265                                1

#define DSL_TILER_SHOW_ALL_SOURCES                                  NULL

#define DSL_CODEC_H264                                              0
#define DSL_CODEC_H265                                              1
#define DSL_CODEC_MPEG4                                             2

#define DSL_CONTAINER_MP4                                           0
#define DSL_CONTAINER_MKV                                           1

// Must match GST_STATE enum values
#define DSL_STATE_NULL                                              1
#define DSL_STATE_READY                                             2
#define DSL_STATE_PAUSED                                            3
#define DSL_STATE_PLAYING                                           4
#define DSL_STATE_CHANGE_ASYNC                                      5
#define DSL_STATE_UNKNOWN                                           UINT32_MAX

#define DSL_PAD_SINK                                                0
#define DSL_PAD_SRC                                                 1

#define DSL_RTP_TCP                                                 0x04
#define DSL_RTP_ALL                                                 0x07
/**
 * @brief time to sleep after a failed reconnection before
 * starting a new re-connection cycle. In units of seconds
 */
#define DSL_RTSP_RECONNECTION_SLEEP_S                               4
/**
 * @brief the maximum time to wait for a RTSP Source to
 * asynchronously transition to a final state of Playing.
 * In units of seconds
 */
#define DSL_RTSP_RECONNECTION_TIMEOUT_S                             30

#define DSL_CAPTURE_TYPE_OBJECT                                     0
#define DSL_CAPTURE_TYPE_FRAME                                      1

// Trigger-Always 'when' constants, pre/post check-for-occurrence
#define DSL_ODE_PRE_OCCURRENCE_CHECK                                0
#define DSL_ODE_POST_OCCURRENCE_CHECK                               1

// Source and Class Trigger filter constants for no-filter
#define DSL_ODE_ANY_SOURCE                                          NULL
#define DSL_ODE_ANY_CLASS                                           INT32_MAX

// Must match NvOSD_Arrow_Head_Direction
#define DSL_ARROW_START_HEAD                                        0
#define DSL_ARROW_END_HEAD                                          1
#define DSL_ARROW_BOTH_HEAD                                         2

// Must match GstPadProbeReturn values
#define DSL_PAD_PROBE_DROP                                          0
#define DSL_PAD_PROBE_OK                                            1
#define DSL_PAD_PROBE_REMOVE                                        2
#define DSL_PAD_PROBE_PASS                                          3
#define DSL_PAD_PROBE_HANDLED                                       4

/**
 * @brief DSL_DEFAULT values initialized on first call to DSL
 */
//TODO move to new defaults schema
#define DSL_DEFAULT_SOURCE_IN_USE_MAX                               8
#define DSL_DEFAULT_SINK_IN_USE_MAX                                 8
#define DSL_DEFAULT_STREAMMUX_BATCH_TIMEOUT                         40000
#define DSL_DEFAULT_STREAMMUX_WIDTH                                 1920
#define DSL_DEFAULT_STREAMMUX_HEIGHT                                1080
#define DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC                     10

#define DSL_DEFAULT_VIDEO_RECORD_CACHE_IN_SEC                       30
#define DSL_DEFAULT_VIDEO_RECORD_DURATION_IN_SEC                    30

EXTERN_C_BEGIN

typedef uint DslReturnType;
typedef uint boolean;

/**
 * @struct dsl_rtsp_connection_data
 * @brief a structure of Connection Stats and Parameters for a given RTSP Source
 */
typedef struct dsl_rtsp_connection_data
{
    /**
     * @brief true if the RTSP Source is currently in a connected state, false otherwise
     */ 
    boolean is_connected; 
    
    /**
     * @brief linux time in seconds for the first successful connection or
     * when the stats were last cleared
     */ 
    time_t first_connected; 

    /**
     * @brief linux time in seconds for the last succesful connection or
     * when the stats were last cleared
     */ 
    time_t last_connected; 

    /**
     * @brief linux time in seconds for the last disconnection or
     * when the stats were last cleared
     */ 
    time_t last_disconnected; 

    /**
     * @brief count of succesful connections from the start of Pipeline
     * play, or from when the stats were last cleared
     */ 
    uint count;
    
    /**
     * @brief true if the RTSP Source is currently in a re-connection cycle, false otherwise
     */ 
    boolean is_in_reconnect; 
    
    /**
     * @brief number of re-connection retries for either the current cycle, if "is_in_reconnect == true"
     * or the last connection if "is_in_reconnect == false"
     */ 
    uint retries;
    
    /**
     * @brief current setting for the time to sleep between re-connection attempts after failure.
     */ 
    uint sleep;
    
    /**
     * @brief current setting for the maximum time to wait for an asynchronous state change to complete
     * before resetting the source and then retrying again after the next sleep period.
     */ 
   uint timeout;
   
}dsl_rtsp_connection_data;

/**
 * @struct dsl_recording_info
 * @brief recording session information provided to the client on callback
 */
typedef struct dsl_recording_info
{
    /**
     * @brief the unique sesions id assigned on record start
     */
    uint sessionId;
    
    /**
     * @brief filename generated for the completed recording. 
     */
    const wchar_t* filename;
    
    /** 
     * @brief directory path for the completed recording
     */
    const wchar_t* dirpath;
    
    /**
     * @brief duration of the recording in milliseconds
     */
    uint64_t duration;
    
    /**
     * @brief either DSL_CONTAINER_MP4 or DSL_CONTAINER_MP4
     */
    uint containerType;
    
    /**
     * @brief width of the recording in pixels
     */
    uint width;

    /**
     * @brief width of the recording in pixels
     */
    uint height;

} dsl_recording_info;

/**
 *
 * @brief callback typedef for a client ODE occurrence handler function. Once 
 * registered, the function will be called on ODE occurrence
 * @param[in] event_id unique ODE occurrence ID, numerically ordered by occurrence
 * @param[in] trigger unique name of the ODE Event Trigger that trigger the occurrence
 * @param[in] pointer to a frame_meta structure that triggered the ODE event
 * @param[in] pointer to a object_meta structure that triggered the ODE event
 * This parameter will be set to NULL for ODE occurrences detected in Post process frame. Absence and Submation ODE's
 * @param[in] client_data opaque pointer to client's user data
 */
typedef void (*dsl_ode_handle_occurrence_cb)(uint64_t event_id, const wchar_t* trigger,
    void* buffer, void* frame_meta, void* object_meta, void* client_data);

/**
 * @brief callback typedef for a client ODE Custom Trigger check-for-occurrence function. Once 
 * registered, the function will be called on every object detected that meets the minimum
 * criteria for the Custom Trigger. The client, determining that criteria is met for ODE occurrence,
 * returns true to invoke all ODE acctions owned by the Custom Trigger
 * @param[in] pointer to a frame_meta structure that triggered the ODE event
 * @param[in] pointer to a object_meta structure that triggered the ODE event
 * This parameter will be set to NULL for ODE occurrences detected in Post process frame. Absence and Submation ODE's
 * @param[in] client_data opaque pointer to client's user data
 */
typedef boolean (*dsl_ode_check_for_occurrence_cb)(void* buffer,
    void* frame_meta, void* object_meta, void* client_data);

/**
 * @brief callback typedef for a client ODE Custom Trigger post-process-frame function. Once 
 * registered, the function will be called on every frame AFTER all Check-For-Occurrence calls have been handles
 * The client, determining that criteria is met for ODE occurrence,  returns true to invoke all ODE acctions owned by the Custom Trigger
 * @param[in] pointer to a frame_meta structure that triggered the ODE event
 * @param[in] pointer to a object_meta structure that triggered the ODE event
 * This parameter will be set to NULL for ODE occurrences detected in Post process frame. Absence and Submation ODE's
 * @param[in] client_data opaque pointer to client's user data
 */
typedef boolean (*dsl_ode_post_process_frame_cb)(void* buffer,
    void* frame_meta, void* client_data);

/**
 * @brief callback typedef for a client to hanlde new Pipeline performance data
 * ,calcaulated by the Meter Pad Probe Handler, at an intervel specified by the client.
 * @param[in] session_fps_averages array of frames-per-second measurements, one per source, specified by list_size 
 * @param[in] interval_fps_averages array of average frames-per-second measurements, one per source, specified by list_size 
 * @param[in] source_count count of both session_fps_averages and avg_fps interval_fps_averages, one Pipeline Source
 * @param[in] client_data opaque pointer to client's user data provide on end-of-session
 */
typedef boolean (*dsl_pph_meter_client_handler_cb)(double* session_fps_averages, double* interval_fps_averages, 
    uint source_count, void* client_data);
    
/**
 * @brief callback typedef for a client pad probe handler function. Once added to a Component, 
 * the function will be called when the component receives a pad probe buffer ready.
 * @param[in] buffer pointer to a stream buffer to process
 * @param[in] client_data opaque pointer to client's user data
 * @return one of DSL_PAD_PROBE values defined above 
 */
typedef uint (*dsl_pph_custom_client_handler_cb)(void* buffer, void* client_data);

/**
 * @brief callback typedef for a client listener function. Once added to a Pipeline, 
 * the function will be called when the Pipeline changes state.
 * @param[in] prev_state one of DSL_PIPELINE_STATE constants for the previous pipeline state
 * @param[in] curr_state one of DSL_PIPELINE_STATE constants for the current pipeline state
 * @param[in] client_data opaque pointer to client's data
 */
typedef void (*dsl_state_change_listener_cb)(uint prev_state, uint curr_state, void* client_data);

/**
 * @brief callback typedef for a client listener function. Once added to a Pipeline, 
 * the function will be called on receipt of EOS message from the Pipeline bus.
 * @param[in] client_data opaque pointer to client's data
 */
typedef void (*dsl_eos_listener_cb)(void* client_data);

/**
 * @brief callback typedef for a client listener function. Once added to a Pipeline, 
 * the function will be called on receipt of Error messages from the Pipeline bus.
 * @param[in] source name of the element or component that is the source of the message
 * @param[in] message error parsed from the message data
 * @param[in] client_data opaque pointer to client's data
 */
typedef void (*dsl_error_message_handler_cb)(const wchar_t* source, const wchar_t* message, void* client_data);

/**
 * @brief callback typedef for a client XWindow KeyRelease event handler function. Once added to a Pipeline, 
 * the function will be called when the Pipeline receives XWindow KeyRelease events.
 * @param[in] key UNICODE key string for the key pressed
 * @param[in] client_data opaque pointer to client's user data
 */
typedef void (*dsl_xwindow_key_event_handler_cb)(const wchar_t* key, void* client_data);

/**
 * @brief callback typedef for a client XWindow ButtonPress event handler function. Once added to a Pipeline, 
 * the function will be called when the Pipeline receives XWindow ButtonPress events.
 * @param[in] button button 1 through 5 including scroll wheel up and down
 * @param[in] xpos from the top left corner of the window
 * @param[in] ypos from the top left corner of the window
 * @param[in] client_data opaque pointer to client's user data
 */
typedef void (*dsl_xwindow_button_event_handler_cb)(uint button, int xpos, int ypos, void* client_data);

/**
 * @brief callback typedef for a client XWindow Delete Message event handler function. Once added to a Pipeline, 
 * the function will be called when the Pipeline receives XWindow Delete Message event.
 * @param[in] client_data opaque pointer to client's user data
 */
typedef void (*dsl_xwindow_delete_event_handler_cb)(void* client_data);


/**
 * @brief callback typedef for a client to listen for notification that a Recording Session has ended.
 * @param[in] info pointer to session info, see... dsl_recording_info above.
 * @param[in] client_data opaque pointer to client's user data provide on end-of-session
 */
typedef void* (*dsl_record_client_listener_cb)(dsl_recording_info* info, void* client_data);


/**
 * @brief creates a uniquely named RGBA Display Color
 * @param[in] name unique name for the RGBA Color
 * @param[in] red red level for the RGB color [0..1]
 * @param[in] blue blue level for the RGB color [0..1]
 * @param[in] green green level for the RGB color [0..1]
 * @param[in] alpha alpha level for the RGB color [0..1]
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_color_new(const wchar_t* name, 
    double red, double green, double blue, double alpha);

/**
 * @brief creates a uniquely named RGBA Display Font
 * @param[in] name unique name for the RGBA Font
 * @param[in] fount standard, unique string name of the actual font type (eg. 'arial')
 * @param[in] size size of the font
 * @param[in] color name of the RGBA Color for the RGBA font
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_font_new(const wchar_t* name, const wchar_t* font, uint size, const wchar_t* color);

/**
 * @brief creates a uniquely named RGBA Display Text
 * @param[in] name unique name of the RGBA Text
 * @param[in] text text string to display
 * @param[in] x_offset starting x positional offset
 * @param[in] y_offset starting y positional offset
 * @param[in] font RGBA font to use for the display dext
 * @param[in] hasBgColor set to true to enable bacground color, false otherwise
 * @param[in] bgColor RGBA Color for the Text background if set
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_text_new(const wchar_t* name, const wchar_t* text, uint x_offset, uint y_offset, 
    const wchar_t* font, boolean has_bg_color, const wchar_t* bg_color);
    
/**
 * @brief creates a uniquely named RGBA Display Line
 * @param[in] name unique name for the RGBA LIne
 * @param[in] x1 starting x positional offest
 * @param[in] y1 starting y positional offest
 * @param[in] x2 ending x positional offest
 * @param[in] y2 ending y positional offest
 * @param[in] width width of the line in pixels
 * @param[in] color RGBA Color for thIS RGBA Line
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_line_new(const wchar_t* name, 
    uint x1, uint y1, uint x2, uint y2, uint width, const wchar_t* color);

/**
 * @brief creates a uniquely named RGBA Display Arrow
 * @param[in] name unique name for the RGBA Arrow
 * @param[in] x1 starting x positional offest
 * @param[in] y1 starting y positional offest
 * @param[in] x2 ending x positional offest
 * @param[in] y2 ending y positional offest
 * @param[in] width width of the Arrow in pixels
 * @param[in] head DSL_ARROW_START_HEAD, DSL_ARROW_END_HEAD, DSL_ARROW_BOTH_HEAD
 * @param[in] color RGBA Color for thIS RGBA Line
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_arrow_new(const wchar_t* name, 
    uint x1, uint y1, uint x2, uint y2, uint width, uint head, const wchar_t* color);

/**
 * @brief creates a uniquely named RGBA Rectangle
 * @param[in] name unique name for the RGBA Rectangle
 * @param[in] left left positional offest
 * @param[in] top positional offest
 * @param[in] width width of the rectangle in Pixels
 * @param[in] height height of the rectangle in Pixels
 * @param[in] border_width width of the rectangle border in pixels
 * @param[in] color RGBA Color for thIS RGBA Line
 * @param[in] hasBgColor set to true to enable bacground color, false otherwise
 * @param[in] bgColor RGBA Color for the Circle background if set
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_rectangle_new(const wchar_t* name, uint left, uint top, uint width, uint height, 
    uint border_width, const wchar_t* color, bool has_bg_color, const wchar_t* bg_color);

/**
 * @brief creates a uniquely named RGBA Circle
 * @param[in] name unique name for the RGBA Circle
 * @param[in] x_center X positional offset to center of Circle
 * @param[in] y_center y positional offset to center of Circle
 * @param[in] radius radius of the RGBA Circle in pixels 
 * @param[in] color RGBA Color for the RGBA Circle
 * @param[in] hasBgColor set to true to enable bacground color, false otherwise
 * @param[in] bgColor RGBA Color for the Circle background if set
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_circle_new(const wchar_t* name, uint x_center, uint y_center, uint radius,
    const wchar_t* color, bool has_bg_color, const wchar_t* bg_color);

/**
 * @brief creates a uniquely named Source Number Display Type
 * @param[in] name unique name of the Display Type
 * @param[in] x_offset starting x positional offset
 * @param[in] y_offset starting y positional offset
 * @param[in] font RGBA font to use for the display text
 * @param[in] hasBgColor set to true to enable bacground color, false otherwise
 * @param[in] bgColor RGBA Color for the Text background if set
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_source_number_new(const wchar_t* name, uint x_offset, uint y_offset, 
    const wchar_t* font, boolean has_bg_color, const wchar_t* bg_color);
    
/**
 * @brief creates a uniquely named Source Name Display Type
 * @param[in] name unique name of the Display Type
 * @param[in] x_offset starting x positional offset
 * @param[in] y_offset starting y positional offset
 * @param[in] font RGBA font to use for the display text
 * @param[in] hasBgColor set to true to enable bacground color, false otherwise
 * @param[in] bgColor RGBA Color for the Text background if set
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_source_name_new(const wchar_t* name, uint x_offset, uint y_offset, 
    const wchar_t* font, boolean has_bg_color, const wchar_t* bg_color);
    
/**
 * @brief creates a uniquely named Source Dimensions Display Type
 * @param[in] name unique name of the Display Type
 * @param[in] x_offset starting x positional offset
 * @param[in] y_offset starting y positional offset
 * @param[in] font RGBA font to use for the display text
 * @param[in] hasBgColor set to true to enable bacground color, false otherwise
 * @param[in] bgColor RGBA Color for the Text background if set
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_source_dimensions_new(const wchar_t* name, uint x_offset, uint y_offset, 
    const wchar_t* font, boolean has_bg_color, const wchar_t* bg_color);

/**
 * @brief Adds a named Display Type (text/shape) to a frames's display metadata, The caller 
 * is responsible for aquiring the display metadata for the current frame.
 * @param name unique name of the Display Type to overlay
 * @param display_meta opaque pointer to the aquired display meta to to add the Display Type to
 * @param frame_meta opaque pointer to a Frame's meta data to add the Display Type
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_meta_add(const wchar_t* name, void* buffer, void* frame_meta);
    
/**
 * @brief deletes a uniquely named Display Type of any type
 * @param[in] name unique name for the Display Type to delete
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_delete(const wchar_t* name);

/**
 * @brief Deletes a Null terminated array of Display Types of any type
 * @param[in] names Null ternimated array of unique names to delete
 * @return DSL_RESULT_SUCCESS on success, on of DSL_RESULT_DISPLAY_RESULT otherwise.
 */
DslReturnType dsl_display_type_delete_many(const wchar_t** names);

/**
 * @brief deletes all Display Types currently in memory
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_delete_all();

/**
 * @brief Returns the size of the list of Display Types
 * @return the number of Display Types in the list
 */
uint dsl_display_type_list_size();

/**
 * @brief Creates a uniquely named ODE Custom Action
 * @param[in] name unique name for the ODE Custom Action 
 * @param[in] client_handler function to call on ODE occurrence
 * @param[in] client_data opaue pointer to client's user data, returned on callback
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_custom_new(const wchar_t* name, 
    dsl_ode_handle_occurrence_cb client_handler, void* client_data);

/**
 * @brief Creates a uniquely named Capture Frame ODE Action
 * @param[in] name unique name for the Capture Frame ODE Action 
 * @param[in] outdir absolute or relative path to image capture directory 
 * @param[in] annotate if true, bounding boxes and labes will be added to the image.
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_capture_frame_new(const wchar_t* name, const wchar_t* outdir, boolean annotate);

/**
 * @brief Creates a uniquely named Capture Object ODE Action
 * @param[in] name unique name for the Capture Object ODE Action 
 * @param[in] outdir absolute or relative path to image capture directory 
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_capture_object_new(const wchar_t* name, const wchar_t* outdir);

/**
 * @brief Creates a uniquely named Display ODE Action
 * @param[in] name unique name for the ODE Display Action 
 * @param[in] offsetX offset in the X direction for the Display text
 * @param[in] offsetY offset in the Y direction for the Display text
 * @param[in] offsetY_with_classId adds an additional offset based on ODE class Id if set true
 * The setting allows multiple ODE Triggers with different class Ids to share the same Display action
 * @param[in] font RGBA Font type to use for the Display text
 * @param[in] has_bg_color if true, displays the background color for the Display Text
 * @param[in] bg_color color to use for the Display Text background color, if has_bg_color
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_display_new(const wchar_t* name, uint offsetX, uint offsetY, 
    boolean offsetY_with_classId, const wchar_t* font, boolean has_bg_color, const wchar_t* bg_color);

/**
 * @brief Creates a uniquely named Add Display Metadata ODE Action to add Display metadata
 * using a uniquely named Display Type 
 * @param[in] name unique name for the Add Display Metadata ODE Action 
 * @param[in] display_type unique name of the Display Type to overlay on ODE occurrence
 * Note: the Display Type must exist prior to constructing the Action.
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_display_meta_add_new(const wchar_t* name, const wchar_t* display_type);

/**
 * @brief Creates a uniquely named Add Many Display Metadata ODE Action to add the 
 * metadata using multiple uniquely named Display Types 
 * @param[in] name unique name for the Add Many Display Metadata ODE Action 
 * @param[in] display_typess NULL terminated list of names of the Display Types to overlay on ODE occurrence
 * Note: the Display Type must exist prior to constructing the Action.
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_display_meta_add_many_new(const wchar_t* name, const wchar_t** display_types);

/**
 * @brief Creates a uniquely named Fill Frame ODE Action, that fills the entire
 * frame with a give RGBA color value
 * @param[in] name unique name for the Fill Frame ODE Action
 * @param[in] red red value for the RGBA background color [1..0]
 * @param[in] green green value for the RGBA background color [1..0]
 * @param[in] blue blue value for the RGBA background color [1..0]
 * @param[in] alpha alpha value for the RGBA background color [1..0]
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_fill_frame_new(const wchar_t* name, const wchar_t* color);

/**
 * @brief Creates a uniquely named Fill Object ODE Action, that fills an object's
 * Background with RGBA color values
 * @param[in] name unique name for the Fill Object ODE Action
 * @param[in] color nane of a RGBA color that must exist prior to creating the Action
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_fill_object_new(const wchar_t* name, const wchar_t* color);

/**
 * @brief Creates a uniquely named Fill Surroundings ODE Action, that fills the entire
 * frame area surroudning an Object's rectangle
 * @param[in] name unique name for the Fill Frame ODE Action
 * @param[in] color nane of a RGBA color that must exist prior to creating the Action
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_fill_surroundings_new(const wchar_t* name, const wchar_t* color);

/**
 * @brief Creates a uniquely named Disable Handler Action that disables
 * a namded Handler
 * @param[in] name unique name for the Fill Backtround ODE Action
 * @param[in] handler unique name of the Handler to disable
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_handler_disable_new(const wchar_t* name, const wchar_t* handler);

/**
 * @brief Creates a uniquely named Hide Object Display ODE Action
 * @param[in] name unique name for the ODE Hide Action 
 * @param[in] if true, hides the Object's Display Text on HandleOccurrence
 * @param[in] if true, hides the Object's Rectangle Border on HandleOccurrence
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_hide_new(const wchar_t* name, boolean text, boolean border);

/**
 * @brief Creates a uniquely named Log ODE Action
 * @param[in] name unique name for the Log ODE Action 
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_log_new(const wchar_t* name);

/**
 * @brief Creates a uniquely named Pause ODE Action
 * @param[in] name unique name for the Pause ODE Action 
 * @param[in] pipeline unique name of the Pipeline to Pause on ODE occurrence
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_pause_new(const wchar_t* name, const wchar_t* pipeline);

/**
 * @brief Creates a uniquely named Print ODE Action
 * @param[in] name unique name for the Print ODE Action 
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_print_new(const wchar_t* name);
    
/**
 * @brief Creates a uniquely named Redact Object ODE Action, that blacks out an 
 * Object's background redacting the rectangle area
 * @param[in] name unique name for the Redact Object ODE Action
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_redact_new(const wchar_t* name);

/**
 * @brief Creates a uniquely named Add Sink Action that adds
 * a named Sink to a named Pipeline
 * @param[in] name unique name for the ODE Add Sink Action 
 * @param[in] pipeline unique name of the Pipeline to add the Source to
 * @param[in] sink unique name of the Sink to add to the Pipeline
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_sink_add_new(const wchar_t* name,
    const wchar_t* pipeline, const wchar_t* sink);

/**
 * @brief Creates a uniquely named Remove Sink Action that removes
 * a named Sink from a named Pipeline
 * @param[in] name unique name for the Sink Remove Action 
 * @param[in] pipeline unique name of the Pipeline to remove the Sink from
 * @param[in] sink unique name of the Sink to remove from the Pipeline
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_sink_remove_new(const wchar_t* name,
    const wchar_t* pipeline, const wchar_t* sink);

/**
 * @brief Creates a uniquely named Start Record Sink ODE Action
 * @param[in] name unique name for the Print ODE Action 
 * @param[in] record_sink unique name of the Record Sink to start recording
 * @param[in] start start time before current time in seconds
 * should be less the Record Sink's cache size
 * @param[in] duration duration of the recording in seconds
 * @param[in] client_data opaque pointer to client data
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_sink_record_start_new(const wchar_t* name,
    const wchar_t* record_sink, uint start, uint duration, void* client_data);

/**
 * @brief Creates a uniquely named Add Source Action that adds
 * a named Source to a named Pipeline
 * @param[in] name unique name for the ODE Add Action 
 * @param[in] pipeline unique name of the Pipeline to add the Source to
 * @param[in] source unique name of the Source to add to the Pipeline
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_source_add_new(const wchar_t* name,
    const wchar_t* pipeline, const wchar_t* source);

/**
 * @brief Creates a uniquely named Remove Source Action that removes
 * a named Source from a named Pipeline
 * @param[in] name unique name for the Source Remove Action 
 * @param[in] pipeline unique name of the Pipeline to remove the Source from
 * @param[in] source unique name of the Source to remove from the Pipeline
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_source_remove_new(const wchar_t* name,
    const wchar_t* pipeline, const wchar_t* source);

/**
 * @brief Creates a uniquely named Start Record Tap ODE Action
 * @param[in] name unique name for the Start Record Tap Action 
 * @param[in] record_tap unique name of the Record Tap to start recording
 * @param[in] start start time before current time in seconds
 * should be less the Record Taps's cache size
 * @param[in] duration duration of the recording in seconds
 * @param[in] client_data opaque pointer to client data
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_tap_record_start_new(const wchar_t* name,
    const wchar_t* record_sink, uint start, uint duration, void* client_data);

/**
 * @brief Creates a uniquely named Add Area ODE Action that adds
 * a named ODE Area to a named ODE Trigger on ODE occurrence
 * @param[in] name unique name for the Add Area ODE Action 
 * @param[in] trigger unique name of the ODE Trigger to add the ODE Area to
 * @param[in] area unique name of the ODE Area to add to the ODE Trigger
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_area_add_new(const wchar_t* name,
    const wchar_t* trigger, const wchar_t* area);

/**
 * @brief Creates a uniquely named Remove Area ODE Action that removes
 * a named ODE Area from a named ODE Trigger on ODE occurrence
 * @param[in] name unique name for the Remvoe Area ODE Action 
 * @param[in] trigger unique name of the ODE Trigger to remove the ODE Area from
 * @param[in] area unique name of the ODE Area to remove from the ODE Trigger
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_area_remove_new(const wchar_t* name,
    const wchar_t* trigger, const wchar_t* area);

/**
 * @brief Creates a uniquely named Disable Trigger ODE Action that disables
 * a named ODE Trigger on ODE occurrence
 * @param[in] name unique name for the Disable ODE Trigger Action 
 * @param[in] trigger unique name of the ODE Trigger to disable
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_trigger_disable_new(const wchar_t* name, const wchar_t* trigger);

/**
 * @brief Creates a uniquely named Enable Trigger ODE Action that enables
 * a named ODE Trigger on ODE occurrence
 * @param[in] name unique name for the ODE Trigger Enable Action 
 * @param[in] trigger unique name of the ODE Trigger to disable
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_trigger_enable_new(const wchar_t* name, const wchar_t* trigger);

/**
 * @brief Creates a uniquely named Reset Trigger ODE Action that disables
 * a named ODE Trigger on ODE occurrence
 * @param[in] name unique name for the Reset ODE Trigger Action 
 * @param[in] trigger unique name of the ODE Trigger to reset
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_trigger_reset_new(const wchar_t* name, const wchar_t* trigger);

/**
 * @brief Creates a uniquely named Disable Action ODE Action that disables
 * a named ODE Action on ODE occurrence
 * @param[in] name unique name for the Trigger Disable Action 
 * @param[in] action unique name of the ODE Action to disable
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_action_disable_new(const wchar_t* name, const wchar_t* action);

/**
 * @brief Creates a uniquely named Enable Action ODE Action that enables
 * a named ODE Action on ODE occurrence
 * @param[in] name unique name for the Enable Action ODE  Action 
 * @param[in] action unique name of the ODE Trigger to disable
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_action_enable_new(const wchar_t* name, const wchar_t* action);

/**
 * @brief Creates a uniquely named Tiler Show Source ODE Action that calls on the 
 * named Tiler to show the Source for the specific frame on ODE occurrence
 * @param[in] name unique name for the ODE Trigger Enable Action 
 * @param[in] tiler unique name of the Tiler to call to show-source
 * @param[in] timeout to pass to the Tiler on show-source
 * @param[in] has_precedence if true will take precedence over a currently show single source. 
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_tiler_source_show_new(const wchar_t* name, 
    const wchar_t* tiler, uint timeout, boolean has_precedence);

/**
 * @brief Gets the current enabled setting for the ODE Action
 * @param[in] name unique name of the ODE Action to query
 * @param[out] enabled true if the ODE Action is currently enabled, false otherwise
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_enabled_get(const wchar_t* name, boolean* enabled);

/**
 * @brief Sets the enabled setting for the ODE Action
 * @param[in] name unique name of the ODE Action to update
 * @param[in] enabled true if the ODE Action is currently enabled, false otherwise
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_enabled_set(const wchar_t* name, boolean enabled);

/**
 * @brief Deletes an ODE Action of any type
 * This service will fail with DSL_RESULT_ODE_ACTION_IN_USE if the Action is currently
 * owned by a ODE Trigger.
 * @param[in] name unique name of the ODE Action to delete
 * @return DSL_RESULT_SUCCESS on success, on of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_delete(const wchar_t* name);

/**
 * @brief Deletes a Null terminated array of ODE Actions of any type
 * This service will fail with DSL_RESULT_ODE_ACTION_IN_USE if any of the Actions 
 * are currently owned by a ODE Trigger.
 * @param[in] names Null ternimated array of unique names to delete
 * @return DSL_RESULT_SUCCESS on success, on of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_delete_many(const wchar_t** names);

/**
 * @brief Deletes all ODE Actions of all types
 * This service will fail with DSL_RESULT_ODE_ACTION_IN_USE if any of the Actions 
 * are currently owned by a ODE Trigger.
 * @return DSL_RESULT_SUCCESS on success, on of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_delete_all();

/**
 * @brief Returns the size of the list of ODE Actions
 * @return the number of ODE Actions in the list
 */
uint dsl_ode_action_list_size();

/**
 * @brief Creates a uniquely named ODE Inclusion Area
 * @param[in] name unique name of the ODE area to create
 * @param[in] rectangle name of an RGBA Display Rectangle
 * @param[in] display set to true to display (overlay) the rectangle on each frame
 * @return DSL_RESULT_SUCCESS on successful create, DSL_RESULT_ODE_AREA_RESULT otherwise.
 */
DslReturnType dsl_ode_area_inclusion_new(const wchar_t* name, 
    const wchar_t* rectangle, boolean display);

/**
 * @brief Creates a uniquely named ODE Inclusion Area
 * @param[in] name unique name of the ODE area to create
 * @param[in] rectangle name of an RGBA Display Rectangle
 * @param[in] display set to true to display (overlay) the rectangle on each frame
 * @return DSL_RESULT_SUCCESS on successful create, DSL_RESULT_ODE_AREA_RESULT otherwise.
 */
DslReturnType dsl_ode_area_exclusion_new(const wchar_t* name, 
    const wchar_t* rectangle, boolean display);

/**
 * @brief Deletes an ODE Area
 * This service will fail with DSL_RESULT_ODE_ACTION_IN_USE if the Area is currently
 * owned by a ODE Trigger.
 * @param[in] name unique name of the ODE Area to delete
 * @return DSL_RESULT_SUCCESS on success, on of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_area_delete(const wchar_t* name);

/**
 * @brief Deletes a Null terminated array of ODE Areas of any type
 * This service will fail with DSL_RESULT_ODE_ACTION_IN_USE if any of the Areas 
 * are currently owned by a ODE Trigger.
 * @param[in] names Null ternimated array of unique names to delete
 * @return DSL_RESULT_SUCCESS on success, on of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_area_delete_many(const wchar_t** names);

/**
 * @brief Deletes all ODE Areas of all types
 * This service will fail with DSL_RESULT_ODE_ACTION_IN_USE if any of the Areas 
 * are currently owned by a ODE Trigger.
 * @return DSL_RESULT_SUCCESS on success, on of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_area_delete_all();

/**
 * @brief Returns the size of the list of ODE Areas
 * @return the number of ODE Actions in the list
 */
uint dsl_ode_area_list_size();

/**
 * @brief Frame-Meta trigger that triggers for every Frame metadata, always. 
 * Note, this is a No-Limit trigger, and setting a Class ID filer will have no effect.
 * Although always triggered, the client selects whether to Trigger an ODE occurrence
 * before (pre) or after (post) processing all Object metadata for all other Triggers.
 * @param[in] name unique name for the ODE Trigger
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE
 * @param[in] when DSL_PRE_CHECK_FOR_OCCURRENCES or DSL_POST_CHECK_FOR_OCCURRENCES
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_always_new(const wchar_t* name, const wchar_t* source, uint when);

/**
 * @brief Occurence trigger that checks for the occurrence of Objects within a frame for a 
 * specified source and object class_id.
 * @param[in] name unique name for the ODE Trigger
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE
 * @param[in] class_id class id filter for this ODE Trigger
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_occurrence_new(const wchar_t* name, const wchar_t* source, uint class_id, uint limit);

/**
 * @brief Absence trigger that checks for the absence of Objects within a frame
 * @param[in] name unique name for the ODE Trigger
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE
 * @param[in] class_id class id filter for this ODE Trigger
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_absence_new(const wchar_t* name, const wchar_t* source, uint class_id, uint limit);

/**
 * @brief Intersection trigger that checks for intersection of all Object detected
 * and triggers an ODE occurrence for each unique overlaping pair.
 * @param[in] name unique name for the ODE Trigger
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE
 * @param[in] class_id class id filter for this ODE Trigger
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_intersection_new(const wchar_t* name, const wchar_t* source, uint class_id, uint limit);

/**
 * @brief Summation trigger that checks for and sums all objects detected within a frame
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE
 * @param[in] name unique name for the ODE Trigger
 * @param[in] class_id class id filter for this ODE Trigger
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
 
DslReturnType dsl_ode_trigger_summation_new(const wchar_t* name, const wchar_t* source, uint class_id, uint limit);

/**
 * @brief Custom ODE Trigger that allows the client to provide a custom "check-for-occurrence' function
 * to be called with Frame Meta and Object Meta data for every object that meets the trigger's
 * criteria: class id, min dimensions, min confidence, etc. The Client can maitain and test with
 * their own criteria, running stats etc, managed with client_data.
 * @param[in] name unique name for the ODE Trigger
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE
 * @param[in] class_id class id filter for this ODE Trigger
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit
 * @param[in] client_checker client custom callback function to Check for the occurrence
 * of an ODE. Set this parameter to NULL to omit object checking/
 * @param[in] client_post_processor client custom callback function to Check for the occurrence
 * of an ODE after all objects have be checked. Set to NULL to omit post processing of Frame metadata
 * @param[in] client_data opaque client data returned to the client on callback
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_custom_new(const wchar_t* name, const wchar_t* source, 
    uint class_id, uint limit, dsl_ode_check_for_occurrence_cb client_checker, 
    dsl_ode_post_process_frame_cb client_post_processor, void* client_data);

/**
 * @brief Miniumu occurence trigger that checks for the occurrence of Objects within a frame
 * against a specified minimum number, and generates an ODE occurence if not met
 * @param[in] name unique name for the ODE Trigger
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE
 * @param[in] class_id class id filter for this ODE Trigger
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit
 * @param[in] minimum the minimum count that must be present before triggering an ODE occurence
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_minimum_new(const wchar_t* name, const wchar_t* source, 
    uint class_id, uint limit, uint minimum);

/**
 * @brief Maximum occurence trigger that checks for the occurrence of Objects within a frame
 * against a specified maximum number, and generates an ODE occurence if exceeded
 * @param[in] name unique name for the ODE Trigger
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE
 * @param[in] class_id class id filter for this ODE Trigger
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit
 * @param[in] maximum the maximum count allowed without triggering ODE occurence
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_maximum_new(const wchar_t* name, const wchar_t* source, 
    uint class_id, uint limit, uint maximum);

/**
 * @brief Range occurence trigger that checks for the occurrence of Objects within a frame
 * against a range of numbers, and generates an ODE occurence if within range
 * @param[in] name unique name for the ODE Trigger
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE
 * @param[in] class_id class id filter for this ODE Trigger
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit
 * @param[in] lower the lower range for triggering ODE occurence
 * @param[in] upper the upper range for triggering ODE occurence
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_range_new(const wchar_t* name, const wchar_t* source, 
    uint class_id, uint limit, uint lower, uint upper);

/**
 * @brief Smallest trigger that checks for the occurrence of Objects within a frame
 * and if at least one is found, Triggers on the Object with smallest rectangle area.
 * @param[in] name unique name for the ODE Trigger
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE
 * @param[in] class_id class id filter for this ODE Trigger
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_smallest_new(const wchar_t* name, const wchar_t* source, uint class_id, uint limit);

/**
 * @brief Largest trigger that checks for the occurrence of Objects within a frame
 * and if at least one is found, Triggers on the Object with larget rectangle area.
 * @param[in] name unique name for the ODE Trigger
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE
 * @param[in] class_id class id filter for this ODE Trigger
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_largest_new(const wchar_t* name, const wchar_t* source, uint class_id, uint limit);


/**
 * @brief Resets the a named ODE Trigger, setting it's triggered count to 0
 * This affects Triggers with fixed limits, whether they have reached their limit or not.
 * @param[in] name unique name of the ODE Trigger to update
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_reset(const wchar_t* name);

/**
 * @brief Gets the current enabled setting for the ODE Trigger
 * @param[in] name unique name of the ODE Trigger to query
 * @param[out] enabled true if the ODE Trigger is currently enabled, false otherwise
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_enabled_get(const wchar_t* name, boolean* enabled);

/**
 * @brief Sets the enabled setting for the ODE Trigger
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] enabled true if the ODE Trigger is currently enabled, false otherwise
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_enabled_set(const wchar_t* name, boolean enabled);

/**
 * @brief Gets the current source_id filter for the ODE Trigger
 * A value of 0 indicates filter disabled
 * @param[in] name unique name of the ODE Trigger to query
 * @param[out] source returns the current source name in use
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_source_get(const wchar_t* name, const wchar_t** source);

/**
 * @brief Sets the source_id for the ODE Trigger to filter on
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] source new source name to filter on
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_source_set(const wchar_t* name, const wchar_t* source);

/**
 * @brief Gets the current class_id filter for the ODE Trigger
 * @param[in] name unique name of the ODE Trigger to query
 * @param[out] class_id returns the current class_id in use
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_class_id_get(const wchar_t* name, uint* class_id);

/**
 * @brief Sets the class_id for the ODE Trigger to filter on
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] class_id new class_id to use
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_class_id_set(const wchar_t* name, uint class_id);

/**
 * @brief Gets the current minimum confidence setting for the ODE Trigger
 * A value of 0.0 (default) indicates the minimum confidence criteria is disabled
 * @param[in] name unique name of the ODE Trigger to query
 * @param[out] min_confidence current minimum confidence criteria
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_confidence_min_get(const wchar_t* name, float* min_confidence);

/**
 * @brief Sets the enabled setting for the ODE Trigger
 * Setting the value of 0.0 indicates the minimum confidence criteria is disabled
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] min_confidence minimum confidence to trigger an ODE occurrnce
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_confidence_min_set(const wchar_t* name, float min_confidence);

/**
 * @brief Gets the current minimum rectangle width and height values for the ODE Trigger
 * A value of 0 = no minimum
 * @param[in] name unique name of the ODE Trigger to query
 * @param[out] min_width returns the current minimun frame width in use
 * @param[out] min_height returns the current minimun frame hight in use
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_dimensions_min_get(const wchar_t* name, float* min_width, float* min_height);

/**
 * @brief Sets the current minimum rectangle width and height values for the ODE Trigger
 * A value of 0 = no minimum
 * @param[in] name unique name of the ODE Trigger to query
 * @param[in] min_width the new minimun frame width to use
 * @param[in] min_height the new minimun frame hight to use
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_dimensions_min_set(const wchar_t* name, float min_width, float min_height);

/**
 * @brief Gets the current maximum rectangle width and height values for the ODE Trigger
 * A value of 0 = no maximum
 * @param[in] name unique name of the ODE Trigger to query
 * @param[out] max_width returns the current maximun frame width in use
 * @param[out] max_height returns the current maximun frame hight in use
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_dimensions_max_get(const wchar_t* name, float* max_width, float* max_height);

/**
 * @brief Sets the current maximum rectangle width and height values for the ODE Trigger
 * A value of 0 = no maximum
 * @param[in] name unique name of the ODE Trigger to query
 * @param[in] max_width the new maximun frame width to use
 * @param[in] max_height the new maximun frame hight to use
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_dimensions_max_set(const wchar_t* name, float max_width, float max_height);

/**
 * @brief Gets the current Inferrence-Done-Only setting for the named trigger
 * @param[in] name unique name of the ODE Trigger to query
 * @param[in] infer_done_only if true, then Inference Done will become minimum criteria
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_infer_done_only_get(const wchar_t* name, boolean* infer_done_only);

/**
 * @brief Sets the current Inferrence-Done-Only setting for the named trigger
 * @param[in] name unique name of the ODE Trigger to query
 * @param[in] infer_done_only if true, then Inference Done will become minimum criteria
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_infer_done_only_set(const wchar_t* name, boolean infer_done_only);

/**
 * @brief Gets the current min frame count (detected in last N out of D frames) for the ODE Trigger
 * A value of 0 = no minimum
 * @param[in] name unique name of the ODE Trigger to query
 * @param[out] min_count_n returns the current minimun frame count numerator in use
 * @param[out] min_count_d returns the current minimun frame count denomintor in use
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_frame_count_min_get(const wchar_t* name, uint* min_count_n, uint* min_count_d);

/**
 * @brief Sets the current min frame count (detected in last N out of D frames) for the ODE Trigger
 * A value of 0 = no minimum
 * @param[in] name unique name of the ODE Trigger to query
 * @param[out] min_count_n sets the current minimun frame count numerator to use
 * @param[out] min_count_d sets the current minimun frame count denomintor to use
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_frame_count_min_set(const wchar_t* name, uint min_count_n, uint min_count_d);

/**
 * @brief Adds a named ODE Action to a named ODE Trigger
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] action unique name of the ODE Action to Add
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_action_add(const wchar_t* name, const wchar_t* action);

/**
 * @brief Adds a Null terminated list of named ODE Actions to a named ODE Trigger
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] actions Null terminated list of unique names of the ODE Actions to add
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_action_add_many(const wchar_t* name, const wchar_t** actions);

/**
 * @brief Removes a named ODE Action from a named ODE Trigger
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] action unique name of the ODE Action to Remove
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_action_remove(const wchar_t* name, const wchar_t* action);

/**
 * @brief Removes a Null terminated list of named ODE Actions from a named ODE Trigger
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] actions Null terminated list of unique names of the ODE Actions to remove
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_action_remove_many(const wchar_t* name, const wchar_t** actions);

/**
 * @brief Removes a named ODE Action from a named ODE Trigger
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] action unique name of the ODE Action to Remove
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_action_remove_all(const wchar_t* name);

/**
 * @brief Adds a named ODE Area to a named ODE Trigger
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] area unique name of the ODE Area to Add
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_area_add(const wchar_t* name, const wchar_t* area);

/**
 * @brief Adds a Null terminated list of named ODE Areas to a named ODE Trigger
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] areas Null terminated list of unique names of the ODE Areas to add
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_area_add_many(const wchar_t* name, const wchar_t** areas);

/**
 * @brief Removes a named ODE Area from a named ODE Trigger
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] area unique name of the ODE Area to Remove
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_area_remove(const wchar_t* name, const wchar_t* area);

/**
 * @brief Removes a Null terminated list of named ODE Areas from a named ODE Trigger
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] areas Null terminated list of unique names of the ODE Areas to remove
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_area_remove_many(const wchar_t* name, const wchar_t** areas);

/**
 * @brief Removes a named ODE Area from a named ODE Trigger
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] area unique name of the ODE Area to Remove
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_area_remove_all(const wchar_t* name);

/**
 * @brief Deletes a uniquely named Trigger. The call will fail if the event is currently in use
 * @brief[in] name unique name of the event to delte
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_delete(const wchar_t* name);

/**
 * @brief Deletes a Null terminated list of Triggers. The call will fail if any of the events are currently in use
 * @brief[in] names Null terminaed list of event names to delte
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_delete_many(const wchar_t** names);

/**
 * @brief Deletes all Triggers. The call will fail if any of the events are currently in use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_delete_all();

/**
 * @brief Returns the size of the list of Triggers
 * @return the number of Triggers in the list
 */
uint dsl_ode_trigger_list_size();

/**
 * @brief creates a new, uniquely named Handler component
 * @param[in] name unique name for the new Handler
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_HANDLER_RESULT otherwise
 */
DslReturnType dsl_pph_ode_new(const wchar_t* name);

/**
 * @brief Adds a named ODE Trigger to a named ODE Handler Component
 * @param[in] name unique name of the ODE Handler to update
 * @param[in] trigger unique name of the Trigger to add
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_HANDLER_RESULT otherwise
 */
DslReturnType dsl_pph_ode_trigger_add(const wchar_t* name, const wchar_t* trigger);

/**
 * @brief Adds a Null terminated listed of named ODE Triggers to a named ODE Handler Component
 * @param[in] name unique name of the ODE Handler to update
 * @param[in] triggers Null terminated list of Trigger names to add
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_HANDLER_RESULT otherwise
 */
DslReturnType dsl_pph_ode_trigger_add_many(const wchar_t* name, const wchar_t** triggers);

/**
 * @brief Removes a named ODE Trigger from a named ODE Handler Component
 * @param[in] name unique name of the ODE Handler to update
 * @param[in] odeType unique name of the Trigger to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_HANDLER_RESULT otherwise
 */
DslReturnType dsl_pph_ode_trigger_remove(const wchar_t* name, const wchar_t* trigger);

/**
 * @brief Removes a Null terminated listed of named ODE Triggers from a named ODE Handler Component
 * @param[in] name unique name of the ODE Handler to update
 * @param[in] triggers Null terminated list of Trigger names to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_HANDLER_RESULT otherwise
 */
DslReturnType dsl_pph_ode_trigger_remove_many(const wchar_t* name, const wchar_t** triggers);

/**
 * @brief Removes all ODE Triggers from a named ODE Handler Component
 * @param[in] name unique name of the ODE Handler to update
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_HANDLER_RESULT otherwise
 */
DslReturnType dsl_pph_ode_trigger_remove_all(const wchar_t* name);

/**
 * @brief creates a new, uniquely named Custom pad-probe-handler to process a buffer
 * @param[in] name unique component name for the new Custom Handler
 * @param[in] client_handler client callback function, called with each buffer that flows over the pad
 * @param[in] client_data opaque pointer to client date returned with the callback
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PPH_RESULT otherwise
 */
DslReturnType dsl_pph_custom_new(const wchar_t* name,
     dsl_pph_custom_client_handler_cb client_handler, void* client_data);
     
/**
 * @brief creates a new, uniquely named Meter pad-probe-handler to calcaulate performance measurements
 * @param[in] name unique component name for the new Meter
 * @param[in] interval interval at which to report performance measurements
 * @param[in] client_handler client callback function, called at "interval" with 
 * performance measurements for each source
 * @param[in] client_data opaque pointer to client date returned with the callback
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PPH_RESULT otherwise
 */
DslReturnType dsl_pph_meter_new(const wchar_t* name, uint interval,
    dsl_pph_meter_client_handler_cb client_handler, void* client_data);
/**
 * @brief gets the current reporting interval for the named Meter Sink
 * @param[in] name unique name of the Meter Sink to query
 * @param[out] interval the current reporting interval in seconds
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PPH_RESULT otherwise
 */
DslReturnType dsl_pph_meter_interval_get(const wchar_t* name, uint* interval);

/**
 * @brief sets the current reportings for the named Meter Sink
 * @param[in] name unique name of the Meter Sink to query
 * @param[out] interval new reporting interval in seconds.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PPH_RESULT otherwise
 */
DslReturnType dsl_pph_meter_interval_set(const wchar_t* name, uint interval);

/**
 * @brief gets the current enabled setting for the named Pad Probe Handler
 * @param[in] name unique name of the Handler to query
 * @param[out] enabled true if the Handler is enabled, false otherwise
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PPH_RESULT otherwise
 */
DslReturnType dsl_pph_enabled_get(const wchar_t* name, boolean* enabled);

/**
 * @brief Sets the Pad Probe Handler's enabled setting
 * @param[in] name unique name of the pad-probe-handler to update
 * @param[out] enabled set true to enable, if in a disabled state, 
 * false to disable if currently in an enbled state. 
 * Attempts to reset to the same/current state will fail
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PPH_RESULT otherwise
 */
DslReturnType dsl_pph_enabled_set(const wchar_t* name, boolean enabled);


/**
 * @brief Deletes a uniquely named Pad Probe Handler. The call will fail if the Handler is currently in use
 * @brief[in] name unique name of the Handler to delte
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_HANDLER_RESULT otherwise.
 */
DslReturnType dsl_pph_delete(const wchar_t* name);

/**
 * @brief Deletes a Null terminated list of Pad Probe Handlers. 
 * The call will fail if any of the Handlers are currently in use
 * @brief[in] names Null terminaed list of Handler names to delte
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_HANDLER_RESULT otherwise.
 */
DslReturnType dsl_pph_delete_many(const wchar_t** names);

/**
 * @brief Deletes all Pad Probe Handlers. The call will fail if any of the Handlers are currently in use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_pph_delete_all();

/**
 * @brief Returns the size of the list of Pad Probe HandlerS
 * @return the number of Handlers in the list
 */
uint dsl_pph_list_size();

/**
 * @brief creates a new, uniquely named CSI Camera Source component
 * @param[in] name unique name for the new Source
 * @param[in] width width of the source in pixels
 * @param[in] height height of the source in pixels
 * @param[in] fps-n frames/second fraction numerator
 * @param[in] fps-d frames/second fraction denominator
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_csi_new(const wchar_t* name,
    uint width, uint height, uint fps_n, uint fps_d);

/**
 * @brief creates a new, uniquely named USB Camera Source component
 * @param[in] name unique name for the new Source
 * @param[in] width width of the source in pixels
 * @param[in] height height of the source in pixels
 * @param[in] fps-n frames/second fraction numerator
 * @param[in] fps-d frames/second fraction denominator
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_usb_new(const wchar_t* name,
    uint width, uint height, uint fps_n, uint fps_d);

/**
 * @brief creates a new, uniquely named URI Source component
 * @param[in] name Unique Resource Identifier (file or live)
 * @param[in] is_live true if source is live false if file
 * @param[in] cudadec_mem_type, use DSL_CUDADEC_MEMORY_TYPE_<type>
 * @param[in] 
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_uri_new(const wchar_t* name, const wchar_t* uri, boolean is_live,
    uint cudadec_mem_type, uint intra_decode, uint drop_frame_interval);

/**
 * @brief creates a new, uniquely named RTSP Source component
 * @param[in] name Unique Resource Identifier (file or live)
 * @param[in] protocol one of the constant protocol values [ DSL_RTP_TCP | DSL_RTP_ALL ]
 * @param[in] cudadec_mem_type, use DSL_CUDADEC_MEMORY_TYPE_<type>
 * @param[in] intra_decode set to True to enable, false to disable
 * @param[in] drop_frame_interval, set to 0 to decode every frame.
 * @param[in] latency in milliseconds
 * @param[in] timeout time to wait between successive frames before determining the 
 * connection is lost. Set to 0 to disable timeout.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_rtsp_new(const wchar_t* name, const wchar_t* uri, uint protocol,
    uint cudadec_mem_type, uint intra_decode, uint drop_frame_interval, uint latency, uint timeout);

/**
 * @brief returns the frame rate of the name source as a fraction
 * Camera sources will return the value used on source creation
 * URL and RTPS sources will return 0 until prior entering a state of play
 * @param[in] name unique name of the source to query
 * @param[out] width of the source in pixels
 * @param[out] height of the source in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_dimensions_get(const wchar_t* name, uint* width, uint* height);

/**
 * @brief returns the frame rate of the named source as a fraction
 * Camera sources will return the value used on source creation
 * URL and RTPS sources will return 0 until prior entering a state of play
 * @param[in] name unique name of the source to query
 * @param[out] fps_n frames per second numerator
 * @param[out] fps_d frames per second denominator
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_frame_rate_get(const wchar_t* name, uint* fps_n, uint* fps_d);

/**
 * @brief Gets the current URI in use by the named Decode Source
 * @param[in] name name of the Source to query
 * @param[out] uri in use by the Decode Source
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_decode_uri_get(const wchar_t* name, const wchar_t** uri);

/**
 * @brief Sets the current URI for the named Decode Source to use
 * @param[in] name name of the Source to update
 * @param[out] uri in use by the Decode Source
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_decode_uri_set(const wchar_t* name, const wchar_t* uri);

/**
 * @brief Adds a named dewarper to a named decode source (URI, RTSP)
 * @param[in] name name of the source object to update
 * @param[in] dewarper name of the dewarper to add
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_decode_dewarper_add(const wchar_t* name, const wchar_t* dewarper);

/**
 * @brief Adds a named dewarper to a named decode source (URI, RTSP)
 * @param[in] name name of the source object to update
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_decode_dewarper_remove(const wchar_t* name);

/**
 * @brief Gets the current buffer timeout for the named RTSP Source
 * @param[in] name name of the source object to query
 * @param[out] timeout current time to wait between successive frames before determining the 
 * connection is lost. If set to 0 then timeout is disabled.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_rtsp_timeout_get(const wchar_t* name, uint* timeout);

/**
 * @brief Sets the current buffer timeout for the named RTSP Source
 * @param[in] name name of the source object to update
 * @param[in] timeout time to wait between successive frames before determining the 
 * connection is lost. Set to 0 to disable timeout.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_rtsp_timeout_set(const wchar_t* name, uint timeout);

/**
 * @brief Gets the current reconnection params in use by the named RTSP Source. The parameters are set
 * to DSL_RTSP_RECONNECT_SLEEP_TIME_MS and DSL_RTSP_RECONNECT_TIMEOUT_MS on source creation.
 * @param[in] name name of the source object to query.
 * @param[out] sleep time, in unit of seconds, to sleep between successively checking the status of the asynchrounus reconnection
 * Also, the retry_sleep time is used after a state change failure
 * @param[out] timeout time, in units of seconds, to wait before terminating the current reconnection try and
 * restarting the reconnection cycle again.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_rtsp_reconnection_params_get(const wchar_t* name, uint* sleep, uint* timeout);

/**
 * @brief Sets the current reconnection params in use by the named RTSP Source. The parameters are set
 * to DSL_RTSP_RECONNECT_SLEEP_TIME and DSL_RTSP_RECONNECT_TIMEOUT on source creation.
 * Note: calling this service while a reconnection cycle is in progess will terminate
 * the current cycle before restarting with the new parmeters.
 * @param[in] retry_sleep time, in unit of seconds, to sleep between successively checking the status of the asynchrounus reconnection
 * Also, the retry_sleep time is used after a state change failure
 * @param[in] retry_timeout time, in units of seconds, to wait before terminating the current reconnection try and
 * restarting the reconnection cycle again.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_rtsp_reconnection_params_set(const wchar_t* name, uint sleep, uint timeout);

/**
 * @brief Gets the current connection stats for the named RTSP Source
 * @param[in] name name of the source object to query
 * @param[out] data the current Connection Stats and Params for the Source. 
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_rtsp_connection_data_get(const wchar_t* name, dsl_rtsp_connection_data* data); 

/**
 * @brief Clears the connection stats for the named RTSP Source.
 * Note: "retries" will not be cleared if is_in_reset == true
 * @param[in] name name of the source object to update
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_rtsp_connection_stats_clear(const wchar_t* name); 

/**
 * @brief adds a callback to be notified on change of RTSP Source state
 * @param[in] name name of the RTSP source to update
 * @param[in] listener pointer to the client's function to call on state change
 * @param[in] client_data opaque pointer to client data passed into the listener function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_rtsp_state_change_listener_add(const wchar_t* name, 
    dsl_state_change_listener_cb listener, void* client_data);

/**
 * @brief removes a callback previously added with dsl_source_rtsp_state_change_listener_add
 * @param[in] name unique name of the RTSP source to update
 * @param[in] listener pointer to the client's function to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_rtsp_state_change_listener_remove(const wchar_t* name, 
    dsl_state_change_listener_cb listener);

/**
 * @brief Adds a named Tap to a named RTSP source
 * @param[in] name name of the source object to update
 * @param[in] tap name of the Tap to add
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_rtsp_tap_add(const wchar_t* name, const wchar_t* tap);

/**
 * @brief Adds a named dewarper to a named decode source (URI, RTSP)
 * @param[in] name name of the source object to update
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_rtsp_tap_remove(const wchar_t* name);

/**
 * @brief returns the name of a Source component from a unqiue Source Id
 * @param[in] source_id unique Source Id to check for
 * @param[out] name the name of Source component if found
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_name_get(uint source_id, const wchar_t** name);

/**
 * @brief pauses a single Source object if the Source is 
 * currently in a state of in-use and Playing..
 * @param[in] name the name of Source component to pause
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_pause(const wchar_t* name);

/**
 * @brief resumes a single Source object if the Source is 
 * currently in a state of in-use and Paused..
 * @param[in] name the name of Source component to resume
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_resume(const wchar_t* name);

/**
 * @brief returns whether the source stream is live or not
 * @param[in] name the name of Source component to query
 * @return True if the source's stream is live
 */
boolean dsl_source_is_live(const wchar_t* name);

/**
 * @brief returns the number of Sources currently in use by 
 * all Pipelines in memory. 
 * @return the current number of Sources in use
 */
uint dsl_source_num_in_use_get();  

/**
 * @brief Returns the maximum number of sources that can be in-use
 * by all parent Pipelines at one time. The maximum number is 
 * impossed by the Jetson hardware in use, see dsl_source_num_in_use_max_set() 
 * @return the current sources in use max setting.
 */
uint dsl_source_num_in_use_max_get();  

/**
 * @brief Sets the maximum number of in-memory sources 
 * that can be in use at any time. The function overrides 
 * the default value on first call. The maximum number is 
 * limited by Hardware. The caller must ensure to set the 
 * number correctly based on the Jetson hardware in use.
 */
boolean dsl_source_num_in_use_max_set(uint max);  

/**
 * @brief create a new, uniquely named Dewarper object
 * @param[in] name unique name for the new Dewarper object
 * @param[in] config_file absolute or relative path to Dewarper config text file
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_GIE_RESULT otherwise.
 */
DslReturnType dsl_dewarper_new(const wchar_t* name, const wchar_t* config_file);

/**
 * @brief creates a new, uniquely named Record Tap component
 * @param[in] name unique component name for the new Record Tap
 * @param[in] outdir absolute or relative path to the recording output dir.
 * @param[in] container one of DSL_MUXER_MPEG4 or DSL_MUXER_MK4
 * @param[in] client_listener client callback for end-of-sesssion notifications.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_tap_record_new(const wchar_t* name, const wchar_t* outdir, 
    uint container, dsl_record_client_listener_cb client_listener);
     
/**
 * @brief starts a new recording session for the named Record Tap
 * @param[in] name unique of the Record Tap to start the session
 * @param[in] start start time in seconds before the current time
 * should be less that the video cache size
 * @param[in] duration in seconds from the current time to record.
 * @param[in] client_data opaque pointer to client data returned
 * on callback to the client listener function provided on Tap creation
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_tap_record_session_start(const wchar_t* name,
    uint start, uint duration, void* client_data);

/**
 * @brief stops a current recording in session
 * @param[in] name unique of the Record Tap to stop
 * @param[in] session unique id for the session to stop
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TAP_RESULT on failure
 */
DslReturnType dsl_tap_record_session_stop(const wchar_t* name);

/**
 * @brief returns the video recording cache size in units of seconds
 * A fixed size cache is created when the Pipeline is linked and played. 
 * The default cache size is set to DSL_DEFAULT_VIDEO_RECORD_CACHE_IN_SEC
 * @param[in] name name of the Record Tap to query
 * @param[out] cache_size current cache size setting
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TAP_RESULT
 */
DslReturnType dsl_tap_record_cache_size_get(const wchar_t* name, uint* cache_size);

/**
 * @brief sets the video recording cache size in units of seconds
 * A fixed size cache is created when the Pipeline is linked and played. 
 * The default cache size is set to DSL_DEFAULT_VIDEO_RECORD_CACHE_IN_SEC
 * @param[in] name name of the Record Tap to update
 * @param[in] cache_size new cache size setting to use on Pipeline play
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TAP_RESULT
 */
DslReturnType dsl_tap_record_cache_size_set(const wchar_t* name, uint cache_size);

/**
 * @brief returns the dimensions, width and height, used for the video recordings
 * @param[in] name name of the Record Tap to query
 * @param[out] width current width of the video recording in pixels
 * @param[out] height current height of the video recording in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TAP_RESULT
 */
DslReturnType dsl_tap_record_dimensions_get(const wchar_t* name, uint* width, uint* height);

/**
 * @brief sets the dimensions, width and height, for the video recordings created
 * values of zero indicate no-transcodes
 * @param[in] name name of the Record Tap to update
 * @param[in] width width to set the video recording in pixels
 * @param[in] height height to set the video in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TAP_RESULT
 */
DslReturnType dsl_tap_record_dimensions_set(const wchar_t* name, uint width, uint height);

/**
 * @brief returns the current recording state of the Record Tap
 * @param[in] name name of the Record Tap to query
 * @param[out] is_on true if the Record Tap is currently recording a session, false otherwise
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_TAP_RESULT on failure
 */
DslReturnType dsl_tap_record_is_on_get(const wchar_t* name, boolean* is_on);

/**
 * @brief returns the current state of the Record Tap reset done flag
 * @param[in] name name of the Record Tap to query
 * @param[out] is_on true if Reset has been done, false otherwise
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_TAP_RESULT on failure
 */
DslReturnType dsl_tap_record_reset_done_get(const wchar_t* name, boolean* reset_done);


/**
 * @brief creates a new, uniquely named Primary GIE object
 * @param[in] name unique name for the new GIE object
 * @param[in] infer_config_file pathspec of the Infer Config file to use
 * @param[in] model_engine_file pathspec of the Model Engine file to use
 * @param[in] interval frame interval to infer on. 0 = every frame, 
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_GIE_RESULT otherwise.
 */
DslReturnType dsl_gie_primary_new(const wchar_t* name, const wchar_t* infer_config_file,
    const wchar_t* model_engine_file, uint interval);

/**
 * @brief Adds a pad-probe-handler to be called to process each frame buffer.
 * A Primary GIE can have multiple Sink and Source pad-probe-handlers
 * @param[in] name unique name of the Primary GIE to update
 * @param[in] handler callback function to process pad probe data
 * @param[in] pad pad to add the handler to; DSL_PAD_SINK | DSL_PAD SRC
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_GIE_RESULT otherwise
 */
DslReturnType dsl_gie_primary_pph_add(const wchar_t* name, const wchar_t* handler, uint pad);

/**
 * @brief Removes a pad-probe-handler from the Primary GIE
 * @param[in] name unique name of the Primary GIE to update
 * @param[in] handler pad-probe-handler to remove
 * @param[in] pad pad to remove the handler from; DSL_PAD_SINK | DSL_PAD SRC
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_GIE_RESULT otherwise
 */
DslReturnType dsl_gie_primary_pph_remove(const wchar_t* name, const wchar_t* handler, uint pad);

/**
 * @brief creates a new, uniquely named Secondary GIE object
 * @param[in] name unique name for the new GIE object
 * @param[in] infer_config_file pathspec of the Infer Config file to use
 * @param[in] model_engine_file pathspec of the Model Engine file to use
 * @param[in] infer_on_gie name of the Primary of Secondary GIE to infer on
 * @param[in] interval frame interval to infer on. 0 = every frame, 
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_GIE_RESULT otherwise.
 */
DslReturnType dsl_gie_secondary_new(const wchar_t* name, const wchar_t* infer_config_file,
    const wchar_t* model_engine_file, const wchar_t* infer_on_gie, uint interval);

/**
 * @brief Gets the current Infer Config File in use by the named Primary or Secondary GIE
 * @param[in] name of Primary or Secondary GIE to query
 * @param[out] infer_config_file Infer Config file currently in use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_GIE_RESULT otherwise.
 */
DslReturnType dsl_gie_infer_config_file_get(const wchar_t* name, const wchar_t** infer_config_file);

/**
 * @brief Sets the Infer Config File to use by the named Primary or Secondary GIE
 * @param[in] name of Primary or Secondary GIE to update
 * @param[in] infer_config_file new Infer Config file to use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_GIE_RESULT otherwise.
 */
DslReturnType dsl_gie_infer_config_file_set(const wchar_t* name, const wchar_t* infer_config_file);

/**
 * @brief Gets the current Model Engine File in use by the named Primary or Secondary GIE
 * @param[in] name of Primary or Secondary GIE to query
 * @param[out] model_engi_file Model Engine file currently in use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_GIE_RESULT otherwise.
 */
DslReturnType dsl_gie_model_engine_file_get(const wchar_t* name, const wchar_t** model_engine_file);

/**
 * @brief Sets the Model Engine File to use by the named Primary or Secondary GIE
 * @param[in] name of Primary or Secondary GIE to update
 * @param[in] model_engine_file new Model Engine file to use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_GIE_RESULT otherwise.
 */
DslReturnType dsl_gie_model_engine_file_set(const wchar_t* name, const wchar_t* model_engine_file);

/**
 * @brief Gets the current Infer Interval in use by the named Primary or Secondary GIE
 * @param[in] name of Primary or Secondary GIE to query
 * @param[out] interval Infer interval value currently in use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_GIE_RESULT otherwise.
 */
DslReturnType dsl_gie_interval_get(const wchar_t* name, uint* interval);

/**
 * @brief Sets the Model Engine File to use by the named Primary or Secondary GIE
 * @param[in] name of Primary or Secondary GIE to update
 * @param[in] interval new Infer Interval value to use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_GIE_RESULT otherwise.
 */
DslReturnType dsl_gie_interval_set(const wchar_t* name, uint interval);

/**
 * @brief Enbles/disables the raw layer-info output to binary file for the named the GIE
 * @param[in] name name of the Primary or Secondary GIE to update
 * @param[in] enabled set to true to enable frame-to-file output for each GIE layer
 * @param[in] path absolute or relative direcory path to write to. 
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_GIE_RESULT otherwise.
 */
DslReturnType dsl_gie_raw_output_enabled_set(const wchar_t* name, boolean enabled, const wchar_t* path);

/**
 * @brief creates a new, uniquely named KTL Tracker object
 * @param[in] name unique name for the new Tracker
 * @param[in] max_width maximum frame width of the input transform buffer
 * @param[in] max_height maximum_frame height of the input tranform buffer
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_ktl_new(const wchar_t* name, uint max_width, uint max_height);

/**
 * @brief creates a new, uniquely named IOU Tracker object
 * @param[in] name unique name for the new Tracker
 * @param[in] config_file fully qualified pathspec to the IOU Lib config text file
 * @param[in] max_width maximum frame width of the input transform buffer
 * @param[in] max_height maximum_frame height of the input tranform buffer
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_iou_new(const wchar_t* name, const wchar_t* config_file, uint max_width, uint max_height);

/**
 * @brief returns the current maximum frame width and height settings for the named IOU Tracker object
 * @param[in] name unique name of the Tracker to query
 * @param[out] max_width maximum frame width of the input transform buffer
 * @param[out] max_height maximum_frame height of the input tranform buffer
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_max_dimensions_get(const wchar_t* name, uint* max_width, uint* max_height);

/**
 * @brief sets the maximum frame width and height settings for the named IOU Tracker object
 * @param[in] name unique name of the Tracker to update
 * @param[in] max_width new maximum frame width of the input transform buffer
 * @param[in] max_height new maximum_frame height of the input tranform buffer
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_max_dimensions_set(const wchar_t* name, uint max_width, uint max_height);

/**
 * @brief returns the current config file in use by the named IOU Tracker object
 * @param[in] name unique name of the Tracker to query
 * @param[out] config_file absolute or relative pathspec to the new config file to use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_iou_config_file_get(const wchar_t* name, const wchar_t** config_file);

/**
 * @brief sets the config file to use by named IOU Tracker object
 * @param[in] name unique name of the Tracker to Update
 * @param[in] config_file absolute or relative pathspec to the new config file to use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_iou_config_file_set(const wchar_t* name, const wchar_t* config_file);

/**
 * @brief Adds a pad-probe-handler to be called to process each frame buffer.
 * A Primary GIE can have multiple Sink and Source pad-probe-handlers
 * @param[in] name unique name of the Primary GIE to update
 * @param[in] handler callback function to process pad probe data
 * @param[in] pad pad to add the handler to; DSL_PAD_SINK | DSL_PAD SRC
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_GIE_RESULT otherwise
 */
DslReturnType dsl_tracker_pph_add(const wchar_t* name, const wchar_t* handler, uint pad);

/**
 * @brief Removes a pad-probe-handler from the Primary GIE
 * @param[in] name unique name of the Primary GIE to update
 * @param[in] handler pad-probe-handler to remove
 * @param[in] pad pad to remove the handler from; DSL_PAD_SINK | DSL_PAD SRC
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_GIE_RESULT otherwise
 */
DslReturnType dsl_tracker_pph_remove(const wchar_t* name, const wchar_t* handler, uint pad);

/**
 * @brief creates a new, uniquely named Optical Flow Visualizer (OFV) obj
 * @param[in] name unique name for the new OFV
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OFD_RESULT otherwise
 */
DslReturnType dsl_ofv_new(const wchar_t* name);


/**
 * @brief creates a new, uniquely named OSD obj
 * @param[in] name unique name for the new OSD
 * @param[in] is_clock_enabled true if clock is visible
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_new(const wchar_t* name, boolean is_clock_enabled);

/**
 * @brief returns the current clock enabled setting for the named On-Screen Display
 * @param[in] name name of the Display to query
 * @param[out] enabled current setting for OSD clock in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_clock_enabled_get(const wchar_t* name, boolean* enabled);

/**
 * @brief sets the the clock enabled setting for On-Screen-Display
 * @param[in] name name of the OSD to update
 * @param[in] enabled new enabled setting for the OSD clock
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_clock_enabled_set(const wchar_t* name, boolean enabled);

/**
 * @brief returns the current X and Y offsets for On-Screen-Display clock
 * @param[in] name name of the OSD to query
 * @param[out] offsetX current offset in the X direction for the OSD clock in pixels
 * @param[out] offsetY current offset in the Y direction for the OSD clock in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_clock_offsets_get(const wchar_t* name, uint* offsetX, uint* offsetY);

/**
 * @brief sets the X and Y offsets for the On-Screen-Display clock
 * @param[in] name name of the OSD to update
 * @param[in] offsetX new offset for the OSD clock in the X direction in pixels
 * @param[in] offsetY new offset for the OSD clock in the X direction in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_clock_offsets_set(const wchar_t* name, uint offsetX, uint offsetY);

/**
 * @brief returns the font name and size for On-Screen-Display clock
 * @param[in] name name of the OSD to query
 * @param[out] font current font string for the OSD clock
 * @param[out] size current font size for the OSD clock
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_clock_font_get(const wchar_t* name, const wchar_t** font, uint* size);

/**
 * @brief sets the font name and size for the On-Screen-Display clock
 * @param[in] name name of the OSD to update
 * @param[in] font new font string to use for the OSD clock
 * @param[in] size new size string to use for the OSD clock
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_clock_font_set(const wchar_t* name, const wchar_t* font, uint size);

/**
 * @brief returns the font name and size for On-Screen-Display clock
 * @param[in] name name of the OSD to query
 * @param[out] red current red color value for the OSD clock
 * @param[out] gren current green color value for the OSD clock
 * @param[out] blue current blue color value for the OSD clock
 * @param[out] alpha current alpha color value for the OSD clock
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_clock_color_get(const wchar_t* name, double* red, double* green, double* blue, double* alpha);

/**
 * @brief sets the font name and size for the On-Screen-Display clock
 * @param[in] name name of the OSD to update
 * @param[in] red new red color value for the OSD clock
 * @param[in] gren new green color value for the OSD clock
 * @param[in] blue new blue color value for the OSD clock
 * @param[out] alpha current alpha color value for the OSD clock
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_clock_color_set(const wchar_t* name, double red, double green, double blue, double alpha);

/**
 * @brief gets the current crop settings for the named On-Screen-Display
 * @param[in] name name of the OSD to query
 * @param[out] left number of pixels to crop from the left
 * @param[out] top number of pixels to crop from the top
 * @param[out] width width of the cropped image in pixels
 * @param[out] height height of the cropped image in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_crop_settings_get(const wchar_t* name, uint* left, uint* top, uint* width, uint* height);

/**
 * @brief Sets the current crop settings for the named On-Screen-Display
 * @param[in] name name of the OSD to query
 * @param[in] left number of pixels to crop from the left
 * @param[in] top number of pixels to crop from the top
 * @param[in] width width of the cropped image in pixels
 * @param[in] height height of the cropped image in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_crop_settings_set(const wchar_t* name, uint left, uint top, uint width, uint height);

/**
 * @brief Adds a pad-probe-handler to be called to process each frame buffer.
 * An On-Screen-Display can have multiple Sink and Source pad-probe-handlers
 * @param[in] name unique name of the OSD to update
 * @param[in] handler callback function to process pad probe data
 * @param[in] pad pad to add the handler to; DSL_PAD_SINK | DSL_PAD SRC
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_pph_add(const wchar_t* name, const wchar_t* handler, uint pad);

/**
 * @brief Removes a pad-probe-handler from the OSD
 * @param[in] name unique name of the OSD to update
 * @param[in] handler pad-probe-handler to remove
 * @param[in] pad pad to remove the handler from; DSL_PAD_SINK | DSL_PAD SRC
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_pph_remove(const wchar_t* name, const wchar_t* handler, uint pad);

/**
 * @brief Creates a new, uniquely named Stream Demuxer Tee component
 * @param[in] name unique name for the new Stream Demuxer Tee
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_DEMUXER_RESULT
 */
DslReturnType dsl_tee_demuxer_new(const wchar_t* name);

/**
 * @brief Creates a new Demuxer Tee and adds a list of Branches
 * @param[in] name name of the Tee to create
 * @param[in] branches NULL terminated array of Branch names to add
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_DEMUXER_RESULT on failure
 */
DslReturnType dsl_tee_demuxer_new_branch_add_many(const wchar_t* name, const wchar_t** branches);


/**
 * @brief Creates a new, uniquely named Stream Splitter Tee component
 * @param[in] name unique name for the new Stream Splitter Tee
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_DEMUXER_RESULT
 */
DslReturnType dsl_tee_splitter_new(const wchar_t* name);

/**
 * @brief Creates a new Demuxer Tee and adds a list of Branches
 * @param[in] name name of the Tee to create
 * @param[in] branches NULL terminated array of Branch names to add
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_DEMUXER_RESULT on failure
 */
DslReturnType dsl_tee_splitter_new_branch_add_many(const wchar_t* name, const wchar_t** branches);

/**
 * @brief adds a single Branch to a Stream Demuxer or Splitter Tee
 * @param[in] name name of the Tee to update
 * @param[in] branch name of Branch to add
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_DEMUXER_RESULT on failure
 */
DslReturnType dsl_tee_branch_add(const wchar_t* name, const wchar_t* branch);

/**
 * @brief adds a list of Branches to a Stream Demuxer or Splitter Tee
 * @param[in] name name of the Tee to update
 * @param[in] branches NULL terminated array of Branch names to add
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_DEMUXER_RESULT on failure
 */
DslReturnType dsl_tee_branch_add_many(const wchar_t* name, const wchar_t** branches);

/**
 * @brief removes a single Branch from a Stream Demuxer or Splitter Tee
 * @param[in] name name of the Tee to update
 * @param[in] branch name of Branch to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_DEMUXER_RESULT on failure
 */
DslReturnType dsl_tee_branch_remove(const wchar_t* name, const wchar_t* branch);

/**
 * @brief removes a list of Branches from a Stream Demuxer or Splitter Tee
 * @param[in] name name of the Tee to update
 * @param[in] branches NULL terminated array of Branch names to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_DEMUXER_RESULT on failure
 */
DslReturnType dsl_tee_branch_remove_many(const wchar_t* name, const wchar_t** branches);

/**
 * @brief removes all Branches from a Stream Demuxer or Splitter Tee
 * @param[in] name name of the Tee to update
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_DEMUXER_RESULT on failure
 */
DslReturnType dsl_tee_branch_remove_all(const wchar_t* name);

/**
 * @brief gets the current number of branches owned by Tee
 * @param[in] tee name of the tee to query
 * @param[out] count current number of branches 
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_DEMUXER_RESULT on failure
 */
DslReturnType dsl_tee_branch_count_get(const wchar_t* name, uint* count);

/**
 * @brief Adds a pad-probe-handler to be called to process each frame buffer.
 * One or more Pad Probe Handlers can be added to the SINK PAD only (single stream).
 * @param[in] name unique name of the Tee to update
 * @param[in] handler callback function to process each frame buffer
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_DEMUXER_RESULT otherwise
 */
DslReturnType dsl_tee_pph_add(const wchar_t* name, const wchar_t* handler);

/**
 * @brief Removes a pad probe handler callback function from a named Tee
 * @param[in] name unique name of the Tee to update
 * @param[in] handler unique name of the pad probe handler to had
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TEE_RESULT otherwise
 */
DslReturnType dsl_tee_pph_remove(const wchar_t* name, const wchar_t* handler);

/**
 * @brief creates a new, uniquely named Display component
 * @param[in] name unique name for the new Display
 * @param[in] width width of the Display in pixels
 * @param[in] height height of the Display in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TILER_RESULT
 */
DslReturnType dsl_tiler_new(const wchar_t* name, uint width, uint height);

/**
 * @brief returns the dimensions, width and height, for the named Tiled Display
 * @param[in] name name of the Display to query
 * @param[out] width current width of the tiler in pixels
 * @param[out] height current height of the tiler in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TILER_RESULT
 */
DslReturnType dsl_tiler_dimensions_get(const wchar_t* name, uint* width, uint* height);

/**
 * @brief sets the dimensions, width and height, for the named Tiled Display
 * @param[in] name name of the Display to update
 * @param[in] width width to set the tiler in pixels
 * @param[in] height height to set the tiler in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TILER_RESULT
 */
DslReturnType dsl_tiler_dimensions_set(const wchar_t* name, uint width, uint height);

/**
 * @brief returns the number of columns and rows for the named Tiled Display
 * @param[in] name name of the Display to query
 * @param[out] cols current number of colums for all Tiles
 * @param[out] rows current number of rows for all Tiles
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TILER_RESULT
 */
DslReturnType dsl_tiler_tiles_get(const wchar_t* name, uint* cols, uint* rows);

/**
 * @brief Sets the number of columns and rows for the named Tiled Display
 * @param[in] name name of the Display to update
 * @param[in] cols current number of colums for all Tiles
 * @param[in] rows current number of rows for all Tiles
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TILER_RESULT
 */
DslReturnType dsl_tiler_tiles_set(const wchar_t* name, uint cols, uint rows);

/** 
 * @brief Gets the current Show Source setting for the named Tiler
 * @param[in] name unique name of the Tiler to query
 * @param[out] source name of the current source shown by the Tiler. 
 * A value of DSL_TILER_ALL_SOURCES (equal to NULL) indicates all sources are shown
 * @param[out] current remaining timeout value, 0 if showing all sources and the timer is not running.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TILER_RESULT
 */
DslReturnType dsl_tiler_source_show_get(const wchar_t* name, 
    const wchar_t** source, uint* timeout);

/** 
 * @brief Shows a single source instead of all tiled sources - the default tile mode.
 * @param[in] name unique name of the Tiler to update
 * @param[in] source unique name of the source to show,
 * @param[in] timeout time to show the source in units of seconds, before showing all-sources again
 * A value of 0 indicates no timeout. 
 * @param[in] has_precedence if true will take precedence over a currently show single source. 
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TILER_RESULT
 */
DslReturnType dsl_tiler_source_show_set(const wchar_t* name, 
    const wchar_t* source, uint timeout, boolean has_precedence);

/** 
 * @brief Shows a single source based on positional selection when the Tiler is currently showing all sources
 * If the Tiler is currenly showing a single source, the Tiler will return to showing all
 * @param[in] name unique name of the Tiler to update
 * @param[in] x_pos relative to given window_width.
 * @param[in] y_pos relative to given window_height
 * @param[in] window_width width of the window the x and y positional coordinates are relative to
 * @param[in] window_height height of the window the x and y positional coordinates are relative to
 * @param[in] timeout time to show the source in units of seconds, before showing all-sources again...
 * A value of 0 indicates no timeout. 
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TILER_RESULT
 */
DslReturnType dsl_tiler_source_show_select(const wchar_t* name, 
    int x_pos, int y_pos, uint window_width, uint window_height, uint timeout);

/** 
 * @brief Shows all sources and stops the show-source timer if running.
 * @param[in] name unique name of the Tiler to update
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TILER_RESULT
 */
DslReturnType dsl_tiler_source_show_all(const wchar_t* name);

/** 
 * @brief Cycles through all sources showing each individually for a period of time.
 * Note: calling any other "tiler_source_show" (other than get) disables cycling.
 * Calling dsl_tiler_source_show_get when cycling will return the current source
 * shown and remaining time before timeout.
 * @param[in] name unique name of the Tiler to update
 * @param[in] timeout time to show a source in units of seconds, before moving to the next.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TILER_RESULT
 */
DslReturnType dsl_tiler_source_show_cycle(const wchar_t* name, uint timeout);

/**
 * @brief Adds a pad-probe-handler to either the Sink or Source pad of the named Tiler
 * A Tiled Display can have multiple Sink and Source pad probe handlers
 * @param[in] name unique name of the Tiled Display to update
 * @param[in] handler unique name of the Batch Meta Handler to add
 * @param[in] pad pad to add the handler to; DSL_PAD_SINK | DSL_PAD SRC

 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TILER_RESULT otherwise
 */
DslReturnType dsl_tiler_pph_add(const wchar_t* name, 
    const wchar_t* handler, uint pad);

/**
 * @brief Removes a pad-probe-handler to either the Sink or Source pad of the named Tiler
 * @param[in] name unique name of the Tiled Dislplay to update
 * @param[in] pad pad to remove the handler from; DSL_PAD_SINK | DSL_PAD SRC
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TILER_RESULT otherwise
 */
DslReturnType dsl_tiler_pph_remove(const wchar_t* name, 
    const wchar_t* handler, uint pad);

/**
 * @brief creates a new, uniquely named Fake Sink component
 * @param[in] name unique component name for the new Fake Sink
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT otherwise
 */
DslReturnType dsl_sink_fake_new(const wchar_t* name);

/**
 * @brief creates a new, uniquely named Ovelay Sink component
 * @param[in] name unique component name for the new Overlay Sink
 * @param[in] display_id unique display ID for this Overlay Sink
 * @param[in] depth overlay depth for this Overlay Sink
 * @param[in] offsetX upper left corner offset in the X direction in pixels
 * @param[in] offsetY upper left corner offset in the Y direction in pixels
 * @param[in] width width of the Ovelay Sink in pixels
 * @param[in] heigth height of the Overlay Sink in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT
 */
DslReturnType dsl_sink_overlay_new(const wchar_t* name, uint overlay_id, uint display_id,
    uint depth, uint offsetX, uint offsetY, uint width, uint height);

/**
 * @brief creates a new, uniquely named Window Sink component
 * @param[in] name unique component name for the new Overlay Sink
 * @param[in] offsetX upper left corner offset in the X direction in pixels
 * @param[in] offsetY upper left corner offset in the Y direction in pixels
 * @param[in] width width of the Window Sink in pixels
 * @param[in] heigth height of the Window Sink in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT
 */
DslReturnType dsl_sink_window_new(const wchar_t* name, 
    uint offsetX, uint offsetY, uint width, uint height);

/**
 * @brief creates a new, uniquely named File Sink component
 * @param[in] name unique component name for the new File Sink
 * @param[in] filepath absolute or relative file path including extension
 * @param[in] codec one of DSL_CODEC_H264, DSL_CODEC_H265, DSL_CODEC_MPEG4
 * @param[in] container one of DSL_MUXER_MPEG4 or DSL_MUXER_MK4
 * @param[in] bitrate in bits per second - H264 and H265 only
 * @param[in] interval iframe interval to encode at
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_file_new(const wchar_t* name, const wchar_t* filepath, 
     uint codec, uint container, uint bitrate, uint interval);

/**
 * @brief creates a new, uniquely named File Record component
 * @param[in] name unique component name for the new Record Sink
 * @param[in] outdir absolute or relative path to the recording output dir.
 * @param[in] codec one of DSL_CODEC_H264, DSL_CODEC_H265, DSL_CODEC_MPEG4
 * @param[in] container one of DSL_MUXER_MPEG4 or DSL_MUXER_MK4
 * @param[in] bitrate in bits per second - H264 and H265 only
 * @param[in] interval iframe interval to encode at
 * @param[in] client_listener client callback for end-of-sesssion notifications.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_record_new(const wchar_t* name, const wchar_t* outdir, uint codec, 
    uint container, uint bitrate, uint interval, dsl_record_client_listener_cb client_listener);
     
/**
 * @brief starts a new recording session for the named Record Sink
 * @param[in] name unique of the Record Sink to start the session
 * @param[in] start start time in seconds before the current time
 * should be less that the video cache size
 * @param[in] duration in seconds from the current time to record.
 * @param[in] client_data opaque pointer to client data returned
 * on callback to the client listener function provided on Sink creation
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_record_session_start(const wchar_t* name,
    uint start, uint duration, void* client_data);

/**
 * @brief stops a current recording in session
 * @param[in] name unique of the Record Sink to stop
 * should be less that the video cache size
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_record_session_stop(const wchar_t* name);

/**
 * @brief returns the video recording cache size in units of seconds
 * A fixed size cache is created when the Pipeline is linked and played. 
 * The default cache size is set to DSL_DEFAULT_VIDEO_RECORD_CACHE_IN_SEC
 * @param[in] name name of the Record Sink to query
 * @param[out] cache_size current cache size setting
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT
 */
DslReturnType dsl_sink_record_cache_size_get(const wchar_t* name, uint* cache_size);

/**
 * @brief sets the video recording cache size in units of seconds
 * A fixed size cache is created when the Pipeline is linked and played. 
 * The default cache size is set to DSL_DEFAULT_VIDEO_RECORD_CACHE_IN_SEC
 * @param[in] name name of the Record Sink to query
 * @param[in] cache_size new cache size setting to use on Pipeline play
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT
 */
DslReturnType dsl_sink_record_cache_size_set(const wchar_t* name, uint cache_size);

/**
 * @brief returns the dimensions, width and height, used for the video recordings
 * @param[in] name name of the Record Sink to query
 * @param[out] width current width of the video recording in pixels
 * @param[out] height current height of the video recording in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TILER_RESULT
 */
DslReturnType dsl_sink_record_dimensions_get(const wchar_t* name, uint* width, uint* height);

/**
 * @brief sets the dimensions, width and height, for the video recordings created
 * values of zero indicate no-transcodes
 * @param[in] name name of the Record Sink to update
 * @param[in] width width to set the video recording in pixels
 * @param[in] height height to set the video in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT
 */
DslReturnType dsl_sink_record_dimensions_set(const wchar_t* name, uint width, uint height);

/**
 * @brief returns the current recording state of the Record Sink
 * @param[in] name name of the Record Sink to query
 * @param[out] is_on true if the Record Sink is currently recording a session, false otherwise
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_record_is_on_get(const wchar_t* name, boolean* is_on);

/**
 * @brief returns the current recording state of the Record Sink
 * @param[in] name name of the Record Sink to query
 * @param[out] is_on true if Reset has been done, false otherwise
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_record_reset_done_get(const wchar_t* name, boolean* reset_done);

/**
 * @brief gets the current codec and video media container formats
 * @param[in] name unique name of the Sink to query
 * @param[out] codec one of DSL_CODEC_H264, DSL_CODEC_H265, DSL_CODEC_MPEG4
 * @param[out] container one of DSL_MUXER_MPEG4 or DSL_MUXER_MK4
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_encode_video_formats_get(const wchar_t* name,
    uint* codec, uint* container);

/**
 * @brief gets the current bit-rate and interval settings for the named File Sink
 * @param[in] name unique name of the File Sink to query
 * @param[out] bitrate current Encoder bit-rate in bits/sec for the named File Sink
 * @param[out] interval current Encoder iframe interval value
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_encode_settings_get(const wchar_t* name,
    uint* bitrate, uint* interval);

/**
 * @brief sets new bit_rate and interval settings for the named File Sink
 * @param[in] name unique name of the File Sink to update
 * @param[in] bitrate new Encoder bit-rate in bits/sec for the named File Sink
 * @param[in] interval new Encoder iframe interval value to use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_encode_settings_set(const wchar_t* name,
    uint bitrate, uint interval);

/**
 * @brief creates a new, uniquely named RTSP Sink component
 * @param[in] name unique coomponent name for the new RTSP Sink
 * @param[in] host address for the RTSP Server
 * @param[in] port UDP port number for the RTSP Server
 * @param[in] port RTSP port number for the RTSP Server
 * @param[in] codec one of DSL_CODEC_H264, DSL_CODEC_H265
 * @param[in] bitrate in bits per second
 * @param[in] interval iframe interval to encode at
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_rtsp_new(const wchar_t* name, const wchar_t* host, 
     uint udpPort, uint rtmpPort, uint codec, uint bitrate, uint interval);

/**
 * @brief gets the current codec and video media container formats
 * @param[in] name unique name of the Sink to query
 * @param[out] port UDP Port number to use
 * @param[out] codec one of DSL_CODEC_H264, DSL_CODEC_H265
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_rtsp_server_settings_get(const wchar_t* name,
    uint* udpPort, uint* rtspPort, uint* codec);

/**
 * @brief gets the current bit-rate and interval settings for the named RTSP Sink
 * @param[in] name unique name of the RTSP Sink to query
 * @param[out] bitrate current Encoder bit-rate in bits/sec for the named RTSP Sink
 * @param[out] interval current Encoder iframe interval value
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_rtsp_encoder_settings_get(const wchar_t* name,
    uint* bitrate, uint* interval);

/**
 * @brief sets new bit_rate and interval settings for the named RTSP Sink
 * @param[in] name unique name of the RTSP Sink to update
 * @param[in] bitrate new Encoder bit-rate in bits/sec for the named RTSP Sink
 * @param[in] interval new Encoder iframe interval value to use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_rtsp_encoder_settings_set(const wchar_t* name,
    uint bitrate, uint interval);

/**
 * @brief Adds a pad-probe-handler to be called to process each frame buffer.
 * One or more Pad Probe Handlers can be added to the SINK PAD only (single stream).
 * @param[in] name unique name of the Sink to update
 * @param[in] handler unique name of the pad probe handler to add
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT otherwise
 */
DslReturnType dsl_sink_pph_add(const wchar_t* name, const wchar_t* handler);

/**
 * @brief Removes a pad probe handler callback function from a named Sink
 * @param[in] name unique name of the Sink to update
 * @param[in] handler unique name of the pad probe handler to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT otherwise
 */
DslReturnType dsl_sink_pph_remove(const wchar_t* name, const wchar_t* handler);

/**
 * @brief Gets the current settings for the "sync" and "async" attributes for the named Sink
 * @param[in] name unique name of the Sink to query
 * @param[out] sync the current setting for the Sink's "sync" attribute
 * @param[out] async the current setting for the Sink's "async" attribute
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_SINK_RESULT otherwise
 */
DslReturnType dsl_sink_sync_settings_get(const wchar_t* name, boolean* sync, boolean* async);

/**
 * @brief Sets the "sync" and "async" attributes for the named Sink
 * @param[in] name unique name of the Sink to update
 * @param[in] sync the new setting for the Sink's "sync" attribute
 * @param[in] async the current setting for the Sink's "async" attribute
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_SINK_RESULT otherwise
 */
DslReturnType dsl_sink_sync_settings_set(const wchar_t* name, boolean sync, boolean async);

/**
 * @brief returns the number of Sinks currently in use by 
 * all Pipelines in memory. 
 * @return the current number of Sinks in use
 */
uint dsl_sink_num_in_use_get();  

/**
 * @brief Returns the maximum number of Sinks that can be in-use
 * by all parent Pipelines at one time. The maximum number is 
 * impossed by the Jetson hardware in use, see dsl_sink_num_in_use_max_set() 
 * @return the current Sinks in use max setting.
 */
uint dsl_sink_num_in_use_max_get();  

/**
 * @brief Sets the maximum number of in-memory Sinks 
 * that can be in use at any time. The function overrides 
 * the default value on first call. The maximum number is 
 * limited by Hardware. The caller must ensure to set the 
 * number correctly based on the Jetson hardware in use.
 */
boolean dsl_sink_num_in_use_max_set(uint max);  

/**
 * @brief deletes a Component object by name
 * @param[in] name name of the Component object to delete
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_COMPONENT_RESULT
 * @info the function checks that the name is not 
 * owned by a pipeline before deleting, and returns
 * DSL_RESULT_COMPONENT_IN_USE as failure
 */
DslReturnType dsl_component_delete(const wchar_t* name);

/**
 * @brief deletes a NULL terminated list of components
 * @param[in] names NULL terminated list of names to delete
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_COMPONENT_RESULT
 */
DslReturnType dsl_component_delete_many(const wchar_t** names);

/**
 * @brief deletes all components in memory
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_COMPONENT_RESULT
 */
DslReturnType dsl_component_delete_all();

/**
 * @brief returns the current number of components
 * @return size of the list of components
 */
uint dsl_component_list_size();

/**
 * @brief Gets the named component's current GPU ID
 * @param[in] name name of the component to query
 * @param[out] gpuid current GPU ID setting
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_COMPONENT_RESULT on failure
 */
DslReturnType dsl_component_gpuid_get(const wchar_t* name, uint* gpuid);

/**
 * @brief Sets the GPU ID for the named component
 * @param[in] name name of the component to update
 * @param[in] gpuid GPU ID value to use
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_COMPONENT_RESULT on failure
 */
DslReturnType dsl_component_gpuid_set(const wchar_t* name, uint gpuid);

/**
 * @brief Sets the GPU ID for a list of components
 * @param[in] names a null terminated list of component names to update
 * @param[in] gpuid GPU ID value to use
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_COMPONENT_RESULT on failure
 */
DslReturnType dsl_component_gpuid_set_many(const wchar_t** names, uint gpuid);

/**
 * @brief creates a new, uniquely named Branch
 * @param[in] name unique name for the new Branch
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_BRANCH_RESULT on failure
 */
DslReturnType dsl_branch_new(const wchar_t* name);

/**
 * @brief creates a new Branch for each name in the names array
 * @param[in] names a NULL terminated array of unique Branch names
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_BRANCH_RESULT on failure
 */
DslReturnType dsl_branch_new_many(const wchar_t** names);

/**
 * @brief creates a new branch and adds a list of components
 * @param[in] name name of the Branch to create and populate
 * @param[in] components NULL terminated array of component names to add
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_BRANCH_RESULT on failure
 */
DslReturnType dsl_branch_new_component_add_many(const wchar_t* name, 
    const wchar_t** components);

/**
 * @brief adds a single components to a Branch 
 * @param[in] branch name of the branch to update
 * @param[in] component component names to add
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_BRANCH_RESULT on failure
 */
DslReturnType dsl_branch_component_add(const wchar_t* name, 
    const wchar_t* component);

/**
 * @brief adds a list of components to a Branch
 * @param[in] name name of the Branch to update
 * @param[in] components NULL terminated array of component names to add
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_BRANCH_RESULT on failure
 */
DslReturnType dsl_branch_component_add_many(const wchar_t* name, 
    const wchar_t** components);

/**
 * @brief removes a Component from a Pipeline
 * @param[in] name name of the Branch to update
 * @param[in] component name of the Component to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_BRANCH_RESULT on failure
 */
DslReturnType dsl_branch_component_remove(const wchar_t* name, 
    const wchar_t* component);

/**
 * @brief removes a list of Components from a Branch
 * @param[in] name name of the Branch to update
 * @param[in] components NULL terminated array of component names to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_BRANCH_RESULT on failure
 */
DslReturnType dsl_branch_component_remove_many(const wchar_t* name, 
    const wchar_t** components);

/**
 * @brief creates a new, uniquely named Pipeline
 * @param[in] pipeline unique name for the new Pipeline
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT
 */
DslReturnType dsl_pipeline_new(const wchar_t* pipeline);

/**
 * @brief creates a new Pipeline for each name pipelines array
 * @param[in] pipelines a NULL terminated array of unique Pipeline names
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT
 */
DslReturnType dsl_pipeline_new_many(const wchar_t** pipelines);

/**
 * @brief creates a new Pipeline and adds a list of components
 * @param[in] name name of the pipeline to update
 * @param[in] components NULL terminated array of component names to add
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT
 */
DslReturnType dsl_pipeline_new_component_add_many(const wchar_t* pipeline, 
    const wchar_t** components);


/**
 * @brief deletes a Pipeline object by name.
 * @param[in] pipeline unique name of the Pipeline to delete.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT otherwise.
 * @info any/all components owned by the pipeline move
 * to a state of not-in-use.
 */
DslReturnType dsl_pipeline_delete(const wchar_t* pipeline);

/**
 * @brief deletes a NULL terminated list of pipelines
 * @param[in] pipelines NULL terminated list of names to delete
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT
 * @info any/all components owned by the pipelines move
 * to a state of not-in-use.
 */
DslReturnType dsl_pipeline_delete_many(const wchar_t** pipelines);

/**
 * @brief deletes all pipelines in memory
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_COMPONENT_RESULT
 * @info any/all components owned by the pipelines move
 * to a state of not-in-use.
 */
DslReturnType dsl_pipeline_delete_all();

/**
 * @brief returns the current number of pipelines
 * @return size of the list of pipelines
 */
uint dsl_pipeline_list_size();

/**
 * @brief adds a single components to a Pipeline 
 * @param[in] pipeline name of the pipeline to update
 * @param[in] component component names to add
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT
 */
DslReturnType dsl_pipeline_component_add(const wchar_t* pipeline, 
    const wchar_t* component);

/**
 * @brief adds a list of components to a Pipeline 
 * @param[in] name name of the pipeline to update
 * @param[in] components NULL terminated array of component names to add
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT
 */
DslReturnType dsl_pipeline_component_add_many(const wchar_t* pipeline, 
    const wchar_t** components);

/**
 * @brief removes a Component from a Pipeline
 * @param[in] pipeline name of the Pipepline to update
 * @param[in] component name of the Component to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT
 */
DslReturnType dsl_pipeline_component_remove(const wchar_t* pipeline, 
    const wchar_t* component);

/**
 * @brief removes a list of Components from a Pipeline
 * @param[in] pipeline name of the Pipeline to update
 * @param[in] components NULL terminated array of component names to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT
 */
DslReturnType dsl_pipeline_component_remove_many(const wchar_t* pipeline, 
    const wchar_t** components);

/**
 * @brief 
 * @param[in] pipeline name of the pipeline to query
 * @return DSL_RESULT_SUCCESS on success, 
 */
DslReturnType dsl_pipeline_streammux_batch_properties_get(const wchar_t* pipeline, 
    uint* batchSize, uint* batchTimeout);

/**
 * @brief 
 * @param[in] pipeline name of the pipeline to update
 * @return DSL_RESULT_SUCCESS on success, 
 */
DslReturnType dsl_pipeline_streammux_batch_properties_set(const wchar_t* pipeline, 
    uint batchSize, uint batchTimeout);

/**
 * @brief 
 * @param[in] pipeline name of the pipeline to query
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT otherwise.
 */
DslReturnType dsl_pipeline_streammux_dimensions_get(const wchar_t* pipeline, 
    uint* width, uint* height);

/**
 * @brief 
 * @param[in] pipeline name of the pipeline to update
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT otherwise.
 */
DslReturnType dsl_pipeline_streammux_dimensions_set(const wchar_t* pipeline, 
    uint width, uint height);

/**
 * @brief clears the Pipelines XWindow
 * @param[in] pipeline name of the pipeline to update
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT otherwise.
 */
DslReturnType dsl_pipeline_xwindow_clear(const wchar_t* pipeline);

/**
 * @brief gets the current Pipeline XWindow Offsets. X and Y offsets will return 0
 * prior to window creation which occurs when the Pipeline is played. 
 * @param[in] pipeline name of the pipeline to query
 * @param[out] x_offset offset in the x direction of the XWindow in pixels
 * @param[out] x_offset offset in the Y direction of the XWindow in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT otherwise.
 */
DslReturnType dsl_pipeline_xwindow_offsets_get(const wchar_t* pipeline, 
    uint* x_offset, uint* y_offset);

/**
 * @brief gets the current Pipeline XWindow dimensions. 
 * @param[in] pipeline name of the pipeline to query
 * @param[out] width width of the XWindow in pixels
 * @param[out] heigth height of the Window in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT otherwise.
 */
DslReturnType dsl_pipeline_xwindow_dimensions_get(const wchar_t* pipeline, 
    uint* width, uint* height);

/**
 * @brief gets the current full-screen-enabled setting for the Pipeline's XWindow
 * @param[in] pipeline name of the pipeline to query
 * @param[out] enabled true if full-screen-mode is currently enabled, false otherwise 
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT otherwise.
 */
DslReturnType dsl_pipeline_xwindow_fullscreen_enabled_get(const wchar_t* pipeline, 
    boolean* enabled);

/**
 * @brief sets the full-screen-enabled setting for the Pipeline's XWindow
 * @param[in] pipeline name of the pipeline to update
 * @param[in] enabled if true, sets the XWindow to full-screen on creation.
 * The service will fail if called after the XWindow has been created.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT otherwise.
 */
DslReturnType dsl_pipeline_xwindow_fullscreen_enabled_set(const wchar_t* pipeline, 
    boolean enabled);

/**
 * @brief returns the current setting, enabled/disabled, for the fixed-aspect-ratio 
 * attribute for the named Tiled Display
 * @param[in] name name of the Display to query
 * @param[out] enable true if the aspect ration is fixed, false if not
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TILER_RESULT
 */
DslReturnType dsl_pipeline_streammux_padding_get(const wchar_t* name, boolean* enabled);

/**
 * @brief updates the current setting - enabled/disabled - for the fixed-aspect-ratio 
 * attribute for the named Tiled Display
 * @param[in] name name of the Display to update
 * @param[out] enable set true to fix the aspect ratio, false to disable
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TILER_RESULT
 */
DslReturnType dsl_pipeline_streammux_padding_set(const wchar_t* name, boolean enabled);

/**
 * @brief pauses a Pipeline if in a state of playing
 * @param[in] pipeline unique name of the Pipeline to pause.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT.
 */
DslReturnType dsl_pipeline_pause(const wchar_t* pipeline);

/**
 * @brief plays a Pipeline if in a state of paused
 * @param[in] pipeline unique name of the Pipeline to play.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_play(const wchar_t* pipeline);

/**
 * @brief Stops a Pipeline if in a state of paused or playing
 * @param[in] pipeline unique name of the Pipeline to stop.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_stop(const wchar_t* pipeline);

/**
 * @brief gets the current state of a Pipeline
 * @param[in] pipeline unique name of the Pipeline to query
 * @param[out] state one of the DSL_STATE_* values representing the current state
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_state_get(const wchar_t* pipeline, uint* state);

/**
 * @brief gets the type of source(s) in use, live, or non-live 
 * @param pipeline unique name of the Pipeline to query
 * @param is_live true if the Pipeline's sources are live, false otherwise
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_is_live(const wchar_t* pipeline, boolean* is_live);

/**
 * @brief dumps a Pipeline's graph to dot file.
 * @param[in] pipeline unique name of the Pipeline to dump
 * @param[in] filename name of the file without extention.
 * The caller is responsible for providing a correctly formated filename
 * The diretory location is specified by the GStreamer debug 
 * environment variable GST_DEBUG_DUMP_DOT_DIR
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */ 
DslReturnType dsl_pipeline_dump_to_dot(const wchar_t* pipeline, wchar_t* filename);

/**
 * @brief dumps a Pipeline's graph to dot file prefixed
 * with the current timestamp.  
 * @param[in] pipeline unique name of the Pipeline to dump
 * @param[in] filename name of the file without extention.
 * The caller is responsible for providing a correctly formated filename
 * The diretory location is specified by the GStreamer debug 
 * environment variable GST_DEBUG_DUMP_DOT_DIR
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */ 
DslReturnType dsl_pipeline_dump_to_dot_with_ts(const wchar_t* pipeline, wchar_t* filename);

/**
 * @brief adds a callback to be notified on End of Stream (EOS)
 * @param[in] pipeline name of the pipeline to update
 * @param[in] listener pointer to the client's function to call on EOS
 * @param[in] client_data opaque pointer to client data passed into the listener function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_eos_listener_add(const wchar_t* pipeline, 
    dsl_eos_listener_cb listener, void* client_data);

/**
 * @brief removes a callback previously added with dsl_pipeline_eos_listener_add
 * @param[in] pipeline name of the pipeline to update
 * @param[in] listener pointer to the client's function to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_eos_listener_remove(const wchar_t* pipeline, 
    dsl_eos_listener_cb listener);

/**
 * @brief Adds a callback to be notified on the event of an error message received by
 * the Pipeline's bus-watcher. The callback is called from the mainloop context allowing
 * clients to change the state of, and components within, the Pipeline
 * @param[in] pipeline name of the pipeline to update
 * @param[in] handler pointer to the client's callback function to add
 * @param[in] client_data opaque pointer to client data passed back to the handler function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_error_message_handler_add(const wchar_t* pipeline, 
    dsl_error_message_handler_cb handler, void* client_data);

/**
 * @brief Removes a callback previously added with dsl_pipeline_error_message_handler_add
 * @param[in] pipeline name of the pipeline to update
 * @param[in] handler pointer to the client's callback function to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_error_message_handler_remove(const wchar_t* pipeline, 
    dsl_error_message_handler_cb handler);

/**
 * @brief Gets the last error message received by the Pipeline's bus watcher
 * @param[in] pipeline name of the pipeline to query
 * @param[out] source name of the GST object that was the source of the error message
 * @param[out] message error message sent from the source object
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_error_message_last_get(const wchar_t* pipeline, 
    const wchar_t** source, const wchar_t** message);

/**
 * @brief adds a callback to be notified on change of Pipeline state
 * @param[in] pipeline name of the pipeline to update
 * @param[in] listener pointer to the client's function to call on state change
 * @param[in] client_data opaque pointer to client data passed into the listener function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_state_change_listener_add(const wchar_t* pipeline, 
    dsl_state_change_listener_cb listener, void* client_data);

/**
 * @brief removes a callback previously added with dsl_pipeline_state_change_listener_add
 * @param[in] pipeline name of the pipeline to update
 * @param[in] listener pointer to the client's function to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_state_change_listener_remove(const wchar_t* pipeline, 
    dsl_state_change_listener_cb listener);

/**
 * @brief adds a callback to be notified on XWindow KeyRelease Event
 * @param[in] pipeline name of the pipeline to update
 * @param[in] handler pointer to the client's function to handle XWindow key events.
 * @param[in] client_data opaque pointer to client data passed into the handler function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_xwindow_key_event_handler_add(const wchar_t* pipeline, 
    dsl_xwindow_key_event_handler_cb handler, void* client_data);

/**
 * @brief removes a callback previously added with dsl_pipeline_xwindow_key_event_handler_add
 * @param[in] pipeline name of the pipeline to update
 * @param[in] handler pointer to the client's function to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_xwindow_key_event_handler_remove(const wchar_t* pipeline, 
    dsl_xwindow_key_event_handler_cb handler);

/**
 * @brief adds a callback to be notified on XWindow ButtonPress Event
 * @param[in] pipeline name of the pipeline to update
 * @param[in] handler pointer to the client's function to call to handle XWindow button events.
 * @param[in] client_data opaque pointer to client data passed into the handler function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_xwindow_button_event_handler_add(const wchar_t* pipeline, 
    dsl_xwindow_button_event_handler_cb handler, void* client_data);

/**
 * @brief removes a callback previously added with dsl_pipeline_xwindow_button_event_handler_add
 * @param[in] pipeline name of the pipeline to update
 * @param[in] handler pointer to the client's function to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_xwindow_button_event_handler_remove(const wchar_t* pipeline, 
    dsl_xwindow_button_event_handler_cb handler);

/**
 * @brief adds a callback to be notified on XWindow Delete Message Event
 * @param[in] pipeline name of the pipeline to update
 * @param[in] handler pointer to the client's function to call to handle XWindow Delete event.
 * @param[in] client_data opaque pointer to client data passed into the handler function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_xwindow_delete_event_handler_add(const wchar_t* pipeline, 
    dsl_xwindow_delete_event_handler_cb handler, void* client_data);

/**
 * @brief removes a callback previously added with dsl_pipeline_xwindow_delete_event_handler_add
 * @param[in] pipeline name of the pipeline to update
 * @param[in] handler pointer to the client's function to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_xwindow_delete_event_handler_remove(const wchar_t* pipeline, 
    dsl_xwindow_delete_event_handler_cb handler);

/**
 * @brief entry point to the GST Main Loop
 * Note: This is a blocking call - executes an endless loop
 */
void dsl_main_loop_run();

/**
 * @brief Terminates the GST Main Loop and releases
 * the caller blocked on dsl_main_loop_run()
 */
void dsl_main_loop_quit();

/**
 * @brief converts a numerical Result Code to a String
 * @param[in] result result code to convert
 * @return String value of result.
 */
const wchar_t* dsl_return_value_to_string(uint result);

/**
 * @brief converts a numerical DSL_STATE_* Value TO A String
 * @param state state value to convert
 * @return String value of state
 */
const wchar_t* dsl_state_value_to_string(uint state);

/**
 * @brief Returns the current version of DSL
 * @return string representation of the current release
 */
const wchar_t* dsl_version_get();

/**
 * @brief Releases/deletes all DSL/GST resources
 */
void dsl_delete_all();


EXTERN_C_END

#endif /* _DSL_API_H */
