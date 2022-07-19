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
#define DSL_RESULT_FAILURE                                          0x00000001
#define DSL_RESULT_API_NOT_IMPLEMENTED                              0x00000002
#define DSL_RESULT_API_NOT_SUPPORTED                                0x00000003
#define DSL_RESULT_API_NOT_ENABLED                                  0x00000004
#define DSL_RESULT_INVALID_INPUT_PARAM                              0x00000005
#define DSL_RESULT_THREW_EXCEPTION                                  0x00000006
#define DSL_RESULT_INVALID_RESULT_CODE                              UINT32_MAX

/**
 * Component API Return Values
 */
#define DSL_RESULT_COMPONENT_RESULT                                 0x00010000
#define DSL_RESULT_COMPONENT_NAME_NOT_UNIQUE                        0x00010001
#define DSL_RESULT_COMPONENT_NAME_NOT_FOUND                         0x00010002
#define DSL_RESULT_COMPONENT_NAME_BAD_FORMAT                        0x00010003
#define DSL_RESULT_COMPONENT_THREW_EXCEPTION                        0x00010004
#define DSL_RESULT_COMPONENT_IN_USE                                 0x00010005
#define DSL_RESULT_COMPONENT_NOT_USED_BY_PIPELINE                   0x00010006
#define DSL_RESULT_COMPONENT_NOT_USED_BY_BRANCH                     0x00010007
#define DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE                   0x00010008
#define DSL_RESULT_COMPONENT_SET_GPUID_FAILED                       0x00010009
#define DSL_RESULT_COMPONENT_SET_NVBUF_MEM_TYPE_FAILED              0x0001000A

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
#define DSL_RESULT_SOURCE_CODEC_PARSER_INVALID                      0x0002000B
#define DSL_RESULT_SOURCE_DEWARPER_ADD_FAILED                       0x0002000C
#define DSL_RESULT_SOURCE_DEWARPER_REMOVE_FAILED                    0x0002000D
#define DSL_RESULT_SOURCE_TAP_ADD_FAILED                            0x0002000E
#define DSL_RESULT_SOURCE_TAP_REMOVE_FAILED                         0x0002000F
#define DSL_RESULT_SOURCE_COMPONENT_IS_NOT_SOURCE                   0x00020010
#define DSL_RESULT_SOURCE_COMPONENT_IS_NOT_DECODE_SOURCE            0x00020011
#define DSL_RESULT_SOURCE_COMPONENT_IS_NOT_FILE_SOURCE              0x00020012
#define DSL_RESULT_SOURCE_CALLBACK_ADD_FAILED                       0x00020013
#define DSL_RESULT_SOURCE_CALLBACK_REMOVE_FAILED                    0x00020014
#define DSL_RESULT_SOURCE_SET_FAILED                                0x00020015


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
#define DSL_RESULT_TRACKER_IS_IN_USE                                0x00030006
#define DSL_RESULT_TRACKER_SET_FAILED                               0x00030007
#define DSL_RESULT_TRACKER_HANDLER_ADD_FAILED                       0x00030008
#define DSL_RESULT_TRACKER_HANDLER_REMOVE_FAILED                    0x00030009
#define DSL_RESULT_TRACKER_PAD_TYPE_INVALID                         0x0003000A
#define DSL_RESULT_TRACKER_COMPONENT_IS_NOT_TRACKER                 0x0003000B

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
#define DSL_RESULT_SINK_COMPONENT_IS_NOT_RENDER_SINK                0x0004000D
#define DSL_RESULT_SINK_OBJECT_CAPTURE_CLASS_ADD_FAILED             0x0004000E
#define DSL_RESULT_SINK_OBJECT_CAPTURE_CLASS_REMOVE_FAILED          0x0004000F
#define DSL_RESULT_SINK_HANDLER_ADD_FAILED                          0x00040010
#define DSL_RESULT_SINK_HANDLER_REMOVE_FAILED                       0x00040011
#define DSL_RESULT_SINK_PLAYER_ADD_FAILED                           0x00040012
#define DSL_RESULT_SINK_PLAYER_REMOVE_FAILED                        0x00040013
#define DSL_RESULT_SINK_MAILER_ADD_FAILED                           0x00040014
#define DSL_RESULT_SINK_MAILER_REMOVE_FAILED                        0x00040015
#define DSL_RESULT_SINK_OVERLAY_NOT_SUPPORTED                       0x00040016
#define DSL_RESULT_SINK_WEBRTC_CLIENT_LISTENER_ADD_FAILED           0x00040017
#define DSL_RESULT_SINK_WEBRTC_CLIENT_LISTENER_REMOVE_FAILED        0x00040018
#define DSL_RESULT_SINK_WEBRTC_CONNECTION_CLOSED_FAILED             0x00040019
#define DSL_RESULT_SINK_MESSAGE_CONFIG_FILE_NOT_FOUND               0x00040020
#define DSL_RESULT_SINK_COMPONENT_IS_NOT_MESSAGE_SINK               0x00040021

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
 * GIE and TIS API Return Values
 */
#define DSL_RESULT_INFER_RESULT                                     0x00060000
#define DSL_RESULT_INFER_NAME_NOT_UNIQUE                            0x00060001
#define DSL_RESULT_INFER_NAME_NOT_FOUND                             0x00060002
#define DSL_RESULT_INFER_NAME_BAD_FORMAT                            0x00060003
#define DSL_RESULT_INFER_CONFIG_FILE_NOT_FOUND                      0x00060004
#define DSL_RESULT_INFER_MODEL_FILE_NOT_FOUND                       0x00060005
#define DSL_RESULT_INFER_THREW_EXCEPTION                            0x00060006
#define DSL_RESULT_INFER_IS_IN_USE                                  0x00060007
#define DSL_RESULT_INFER_SET_FAILED                                 0x00060008
#define DSL_RESULT_INFER_HANDLER_ADD_FAILED                         0x00060009
#define DSL_RESULT_INFER_HANDLER_REMOVE_FAILED                      0x0006000A
#define DSL_RESULT_INFER_PAD_TYPE_INVALID                           0x0006000B
#define DSL_RESULT_INFER_COMPONENT_IS_NOT_INFER                     0x0006000C
#define DSL_RESULT_INFER_OUTPUT_DIR_DOES_NOT_EXIST                  0x0006000D
#define DSL_RESULT_INFER_ID_NOT_FOUND                               0x0006000E

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
#define DSL_RESULT_PIPELINE_MAIN_LOOP_REQUEST_FAILED                0x00080014

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
#define DSL_RESULT_ODE_TRIGGER_CALLBACK_ADD_FAILED                  0x000F000D
#define DSL_RESULT_ODE_TRIGGER_CALLBACK_REMOVE_FAILED               0x000F000E
#define DSL_RESULT_ODE_TRIGGER_PARAMETER_INVALID                    0x000E000F
#define DSL_RESULT_ODE_TRIGGER_IS_NOT_AB_TYPE                       0x000E0010
#define DSL_RESULT_ODE_TRIGGER_IS_NOT_TRACK_TRIGGER                 0x000E0011
#define DSL_RESULT_ODE_TRIGGER_ACCUMULATOR_ADD_FAILED               0x000E0012
#define DSL_RESULT_ODE_TRIGGER_ACCUMULATOR_REMOVE_FAILED            0x000E0013
#define DSL_RESULT_ODE_TRIGGER_HEAT_MAPPER_ADD_FAILED               0x000E0014
#define DSL_RESULT_ODE_TRIGGER_HEAT_MAPPER_REMOVE_FAILED            0x000E0015

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
#define DSL_RESULT_ODE_ACTION_CALLBACK_ADD_FAILED                   0x000F000A
#define DSL_RESULT_ODE_ACTION_CALLBACK_REMOVE_FAILED                0x000F000B
#define DSL_RESULT_ODE_ACTION_PLAYER_ADD_FAILED                     0x000F000C
#define DSL_RESULT_ODE_ACTION_PLAYER_REMOVE_FAILED                  0x000F000D
#define DSL_RESULT_ODE_ACTION_MAILER_ADD_FAILED                     0x000F000E
#define DSL_RESULT_ODE_ACTION_MAILER_REMOVE_FAILED                  0x000F000F
#define DSL_RESULT_ODE_ACTION_PARAMETER_INVALID                     0x000F0010

/**
 * ODE Area API Return Values
 */
#define DSL_RESULT_ODE_AREA_RESULT                                  0x00100000
#define DSL_RESULT_ODE_AREA_NAME_NOT_UNIQUE                         0x00100001
#define DSL_RESULT_ODE_AREA_NAME_NOT_FOUND                          0x00100002
#define DSL_RESULT_ODE_AREA_THREW_EXCEPTION                         0x00100003
#define DSL_RESULT_ODE_AREA_IN_USE                                  0x00100004
#define DSL_RESULT_ODE_AREA_SET_FAILED                              0x00100005
#define DSL_RESULT_ODE_AREA_PARAMETER_INVALID                       0x00100006

#define DSL_RESULT_DISPLAY_TYPE_RESULT                              0x00200000
#define DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE                     0x00200001
#define DSL_RESULT_DISPLAY_TYPE_NAME_NOT_FOUND                      0x00200002
#define DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION                     0x00200003
#define DSL_RESULT_DISPLAY_TYPE_IN_USE                              0x00200004
#define DSL_RESULT_DISPLAY_TYPE_NOT_THE_CORRECT_TYPE                0x00200005
#define DSL_RESULT_DISPLAY_TYPE_IS_BASE_TYPE                        0x00200006
#define DSL_RESULT_DISPLAY_PARAMETER_INVALID                        0x00200008


/**
 * Tap API Return Values
 */
#define DSL_RESULT_TAP_RESULT                                       0x00300000
#define DSL_RESULT_TAP_NAME_NOT_UNIQUE                              0x00300001
#define DSL_RESULT_TAP_NAME_NOT_FOUND                               0x00300002
#define DSL_RESULT_TAP_THREW_EXCEPTION                              0x00300003
#define DSL_RESULT_TAP_IN_USE                                       0x00300004
#define DSL_RESULT_TAP_SET_FAILED                                   0x00300005
#define DSL_RESULT_TAP_COMPONENT_IS_NOT_TAP                         0x00300006
#define DSL_RESULT_TAP_FILE_PATH_NOT_FOUND                          0x00300007
#define DSL_RESULT_TAP_CONTAINER_VALUE_INVALID                      0x00300008
#define DSL_RESULT_TAP_PLAYER_ADD_FAILED                            0x00300009
#define DSL_RESULT_TAP_PLAYER_REMOVE_FAILED                         0x0030000A
#define DSL_RESULT_TAP_MAILER_ADD_FAILED                            0x0030000B
#define DSL_RESULT_TAP_MAILER_REMOVE_FAILED                         0x0030000C

/**
 * Player API Return Values
 */
#define DSL_RESULT_PLAYER_RESULT                                    0x00400000
#define DSL_RESULT_PLAYER_NAME_NOT_UNIQUE                           0x00400001
#define DSL_RESULT_PLAYER_NAME_NOT_FOUND                            0x00400002
#define DSL_RESULT_PLAYER_NAME_BAD_FORMAT                           0x00400003
#define DSL_RESULT_PLAYER_IS_NOT_RENDER_PLAYER                      0x00400004
#define DSL_RESULT_PLAYER_IS_NOT_IMAGE_PLAYER                       0x00400005
#define DSL_RESULT_PLAYER_IS_NOT_VIDEO_PLAYER                       0x00400006
#define DSL_RESULT_PLAYER_THREW_EXCEPTION                           0x00400007
#define DSL_RESULT_PLAYER_IN_USE                                    0x00400008
#define DSL_RESULT_PLAYER_XWINDOW_GET_FAILED                        0x00400009
#define DSL_RESULT_PLAYER_XWINDOW_SET_FAILED                        0x0040000A
#define DSL_RESULT_PLAYER_CALLBACK_ADD_FAILED                       0x0040000B
#define DSL_RESULT_PLAYER_CALLBACK_REMOVE_FAILED                    0x0040000C
#define DSL_RESULT_PLAYER_FAILED_TO_PLAY                            0x0040000D
#define DSL_RESULT_PLAYER_FAILED_TO_PAUSE                           0x0040000E
#define DSL_RESULT_PLAYER_FAILED_TO_STOP                            0x0040000F
#define DSL_RESULT_PLAYER_RENDER_FAILED_TO_PLAY_NEXT                0x00400010
#define DSL_RESULT_PLAYER_SET_FAILED                                0x00400011

/**
 * SMTP Mailer API Return Values
 */
#define DSL_RESULT_MAILER_RESULT                                    0x00500000
#define DSL_RESULT_MAILER_NAME_NOT_UNIQUE                           0x00500001
#define DSL_RESULT_MAILER_NAME_NOT_FOUND                            0x00500002
#define DSL_RESULT_MAILER_THREW_EXCEPTION                           0x00500003
#define DSL_RESULT_MAILER_IN_USE                                    0x00500004
#define DSL_RESULT_MAILER_SET_FAILED                                0x00500005
#define DSL_RESULT_MAILER_PARAMETER_INVALID                         0x00500006

/**
 * Segmentation Visualizer API Return Values
 */
#define DSL_RESULT_SEGVISUAL_RESULT                                 0x00600000
#define DSL_RESULT_SEGVISUAL_NAME_NOT_UNIQUE                        0x00600001
#define DSL_RESULT_SEGVISUAL_NAME_NOT_FOUND                         0x00600002
#define DSL_RESULT_SEGVISUAL_THREW_EXCEPTION                        0x00600003
#define DSL_RESULT_SEGVISUAL_IN_USE                                 0x00600004
#define DSL_RESULT_SEGVISUAL_SET_FAILED                             0x00600005
#define DSL_RESULT_SEGVISUAL_PARAMETER_INVALID                      0x00600006
#define DSL_RESULT_SEGVISUAL_HANDLER_ADD_FAILED                     0x00600007
#define DSL_RESULT_SEGVISUAL_HANDLER_REMOVE_FAILED                  0x00600008

/**
 * Websocket Server Manger API Return Values
 */
#define DSL_RESULT_WEBSOCKET_SERVER_RESULT                          0x00700000
#define DSL_RESULT_WEBSOCKET_SERVER_THREW_EXCEPTION                 0x00700001
#define DSL_RESULT_WEBSOCKET_SERVER_SET_FAILED                      0x00700002
#define DSL_RESULT_WEBSOCKET_SERVER_CLIENT_LISTENER_ADD_FAILED      0x00700003
#define DSL_RESULT_WEBSOCKET_SERVER_CLIENT_LISTENER_REMOVE_FAILED   0x00700004

/**
 * Message Broker API Return Values
 */
#define DSL_RESULT_BROKER_RESULT                                    0x00800000
#define DSL_RESULT_BROKER_NAME_NOT_UNIQUE                           0x00800001
#define DSL_RESULT_BROKER_NAME_NOT_FOUND                            0x00800002
#define DSL_RESULT_BROKER_THREW_EXCEPTION                           0x00800003
#define DSL_RESULT_BROKER_IN_USE                                    0x00800004
#define DSL_RESULT_BROKER_SET_FAILED                                0x00800005
#define DSL_RESULT_BROKER_PARAMETER_INVALID                         0x00800006
#define DSL_RESULT_BROKER_SUBSCRIBER_ADD_FAILED                     0x00800007
#define DSL_RESULT_BROKER_SUBSCRIBER_REMOVE_FAILED                  0x00800008
#define DSL_RESULT_BROKER_LISTENER_ADD_FAILED                       0x00800009
#define DSL_RESULT_BROKER_LISTENER_REMOVE_FAILED                    0x0080000A
#define DSL_RESULT_BROKER_CONFIG_FILE_NOT_FOUND                     0x0080000B
#define DSL_RESULT_BROKER_PROTOCOL_LIB_NOT_FOUND                    0x0080000C
#define DSL_RESULT_BROKER_CONNECT_FAILED                            0x0080000D
#define DSL_RESULT_BROKER_DISCONNECT_FAILED                         0x0080000E
#define DSL_RESULT_BROKER_MESSAGE_SEND_FAILED                       0x0080000F

/**
 * ODE Accumulator API Return Values
 */
#define DSL_RESULT_ODE_ACCUMULATOR_RESULT                           0x00900000
#define DSL_RESULT_ODE_ACCUMULATOR_NAME_NOT_UNIQUE                  0x00900001
#define DSL_RESULT_ODE_ACCUMULATOR_NAME_NOT_FOUND                   0x00900002
#define DSL_RESULT_ODE_ACCUMULATOR_THREW_EXCEPTION                  0x00900003
#define DSL_RESULT_ODE_ACCUMULATOR_IN_USE                           0x00900004
#define DSL_RESULT_ODE_ACCUMULATOR_SET_FAILED                       0x00900005
#define DSL_RESULT_ODE_ACCUMULATOR_IS_NOT_ODE_ACCUMULATOR           0x00900006
#define DSL_RESULT_ODE_ACCUMULATOR_ACTION_ADD_FAILED                0x00900007
#define DSL_RESULT_ODE_ACCUMULATOR_ACTION_REMOVE_FAILED             0x00900008
#define DSL_RESULT_ODE_ACCUMULATOR_ACTION_NOT_IN_USE                0x00900009

/**
 * ODE Heat-Mapper API Return Values
 */
#define DSL_RESULT_ODE_HEAT_MAPPER_RESULT                           0x00A00000
#define DSL_RESULT_ODE_HEAT_MAPPER_NAME_NOT_UNIQUE                  0x00A00001
#define DSL_RESULT_ODE_HEAT_MAPPER_NAME_NOT_FOUND                   0x00A00002
#define DSL_RESULT_ODE_HEAT_MAPPER_THREW_EXCEPTION                  0x00A00003
#define DSL_RESULT_ODE_HEAT_MAPPER_IN_USE                           0x00A00004
#define DSL_RESULT_ODE_HEAT_MAPPER_SET_FAILED                       0x00A00005
#define DSL_RESULT_ODE_HEAT_MAPPER_IS_NOT_ODE_HEAT_MAPPER           0x00A00006
#define DSL_RESULT_ODE_HEAT_MAPPER_ACTION_ADD_FAILED                0x00A00007
#define DSL_RESULT_ODE_HEAT_MAPPER_ACTION_REMOVE_FAILED             0x00A00008
#define DSL_RESULT_ODE_HEAT_MAPPER_ACTION_NOT_IN_USE                0x00A00009

/**
 * ODE Preprocessor API Return Values
 */
#define DSL_RESULT_PREPROC_RESULT                                   0x00B00000
#define DSL_RESULT_PREPROC_NAME_NOT_UNIQUE                          0x00B00001
#define DSL_RESULT_PREPROC_NAME_NOT_FOUND                           0x00B00002
#define DSL_RESULT_PREPROC_CONFIG_FILE_NOT_FOUND                    0x00060003
#define DSL_RESULT_PREPROC_THREW_EXCEPTION                          0x00B00004
#define DSL_RESULT_PREPROC_IN_USE                                   0x00B00005
#define DSL_RESULT_PREPROC_SET_FAILED                               0x00B00006
#define DSL_RESULT_PREPROC_IS_NOT_PREPROC                           0x00B00007

/**
 * GPU Types
 */
#define DSL_GPU_TYPE_INTEGRATED                                     0
#define DSL_GPU_TYPE_DISCRETE                                       1

/**
 * NVIDIA Buffer Memory Types
 */
#define DSL_NVBUF_MEM_TYPE_DEFAULT                                  0
#define DSL_NVBUF_MEM_TYPE_PINNED                                   1
#define DSL_NVBUF_MEM_TYPE_DEVICE                                   2
#define DSL_NVBUF_MEM_TYPE_UNIFIED                                  3

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

// Must match GstPadProbeReturn values
#define DSL_PAD_PROBE_DROP                                          0
#define DSL_PAD_PROBE_OK                                            1
#define DSL_PAD_PROBE_REMOVE                                        2
#define DSL_PAD_PROBE_PASS                                          3
#define DSL_PAD_PROBE_HANDLED                                       4


#define DSL_RTP_TCP                                                 0x04
#define DSL_RTP_ALL                                                 0x07

/**
 * @brief Default On-Screen Display (OSD) property values
 */
#define DSL_DEFAULT_OSD_PROCESS_MODE                                0
#define DSL_DEFAULT_OSD_CLOCK_FONT_TYPE                             "Serif"
#define DSL_DEFAULT_OSD_CLOCK_FONT_SIZE                             12
#define DSL_DEFAULT_OSD_CLOCK_OFFSET_X                              20
#define DSL_DEFAULT_OSD_CLOCK_OFFSET_Y                              20
#define DSL_DEFAULT_OSD_CLOCK_COLOR                                 {}

/**
 * @brief Websocket Port number for the Soup Server Manager
 * If set to 0, Manager will find an unused port to listen on.
 */
#define DSL_WEBSOCKET_SERVER_DEFAULT_WEBSOCKET_PORT                 60001

/**
 * @brief WebRTC Websocket connection states, used by the 
 * WebRTC to communicate current state to listening clients 
 */
#define DSL_SOCKET_CONNECTION_STATE_CLOSED                          0
#define DSL_SOCKET_CONNECTION_STATE_INITIATED                       1
#define DSL_SOCKET_CONNECTION_STATE_FAILED                          2

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

/**
 * @brief Predefined Color Constants - rows 1 and 2.
 */
#define DSL_COLOR_PREDEFINED_BLACK                                  0
#define DSL_COLOR_PREDEFINED_GRAY_50                                1
#define DSL_COLOR_PREDEFINED_DARK_RED                               2
#define DSL_COLOR_PREDEFINED_RED                                    3
#define DSL_COLOR_PREDEFINED_ORANGE                                 4
#define DSL_COLOR_PREDEFINED_YELLOW                                 5
#define DSL_COLOR_PREDEFINED_GREEN                                  6
#define DSL_COLOR_PREDEFINED_TURQUOISE                              7
#define DSL_COLOR_PREDEFINED_INDIGO                                 8
#define DSL_COLOR_PREDEFINED_PURPLE                                 9

#define DSL_COLOR_PREDEFINED_WHITE                                  10
#define DSL_COLOR_PREDEFINED_GRAY_25                                11
#define DSL_COLOR_PREDEFINED_BROWN                                  12
#define DSL_COLOR_PREDEFINED_ROSE                                   13
#define DSL_COLOR_PREDEFINED_GOLD                                   14
#define DSL_COLOR_PREDEFINED_LIGHT_YELLOW                           15
#define DSL_COLOR_PREDEFINED_LIME                                   16
#define DSL_COLOR_PREDEFINED_LIGHT_TURQUOISE                        17
#define DSL_COLOR_PREDEFINED_BLUE_GRAY                              18
#define DSL_COLOR_PREDEFINED_LAVENDER                               19

/**
 * @brief Hue constants used to define random RGB colors.
 */
#define DSL_COLOR_HUE_RED                                           0
#define DSL_COLOR_HUE_RED_ORANGE                                    1
#define DSL_COLOR_HUE_ORANGE                                        2
#define DSL_COLOR_HUE_ORANGE_YELLOW                                 3
#define DSL_COLOR_HUE_YELLOW                                        4
#define DSL_COLOR_HUE_YELLOW_GREEN                                  5
#define DSL_COLOR_HUE_GREEN                                         6
#define DSL_COLOR_HUE_GREEN_CYAN                                    7
#define DSL_COLOR_HUE_CYAN                                          8
#define DSL_COLOR_HUE_CYAN_BLUE                                     9
#define DSL_COLOR_HUE_BLUE                                          10
#define DSL_COLOR_HUE_BLUE_MAGENTA                                  11
#define DSL_COLOR_HUE_MAGENTA                                       12
#define DSL_COLOR_HUE_MAGENTA_PINK                                  13
#define DSL_COLOR_HUE_PINK                                          14
#define DSL_COLOR_HUE_PINK_RED                                      15
#define DSL_COLOR_HUE_RANDOM                                        16
#define DSL_COLOR_HUE_BLACK_AND_WHITE                               17
#define DSL_COLOR_HUE_BROWN                                         18

/**
 * @brief Luminosity constants used to create predefined and random RGB colors.
 */
#define DSL_COLOR_LUMINOSITY_DARK                                   0
#define DSL_COLOR_LUMINOSITY_NORMAL                                 1
#define DSL_COLOR_LUMINOSITY_LIGHT                                  2
#define DSL_COLOR_LUMINOSITY_BRIGHT                                 3
#define DSL_COLOR_LUMINOSITY_RANDOM                                 4

/**
 * @brief Predefined Color Palette constants used to create a predefined 
 * RGB color palette.
 */
#define DSL_COLOR_PREDEFINED_PALETTE_SPECTRAL                       0
#define DSL_COLOR_PREDEFINED_PALETTE_RED                            1
#define DSL_COLOR_PREDEFINED_PALETTE_GREEN                          2
#define DSL_COLOR_PREDEFINED_PALETTE_BLUE                           3
#define DSL_COLOR_PREDEFINED_PALETTE_GREY                           4


/**
 * @brief On-Screen Heat-Map legend locations.
 */
#define DSL_HEAT_MAP_LEGEND_LOCATION_TOP                            0
#define DSL_HEAT_MAP_LEGEND_LOCATION_RIGHT                          1
#define DSL_HEAT_MAP_LEGEND_LOCATION_BOTTOM                         2
#define DSL_HEAT_MAP_LEGEND_LOCATION_LEFT                           3
 
/**
 * @brief On-Screen Heat-Map legend locations.
 */
#define DSL_CAPTURE_TYPE_OBJECT                                     0
#define DSL_CAPTURE_TYPE_FRAME                                      1

// Trigger-Always 'when' constants, pre/post check-for-occurrence
#define DSL_ODE_PRE_OCCURRENCE_CHECK                                0
#define DSL_ODE_POST_OCCURRENCE_CHECK                               1

/**
 * @brief Source and Class Trigger filter constants for no-filter
 */
#define DSL_ODE_ANY_SOURCE                                          NULL
#define DSL_ODE_ANY_CLASS                                           INT32_MAX
#define DSL_ODE_TRIGGER_LIMIT_NONE                                  0
#define DSL_ODE_TRIGGER_LIMIT_ONE                                   1

/**
 * @brief ODE Trigger limit state values - for Triggers with limits
 */
#define DSL_ODE_TRIGGER_LIMIT_EVENT_LIMIT_REACHED                   0
#define DSL_ODE_TRIGGER_LIMIT_EVENT_LIMIT_CHANGED                   1
#define DSL_ODE_TRIGGER_LIMIT_EVENT_COUNT_RESET                     2

/**
 * @brief Unique class relational identifiers for Class A/B testing
 */
#define DSL_CLASS_A                                                 0
#define DSL_CLASS_B                                                 1

#define DSL_AREA_TYPE_INCLUSION                                     0
#define DSL_AREA_TYPE_EXCLUSION                                     1
#define DSL_AREA_TYPE_LINE                                          3     

#define DSL_AREA_CROSS_DIRECTION_NONE                               0
#define DSL_AREA_CROSS_DIRECTION_IN                                 1
#define DSL_AREA_CROSS_DIRECTION_OUT                                2

/**
 * @brief Defines a Point's location relative to an ODE Area.
 */
#define DSL_AREA_POINT_LOCATION_ON_LINE                             0
#define DSL_AREA_POINT_LOCATION_INSIDE                              1
#define DSL_AREA_POINT_LOCATION_OUTSIDE                             2

/**
 * @brief Defines the ODE Area Line Cross directions.
 */
#define DSL_AREA_CROSS_DIRECTION_NONE                               0
#define DSL_AREA_CROSS_DIRECTION_IN                                 1
#define DSL_AREA_CROSS_DIRECTION_OUT                                2

// Must match NvOSD_Arrow_Head_Direction
#define DSL_ARROW_START_HEAD                                        0
#define DSL_ARROW_END_HEAD                                          1
#define DSL_ARROW_BOTH_HEAD                                         2

#define DSL_DEFAULT_SOURCE_IN_USE_MAX                               8
#define DSL_DEFAULT_SINK_IN_USE_MAX                                 8

/**
 * @brief Image Source Type constants
 */
#define DSL_IMAGE_TYPE_SINGLE                                       0
#define DSL_IMAGE_TYPE_MULTI                                        1
#define DSL_IMAGE_TYPE_STREAM                                       2

/**
 * @brief Image Source File Format constants
 */
#define DSL_IMAGE_FORMAT_JPG                                        0
#define DSL_IMAGE_FORMAT_PNG                                        1

/**
 * @brief Image Source File Extention constants
 */
#define DSL_IMAGE_EXT_JPG                                           "jpg"
#define DSL_IMAGE_EXT_PNG                                           "png"


#define DSL_DEFAULT_STREAMMUX_DEFAULT_NVBUF_MEMORY_TYPE             DSL_NVBUF_MEM_TYPE_DEFAULT
#define DSL_DEFAULT_STREAMMUX_BATCH_TIMEOUT                         40000
#define DSL_DEFAULT_STREAMMUX_WIDTH                                 1920
#define DSL_DEFAULT_STREAMMUX_HEIGHT                                1080
#define DSL_DEFAULT_STREAMMUX_MAX_NUM_SERFACES_PER_FRAME            1

#define DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC                     10
#define DSL_DEFAULT_WAIT_FOR_EOS_TIMEOUT_IN_SEC                     2

#define DSL_DEFAULT_VIDEO_RECORD_CACHE_IN_SEC                       30
#define DSL_DEFAULT_VIDEO_RECORD_DURATION_IN_SEC                    30

#define DSL_BBOX_POINT_CENTER                                       0
#define DSL_BBOX_POINT_NORTH_WEST                                   1
#define DSL_BBOX_POINT_NORTH                                        2
#define DSL_BBOX_POINT_NORTH_EAST                                   3
#define DSL_BBOX_POINT_EAST                                         4
#define DSL_BBOX_POINT_SOUTH_EAST                                   5
#define DSL_BBOX_POINT_SOUTH                                        6
#define DSL_BBOX_POINT_SOUTH_WEST                                   7
#define DSL_BBOX_POINT_WEST                                         8
#define DSL_BBOX_POINT_ANY                                          9

#define DSL_BBOX_EDGE_TOP                                           0
#define DSL_BBOX_EDGE_BOTTOM                                        1
#define DSL_BBOX_EDGE_LEFT                                          2
#define DSL_BBOX_EDGE_RIGHT                                         3

/**
 * @brief Methods of calculating distance between object BBoxes
 */
#define DSL_DISTANCE_METHOD_FIXED_PIXELS                            0
#define DSL_DISTANCE_METHOD_PERCENT_WIDTH_A                         1
#define DSL_DISTANCE_METHOD_PERCENT_WIDTH_B                         2
#define DSL_DISTANCE_METHOD_PERCENT_HEIGHT_A                        3
#define DSL_DISTANCE_METHOD_PERCENT_HEIGHT_B                        4

#define DSL_OBJECT_TRACE_TEST_METHOD_END_POINTS                     0
#define DSL_OBJECT_TRACE_TEST_METHOD_ALL_POINTS                     1

/**
 * @brief the maximum number of coordinates when defining a Display Type
 */
#define DSL_MAX_POLYGON_COORDINATES                                 16
#define DSL_MAX_MULTI_LINE_COORDINATES                              16

/**
 * @brief default maximum number of bbox frames for an ODE Tracking
 * Trigger to maintain as an object trace / history
 */
#define DSL_DEFAULT_TRACKING_TRIGGER_MAX_TRACE_POINTS               10

/**
 * @brief the maximum number of messages that can be queued up
 * by all Mailers running in the main-loop context before
 * new messages are dropped. Messages are pulled from the queue
 * in a low priority background thread which may get starved
 * out if the Tracker and OSD are consuming the CPU
 */
#define DSL_SMTP_MAX_PENDING_MESSAGES                               10

/**
 * @brief Sink Types for Render Players
 */
#define DSL_RENDER_TYPE_OVERLAY                                     0
#define DSL_RENDER_TYPE_WINDOW                                      1

/**
 * @brief Smart Recording Events - to identify which event
 * has occurred when processing dsl_recording_info
 */
#define DSL_RECORDING_EVENT_START                                   0
#define DSL_RECORDING_EVENT_END                                     1

/**
 * @brief Maximum time to wait for a recording session to stop
 * when the client is blocked on one of the session stop services
 */
#define DSL_RECORDING_STOP_WAIT_TIMEOUT_MS                          1000

/**
 * @brief Maximum time to wait for the recording reset done flag
 * when the client is blocked on one of the session stop services
 */
#define DSL_RECORDING_RESET_WAIT_TIMEOUT_MS                         100

/**
 * @brief File Format Options when saving Event Data to file.
 */
#define DSL_EVENT_FILE_FORMAT_TEXT                                  0
#define DSL_EVENT_FILE_FORMAT_CSV                                   1
#define DSL_EVENT_FILE_FORMAT_MOTC                                  2

/**
 * @brief File Open/Write Mode Options when saving Event Data 
 * to file, and when adding custom Object Label Content
 * 
 */
#define DSL_WRITE_MODE_APPEND                                       0
#define DSL_WRITE_MODE_TRUNCATE                                     1

/**
 * @brief Metric Content Options for Object Label customization
 * and Display Action string formatting
  */
#define DSL_METRIC_OBJECT_CLASS                                     0
#define DSL_METRIC_OBJECT_TRACKING_ID                               1
#define DSL_METRIC_OBJECT_LOCATION                                  2
#define DSL_METRIC_OBJECT_DIMENSIONS                                3
#define DSL_METRIC_OBJECT_CONFIDENCE_INFERENCE                      4
#define DSL_METRIC_OBJECT_CONFIDENCE_TRACKER                        5
#define DSL_METRIC_OBJECT_PERSISTENCE                               6
#define DSL_METRIC_OBJECT_DIRECTION                                 7

/**
 * @brief Metric Content Options for Trigger Output customization
 * used by the the Display Action when reporting Frame level
 * metrics. This constant provides meaningful information for the 
 * Summation, Accumulation, New High, and New Low Triggers for 
 * reporting the number of occurrences when post-processing each 
 * frame. For most other Triggers, this value will always be 1.
 * For the Absence Trigger, occurrences will always be 0. 
 */
#define DSL_METRIC_OBJECT_OCCURRENCES                               8

#define DSL_METRIC_OBJECT_OCCURRENCES_DIRECTION_IN                  9
#define DSL_METRIC_OBJECT_OCCURRENCES_DIRECTION_OUT                 10

/**
 * @brief Message converter payload schema types used by all Message Sinks.
 */
#define DSL_MSG_PAYLOAD_DEEPSTREAM                                  0
#define DSL_MSG_PAYLOAD_DEEPSTREAM_MINIMAL                          1
#define DSL_MSG_PAYLOAD_CUSTOM                                      257

/**
 * @brief Message Broker Status/Result codes
 */
#define DSL_STATUS_BROKER_OK                                        0
#define DSL_STATUS_BROKER_ERROR                                     1
#define DSL_STATUS_BROKER_RECONNECTING                              2
#define DSL_STATUS_BROKER_NOT_SUPPORTED                             3


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
     * @brief specifies which recording event has occurred. One of 
     * DSL_RECORDING_EVENT_START or DSL_RECORDING_EVENT_END
     */
    uint recording_event;
    
    /**
     * @brief the unique sesions id assigned on record start
     */
    uint session_id;
    
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
    uint container_type;
    
    /**
     * @brief width of the recording in pixels
     */
    uint width;

    /**
     * @brief height of the recording in pixels
     */
    uint height;

} dsl_recording_info;

/**
 * @struct dsl_capture_info
 * @brief Image capture information provided to the client on callback
 */
typedef struct dsl_capture_info
{
    /**
     * @brief the unique capture id assigned on file save
     */
    uint captureId;

    /**
     * @brief filename generated for the captured image. 
     */
    const wchar_t* filename;
    
    /** 
     * @brief directory path for the captured image
     */
    const wchar_t* dirpath;
    
    /**
     * @brief width of the image in pixels
     */
    uint width;

    /**
     * @brief height of the image in pixels
     */
    uint height;

} dsl_capture_info;

/**
 * @struct dsl_webrtc_connection_data
 * @brief a structure of Connection date for a given WebRTC Sink
 */
typedef struct _dsl_webrtc_connection_data
{
    /**
     * @brief the current state of the WebRTC Sink's Websocket connection
     * one of the DSL_SOCKET_CONNECTION_STATE* values
     */ 
    uint current_state; 

} dsl_webrtc_connection_data;

/**
 * @struct _dsl_coordinate
 * @brief defines a frame coordinate by it's x and y pixel position
 */
typedef struct _dsl_coordinate
{
    uint x;
    uint y;
} dsl_coordinate;

/**
 * @struct dsl_ode_occurrence_source_info
 * @brief Video Source information for the ODE Occurrence provided to the 
 * client on callback.
 */
typedef struct _dsl_ode_occurrence_source_info
{
    /**
     * @brief unique source id for this ODE occurrence.
     */
    uint source_id;
    
    /**
     * @brief the location of the frame in the batch for this ODE occurrence 
     */
    uint batch_id;
    
    /**
     * @brief pad or port index of the Gst-streammux plugin for this ODE occurrence
     */
    uint pad_index;
    
    /**
     * @brief current frame number of the source for this ODE occurrence.
     */
    uint frame_num;
    
    /**
     * @brief width of the frame at input to Gst-streammux for this ODE occurrence.
     */
    uint frame_width;
    
    /**
     * @brief height of the frame at input to Gst-streammux for this ODE occurrence.
     */
    uint frame_height;
    
    /**
     * @brief true if inference was done on the frame for this ODE occurrence.
     */
    boolean inference_done;
    
} dsl_ode_occurrence_source_info;

/**
 * @struct dsl_ode_occurrence_object_info
 * @brief Detected Object information for the ODE Occurrence provided to the 
 * client on callback.
 */
typedef struct _dsl_ode_occurrence_object_info
{
    /**
     * @brief class id for the detected object
     */
    uint class_id;
    
    /**
     * @brief unique id of the inference component that generated the object data.
     */
    uint inference_component_id;
    
    /**
     * @brief unique tracking id as assigned by the multi-object-tracker (MOT)
     */
    uint tracking_id;
    
    /**
     * @brief unique label for the detected object
     */
    const wchar_t* label;
    
    /**
     * @brief current "time in frame" if tracked - Persistence and Cross Triggers
     */
    uint persistence;
    
    /**
     * @brief direction of the Object if line cross event - Cross Trigger only.
     */
    uint direction;
    
    /**
     * @brief inference confidence as calculated by the last detector.
     */
    float inference_confidence;
    
    /**
     * @brief tracker confidence if current frame was not inferred on.
     */
    float tracker_confidence;

    /**
     * @brief the Object's bounding box left coordinate in pixels.
     */
    uint left;
    
    /**
     * @brief the Object's bounding box top coordinate in pixels.
     */
    uint top;
    
    /**
     * @brief the Object's bounding box width in pixels.
     */
    uint width;
    
    /**
     * @brief the Object's bounding box height in pixels.
     */
    uint height;
    
} dsl_ode_occurrence_object_info;

/**
 * @struct _dsl_ode_occurrence_accumulative_info
 * @brief Accumulative ODE Occurrence metrics provided to the 
 * client on callback.
 */
typedef struct _dsl_ode_occurrence_accumulative_info
{
    /**
     * @brief the total number of object detection occurrences for the 
     * frame-level ODE occurrence - Count, New-High, New-Low Triggers
     * or from an ODE accumulator.
     */
    uint occurrences_total;
    
    /**
     * @brief the number of Line-Cross ODE occurrences in the "in-direction".
     * Requires an ODE Cross-Trigger with ODE Accumulator
     */
    uint occurrences_in;

    /**
     * @brief the number of Line-Cross ODE occurrences in the "out-direction".
     * Requires an ODE Cross-Trigger with ODE Accumulator
     */
    uint occurrences_out;

} dsl_ode_occurrence_accumulative_info;


/**
 * @struct dsl_ode_occurrence_criteria_info
 * @brief ODE Trigger Criteria used for the ODE Occurrence.
 */
typedef struct _dsl_ode_occurrence_criteria_info
{
    /**
     * @brief class id filter for ODE occurrence
     */
    uint class_id;
    
    /**
     * @brief inference id filter for ODE occurrence
     */
    uint inference_component_id;
    
    /**
     * @brief the minimum inference confidence to trigger an ODE occurrence.
     */
    float min_inference_confidence;
    
    /**
     * @brief the minimum tracker confidence to trigger an ODE occurrence.
     */
    float min_tracker_confidence;
    
    /**
     * @brief inference must be performed to trigger an ODE occurrence.
     */
    boolean inference_done_only;
     
    /**
     * @brief the minimum bounding box width to trigger an ODE occurrence.
     */
    uint min_width;
    
    /**
     * @brief the minimum bounding box height to trigger an ODE occurrence.
     */
    uint min_height;
    
    /**
     * @brief the maximum bounding box width to trigger an ODE occurrence.
     */
    uint max_width;
    
    /**
     * @brief the maximum bounding box height to trigger an ODE occurrence.
     */
    uint max_height;
    
    /**
     * @brief the interval for checking for an ODE occurrence.
     */
    uint interval;
    
} dsl_ode_occurrence_criteria_info;

/**
 * @struct dsl_ode_occurrence_info
 * @brief ODE Occurrence information provided to the client on callback
 */
typedef struct _dsl_ode_occurrence_info
{
    /**
     * @brief the unique name of the ODE Trigger that triggered the occurrence
     */
    const wchar_t* trigger_name;
    
    /**
     * @brief unique occurrence Id for this occurrence.
     */
    uint64_t unique_ode_id;
    
    /**
     * @brief Network Time for this event.
     */
    uint64_t ntp_timestamp;
    
    /**
     * @brief Video Source information this ODE Occurrence
     */
    dsl_ode_occurrence_source_info source_info;

    /**
     * @brief true if the ODE occurrence information is for a specific object,
     * false for frame-level multi-object events. (absence, new-high count, etc.). 
     */
    boolean is_object_occurrence;
    
    // NOTE: object_info and accumulative_info are mutually exclusive
    // determined by the boolean is_object_occurrence flag above.
    
    /**
     * @brief Object information if object_occurrence == true
     */
    dsl_ode_occurrence_object_info object_info;
    
    /**
     * @brief Accumulative information if object_occurrence == false
     */
    dsl_ode_occurrence_accumulative_info accumulative_info;
    
    /**
     * @brief Trigger Criteria information for this ODE occurrence.
     */
    dsl_ode_occurrence_criteria_info criteria_info;
       
} dsl_ode_occurrence_info;

//------------------------------------------------------------------------------------

/**
 * @brief Callback typedef for a client ODE occurrence handler function. 
 * Once registered by calling dsl_ode_action_custom_new, the function will be called 
 * on ODE occurrence with the full set of parameters received by the ODE Action. 
 * @param[in] event_id unique ODE occurrence ID, numerically ordered by occurrence.
 * @param[in] trigger unique name of the ODE Event Trigger that triggered the occurrence.
 * @param[in] buffer pointer to the frame buffer of type GstBuffer.
 * @param[in] display_meta pointer to a NvDsDisplayMeta structure.
 * @param[in] frame_meta pointer to the NvDsFrameMeta structure that triggered the ODE event.
 * @param[in] object_meta pointer to the NvDsObjectMeta structure that triggered the ODE event.
 * Note: This parameter will be set to NULL for ODE occurrences detected in Post process frame. 
 * Absence and Submation ODE's
 * @param[in] client_data opaque pointer to client's user data
 */
typedef void (*dsl_ode_handle_occurrence_cb)(uint64_t event_id, const wchar_t* trigger,
    void* buffer, void* display_meta, void* frame_meta, void* object_meta, void* client_data);

/**
 * @brief Callback typedef for a client ODE occurrence monitor function. 
 * Once registered by calling dsl_ode_action_monitor_new, the function will be called 
 * on ODE occurrence with all occurrence information using the dsl_ode_occurrence_info
 * structure. 
 * @param[in] occurrence_info occurrence information on ODE occurrence.
 * @param[in] client_data opaque pointer to client's user data
 */    
typedef void (*dsl_ode_monitor_occurrence_cb)(dsl_ode_occurrence_info* occurrence_info,
    void* client_data);    

/**
 * @brief Callback typedef for a client ODE Custom Trigger check-for-occurrence function. Once 
 * registered, the function will be called on every object detected that meets the minimum
 * criteria for the Custom Trigger. The client, determining that criteria is met for ODE occurrence,
 * returns true to invoke all ODE acctions owned by the Custom Trigger
 * @param[in] pointer to a frame_meta structure that triggered the ODE event
 * @param[in] pointer to a object_meta structure that triggered the ODE event
 * This parameter will be set to NULL for ODE occurrences detected in Post process frame. 
 * Absence and Submation ODE's
 * @param[in] client_data opaque pointer to client's user data
 */
typedef boolean (*dsl_ode_check_for_occurrence_cb)(void* buffer,
    void* frame_meta, void* object_meta, void* client_data);

/**
 * @brief Callback typedef for a client ODE Custom Trigger post-process-frame function. Once 
 * registered, the function will be called on every frame AFTER all Check-For-Occurrence calls 
 * have been handles The client, determining that criteria is met for ODE occurrence,  
 * returns true to invoke all ODE acctions owned by the Custom Trigger
 * @param[in] pointer to a frame_meta structure that triggered the ODE event
 * @param[in] pointer to a object_meta structure that triggered the ODE event
 * This parameter will be set to NULL for ODE occurrences detected in Post process frame. 
 * Absence and Submation ODE's
 * @param[in] client_data opaque pointer to client's user data
 */
typedef boolean (*dsl_ode_post_process_frame_cb)(void* buffer,
    void* frame_meta, void* client_data);
    
/**
 * @brief Callback typedef for a client listener function. Once added to an
 * ODE Trigger or ODE Action, this function will be called on change 
 * of the enabled state.
 * @param[in] enabled true if the ODE Object was enabled, false if disalbed.
 * @param[in] client_data opaque pointer to client's user data.
 */
 typedef void (*dsl_ode_enabled_state_change_listener_cb)
    (boolean enabled, void* client_data);

/**
 * @brief Callback typedef for a client listener function. Once added to an
 * ODE Trigger, this function will be called on every Trigger Limit event;
 * LIMIT_REACHED, LIMIT_CHANGED, and COUNT_RESET
 * @param[in] event one of the DSL_ODE_TRIGGER_LIMIT_EVENT constants.
 * @param[in] limit the current, or new limit for the Trigger.
 * @param[in] client_data opaque pointer to client's user data.
 */
 typedef void (*dsl_ode_trigger_limit_event_listener_cb)
    (uint event, uint limit, void* client_data);

/**
 * @brief callback typedef for a client to handle new Pipeline performance data
 * calcaulated by the Meter Pad Probe Handler, at an intervel specified by the client.
 * @param[in] session_fps_averages array of frames-per-second measurements, 
 * one per source, specified by list_size 
 * @param[in] interval_fps_averages array of average frames-per-second measurements, 
 * one per source, specified by list_size 
 * @param[in] source_count count of both session_fps_averages and avg_fps 
 * interval_fps_averages, one Pipeline Source
 * @param[in] client_data opaque pointer to client's user data provide on end-of-session
 */
typedef boolean (*dsl_pph_meter_client_handler_cb)(double* session_fps_averages, 
    double* interval_fps_averages, uint source_count, void* client_data);
    
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
typedef void (*dsl_error_message_handler_cb)(const wchar_t* source, 
    const wchar_t* message, void* client_data);

/**
 * @brief callback typedef for a client XWindow KeyRelease event handler function. 
 * Once added to a Pipeline, the function will be called when the Pipeline receives 
 * XWindow KeyRelease events.
 * @param[in] key UNICODE key string for the key pressed
 * @param[in] client_data opaque pointer to client's user data
 */
typedef void (*dsl_xwindow_key_event_handler_cb)(const wchar_t* key, void* client_data);

/**
 * @brief callback typedef for a client XWindow ButtonPress event handler function. 
 * Once added to a Pipeline, the function will be called when the Pipeline receives 
 * XWindow ButtonPress events.
 * @param[in] button button 1 through 5 including scroll wheel up and down
 * @param[in] xpos from the top left corner of the window
 * @param[in] ypos from the top left corner of the window
 * @param[in] client_data opaque pointer to client's user data
 */
typedef void (*dsl_xwindow_button_event_handler_cb)(uint button, 
    int xpos, int ypos, void* client_data);

/**
 * @brief callback typedef for a client XWindow Delete Message event handler function. 
 * Once added to a Pipeline, the function will be called when the Pipeline receives 
 * XWindow Delete Message event.
 * @param[in] client_data opaque pointer to client's user data
 */
typedef void (*dsl_xwindow_delete_event_handler_cb)(void* client_data);


/**
 * @brief callback typedef for a client to listen for notifications of Recording 
 * events, either DSL_RECORDING_EVENT_START or DSL_RECORDING_EVENT_STOP 
 * @param[in] info pointer to session info, see dsl_recording_info struct.
 * @param[in] client_data opaque pointer to client's user data.
 */
typedef void* (*dsl_record_client_listener_cb)(dsl_recording_info* info, void* client_data);

/**
 * @brief callback typedef for a client to listen for notification that an 
 * JPEG Image has been captured and saved to file.
 * @param[in] info pointer to capture info, see dsl_capture_info struct.
 * @param[in] client_data opaque pointer to client's user data.
 */
typedef void (*dsl_capture_complete_listener_cb)(dsl_capture_info* info, void* client_data);

/**
 * @brief callback typedef for a client to listen for Player termination events.
 * @param[in] client_data opaque pointer to client's user data
 */
typedef void (*dsl_player_termination_event_listener_cb)(void* client_data);

/**
 * @brief callback typedef for a client to listen for incoming Websocket connection events.
 * Important Note: Clients will be notified of the incoming connection prior to checking
 * for any available WebRTC Signaling Transceivers (WebRTC Sinks). This allows Client 
 * listeners to create and add a new WebRTC sink "on demand" - before returning. 
 * @param[in] path path for the incoming connection.
 * @param[in] client_data opaque pointer to client's user data
 */
typedef void (*dsl_websocket_server_client_listener_cb)(const wchar_t* path, 
    void* client_data);
    
/**
 * @brief callback typedef for a client to listen for WebRTC Sink connection events.
 * @param[in] info pointer to connection info, see dsl_webrtc_connection_data struct.
 * @param[in] client_data opaque pointer to client's user data
 */
typedef void (*dsl_sink_webrtc_client_listener_cb)(dsl_webrtc_connection_data* info, 
    void* client_data);

/**
 * @brief callback typedef for a client to receive incoming messages
 * filtered by topic. 
 * @param[in] client_data opaque pointer to client's user data.
 * @param[in] status status of the received messages, one of DSL_STATUS_BROKER
 * @param[in] message pointer to the message received.
 * @param[in] length length of the message received in bytes.
 */
typedef void (*dsl_message_broker_subscriber_cb)(void* client_data, 
    uint status, void* message, uint length, const wchar_t* topic);
    
/**
 * @brief callback typedef for a client to receive and handle
 * incoming message processing errors. 
 * @param[in] client_data opaque pointer to client's user data.
 * @param[in] status status code returned by the Message Broker
 */
typedef void (*dsl_message_broker_connection_listener_cb)(void* client_data, 
    uint status);

/**
 * @brief callback typedef for a client to receive an asynchronus result
 * from calling dsl_message_broker_message_send.
 * @param[in] client_data opaque pointer to client's user data.
 * @param[in] status status code returned by the Message Broker
 */
typedef void (*dsl_message_broker_send_result_listener_cb)(void* client_data, 
    uint status);

/**
 * @brief callback typedef for a client to provide RGBA color parameters
 * on demand from an RGBA On-Demand Color Display Type.
 * @param[out] red red level for the RGB color [0..1].
 * @param[out] blue blue level for the RGB color [0..1].
 * @param[out] green green level for the RGB color [0..1].
 * @param[out] alpha alpha level for the RGB color [0..1].
 * @param[in] client_data opaque pointer to client's user data.
 */
typedef void (*dsl_display_type_rgba_color_provider_cb)(double* red, 
    double* green, double* blue, double* alpha, void* client_data);

// -----------------------------------------------------------------------------------
// Start of DSL Services 

/**
 * @brief Creates a uniquely named Custom RGBA Display Color.
 * Note: this is a static color for the life of the color object. 
 * @param[in] name unique name for the RGBA Color.
 * @param[in] red red level for the RGB color [0..1].
 * @param[in] blue blue level for the RGB color [0..1].
 * @param[in] green green level for the RGB color [0..1].
 * @param[in] alpha alpha level for the RGB color [0..1].
 * @return DSL_RESULT_SUCCESS on successful creation, one of 
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_color_custom_new(const wchar_t* name, 
    double red, double green, double blue, double alpha);

/**
 * @brief Creates a uniquely named RGBA Predefined Display Color. 
 * Note: this is a static color for the life of the color object. 
 * @param[in] name unique name for the RGBA Predefined Color.
 * @param[in] color_id one of the DSL_COLOR_PREDEFINED_* contants.
 * @param[in] alpha alpha level for the RGBA color [0..1].
 * @return DSL_RESULT_SUCCESS on successful creation, one of 
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_color_predefined_new(const wchar_t* name, 
    uint color_id, double alpha);

/**
 * @brief Creates a uniquely named RGBA Random Display Color.
 * Note: this is a dynamic color that regenerates on on new instance events.
 * @param[in] name unique name for the RGBA Random Color.
 * @param[in] hue one of the DSL_COLOR_HUE_* constants, use DSL_COLOR_HUE_RANDOM
 * for a full random color spectrum.
 * @param[in] luminosity one of the DSL_LUMINOSITY_* constants, use 
 * DSL_LUMINOSITY_RANDOM for random luminosity.
 * @param[in] alpha alpha level for the RGB Random color [0..1].
 * @param[in] seed value to seed the random generator.
 * @return DSL_RESULT_SUCCESS on successful creation, one of
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_color_random_new(const wchar_t* name, 
    uint hue, uint luminosity, double alpha, uint seed);

/**
 * @brief Creates a uniquely named RGBA On Demand Color. 
 * Note: this is a dynamic color that calls on the client color provider
 * callback funtion on new instance events.
 * @param[in] name unique name for the RGBA On-Demand Color.
 * @param[in] provider client callback function to provide RGBA color 
 * values on next color needed.
 * @param[in] client_data opaque pointer to client's user data.
 * @return DSL_RESULT_SUCCESS on successful creation, one of 
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_color_on_demand_new(const wchar_t* name, 
    dsl_display_type_rgba_color_provider_cb provider, void* client_data);

/**
 * @brief Creates a uniquely named RGBA Display Color Palette. The palette can
 * constist of a combination of Client defined and predefined static RGBA colors.
 * Note: this is a dynamic color that cycles through the provided palette 
 * of colors on new instance events.
 * @param[in] name unique name for the RGBA Color Palette.
 * @param[in] colors a null terminated list of RGBA Colors. 
 * @return DSL_RESULT_SUCCESS on successful creation, one of 
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_color_palette_new(const wchar_t* name, 
    const wchar_t** colors);

/**
 * @brief Creates a uniquely named Predefined RGBA Display Color Palette.
 * Note: this is a dynamic color that cycles through the provided palette 
 * of colors on new instance events. 
 * @param[in] name unique name for the RGBA Color Palette.
 * @param[in] palette_id one of the DSL_COLOR_PREDEFINED_PALETTE* constants. 
 * @param[in] alpha alpha level for the RGBA Predefined Color Palette.
 * @return DSL_RESULT_SUCCESS on successful creation, one of 
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_color_palette_predefined_new(const wchar_t* name, 
    uint palette_id, double alpha);


/**
 * @brief Creates a uniquely named Random RGBA Display Color Palette.
 * Note: this is a dynamic color that cycles through the palette of random
 * colors on new instance events. 
 * @param[in] name unique name for the RGBA Random Color Palette.
 * @param[in] size size of the color palette to create. 
 * @param[in] hue one of the DSL_COLOR_HUE_* constants, use DSL_COLOR_HUE_RANDOM
 * for a full random color spectrum.
 * @param[in] luminosity one of the DSL_LUMINOSITY_* constants, use 
 * DSL_LUMINOSITY_RANDOM for random luminosity.
 * @param[in] alpha alpha level for the RGB Random color [0..1].
 * @param[in] seed value to seed the random generator.
 * @return DSL_RESULT_SUCCESS on successful creation, one of
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_color_palette_random_new(const wchar_t* name, 
    uint size, uint hue, uint luminosity, double alpha, uint seed);
    
/**
 * @brief Gets the current index value for the named RGBA Color Palette 
 * @param[in] name unique name of the RGBA Color Palette to query
 * @param[out] index current index into the Color Palette's set of colors.
 * @return DSL_RESULT_SUCCESS on successful query, one of 
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_color_palette_index_get(const wchar_t* name, 
    uint* index);

/**
 * @brief Sets the index value for the named RGBA Color Palette 
 * @param[in] name unique name of the RGBA Color Palette to update
 * @param[in] index new index into for the Color Palette's set of colors.
 * @return DSL_RESULT_SUCCESS on successful query, one of 
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_color_palette_index_set(const wchar_t* name, 
    uint index);

/**
 * @brief Sets a Dynamic RGBA Color Type to its next color. For Palette,
 * Random, and On-Demand RGBA Colors.
 * @param[in] name unique name of the dynamic RGBA Color to update.
 * @return DSL_RESULT_SUCCESS on successful creation, one of 
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_color_next_set(const wchar_t* name);

/**
 * @brief Creates a uniquely named RGBA Display Font.
 * @param[in] name unique name for the RGBA Font.
 * @param[in] fount standard, unique string name of the actual font type (eg. 'arial').
 * @param[in] size size of the font.
 * @param[in] color name of the RGBA Color for the RGBA font.
 * @return DSL_RESULT_SUCCESS on successful creation, one of 
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_font_new(const wchar_t* name, 
    const wchar_t* font, uint size, const wchar_t* color);

/**
 * @brief Creates a uniquely named RGBA Display Text.
 * @param[in] name unique name of the RGBA Text.
 * @param[in] text text string to display.
 * @param[in] x_offset starting x positional offset.
 * @param[in] y_offset starting y positional offset.
 * @param[in] font RGBA font to use for the display text.
 * @param[in] has_bg_color set to true to enable background color, false otherwise.
 * @param[in] bg_color RGBA Color for the Text background if set.
 * @return DSL_RESULT_SUCCESS on successful creation, one of 
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_text_new(const wchar_t* name, 
    const wchar_t* text, uint x_offset, uint y_offset, const wchar_t* font, 
    boolean has_bg_color, const wchar_t* bg_color);
    
/**
 * @brief Creates a uniquely named RGBA Display Line.
 * @param[in] name unique name for the RGBA LIne.
 * @param[in] x1 starting x positional offest.
 * @param[in] y1 starting y positional offest.
 * @param[in] x2 ending x positional offest.
 * @param[in] y2 ending y positional offest.
 * @param[in] width width of the line in pixels.
 * @param[in] color RGBA Color for thIS RGBA Line.
 * @return DSL_RESULT_SUCCESS on successful creation, one of 
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_line_new(const wchar_t* name, 
    uint x1, uint y1, uint x2, uint y2, uint width, const wchar_t* color);

/**
 * @brief Creates a uniquely named RGBA Display Arrow.
 * @param[in] name unique name for the RGBA Arrow.
 * @param[in] x1 starting x positional offest.
 * @param[in] y1 starting y positional offest.
 * @param[in] x2 ending x positional offest.
 * @param[in] y2 ending y positional offest.
 * @param[in] width width of the Arrow in pixels.
 * @param[in] head DSL_ARROW_START_HEAD, DSL_ARROW_END_HEAD, DSL_ARROW_BOTH_HEAD
 * @param[in] color RGBA Color for thIS RGBA Line.
 * @return DSL_RESULT_SUCCESS on successful creation, one of 
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_arrow_new(const wchar_t* name, 
    uint x1, uint y1, uint x2, uint y2, uint width, uint head, const wchar_t* color);

/**
 * @brief Creates a uniquely named RGBA Rectangle.
 * @param[in] name unique name for the RGBA Rectangle.
 * @param[in] left left positional offest.
 * @param[in] top positional offest.
 * @param[in] width width of the rectangle in Pixels.
 * @param[in] height height of the rectangle in Pixels.
 * @param[in] border_width width of the rectangle border in pixels.
 * @param[in] color RGBA Color for thIS RGBA Line.
 * @param[in] hasBgColor set to true to enable bacground color, false otherwise.
 * @param[in] bgColor RGBA Color for the Circle background if set.
 * @return DSL_RESULT_SUCCESS on successful creation, one of 
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_rectangle_new(const wchar_t* name, 
    uint left, uint top, uint width, uint height, uint border_width, const wchar_t* color, 
    bool has_bg_color, const wchar_t* bg_color);

/**
 * @brief Creates a uniquely named RGBA Polygon.
 * @param[in] name unique name for the RGBA Polygon.
 * @param[in] coordinate an array of dsl_coordinate structures.
 * @param[in] num_coordinates the number of xy coordinates in the array.
 * @param[in] border_width width of the polygon border in pixels.
 * @param[in] color RGBA Color for the polygon border.
 * @return DSL_RESULT_SUCCESS on successful creation, one of 
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_polygon_new(const wchar_t* name, 
    const dsl_coordinate* coordinates, uint num_coordinates, uint border_width, 
    const wchar_t* color);

/**
 * @brief Creates a uniquely named RGBA Multi-Line.
 * @param[in] name unique name for the RGBA Multi-Line.
 * @param[in] coordinate an array of dsl_coordinate structures.
 * @param[in] num_coordinates the number of xy coordinates in the array.
 * @param[in] border_width width of the multi-line in pixels.
 * @param[in] color RGBA Color for the multi-line.
 * @return DSL_RESULT_SUCCESS on successful creation, one of 
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_line_multi_new(const wchar_t* name, 
    const dsl_coordinate* coordinates, uint num_coordinates, uint border_width, 
    const wchar_t* color);

/**
 * @brief Creates a uniquely named RGBA Circle.
 * @param[in] name unique name for the RGBA Circle.
 * @param[in] x_center X positional offset to center of Circle.
 * @param[in] y_center y positional offset to center of Circle.
 * @param[in] radius radius of the RGBA Circle in pixels.
 * @param[in] color RGBA Color for the RGBA Circle.
 * @param[in] hasBgColor set to true to enable bacground color, false otherwise.
 * @param[in] bgColor RGBA Color for the Circle background if set.
 * @return DSL_RESULT_SUCCESS on successful creation, one of 
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_circle_new(const wchar_t* name, 
    uint x_center, uint y_center, uint radius, const wchar_t* color, bool has_bg_color, 
    const wchar_t* bg_color);

/**
 * @brief Creates a uniquely named Source Number Display Type.
 * @param[in] name unique name of the Display Type.
 * @param[in] x_offset starting x positional offset.
 * @param[in] y_offset starting y positional offset.
 * @param[in] font RGBA font to use for the display text.
 * @param[in] hasBgColor set to true to enable bacground color, false otherwise.
 * @param[in] bg_color RGBA Color for the Text background if set.
 * @return DSL_RESULT_SUCCESS on successful creation, one of 
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_source_number_new(const wchar_t* name, 
    uint x_offset, uint y_offset, const wchar_t* font, boolean has_bg_color, 
    const wchar_t* bg_color);
    
/**
 * @brief Creates a uniquely named Source Name Display Type.
 * @param[in] name unique name of the Display Type.
 * @param[in] x_offset starting x positional offset.
 * @param[in] y_offset starting y positional offset.
 * @param[in] font RGBA font to use for the display text.
 * @param[in] hasBgColor set to true to enable bacground color, false otherwise.
 * @param[in] bg_color RGBA Color for the Text background if set.
 * @return DSL_RESULT_SUCCESS on successful creation, one of 
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_source_name_new(const wchar_t* name, 
    uint x_offset, uint y_offset, const wchar_t* font, boolean has_bg_color, 
    const wchar_t* bg_color);
    
/**
 * @brief Creates a uniquely named Source Dimensions Display Type.
 * @param[in] name unique name of the Display Type.
 * @param[in] x_offset starting x positional offset.
 * @param[in] y_offset starting y positional offset.
 * @param[in] font RGBA font to use for the display text.
 * @param[in] hasBgColor set to true to enable bacground color, false otherwise.
 * @param[in] bg_color RGBA Color for the Text background if set.
 * @return DSL_RESULT_SUCCESS on successful creation, one of 
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_source_dimensions_new(const wchar_t* name, 
    uint x_offset, uint y_offset, const wchar_t* font, boolean has_bg_color, 
    const wchar_t* bg_color);

/**
 * @brief Adds a shadow to the Text Display Type creating a raised effect.
 * @param[in] name unique name of the Text or Source-Info Display Type.
 * @param[in] x_offset shadow offset in the x direction.
 * @param[in] y_offset shadow offset in the y direction.
 * @param[in] color RGBA Color for the text shadow.
 * @return DSL_RESULT_SUCCESS on successful creation, one of 
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_rgba_text_shadow_add(const wchar_t* name, 
    uint x_offset, uint y_offset, const wchar_t* color);
    
///**
// * @brief Adds a named Display Type (text/shape) to a frames's display metadata, The caller 
// * is responsible for aquiring the display metadata for the current frame.
// * @param name unique name of the Display Type to overlay
// * @param display_meta opaque pointer to the aquired display meta to to add the Display Type to
// * @param frame_meta opaque pointer to a Frame's meta data to add the Display Type
// * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
// */
//DslReturnType dsl_display_type_meta_add(const wchar_t* name, void* buffer, void* frame_meta);
    
/**
 * @brief Deletes a uniquely named Display Type of any type.
 * @param[in] name unique name for the Display Type to delete.
 * @return DSL_RESULT_SUCCESS on successful deletion, one of 
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_delete(const wchar_t* name);

/**
 * @brief Deletes a Null terminated array of Display Types of any type.
 * @param[in] names Null ternimated array of unique names to delete.
 * @return DSL_RESULT_SUCCESS on successful deletion, one of 
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_delete_many(const wchar_t** names);

/**
 * @brief Deletes all Display Types currently in memory.
 * @return DSL_RESULT_SUCCESS on successful deletion, one of 
 * DSL_RESULT_DISPLAY_TYPE_RESULT otherwise.
 */
DslReturnType dsl_display_type_delete_all();

/**
 * @brief Returns the size of the list of Display Types.
 * @return the number of Display Types in the list.
 */
uint dsl_display_type_list_size();

/**
 * @brief Creates a uniquely named Capture Frame ODE Action
 * @param[in] name unique name for the Capture Frame ODE Action
 * @param[in] outdir absolute or relative path to image capture directory 
 * @param[in] annotate if true, bounding boxes and labes will be added to the image.
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_capture_frame_new(const wchar_t* name, 
    const wchar_t* outdir, boolean annotate);

/**
 * @brief Creates a uniquely named Capture Object ODE Action
 * @param[in] name unique name for the Capture Object ODE Action 
 * @param[in] outdir absolute or relative path to image capture directory 
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_capture_object_new(const wchar_t* name, 
    const wchar_t* outdir);

/**
 * @brief Adds a callback to be notified on Image Capture complete.
 * @param[in] name unique name of the Capture Action to update
 * @param[in] listener pointer to the client's function to call on capture complete
 * @param[in] client_data opaque pointer to client data passed into the listener function
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_capture_complete_listener_add(const wchar_t* name, 
    dsl_capture_complete_listener_cb listener, void* client_data);

/**
 * @brief Removes a callback previously added with dsl_ode_action_capture_complete_listener_add
 * @param[in] name unique name of the Capture Action to update
 * @param[in] listener pointer to the client's function to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_capture_complete_listener_remove(const wchar_t* name, 
    dsl_capture_complete_listener_cb listener);

/**
 * @brief Adds an Image Player, Render or RTSP type, to a named Capture Action.
 * Once added, each captured image's file_path will be added (or queued) with
 * the Image Player to be played according to the Players settings. 
 * @param[in] name unique name of the Capture Action to update
 * @param[in] player unique name of the Image Player to add
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_capture_image_player_add(const wchar_t* name, 
    const wchar_t* player);
    
/**
 * @brief Removes an Image Player, Render or RTSP type, from a named Capture Action.
 * @param[in] name unique name of the Capture Action to update
 * @param[in] player unique name of the Image Player to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_capture_image_player_remove(const wchar_t* name, 
    const wchar_t* player);
    
/**
 * @brief Adds a SMTP mailer to a named Capture Action. Once added, each
 * captured image's file_path and details will be sent out according to the 
 * Mailer's settings. The image file can be attached to the email as an option.
 * @param[in] name unique name of the Capture Action to update
 * @param[in] mailer unique name of the Mailer to add
 * @param[in] subject subject line to use for all outgoing mail
 * @param[in] attach set to true to attach the image file, false otherwise
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_capture_mailer_add(const wchar_t* name, 
    const wchar_t* mailer, const wchar_t* subject, boolean attach);
    
/**
 * @brief Removes a named Mailer from a named Capture Action.
 * @param[in] name unique name of the Capture Action to update
 * @param[in] mailer unique name of the Mailer to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_capture_mailer_remove(const wchar_t* name, 
    const wchar_t* mailer);

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
 * @brief Creates a uniquely named "Customize Object Label" ODE Action that updates 
 * an Object's label to display specific content.
 * @param[in] name unique name for the "Customize Object Label ODE Action. 
 * @param[in] content_types an array of DSL_OBJECT_LABEL_<type> constants.
 * @param[in] size of the content_types array 
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_customize_label_new(const wchar_t* name,  
    const uint* content_types, uint size);

/**
 * @brief Gets the current content_types, size and write mode settings for the 
 * "Customize Object Label" ODE Action 
 * @param[in] name unique name for the "Customize Object Label ODE Action to query. 
 * @param[out] content_types an array of DSL_OBJECT_LABEL_<type> constants.
 * @param[inout] size max size of the content_types array on call, actual size on return
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_customize_label_get(const wchar_t* name,  
    uint* content_types, uint* size);
    
/**
 * @brief Sets the content_types, size and write mode settings for the 
 * "Customize Object Label" ODE Action,
 * @param[in] name unique name for the "Customize Object Label ODE Action. 
 * @param[in] content_types an array of DSL_OBJECT_LABEL_<type> constants.
 * @param[in] size of the content_types array 
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_customize_label_set(const wchar_t* name,  
    const uint* content_types, uint size);
    
/**
 * @brief Creates a uniquely named Display ODE Action
 * @param[in] name unique name for the ODE Display Action 
 * @param[in] format_string string with format tokens for display
 * @param[in] offset_x offset in the X direction for the Display text
 * @param[in] offset_y offset in the Y direction for the Display text
 * @param[in] font RGBA Font type to use for the Display text
 * @param[in] has_bg_color if true, displays the background color for the Display Text
 * @param[in] bg_color color to use for the Display Text background color, if has_bg_color
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_display_new(const wchar_t* name, 
    const wchar_t* format_string, uint offset_x, uint offset_y, 
    const wchar_t* font, boolean has_bg_color, const wchar_t* bg_color);

/**
 * @brief Creates a uniquely named Add Display Metadata ODE Action to add Display metadata
 * using a uniquely named Display Type 
 * @param[in] name unique name for the Add Display Metadata ODE Action 
 * @param[in] display_type unique name of the Display Type to overlay on ODE occurrence
 * Note: the Display Type must exist prior to constructing the Action.
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_display_meta_add_new(const wchar_t* name, 
    const wchar_t* display_type);

/**
 * @brief Creates a uniquely named Add Many Display Metadata ODE Action to add the 
 * metadata using multiple uniquely named Display Types 
 * @param[in] name unique name for the Add Many Display Metadata ODE Action 
 * @param[in] display_typess NULL terminated list of names of the Display 
 * Types to overlay on ODE occurrence
 * Note: the Display Type must exist prior to constructing the Action.
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_display_meta_add_many_new(const wchar_t* name, 
    const wchar_t** display_types);

/**
 * @brief Creates a uniquely named Email ODE Action, that sends an email message using the
 * SMTP Mailer Object specified by its unique name.
 * @param[in] name unique name for the Email ODE Action
 * @param[in] mailer unique name of the SMTP Mailer Object to use.
 * @param[in] subject text to use as the subject line for all messages sent from this Action
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_email_new(const wchar_t* name, 
    const wchar_t* mailer, const wchar_t* subject);

/**
 * @brief Creates a uniquely named File ODE Action, that write the ODE Event Info to file.
 * @param[in] name unique name for the File ODE Action
 * @param[in] file_path absolute or relative file path of the output file to use
 * The file will be created if one does exists, or opened for append if found.
 * @param[in] mode file open/write mode, one of DSL_EVENT_FILE_MODE_* options
 * @param[in] format one of the DSL_EVENT_FILE_FORMAT_* options
 * @param[in] force_flush  if true, the action will schedule a flush to be performed 
 * by the idle thread. NOTE: although the flush event occurs in a background thread,
 * flushing is still a CPU intensive operation and should be used sparingly, when tailing
 * the file for runtime debugging as an example. Set to 0 to disable forced flushing, 
 * and to allow the operating system to more effectively handle the stream flushing.
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_file_new(const wchar_t* name, 
    const wchar_t* file_path, uint mode, uint format, boolean force_flush);
    
/**
 * @brief Creates a uniquely named Fill Frame ODE Action, that fills the entire
 * frame with a give RGBA color value
 * @param[in] name unique name for the Fill Frame ODE Action
 * @param[in] color name of the RGBA Color to use for the fill action
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_fill_frame_new(const wchar_t* name, const wchar_t* color);

/**
 * @brief Creates a uniquely named Fill Surroundings ODE Action, that fills the entire
 * frame area surroudning an Object's rectangle
 * @param[in] name unique name for the Fill Frame ODE Action
 * @param[in] color nane of a RGBA color that must exist prior to creating the Action
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_fill_surroundings_new(const wchar_t* name, const wchar_t* color);

/**
 * @brief Creates a uniquely named "Format Bounding Box" ODE Action that updates 
 * an Object's RGBA bounding box line width and color. 
 * Note: setting the line width to 0 will exclude/hide the object's bounding box from view.
 * @param[in] name unique name for the "Format Bounding Box" ODE Action. 
 * @param[in] border_width border line width for the object's bounding box. 
 * Set to 0 to exclude/hide the border from view.
 * @param[in] border_color RGBA Color for the bounding box.
 * @param[in] has_bg_color set to true to enable background color, false otherwise
 * @param[in] bg_color RGBA Color for the Text background if set
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_format_bbox_new(const wchar_t* name, uint border_width, 
    const wchar_t* border_color, boolean has_bg_color, const wchar_t* bg_color);

/**
 * @brief Creates a uniquely named "Format Label" ODE Action that updates 
 * an Object's RGBA Label Font
 * @param[in] name unique name for the Format Label ODE Action 
 * @param[in] font RGBA font to use for the object's label
 * @param[in] has_bg_color set to true to enable background color, false otherwise
 * @param[in] bg_color RGBA Color for the Text background if has_bg_color = true
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_format_label_new(const wchar_t* name, 
    const wchar_t* font, boolean has_bg_color, const wchar_t* bg_color);
    

/**
 * @brief Creates a uniquely named Disable Handler Action that disables
 * a namded Handler
 * @param[in] name unique name for the Fill Backtround ODE Action
 * @param[in] handler unique name of the Handler to disable
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_handler_disable_new(const wchar_t* name, const wchar_t* handler);

/**
 * @brief Creates a uniquely named Log ODE Action
 * @param[in] name unique name for the Log ODE Action 
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_log_new(const wchar_t* name);

/**
 * @brief Creates a uniquely named Message Meta Add ODE Action that attaches NvDsEventMsgMeta
 * to the NvDsFrameMeta on ODE occurrence.
 * @param[in] name unique name for the Message ODE Action 
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_message_meta_add_new(const wchar_t* name);

///**
// * @brief Gets the current meta-type identifier in use by the named Message Sink.
// * @param[in] name unique name of the Message ODE Action to query.
// * @param[out] meta_type the current meta-type in use, default = NVDS_EVENT_MSG_META.
// * @return DSL_RESULT_SUCCESS on successful query, one of the 
// * DSL_RESULT_ODE_ACTION_RESULT values otherwise.
// */
//DslReturnType dsl_ode_action_message_meta_type_get(const wchar_t* name,
//    uint* meta_type);
//
///**
// * @brief Sets the meta-type identifier for the named Message Sink to use
// * @param[in] name unique name of the Message ODE Action to update
// * @param[in] meta_type the new meta-type to use, must > or = NVDS_START_USER_META.
// * @return DSL_RESULT_SUCCESS on successful update, one of the 
// * DSL_RESULT_ODE_ACTION_RESULT values otherwise.
// */
//DslReturnType dsl_ode_action_message_meta_type_set(const wchar_t* name,
//    uint meta_type);

/**
 * @brief Creates a uniquely named Monitor ODE Action.
 * @param[in] name unique name for the Monitor ODE Action. 
 * @param[in] client_monitor function to call on ODE occurrence. 
 * @param[in] client_data opaue pointer to client's user data, returned on callback.
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_monitor_new(const wchar_t* name, 
    dsl_ode_monitor_occurrence_cb client_monitor, void* client_data);

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
 * @param[in] force_flush  if true, the action will schedule a flush to be performed 
 * by the idle thread. NOTE: although the flush event occurs in a background thread,
 * flushing is still a CPU intensive operation and should be used sparingly, when tailing
 * the console ouput for runtime debugging as an example. Set to 0 to disable forced flushing, 
 * and to allow the operating system to more effectively handle the process.
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_print_new(const wchar_t* name, boolean force_flush);
    
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
 * @brief Creates a uniquely named Stop Record Sink ODE Action
 * @param[in] name unique name for the Print ODE Action 
 * @param[in] record_sink unique name of the Record Sink to stop recording
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_sink_record_stop_new(const wchar_t* name,
    const wchar_t* record_sink);

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
    const wchar_t* record_tap, uint start, uint duration, void* client_data);

/**
 * @brief Creates a uniquely named Stop Record Tap ODE Action
 * @param[in] name unique name for the Stop Record Tap Action 
 * @param[in] record_tap unique name of the Record Tap to stop recording
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_ode_action_tap_record_stop_new(const wchar_t* name,
    const wchar_t* record_tap);

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
 * @brief Adds a callback to be notified on change of enabled state for a named
 * ODE Action. 
 * @param[in] name name of the ODE Action to update.
 * @param[in] listener pointer to the client's function to call on state change
 * @param[in] client_data opaque pointer to client data passed into the listener function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_ode_action_enabled_state_change_listener_add(const wchar_t* name,
    dsl_ode_enabled_state_change_listener_cb listener, void* client_data);

/**
 * @brief Removes a callback previously added with a call to
 * dsl_ode_action_enabled_state_change_listener_add.
 * @param[in] name name of the ODE Action to update.
 * @param[in] listener pointer to the client's function to remove.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_ode_action_enabled_state_change_listener_remove(const wchar_t* name,
    dsl_ode_enabled_state_change_listener_cb listener);
    
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
 * @param[in] polygon name of an RGBA Polygon Type used to define the Area
 * @param[in] show set to true to show (overlay) the type on each frame
 * @param[in] bbox_test_point one of DSL_BBOX_POINT values defining which point of a
 * object's bounding box to use when testing for inclusion Area
 * @return DSL_RESULT_SUCCESS on successful create, DSL_RESULT_ODE_AREA_RESULT otherwise.
 */
DslReturnType dsl_ode_area_inclusion_new(const wchar_t* name, 
    const wchar_t* polygon, boolean show, uint bbox_test_point);

/**
 * @brief Creates a uniquely named ODE Exclusion Area
 * @param[in] name unique name of the ODE area to create
 * @param[in] polygon name of an RGBA Polygon Type used to define the Area
 * @param[in] show set to true to show (overlay) the type on each frame
 * @param[in] bbox_test_point one of DSL_BBOX_POINT values defining which point of a
 * object's bounding box to use when testing for exclusion from Area
 * @return DSL_RESULT_SUCCESS on successful create, DSL_RESULT_ODE_AREA_RESULT otherwise.
 */
DslReturnType dsl_ode_area_exclusion_new(const wchar_t* name, 
    const wchar_t* polygon, boolean show, uint bbox_test_point);

/**
 * @brief Creates a uniquely named ODE Line Area
 * @param[in] name unique name of the ODE Line Area to create
 * @param[in] line name of an RGBA Line used to define location, dimensions, color
 * @param[in] show set to true to show (overlay) the line on each frame
 * @param[in] bbox_test_point one of DSL_BBOX_POINT values defining which point of a
 * object's bounding box to use when testing for line crossing
 * @return DSL_RESULT_SUCCESS on successful create, DSL_RESULT_ODE_AREA_RESULT otherwise.
 */
DslReturnType dsl_ode_area_line_new(const wchar_t* name,
    const wchar_t* line, boolean show, uint bbox_test_point);

/**
 * @brief Creates a uniquely named ODE Multi-Line Area
 * @param[in] name unique name of the ODE Multi Line Area to create
 * @param[in] multi_line name of an RGBA Multi-Line used to define location, 
 * dimensions, color
 * @param[in] show set to true to show (overlay) the line on each frame
 * @param[in] bbox_test_point one of DSL_BBOX_POINT values defining which point of a
 * object's bounding box to use when testing for line crossing
 * @return DSL_RESULT_SUCCESS on successful create, DSL_RESULT_ODE_AREA_RESULT otherwise.
 */
DslReturnType dsl_ode_area_line_multi_new(const wchar_t* name,
    const wchar_t* multi_line, boolean show, uint bbox_test_point);

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
 * @param[in] when DSL_ODE_PRE_OCCURRENCE_CHECK or DSL_ODE_POST_OCCURRENCE_CHECK
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_always_new(const wchar_t* name, const wchar_t* source, uint when);

/**
 * @brief Absence trigger that checks for the absence of Objects within a frame
 * @param[in] name unique name for the ODE Trigger
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE
 * @param[in] class_id class id filter for this ODE Trigger
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_absence_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit);

/**
 * @brief Count trigger that checks for the occurrence of Objects within a frame
 * and tests if the count is within a specified range.
 * @param[in] name unique name for the ODE Trigger
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE
 * @param[in] class_id class id filter for this ODE Trigger
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit
 * @param[in] minimum the minimum count for triggering ODE occurrence, 0 = no minimum
 * @param[in] maximum the maximum count for triggering ODE occurrence, 0 = no maximum
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_count_new(const wchar_t* name, const wchar_t* source, 
    uint class_id, uint limit, uint minimum, uint maximum);

/**
 * @brief Gets the current minimum and maximum count settings in use 
 * by the named Count Trigger
 * @param[in] name unique name of the Count Trigger to query
 * @param[out] minimum the current minimum count for triggering ODE occurrence, 0 = no minimum
 * @param[out] maximum the current maximum count for triggering ODE occurrence, 0 = no maximum
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_count_range_get(const wchar_t* name, 
    uint* minimum, uint* maximum);

/**
 * @brief Sets the minimum and maximum count settings to use for a 
 * named Count Trigger
 * @param[in] name unique name of the Count Trigger to update
 * @param[in] minimum the new minimum count for triggering ODE occurrence, 0 = no minimum
 * @param[in] maximum the new maximum count for triggering ODE occurrence, 0 = no maximum
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_count_range_set(const wchar_t* name, 
    uint minimum, uint maximum);

/**
 * @brief Occurence trigger that checks for a new instance of an Object for a 
 * specified source and object class_id. Instance identification is based on Tracking Id
 * @param[in] name unique name for the ODE Trigger
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE
 * @param[in] class_id class id filter for this ODE Trigger
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_instance_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit);

/**
 * @brief Intersection trigger that checks for the intersection of detected Objects 
 * and triggers an ODE occurrence for each unique overlaping pair. Detected objects are
 * tested using an A-B comparison of class_ids as specified. 
 * @param[in] name unique name for the ODE Trigger
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE
 * @param[in] class_id_a class id A filter for this ODE Trigger
 * @param[in] class_id_b class id B filter for this ODE Trigger
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_intersection_new(const wchar_t* name, 
    const wchar_t* source, uint class_id_a, uint class_id_b, uint limit);

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
 * @brief Occurence trigger that checks for the occurrence of Objects within a frame for a 
 * specified source and object class_id.
 * @param[in] name unique name for the ODE Trigger
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE
 * @param[in] class_id class id filter for this ODE Trigger
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_occurrence_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit);

/**
 * @brief Smallest trigger that checks for the occurrence of Objects within a frame
 * and if at least one is found, Triggers on the Object with smallest rectangle area.
 * @param[in] name unique name for the ODE Trigger
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE
 * @param[in] class_id class id filter for this ODE Trigger
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_smallest_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit);

/**
 * @brief Largest trigger that checks for the occurrence of Objects within a frame
 * and if at least one is found, Triggers on the Object with larget rectangle area.
 * @param[in] name unique name for the ODE Trigger
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE
 * @param[in] class_id class id filter for this ODE Trigger
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_largest_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit);

/**
 * @brief Summation trigger that checks for and sums all objects detected within a frame
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE
 * @param[in] name unique name for the ODE Trigger
 * @param[in] class_id class id filter for this ODE Trigger
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_summation_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit);


/**
 * @brief New high-count trigger that checks for the occurrence of a new high count of objects within 
 * a frame for a specified source and object class_id.
 * @param[in] name unique name for the ODE Trigger
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE
 * @param[in] class_id class id filter for this ODE Trigger
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit
 * @param[in] preset initial high count to start with. High count will be reset to the preset on trigger reset.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_new_high_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit, uint preset);

/**
 * @brief New low-count trigger that checks for the occurrence of a new low count of objects within 
 * a frame for a  specified source and object class_id. This trigger can be added in a disabled state and 
 * then enabled by a new-high count trigger on first new-high count occurrence.
 * @param[in] name unique name for the ODE Trigger
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE
 * @param[in] class_id class id filter for this ODE Trigger
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit
 * @param[in] preset initial low count to start with. High count will be reset to the preset on trigger reset.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_new_low_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit, uint preset);

/**
 * @brief Distance trigger that checks for the occurrence of two Objects that are below a minimum and/or
 * above a maximum specified distance, and generates an ODE occurrence if detected. Detected objects are
 * tested using an A-B comparison of class_ids as specified. 
 * @param[in] name unique name for the ODE Trigger
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE
 * @param[in] class_id_a class id A filter for this ODE Trigger
 * @param[in] class_id_b class id B filter for this ODE Trigger
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit
 * @param[in] minimum the minimum distance between objects in either pixels or percentage of BBox point
 * as specified by the test_method parameter below.
 * @param[in] maximum the maximum distance between objects in either pixels or percentage of BBox point
 * as specified by the test_method parameter below.
 * @param[in] test_point the point on the bounding box rectangle to use for measurement, one of DSL_BBOX_POINT
 * @param[in] test_method method of measuring the distance between objects, one of DSL_DISTANCE_METHOD
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_distance_new(const wchar_t* name, const wchar_t* source, 
    uint class_id_a, uint class_id_b, uint limit, uint minimum, uint maximum, 
    uint test_point, uint test_method);
    
/**
 * @brief Gets the current minimum and maximum distance settings in use 
 * be the named Distance Trigger
 * @param[in] name unique name of the Distance Trigger to query
 * @param[out] minimum the minimum distance between objects in either pixels or 
 * percentage of BBox point as specified by the test_method parameter below.
 * @param[out] maximum the minimum distance between objects in either pixels or
 * percentage of BBox point as specified by the test_method parameter below.
 */
DslReturnType dsl_ode_trigger_distance_range_get(const wchar_t* name, 
    uint* minimum, uint* maximum);

/**
 * @brief Sets the minimum and maximum distance settings to use for a 
 * named Distance Trigger
 * @param[in] name unique name of the Distance Trigger to update
 * @param[in] minimum the minimum distance between objects in either pixels or 
 * percentage of BBox point as specified by the test_method parameter below.
 * @param[in] minimum the minimum distance between objects in either pixels or 
 * percentage of BBox point as specified by the test_method parameter below.
 */
DslReturnType dsl_ode_trigger_distance_range_set(const wchar_t* name, 
    uint minimum, uint maximum);
    
/**
 * @brief Gets the current Test Point and Test Methods parameters in 
 * use by the named Distance Trigger
 * @param[in] name unique name of the Distance Trigger to Query
 * @param[out] test_point the point on the bounding box rectangle used for 
 * measurement, one of DSL_BBOX_POINT
 * @param[out] test_method method of measuring the distance between objects, 
 * one of DSL_DISTANCE_METHOD
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_distance_test_params_get(const wchar_t* name, 
    uint* test_point, uint* test_method);    

/**
 * @brief sets the current Test Point and Test Methods parameters in 
 * use by the named Distance Trigger
 * @param[in] name unique name of the Distance Trigger to Query
 * @param[in] test_point the point on the bounding box rectangle to use 
 * for measurement, one of DSL_BBOX_POINT
 * @param[in] test_method method of measuring the distance between objects, 
 * one of DSL_DISTANCE_METHOD
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_distance_test_params_set(const wchar_t* name, 
    uint test_point, uint test_method);    

/**
 * @brief Cross trigger that tracks Objects and triggers on the occurrence that an 
 * object crosses the trigger's Line or Polygon Area.
 * @param[in] name unique name for the ODE Trigger
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE
 * @param[in] class_id class id filter for this ODE Trigger
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit
 * @param[in] min_frame_count setting for the minimum number of past 
 * consective frames on both sides of a line (line or polygon area) to trigger an ODE.
 * @param[in] max_frame_count maximum number of past (non-consecutive) frames to 
 * maintain for each tracked objects. 
 * @param[in] test_method one of the DSL_OBJECT_TRACE_TEST_METHOD_* constants.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_cross_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit, uint min_frame_count, 
    uint max_frame_count, uint test_method);

/**
 * @brief Persistence trigger that checks for the persistence of Objects tracked 
 * for a specified source and object class_id. Each object tracked for ">= minimum 
 * and <= maximum time will trigger an ODE occurrence.
 * @param[in] name unique name for the ODE Trigger
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE
 * @param[in] class_id class id filter for this ODE Trigger
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit
 * @param[in] minimum the minimum amount of time a unique object must remain detected 
 * before triggering an ODE occurrence - in units of seconds. 0 = no minimum
 * @param[in] maximum the maximum amount of time a unique object can remain detected 
 * before triggering an ODE occurrence - in units of seconds. 0 = no maximum
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_persistence_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit, uint minimum, uint maximum);

/**
 * @brief Gets the current minimum and maximum time settings in use 
 * by the named Persistence Trigger
 * @param[in] name unique name of the Persistence Trigger to query
 * @param[out] minimum the minimum amount of time a unique object must remain detected 
 * before triggering an ODE occurrence - in units of seconds. 0 = no minimum
 * @param[out] maximum the maximum amount of time a unique object can remain detected 
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_persistence_range_get(const wchar_t* name, 
    uint* minimum, uint* maximum);

/**
 * @brief Sets the minimum and maximum time settings to use for a 
 * named Persistence Trigger
 * @param[in] name unique name of the Persitence Trigger to update
 * @param[in] minimum the minimum amount of time a unique object must remain detected 
 * before triggering an ODE occurrence - in units of seconds. 0 = no minimum
 * @param[in] maximum the maximum amount of time a unique object can remain detected 
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_persistence_range_set(const wchar_t* name, 
    uint minimum, uint maximum);
    
/**
 * @brief Latest Trigger that checks for the persistence of Objects tracked 
 * and will trigger on the Object with the least time of persistence (latest)
 * if at least one is found.
 * @param[in] name unique name for the ODE Trigger
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE.
 * @param[in] class_id class id filter for this ODE Trigger.
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_latest_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit);

/**
 * @brief Earliest Trigger that checks for the persistence of Objects tracked 
 * and will trigger on the Object with the greatest time of persistence (earliest) 
 * if at least one is found.
 * @param[in] name unique name for the ODE Trigger.
 * @param[in] source unique source name filter for the ODE Trigger, NULL = ANY_SOURCE.
 * @param[in] class_id class id filter for this ODE Trigger.
 * @param[in] limit limits the number of ODE occurrences, a value of 0 = NO limit
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_earliest_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit);

/**
 * @brief Gets the current test settings for the named cross trigger.
 * @param[in] name unique name for the ODE Trigger to query
 * @param[out] min_frame_count current setting for the minimum number of past 
 * consective frames on both sides of a line (line or polygon area) to trigger an ODE.
 * @param[out] max_trace_points current setting for the maximum number of past 
 * trace points to maintain for each tracked objects. 
 * @param[out] test_method one of the DSL_OBJECT_TRACE_TEST_METHOD_* constants.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_cross_test_settings_get(const wchar_t* name, 
    uint* min_frame_count, uint* max_trace_points, uint* test_method);
    
/**
 * @brief Sets the max trace-points setting for the named cross trigger.
 * @param[in] name unique name for the ODE Trigger to update
 * @param[in] min_frame_count new setting for the minimum number of past 
 * consective frames on both sides of a line (line or polygon area) to trigger an ODE.
 * @param[in] max_trace_points new setting for the maximum number of past 
 * trace points to maintain for each tracked objects. 
 * @param[in] test_method one of the DSL_OBJECT_TRACE_TEST_METHOD_* constants.
* @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_cross_test_settings_set(const wchar_t* name, 
    uint min_frame_count, uint max_trace_points, uint test_method);
    
/**
 * @brief Gets the current trace settings for the named cross trigger.
 * @param[in] name unique name for the ODE Trigger to query
 * @param[out] enabled true if object trace display is enabled, default = disabled 
 * @param[out] color name of the color to use for object trace display, default = no-color.
 * @param[out] line_width width of the object trace if display is enabled.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_cross_view_settings_get(const wchar_t* name, 
    boolean* enabled, const wchar_t** color, uint* line_width);
    
/**
 * @brief Sets the trace settings for the named cross trigger.
 * @param[in] name unique name for the ODE Trigger to update
 * @param[in] enabled set to true to enable object trace display, false otherwise
 * @param[in] color name of the color to use for object trace display.
 * @param[in] line_width width of the object trace if display is enabled.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_cross_view_settings_set(const wchar_t* name, 
    boolean enabled, const wchar_t* color, uint line_width);
    

/**
 * @brief Resets the a named ODE Trigger, setting it's triggered count to 0
 * This affects Triggers with fixed limits, whether they have reached their limit or not.
 * @param[in] name unique name of the ODE Trigger to update
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_reset(const wchar_t* name);

/**
 * @brief Gets the current auto-reset timer setting for the named ODE Trigger. If set, 
 * the Trigger, upon reaching its limit, will start a timer to then auto-reset on expiration.
 * @param[in] name unique name of the ODE Trigger to update
 * @param[out] time in seconds after reaching trigger limit before auto reset. 0 = disabled (default)
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_reset_timeout_get(const wchar_t* name, uint *timeout);

/**
 * @brief Sets the auto-reset timer setting for the named ODE Trigger. If set, 
 * the Trigger, upon reaching its limit, will start a timer to then auto-reset on expiration.
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] time in seconds after reaching trigger limit before auto reset. Set to 0 to disable (default)
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_reset_timeout_set(const wchar_t* name, uint timeout);

/**
 * @brief Adds a callback to be notified on every ODE Trigger limit event;
 * LIMIT_REACED, LIMIT_CHANGED, and COUNT_REST
 * @param[in] name name of the ODE Trigger to update
 * @param[in] listener pointer to the client's function to call on trigger limit event
 * @param[in] client_data opaque pointer to client data passed into the listener function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_limit_event_listener_add(const wchar_t* name,
    dsl_ode_trigger_limit_event_listener_cb listener, void* client_data);

/**
 * @brief Removes a callback previously added with a call to
 * dsl_ode_trigger_limit_event_listener_add.
 * @param[in] name name of the ODE Trigger to update.
 * @param[in] listener pointer to the client's function to remove.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_limit_event_listener_remove(const wchar_t* name,
    dsl_ode_trigger_limit_event_listener_cb listener);

/**
 * @brief Gets the current enabled setting for the ODE Trigger.
 * @param[in] name unique name of the ODE Trigger to query.
 * @param[out] enabled true if the ODE Trigger is currently enabled, false otherwise.
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_enabled_get(const wchar_t* name, boolean* enabled);

/**
 * @brief Sets the enabled setting for the ODE Trigger.
 * @param[in] name unique name of the ODE Trigger to update.
 * @param[in] enabled true if the ODE Trigger is currently enabled, false otherwise.
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_enabled_set(const wchar_t* name, boolean enabled);

/**
 * @brief Adds a callback to be notified on change of enabled state for a named
 * ODE Trigger. 
 * @param[in] name name of the ODE Trigger to update.
 * @param[in] listener pointer to the client's function to call on state change
 * @param[in] client_data opaque pointer to client data passed into the listener function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_enabled_state_change_listener_add(const wchar_t* name,
    dsl_ode_enabled_state_change_listener_cb listener, void* client_data);

/**
 * @brief Removes a callback previously added with a call to
 * dsl_ode_trigger_enabled_state_change_listener_add.
 * @param[in] name name of the ODE Trigger to update.
 * @param[in] listener pointer to the client's function to remove.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_enabled_state_change_listener_remove(const wchar_t* name,
    dsl_ode_enabled_state_change_listener_cb listener);

/**
 * @brief Gets the current source name filter for the ODE Trigger
 * A value of NULL indicates filter disabled.
 * @param[in] name unique name of the ODE Trigger to query.
 * @param[out] source returns the current source name in use.
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_source_get(const wchar_t* name, const wchar_t** source);

/**
 * @brief Sets the source name for the ODE Trigger to filter on
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] source new source name to filter on. Set to NULL to disaple filter.
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_source_set(const wchar_t* name, const wchar_t* source);

/**
 * @brief Gets the current infer component name filter for the ODE Trigger
 * A value of NULL indicates filter disabled (default)
 * @param[in] name unique name of the ODE Trigger to query
 * @param[out] infer returns the current infer component name in use
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_infer_get(const wchar_t* name, const wchar_t** infer);

/**
 * @brief Sets the infer component name for the ODE Trigger to filter on
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] infer new infer component name to filter on. Set to NULL to disaple filter.
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_infer_set(const wchar_t* name, const wchar_t* infer);

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
 * @brief Gets the current class_id_a and class_id_b filters for the ODE Trigger
 * @param[in] name unique name of the Intersection ODE Trigger to query
 * @param[out] class_id_a returns the current class_id for Class A
 * @param[out] class_id_b returns the current class_id for Class B
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_class_id_ab_get(const wchar_t* name, 
    uint* class_id_a, uint* class_id_b);

/**
 * @brief Sets the class_id_a and class_id_b filters for the ODE Trigger to filter on
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] class_id_a returns the current class_id for Class A
 * @param[in] class_id_b returns the current class_id for Class B
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_class_id_ab_set(const wchar_t* name, 
    uint class_id_a, uint class_id_b);

/**
 * @brief Gets the current limit setting for the ODE Trigger
 * @param[in] name unique name of the ODE Trigger to query
 * @param[out] limit returns the current trigger_limit in use
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_limit_get(const wchar_t* name, uint* limit);

/**
 * @brief Sets the limit for the ODE Trigger
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] limit new limit to use. Setting the limit to a 
 * value less that the current trigger count will effectively 
 * disable the trigger until reset.
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_limit_set(const wchar_t* name, uint limit);

/**
 * @brief Gets the current minimum confidence setting for the ODE Trigger
 * A value of 0.0 (default) indicates the minimum confidence criteria is disabled
 * @param[in] name unique name of the ODE Trigger to query
 * @param[out] min_confidence current minimum confidence criteria
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_confidence_min_get(const wchar_t* name, 
    float* min_confidence);

/**
 * @brief Sets the minimum confidence setting for the ODE Trigger.
 * Setting the value of 0.0 indicates the minimum confidence criteria is disabled
 * Note: the confidence level is only checked with the reported value is > 0.0
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] min_confidence minimum confidence to trigger an ODE occurrence
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_confidence_min_set(const wchar_t* name, 
    float min_confidence);

/**
 * @brief Gets the current minimum Tracker confidence setting for the ODE Trigger
 * A value of 0.0 (default) indicates the minimum Tracker confidence criteria is disabled
 * @param[in] name unique name of the ODE Trigger to query
 * @param[out] min_confidence current minimum Tracker confidence criteria
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_tracker_confidence_min_get(const wchar_t* name, 
    float* min_confidence);

/**
 * @brief Sets the minimum Tracker confidence setting for the ODE Trigger
 * Set the value of 0.0 to disable minimum Tracker confidence criteria.
 * Note: the confidence level is only checked with the reported value is > 0.0
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] min_confidence minimum Tracker confidence to trigger an ODE occurrence
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_tracker_confidence_min_set(const wchar_t* name, 
    float min_confidence);

/**
 * @brief Gets the current minimum rectangle width and height values for the ODE Trigger
 * A value of 0 = no minimum
 * @param[in] name unique name of the ODE Trigger to query
 * @param[out] min_width returns the current minimun frame width in use
 * @param[out] min_height returns the current minimun frame hight in use
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_dimensions_min_get(const wchar_t* name, 
    float* min_width, float* min_height);

/**
 * @brief Sets the current minimum rectangle width and height values for the ODE Trigger
 * A value of 0 = no minimum
 * @param[in] name unique name of the ODE Trigger to query
 * @param[in] min_width the new minimun frame width to use
 * @param[in] min_height the new minimun frame hight to use
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_dimensions_min_set(const wchar_t* name, 
    float min_width, float min_height);

/**
 * @brief Gets the current maximum rectangle width and height values for the ODE Trigger
 * A value of 0 = no maximum
 * @param[in] name unique name of the ODE Trigger to query
 * @param[out] max_width returns the current maximun frame width in use
 * @param[out] max_height returns the current maximun frame hight in use
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_dimensions_max_get(const wchar_t* name, 
    float* max_width, float* max_height);

/**
 * @brief Sets the current maximum rectangle width and height values for the ODE Trigger
 * A value of 0 = no maximum
 * @param[in] name unique name of the ODE Trigger to query
 * @param[in] max_width the new maximun frame width to use
 * @param[in] max_height the new maximun frame hight to use
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_dimensions_max_set(const wchar_t* name, 
    float max_width, float max_height);

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
 * @brief Gets the current process interval setting for the named ODE Trigger
 * If set, the Trigger will only process every  
 * @param[in] name unique name of the ODE Trigger to query
 * @param[out] interval the current interval to use, Default = 0
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_interval_get(const wchar_t* name, uint* interval);

/**
 * @brief Sets the process interval for the named ODE Trigger
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] interval new interval to use. Setting the interval will reset the
 * frame counter to zero, meaning a new value of n will skip the next n-1 frames
 * from the current frame at the time this service is called.
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_interval_set(const wchar_t* name, uint interval);

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
DslReturnType dsl_ode_trigger_area_remove_many(const wchar_t* name, 
    const wchar_t** areas);

/**
 * @brief Removes a named ODE Area from a named ODE Trigger
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] area unique name of the ODE Area to Remove
 * @return DSL_RESULT_SUCCESS on successful update, 
 * DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_area_remove_all(const wchar_t* name);

/**
 * @brief Adds a named ODE Accumulator to a named ODE Trigger
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] accumulator unique name of the ODE Accumulator to add
 * @return DSL_RESULT_SUCCESS on successful update, 
 * DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_accumulator_add(const wchar_t* name, 
    const wchar_t* accumulator);

/**
 * @brief Removes the ODE Accumulator from a named ODE Trigger
 * @param[in] name unique name of the ODE Trigger to update
 * @return DSL_RESULT_SUCCESS on successful update, 
 * DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_accumulator_remove(const wchar_t* name);

/**
 * @brief Adds a named ODE Heat-Mapper to a named ODE Trigger
 * @param[in] name unique name of the ODE Trigger to update
 * @param[in] heat_mapper unique name of the ODE Heat-Mapper to add
 * @return DSL_RESULT_SUCCESS on successful update, 
 * DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_heat_mapper_add(const wchar_t* name, 
    const wchar_t* heat_mapper);

/**
 * @brief Removes the ODE Heat-Mapper from a named ODE Trigger
 * @param[in] name unique name of the ODE Trigger to update
 * @return DSL_RESULT_SUCCESS on successful update, 
 * DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_heat_mapper_remove(const wchar_t* name);

/**
 * @brief Deletes a uniquely named Trigger. The call will fail if the Triggers is 
 * currently in use
 * @brief[in] name unique name of the event to delte
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_delete(const wchar_t* name);

/**
 * @brief Deletes a Null terminated list of Triggers. The call will fail if any of 
 * the Triggers are currently in use
 * @brief[in] names Null terminaed list of event names to delte
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_delete_many(const wchar_t** names);

/**
 * @brief Deletes all Triggers. The call will fail if any of the Triggers are 
 * currently in use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_TRIGGER_RESULT otherwise.
 */
DslReturnType dsl_ode_trigger_delete_all();

/**
 * @brief Returns the size of the list of Triggers
 * @return the number of Triggers in the list
 */
uint dsl_ode_trigger_list_size();

/**
 * @brief Creates a new ODE Accumulator that when added to an ODE Trigger, accumulates
 * the count(s) of ODE Occurrence and calls on all actions during the Trigger's post 
 * processing of each frame.
 * @param[in] name unique name for the ODE Accumulator
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_ACCUMULATOR_RESULT otherwise.
 */
DslReturnType dsl_ode_accumulator_new(const wchar_t* name);

/**
 * @brief Adds a named ODE Action to a named ODE Accumulator
 * @param[in] name unique name of the ODE Accumulator to update
 * @param[in] action unique name of the ODE Action to Add
 * @return DSL_RESULT_SUCCESS on successful update, D
 * SL_RESULT_ODE_ACCUMULATOR_RESULT otherwise.
 */
DslReturnType dsl_ode_accumulator_action_add(const wchar_t* name, 
    const wchar_t* action);

/**
 * @brief Adds a Null terminated list of named ODE Actions to a named ODE Accumulator
 * @param[in] name unique name of the ODE Accumulator to update
 * @param[in] actions Null terminated list of unique names of the ODE Actions to add
 * @return DSL_RESULT_SUCCESS on successful update, 
 * DSL_RESULT_ODE_ACCUMULATOR_RESULT otherwise.
 */
DslReturnType dsl_ode_accumulator_action_add_many(const wchar_t* name, 
    const wchar_t** actions);

/**
 * @brief Removes a named ODE Action from a named ODE Accumulator
 * @param[in] name unique name of the ODE Accumulator to update
 * @param[in] action unique name of the ODE Action to Remove
 * @return DSL_RESULT_SUCCESS on successful update, 
 * DSL_RESULT_ODE_ACCUMULATOR_RESULT otherwise.
 */
DslReturnType dsl_ode_accumulator_action_remove(const wchar_t* name, 
    const wchar_t* action);

/**
 * @brief Removes a Null terminated list of named ODE Actions from a named 
 * ODE Accumulator
 * @param[in] name unique name of the ODE Accumulator to update
 * @param[in] actions Null terminated list of unique names of the 
 * ODE Actions to remove
 * @return DSL_RESULT_SUCCESS on successful update, 
 * DSL_RESULT_ODE_ACCUMULATOR_RESULT otherwise.
 */
DslReturnType dsl_ode_accumulator_action_remove_many(const wchar_t* name, 
    const wchar_t** actions);

/**
 * @brief Removes a named ODE Action from a named ODE Accumulator
 * @param[in] name unique name of the ODE Accumulator to update
 * @param[in] action unique name of the ODE Action to Remove
 * @return DSL_RESULT_SUCCESS on successful update, 
 * DSL_RESULT_ODE_ACCUMULATOR_RESULT otherwise.
 */
DslReturnType dsl_ode_accumulator_action_remove_all(const wchar_t* name);

/**
 * @brief Deletes a uniquely named ODE Accumulator. The call will fail if 
 * the ODE Accumulator is currently in use
 * @brief[in] name unique name of the ODE Accumulator to delete
 * @return DSL_RESULT_SUCCESS on successful delete, 
 * DSL_RESULT_ODE_ACCUMULATOR_RESULT otherwise.
 */
DslReturnType dsl_ode_accumulator_delete(const wchar_t* name);

/**
 * @brief Deletes a Null terminated list of ODE Accumulator. The call will fail 
 * if any of the ODE Accumulators are currently in use
 * @brief[in] names Null terminaed list of ODE Accumulator names to delete
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_ACCUMULATOR_RESULT otherwise.
 */
DslReturnType dsl_ode_accumulator_delete_many(const wchar_t** names);

/**
 * @brief Deletes all ODE Accumulators. The call will fail if any of the 
 * ODE Accumulators are currently in use.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_ACCUMULATOR_RESULT otherwise.
 */
DslReturnType dsl_ode_accumulator_delete_all();

/**
 * @brief Returns the size of the list of ODE Accumulators
 * @return the number of ODE Accumulators in the list
 */
uint dsl_ode_accumulator_list_size();

/**
 * @brief Creates a new ODE Heat-Mapper that when added to an ODE Trigger, maps the
 * bounding-box test point for objects that trigger an ODE Occurrence over time. 
 * The source-frame, with width and height, is partitioned into a two-dimensional grid of 
 * rectangles as defined by the input parameters cols and rows. A color value from the
 * provided RGBA Color Palette is assigned to each rectangle based on the percentage
 * of test-points per rectangle. The RGBA Display Metadata is added to the Frame's metadata
 *  when the Trigger post processes each frame.
 * @param[in] name unique name for the ODE Heat-Mapper
 * @param[in] cols number of columns for the two-dimensional map.
 * @param[in] rows number of rows for the two-dimensional map
 * @param[in] bbox_test_point one of DSL_BBOX_POINT values defining which point of a
 * object's bounding box to use as coordinates for mapping.
 * @param[in] color_palette a palette of RGBA Colors to assign to the grid rectangles
 * based on the percentage of center-points per-square.  
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_HEAT_MAPPER_RESULT otherwise.
 */
DslReturnType dsl_ode_heat_mapper_new(const wchar_t* name, 
    uint cols, uint rows, uint bbox_test_point, const wchar_t* color_palette);

/**
 * @brief Gets the RGBA Color Palette in use by the named ODE Heat-Mapper
 * @param[in] name unique name of the ODE Heat-Mapper to query
 * @param[out] color_palette name of the RGBA Color Palette in use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_HEAT_MAPPER_RESULT otherwise.
 */
DslReturnType dsl_ode_heat_mapper_color_palette_get(const wchar_t* name, 
    const wchar_t** color_palette);

/**
 * @brief Sets the the RGBA Color Palette to use by the named ODE Heat-Mapper
 * @param[in] name unique name of the ODE Heat-Mapper to update
 * @param[in] color_palette name of the RGBA Color Palette to use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_HEAT_MAPPER_RESULT otherwise.
 */
DslReturnType dsl_ode_heat_mapper_color_palette_set(const wchar_t* name, 
    const wchar_t* color_palette);

/**
 * @brief Gets the current Legend settings for the named ODE Heat-Mapper.
 * @param[in] name unique name of the ODE Heat-Mapper to query.
 * @param[out] enabled true if display is enabled, false otherwise.
 * @param[out] location current frame location, one of DSL_HEAT_MAP_LEGEND_LOCATION_*
 * @param[out] width width of each entry in the legend in units of columns.
 * @param[out] height height of each entry in the legend in units of rows.
 * @return DSL_RESULT_SUCCESS on successful query, 
 * DSL_RESULT_ODE_HEAT_MAPPER_RESULT otherwise.
 */
DslReturnType dsl_ode_heat_mapper_legend_settings_get(const wchar_t* name, 
    boolean* enabled, uint* location, uint* width, uint* height);

/**
 * @brief Gets the current Legend settings for the named ODE Heat-Mapper.
 * @param[in] name unique name of the ODE Heat-Mapper to update.
 * @param[in] enabled set to true to enable display, false to disable.
 * @param[in] location frame location, one of DSL_HEAT_MAP_LEGEND_LOCATION_*
 * @param[in] width width of each entry in the legend in units of columns.
 * @param[in] height height of each entry in the legend in units of rows.
 * @return DSL_RESULT_SUCCESS on successful query, 
 * DSL_RESULT_ODE_HEAT_MAPPER_RESULT otherwise.
 */
DslReturnType dsl_ode_heat_mapper_legend_settings_set(const wchar_t* name, 
    boolean enabled, uint location, uint width, uint height);

/**
 * @brief Calls on an ODE Heat-Mapper to clear its current heat-map metrics
 * returning the map to its initial all-zero state. 
 * @param[in] name unique name of the ODE Heat-Mapper to call on.
 * @return DSL_RESULT_SUCCESS on success, 
 * DSL_RESULT_ODE_HEAT_MAPPER_RESULT otherwise.
 */
DslReturnType dsl_ode_heat_mapper_metrics_clear(const wchar_t* name);

/**
 * @brief Get the current heat-map metrics from an ODE Heat-Mapper
 * @param[in] name unique name of the ODE Heat-Mapper to query.
 * @param[out] buffer a linear buffer of metric map data. Each row or 
 * map data is serialized to a single buffer of size cols*rows. 
 * Each element in the buffer indicates the total number of occurrences
 * accumulated for the position in the map.
 * @param[out] size size of buffer - cols*rows.
 * @return DSL_RESULT_SUCCESS on success, 
 * DSL_RESULT_ODE_HEAT_MAPPER_RESULT otherwise.
 */
DslReturnType dsl_ode_heat_mapper_metrics_get(const wchar_t* name,
    const uint64_t** buffer, uint* size);

/**
 * @brief Calls on an ODE Heat-Mapper to print its current heat-map metrics
 * to the console. 
 * @param[in] name unique name of the ODE Heat-Mapper to call on.
 * @return DSL_RESULT_SUCCESS on success, 
 * DSL_RESULT_ODE_HEAT_MAPPER_RESULT otherwise.
 */
DslReturnType dsl_ode_heat_mapper_metrics_print(const wchar_t* name);

/**
 * @brief Calls on an ODE Heat-Mapper to log its current heat-map metrics
 * at the INFO log level. 
 * @param[in] name unique name of the ODE Heat-Mapper to call on.
 * @return DSL_RESULT_SUCCESS on success, 
 * DSL_RESULT_ODE_HEAT_MAPPER_RESULT otherwise.
 */
DslReturnType dsl_ode_heat_mapper_metrics_log(const wchar_t* name);

/**
 * @brief Calls on an ODE Heat-Mapper to write its current heat-map metrics to file.
 * @param[in] name unique name of the ODE Heat-Mapper to call on.
 * @param[in] file_path absolute or relative file path of the output file to use
 * The file will be created if one does exists, or opened/used if found.
 * @param[in] mode file open/write mode, one of DSL_EVENT_FILE_MODE_* options
 * @param[in] format one of the DSL_EVENT_FILE_FORMAT_* options
 * @return DSL_RESULT_SUCCESS on success, 
 * DSL_RESULT_ODE_HEAT_MAPPER_RESULT otherwise.
 */
DslReturnType dsl_ode_heat_mapper_metrics_file(const wchar_t* name,
    const wchar_t* file_path, uint mode, uint format);

/**
 * @brief Deletes a uniquely named ODE Heat-Mapper. The call will fail if 
 * the ODE Heat-Mapper is currently in use.
 * @brief[in] name unique name of the ODE Heat-Mapper to delete
 * @return DSL_RESULT_SUCCESS on successful delete, 
 * DSL_RESULT_ODE_HEAT_MAPPER_RESULT otherwise.
 */
DslReturnType dsl_ode_heat_mapper_delete(const wchar_t* name);

/**
 * @brief Deletes a Null terminated list of ODE Heat-Mappers. The call will fail 
 * if any of the ODE Heat-Mappers are currently in use.
 * @brief[in] names Null terminated list of ODE Heat-Mapper names to delete
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_HEAT_MAPPER_RESULT otherwise.
 */
DslReturnType dsl_ode_heat_mapper_delete_many(const wchar_t** names);

/**
 * @brief Deletes all ODE Heat-Mappers. The call will fail if any of the 
 * ODE Heat-Mappers are currently in use.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_HEAT_MAPPER_RESULT otherwise.
 */
DslReturnType dsl_ode_heat_mapper_delete_all();

/**
 * @brief Returns the size of the list of ODE Heat-Mappers.
 * @return the number of ODE Heat-Mapper in the list.
 */
uint dsl_ode_heat_mapper_list_size();

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
 * @brief Gets the current setting for the number of Display Meta structures that
 * are allocated for each frame. Each structure can hold up to 16 display elements
 * for each display type (lines, arrows, rectangles, etc.). The default size is one.
 * Note: each allocation adds overhead to the processing of each frame. 
 * @param[in] name unique name of the ODE Handler to query.
 * @param[out] count current count of Display Meta structures allocated per frame
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_HANDLER_RESULT otherwise
 */
DslReturnType dsl_pph_ode_display_meta_alloc_size_get(const wchar_t* name, uint* size);

/**
 * @brief Sets the current setting for the number of Display Meta structures that
 * are allocated for each frame. Each structure can hold up to 16 display elements
 * for each display type (lines, arrows, rectangles, etc.). The default size is one.
 * Note: each allocation adds overhead to the processing of each frame. 
 * @param[in] name unique name of the ODE Handler to update.
 * @param[in] size number of Display Meta structures allocated per frame
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_HANDLER_RESULT otherwise
 */
DslReturnType dsl_pph_ode_display_meta_alloc_size_set(const wchar_t* name, uint size);

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
 * @param[in] name unique name for the new URI Source
 * @param[in] uri Unique Resource Identifier (file or live)
 * @param[in] is_live true if source is live false if file
 * @param[in] intra_decode set to True to enable, false to disable
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_uri_new(const wchar_t* name, const wchar_t* uri, 
    boolean is_live, uint intra_decode, uint drop_frame_interval);

/**
 * @brief creates a new, uniquely named File Source component
 * @param[in] name Unique name for the File Source
 * @param[in] file_path absolute or relative path to the media file to play
 * @param[in] repeat_enabled set to true to repeat source on EOS
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_file_new(const wchar_t* name, 
    const wchar_t* file_path, boolean repeat_enabled);

/**
 * @brief Gets the current File Path in use by the named File Source
 * @param[in] name name of the File Source to query
 * @param[out] FilePath in use by the File Source
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_file_path_get(const wchar_t* name, const wchar_t** file_path);

/**
 * @brief Sets the current File Path for the named File Source to use
 * @param[in] name name of the File Source to update
 * @param[in] file_path new file path to use by the File Source
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_file_path_set(const wchar_t* name, const wchar_t* file_path);


/**
 * @brief Gets the current Repeat on EOS Enabled setting for the File Source
 * @param[in] name name of the File Source to query
 * @param[out] enabled true if Repeat on EOS is enabled, false otherwise 
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_file_repeat_enabled_get(const wchar_t* name, boolean* enabled);

/**
 * @brief Sets the current Repeat on EOS Enabled setting for the File Source
 * @param[in] name name of the File Source to update
 * @param[in] enabled set to true to enable Repeat on EOS, false to disable. 
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_file_repeat_enabled_set(const wchar_t* name, boolean enabled);

/**
 * @brief creates a new, uniquely named Image Source component that
 * decodes a single image producing a single frame followed by EOS
 * @param[in] name Unique name for the Image Source
 * @param[in] file_path absolute or relative path to the image file to play
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_image_new(const wchar_t* name, 
    const wchar_t* file_path);

/**
 * @brief creates a new, uniquely named Multi Image Source component that
 * decodes multiple images specified by folder/filename-pattern.
 * @param[in] name Unique name for the Image Frame Source
 * @param[in] file_path use the printf style %d in the absolute or relative path. 
 * Eample: "./my_images/image.%d04.mjpg", where the files in "./my_images/"
 * are named "image.0000.mjpg", "image.0001.mjpg", "image.0002.mjpg" etc.
 * @param[in] fps-n frames/second fraction numerator
 * @param[in] fps-d frames/second fraction denominator
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_image_multi_new(const wchar_t* name, 
    const wchar_t* file_path, uint fps_n, uint fps_d);

/**
 * @brief creates a new, uniquely named Image Stream Source component that
 * streams an image at a specified framerate
 * @param[in] name Unique name for the Image Source
 * @param[in] file_path absolute or relative path to the image file to play
 * @param[in] is_live set to true to act as live source, false otherwise
 * @param[in] fps_n frames/second fraction numerator
 * @param[in] fps_d frames/second fraction denominator
 * @param[in] timeout source will send an EOS event on timeout, set to 0 to disable
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_image_stream_new(const wchar_t* name, 
    const wchar_t* file_path, boolean is_live, uint fps_n, uint fps_d, uint timeout);

/**
 * @brief Gets the current Timeout setting for the Image Stream Source
 * @param[in] name name of the Image Source to query
 * @param[out] timeout current timeout value for the EOS Timer, 0 means the
 * timer is disabled
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_image_stream_timeout_get(const wchar_t* name, uint* timeout);

/**
 * @brief Sets the current Timeout setting for the Image Stream Source
 * @param[in] name name of the Image Source to update
 * @param[in] timeout new timeout value for the EOS Timer (in seconds), 0 to disable. 
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_image_stream_timeout_set(const wchar_t* name, uint timeout);
    
/**
 * @brief creates a new, uniquely named RTSP Source component
 * @param[in] name Unique Resource Identifier (file or live)
 * @param[in] protocol one of the constant protocol values [ DSL_RTP_TCP | DSL_RTP_ALL ]
 * @param[in] intra_decode set to True to enable, false to disable.
 * @param[in] drop_frame_interval, set to 0 to decode every frame.
 * @param[in] latency in milliseconds.
 * @param[in] timeout time to wait between successive frames before determining the 
 * connection is lost. Set to 0 to disable timeout.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_rtsp_new(const wchar_t* name, const wchar_t* uri, uint protocol,
    uint intra_decode, uint drop_frame_interval, uint latency, uint timeout);

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
 * @param[in] uri to use by the Decode Source
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
 * @brief returns the name of a Source component from a unique Source Id
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
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_INFER_RESULT otherwise.
 */
DslReturnType dsl_dewarper_new(const wchar_t* name, const wchar_t* config_file);

/**
 * @brief creates a new, uniquely named Record Tap component
 * @param[in] name unique component name for the new Record Tap
 * @param[in] outdir absolute or relative path to the recording output dir.
 * @param[in] container one of DSL_MUXER_MPEG4 or DSL_MUXER_MK4
 * @param[in] client_listener client callback for notifications of recording
 * events, DSL_RECORDING_EVENT_START and DSL_RECORDING_EVENT_STOP.
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
 * @param[in] name unique name of the Record Tap to stop
 * @param[in] sync if set to true this call will block until the 
 * stop operation has completed or timeout DSL_RECORDING_STOP_WAIT_TIMEOUT.
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_TAP_RESULT on failure
 */
DslReturnType dsl_tap_record_session_stop(const wchar_t* name, boolean sync);

/**
 * @brief returns the video recording output directory for the named Record Tap
 * @param[in] name name of the Record Tap to query
 * @param[out] outdir current output directory set for the Record Tap
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_TAP_RESULT on failure
  */
DslReturnType dsl_tap_record_outdir_get(const wchar_t* name, const wchar_t** outdir);

/**
 * @brief returns the video recording output directory in use by the named Record Tap
 * @param[in] name name of the Record Tap to update
 * @param[in] outdir new output directory to use by the Record Tap
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_TAP_RESULT on failure
 */
DslReturnType dsl_tap_record_outdir_set(const wchar_t* name, const wchar_t* outdir);

/**
 * @brief returns the video recording container type for the named Record Tap
 * @param[in] name name of the Record Tap to query
 * @param[out] container current setting, one of DSL_MUXER_MPEG4 or DSL_MUXER_MK4
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_TAP_RESULT on failure
  */
DslReturnType dsl_tap_record_container_get(const wchar_t* name, uint* container);

/**
 * @brief returns the video recording container type for the named Record Tap
 * @param[in] name name of the Record Tap to query
 * @param[in] container new setting, one of DSL_MUXER_MPEG4 or DSL_MUXER_MK4
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_TAP_RESULT on failure
 */
DslReturnType dsl_tap_record_container_set(const wchar_t* name,  uint container);

/**
 * @brief returns the video recording cache size in units of seconds
 * A fixed size cache is created when the Pipeline is linked and played. 
 * The default cache size is set to DSL_DEFAULT_VIDEO_RECORD_CACHE_IN_SEC
 * @param[in] name name of the Record Tap to query
 * @param[out] cache_size current cache size setting
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_TAP_RESULT on failure
 */
DslReturnType dsl_tap_record_cache_size_get(const wchar_t* name, uint* cache_size);

/**
 * @brief sets the video recording cache size in units of seconds
 * A fixed size cache is created when the Pipeline is linked and played. 
 * The default cache size is set to DSL_DEFAULT_VIDEO_RECORD_CACHE_IN_SEC
 * @param[in] name name of the Record Tap to update
 * @param[in] cache_size new cache size setting to use on Pipeline play
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_TAP_RESULT on failure
 */
DslReturnType dsl_tap_record_cache_size_set(const wchar_t* name, uint cache_size);

/**
 * @brief returns the dimensions, width and height, used for the video recordings
 * @param[in] name name of the Record Tap to query
 * @param[out] width current width of the video recording in pixels
 * @param[out] height current height of the video recording in pixels
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_TAP_RESULT on failure
 */
DslReturnType dsl_tap_record_dimensions_get(const wchar_t* name, uint* width, uint* height);

/**
 * @brief sets the dimensions, width and height, for the video recordings created
 * values of zero indicate no-transcodes
 * @param[in] name name of the Record Tap to update
 * @param[in] width width to set the video recording in pixels
 * @param[in] height height to set the video in pixels
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_TAP_RESULT on failure
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
 * @brief Adds a Video Player, Render or RTSP type, to a named Record Tap.
 * Once added, each recorded video's file_path will be added (or queued) with
 * the Video Player to be played according to the Players settings. 
 * @param[in] name unique name of the Record Tap to update
 * @param[in] player unique name of the Video Player to add
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_tap_record_video_player_add(const wchar_t* name, 
    const wchar_t* player);
    
/**
 * @brief Removes a Video Player, Render or RTSP type, from a named Record Tap.
 * @param[in] name unique name of the Record Tap to update
 * @param[in] player unique name of the Video Player to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_tap_record_video_player_remove(const wchar_t* name, 
    const wchar_t* player);

/**
 * @brief Adds a SMTP mailer to a named Record Tap. Once added, the Tap will 
 * use the Mailer to send out the file_path and details of each saved recording 
 * according to the Mailer's settings.
 * @param[in] name unique name of the Record Tap to update
 * @param[in] mailer unique name of the Mailer to add
 * @param[in] subject subject line to use for all outgoing mail
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT otherwise.
 */
DslReturnType dsl_tap_record_mailer_add(const wchar_t* name, 
    const wchar_t* mailer, const wchar_t* subject);
    
/**
 * @brief Removes a named Mailer from a named Record Tap.
 * @param[in] name unique name of the Record Tap to update
 * @param[in] mailer unique name of the Mailer to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT otherwise.
 */
DslReturnType dsl_tap_record_mailer_remove(const wchar_t* name, 
    const wchar_t* mailer);

/**
 * @brief creates a new, uniquely named Preprocessor component
 * @param[in] name unique name for the new Tracker
 * @param[in] config_file relative or absolute pathspec to 
 * the Preprocessor config text file to use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PREPROC_RESULT otherwise.
 */
DslReturnType dsl_preproc_new(const wchar_t* name, const wchar_t* config_file);

/**
 * @brief Gets the current Config File in use by the named Preprocessor
 * @param[in] name of Preprocessor to query
 * @param[out] config_file Config file currently in use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PREPROC_RESULT otherwise.
 */
DslReturnType dsl_preproc_config_file_get(const wchar_t* name, 
    const wchar_t** config_file);

/**
 * @brief Sets the Config File to use by the named Preprocessor
 * @param[in] name of Preprocessor to update.
 * @param[in] config_file new Config file to use.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PREPROC_RESULT otherwise.
 */
DslReturnType dsl_preproc_config_file_set(const wchar_t* name, 
    const wchar_t* config_file);

/**
 * @brief Gets the current enabled settings for the named Preprocessor.
 * @param[in] name unique name of the Preprocessor to query
 * @param[out] true if preprocessing is enabled, fale otherwise
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PREPROC_RESULT otherwise.
 */
DslReturnType dsl_preproc_enabled_get(const wchar_t* name, 
    boolean* enabled);

/**
 * @brief Sets the enabled settings for the named Preprocessor.
 * @param[in] name unique name of the Preprocessor to update
 * @param[out] enabled set to true to enable preprocessing, fale to
 * set the Preprocessor into passthrough mode
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PREPROC_RESULT otherwise.
 */
DslReturnType dsl_preproc_enabled_set(const wchar_t* name, 
    boolean enabled);

/**
 * @brief Gets the unique Id assigned to the named Preprocessor when created.
 * The Id is used to identify metadata generated by the Preprocessor.
 * Id's start at 0 and are incremented with each new Preprocessor created.
 * Id's will be reused if the Preprocessor is deleted.
 * @param[in] name unique name of the Preprocessor to query
 * @param[out] id unique id assigned to the named Preprocessor
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PREPROC_RESULT otherwise.
 */
DslReturnType dsl_preproc_unique_id_get(const wchar_t* name, 
    uint* id);

/**
 * @brief Creates a new, uniquely named Segmentation Visualizer. Once created,
 * the Segmentation Visualizer can be added to a Primary GIE. 
 * @param[in] name unique name for the new Segmentation Visualizer
 * @param[in] width output width in pixels, typically same as input to GIE
 * @param[in] height output height in pixels, typically same as input to GIE
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SIGVISUAL_RESULT
 */
DslReturnType dsl_segvisual_new(const wchar_t* name, uint width, uint height);

/**
 * @brief Returns the output dimensions, width and height, for the named
 * Segmentation Visualizer.  
 * @param[in] name name of the Segmentation Visualizer to query
 * @param[out] width current output width in pixels
 * @param[out] height current output height in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SIGVISUAL_RESULT
 */
DslReturnType dsl_segvisual_dimensions_get(const wchar_t* name, 
    uint* width, uint* height);

/**
 * @brief Sets the output dimensions, width and height, for the named 
 * Segmentation Visualizer.
 * @param[in] name name of the Display to update
 * @param[in] width new output width in pixels
 * @param[in] height new output height in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SIGVISUAL_RESULT
 */
DslReturnType dsl_segvisual_dimensions_set(const wchar_t* name, 
    uint width, uint height);

/**
 * @brief Adds a pad-probe-handler to be called to process each frame buffer.
 * One or more Pad Probe Handlers can be added to the SOURCE PAD only.
 * @param[in] name unique name of the Segmentation visualizer to update
 * @param[in] handler unique name of the pad probe handler to add
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT otherwise
 */
DslReturnType dsl_segvisual_pph_add(const wchar_t* name, const wchar_t* handler);

/**
 * @brief Removes a pad probe handler callback function from a named
 * Segmentation visualizer.
 * @param[in] name unique name of the Segmentation visualizer to update
 * @param[in] handler unique name of the pad probe handler to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT otherwise
 */
DslReturnType dsl_segvisual_pph_remove(const wchar_t* name, const wchar_t* handler);
    
/**
 * @brief creates a new, uniquely named Primary GIE object
 * @param[in] name unique name for the new GIE object
 * @param[in] infer_config_file pathspec of the Infer Config file to use
 * @param[in] model_engine_file pathspec of the Model Engine file to use
 * Set to NULL or empty string "" to leave unspecified, indicating that
 * the model should be created based on the infer_config_file settings
 * @param[in] interval frame interval to infer on. 0 = every frame, 
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_INFER_RESULT otherwise.
 */
DslReturnType dsl_infer_gie_primary_new(const wchar_t* name, 
    const wchar_t* infer_config_file, const wchar_t* model_engine_file, uint interval);

/**
 * @brief creates a new, uniquely named Primary Triton Inference Server (TIS) object
 * @param[in] name unique name for the new TIS object
 * @param[in] infer_config_file pathspec of the Infer Config file to use
 * @param[in] interval frame interval to infer on. 0 = every frame, 
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TIS_RESULT otherwise.
 */
DslReturnType dsl_infer_tis_primary_new(const wchar_t* name, 
    const wchar_t* infer_config_file, uint interval);

/**
 * @brief creates a new, uniquely named Secondary GIE object
 * @param[in] name unique name for the new GIE object
 * @param[in] infer_config_file pathspec of the Infer Config file to use
 * @param[in] model_engine_file pathspec of the Model Engine file to use
 * Set to NULL or empty string "" to leave unspecified, indicating that
 * the model should be created based on the infer_config_file settings
 * @param[in] infer_on_gie name of the Primary or Secondary GIE to infer on
 * @param[in] interval frame interval to infer on. 0 = every frame, 
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_INFER_RESULT otherwise.
 */
DslReturnType dsl_infer_gie_secondary_new(const wchar_t* name, 
    const wchar_t* infer_config_file, const wchar_t* model_engine_file, 
    const wchar_t* infer_on_gie, uint interval);

/**
 * @brief creates a new, uniquely named Secondary TIS object
 * @param[in] name unique name for the new TIS object
 * @param[in] infer_config_file pathspec of the Infer Config file to use
 * @param[in] infer_on_tis name of the Primary or Secondary TIS to infer on
 * @param[in] interval frame interval to infer on. 0 = every frame, 
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_INFER_RESULT otherwise.
 */
DslReturnType dsl_infer_tis_secondary_new(const wchar_t* name, 
    const wchar_t* infer_config_file, const wchar_t* infer_on_tis, uint interval);

/**
 * @brief Gets the client defined batch-size setting for the named GIE or TIS. If
 * not set (0), the Pipeline will set the batch-size to the same as the Streammux 
 * batch-size which - by default - is derived from the number of sources when the 
 * Pipeline is called to play. The Streammux batch-size can be set (overridden)
 * by the client as well.
 * @param[in] name unique name of the GIE or TIS to update.
 * @param[in] size value to set the batch-size setting for the named GIE or TIS
 * @return DSL_RESULT_SUCCESS on successful update, one of 
 * DSL_RESULT_INFER_RESULT on failure. 
 */
DslReturnType dsl_infer_batch_size_get(const wchar_t* name, uint* size);

/**
 * @brief Sets (overides) the batch-size setting for the named GIE or TIS. If
 * not set (0), the Pipeline will set the batch-size to the same as the Streammux 
 * batch-size which - by default - is derived from the number of sources when the 
 * Pipeline is called to play. The Streammux batch-size can be set (overridden)
 * by the client as well.
 * @param[in] name unique name of the GIE or TIS to update.
 * @param[in] size value to set the batch-size setting for the named GIE or TIS
 * @return DSL_RESULT_SUCCESS on successful update, one of 
 * DSL_RESULT_INFER_RESULT on failure. 
 */
DslReturnType dsl_infer_batch_size_set(const wchar_t* name, uint size);

/**
 * @brief Queries a GIE or TIS for its unique Id 
 * @param[in] name unique name of the GIE or TIS to query.
 * @param[out] id unique id for the named GIE or TIS.
 * @return DSL_RESULT_SUCCESS on successful query, one of 
 * DSL_RESULT_INFER_RESULT on failure. 
 */
DslReturnType dsl_infer_unique_id_get(const wchar_t* name, uint* id);

/**
 * @brief Adds a pad-probe-handler to be called to process each frame buffer.
 * A Primary GIE can have multiple Sink and Source pad-probe-handlers
 * @param[in] name unique name of the Primary GIE to update
 * @param[in] handler callback function to process pad probe data
 * @param[in] pad pad to add the handler to; DSL_PAD_SINK | DSL_PAD SRC
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_INFER_RESULT otherwise
 */
DslReturnType dsl_infer_primary_pph_add(const wchar_t* name, 
    const wchar_t* handler, uint pad);

/**
 * @brief Removes a pad-probe-handler from the Primary GIE
 * @param[in] name unique name of the Primary GIE to update
 * @param[in] handler pad-probe-handler to remove
 * @param[in] pad pad to remove the handler from; DSL_PAD_SINK | DSL_PAD SRC
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_INFER_RESULT otherwise
 */
DslReturnType dsl_infer_primary_pph_remove(const wchar_t* name, 
    const wchar_t* handler, uint pad);

/**
 * @brief Gets the current Infer Config File in use by the named Primary or Secondary GIE
 * @param[in] name unique name of Primary or Secondary GIE to query
 * @param[out] infer_config_file Infer Config file currently in use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_INFER_RESULT otherwise.
 */
DslReturnType dsl_infer_config_file_get(const wchar_t* name, 
    const wchar_t** infer_config_file);

/**
 * @brief Sets the Infer Config File to use by the named Primary or Secondary GIE
 * @param[in] name unique name of Primary or Secondary GIE to update
 * @param[in] infer_config_file new Infer Config file to use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_INFER_RESULT otherwise.
 */
DslReturnType dsl_infer_config_file_set(const wchar_t* name, 
    const wchar_t* infer_config_file);

/**
 * @brief Gets the current Model Engine File in use by the named Primary or Secondary GIE
 * @param[in] name unique name of Primary or Secondary GIE to query
 * @param[out] model_engi_file Model Engine file currently in use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_INFER_RESULT otherwise.
 */
DslReturnType dsl_infer_gie_model_engine_file_get(const wchar_t* name, 
    const wchar_t** model_engine_file);

/**
 * @brief Sets the Model Engine File to use by the named Primary or Secondary GIE
 * @param[in] name unique name of Primary or Secondary GIE to update
 * @param[in] model_engine_file new Model Engine file to use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_INFER_RESULT otherwise.
 */
DslReturnType dsl_infer_gie_model_engine_file_set(const wchar_t* name, 
    const wchar_t* model_engine_file);

/**
 * @brief Gets the current input amd output tensor-meta settings in use by the 
 * named Primary or Secondary GIE.
 * @param[in] name unique name of Primary or Secondary GIE to query.
 * @param[out] input_enabled if true preprocessing input tensors attached as 
 * metadata instead of preprocessing inside the plugin, false otherwise.
 * @param[out] output_enabled if true tensor outputs will be attached as 
 * meta on GstBuffer.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_INFER_RESULT otherwise.
 */
DslReturnType dsl_infer_gie_tensor_meta_settings_get(const wchar_t* name, 
    boolean* input_enabled, boolean* output_enabled);

/**
 * @brief Sets the current input amd output tensor-meta settings for the 
 * named Primary or Secondary GIE to use.
 * @param[in] name unique name of Primary or Secondary GIE to query
 * @param[in] input_enabled set to true preprocess input tensors attached as 
 * metadata instead of preprocessing inside the plugin, false otherwise.
 * @param[in] output_enabled set to true to attach tensor outputs as 
 * meta on GstBuffer.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_INFER_RESULT otherwise.
 */
DslReturnType dsl_infer_gie_tensor_meta_settings_set(const wchar_t* name, 
    boolean input_enabled, boolean output_enabled);

/**
 * @brief Gets the current Infer Interval in use by the named Primary or Secondary GIE
 * @param[in] name of Primary or Secondary GIE to query
 * @param[out] interval Infer interval value currently in use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_INFER_RESULT otherwise.
 */
DslReturnType dsl_infer_interval_get(const wchar_t* name, uint* interval);

/**
 * @brief Sets the Model Engine File to use by the named Primary or Secondary GIE
 * @param[in] name of Primary or Secondary GIE to update
 * @param[in] interval new Infer Interval value to use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_INFER_RESULT otherwise.
 */
DslReturnType dsl_infer_interval_set(const wchar_t* name, uint interval);

/**
 * @brief Enbles/disables the raw layer-info output to binary file for the named the GIE
 * @param[in] name name of the Primary or Secondary GIE to update
 * @param[in] enabled set to true to enable frame-to-file output for each GIE layer
 * @param[in] path absolute or relative direcory path to write to. 
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_INFER_RESULT otherwise.
 */
DslReturnType dsl_infer_raw_output_enabled_set(const wchar_t* name, 
    boolean enabled, const wchar_t* path);

/**
 * @brief creates a new, uniquely named DCF Tracker object
 * @param[in] name unique name for the new Tracker
 * @param[in] config_file (optional) relative or absolute pathspec to 
 * the NvDCF Lib config text file
 * @param[in] width operational frame width for Tracker
 * @param[in] height operational frame height for the Tracker
 * @param[in] batch_processing_enabled set to true to enable batch_mode 
 * processing, false for single stream mode
 * @param[in] past_frame_reporting_enabled set to true to enable 
 * reporting of past frame data when available, false otherwise.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_dcf_new(const wchar_t* name, 
    const wchar_t* config_file, uint width, uint height,
    boolean batch_processing_enabled, boolean past_frame_reporting_enabled);

/**
 * @brief creates a new, uniquely named KTL Tracker object
 * @param[in] name unique name for the new for Tracker
 * @param[in] width operational frame width for Tracker
 * @param[in] height operational frame height for the Tracker
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_ktl_new(const wchar_t* name, uint width, uint height);

/**
 * @brief creates a new, uniquely named IOU Tracker object
 * @param[in] name unique name for the new Tracker
 * @param[in] config_file (optional) relative or absolute pathspec to 
 * the IOU Lib config text file
 * @param[in] width operational frame width for the Tracker
 * @param[in] height operational frame height for the Tracker
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_iou_new(const wchar_t* name, 
    const wchar_t* config_file, uint width, uint height);

/**
 * @brief returns the current frame width and height settings for the named KTL 
 * or IOU Tracker
 * @param[in] name unique name of the Tracker to query
 * @param[in] width operational frame width for the Tracker
 * @param[in] height operational frame height for the Tracker
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_dimensions_get(const wchar_t* name, uint* width, uint* height);

/**
 * @brief sets the frame width and height settings for the named KTL or IOU Tracker
 * @param[in] name unique name of the Tracker to update
 * @param[in] width output frame width for GIEs and TISs to work on
 * @param[in] height output frame height for GIEs and TIS to work on
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_dimensions_set(const wchar_t* name, uint width, uint height);

/**
 * @brief Gets the current "enable-batch-process" settings for the named KTL 
 * or IOU Tracker object. 
 * @param[in] name unique name of the Tracker to query
 * @param[out] true if batch-processing is enabled, fale otherwise
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_dcf_batch_processing_enabled_get(const wchar_t* name, 
    boolean* enabled);

/**
 * @brief Sets the "enable-batch-process" settings for the named KTL 
 * or IOU Tracker
 * Note: This call is only effective if the low-level library supports 
 * both batch and per-stream processing.
 * @param[in] name unique name of the Tracker to query
 * @param[out] true to enable batch-processing enabled, fale otherwise
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_dcf_batch_processing_enabled_set(const wchar_t* name, 
    boolean enabled);

/**
 * @brief Gets the current "enable-past-frame" settings for the named KTL 
 * or IOU Tracker
 * @param[in] name unique name of the Tracker to query
 * @param[out] true if past-frame reporting is enabled, fale otherwise.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_dcf_past_frame_reporting_enabled_get(const wchar_t* name, 
    boolean* enabled);

/**
 * @brief Sets current "enable-past-frame" settings for the named KTL 
 * or IOU Tracker object
 * Note: This call is only effective if the low-level library supports 
 * past frame reporting.
 * @param[in] name unique name of the Tracker to query
 * @param[out] true if past frame reporting is enabled, fale otherwise.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_dcf_past_frame_reporting_enabled_set(const wchar_t* name, 
    boolean enabled);

/**
 * @brief returns the current config file in use by the named Tracker object
 * @param[in] name unique name of the Tracker to query
 * @param[out] config_file absolute or relative pathspec to the new config file to use
 * Note: the config_file is an optional setting. An empty string will be returned if omitted
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_config_file_get(const wchar_t* name, const wchar_t** config_file);

/**
 * @brief sets the config file to use by named IOU of DCF Tracker object. Calling this
 * service on a KTL Tracker will have no affect.
 * @param[in] name unique name of the Tracker to Update
 * @param[in] config_file absolute or relative pathspec to the new config file to use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_config_file_set(const wchar_t* name, 
    const wchar_t* config_file);

/**
 * @brief Adds a pad-probe-handler to be called to process each frame buffer.
 * A Primary GIE can have multiple Sink and Source pad-probe-handlers
 * @param[in] name unique name of the Primary GIE to update
 * @param[in] handler callback function to process pad probe data
 * @param[in] pad pad to add the handler to; DSL_PAD_SINK | DSL_PAD SRC
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_INFER_RESULT otherwise
 */
DslReturnType dsl_tracker_pph_add(const wchar_t* name, const wchar_t* handler, uint pad);

/**
 * @brief Removes a pad-probe-handler from the Primary GIE
 * @param[in] name unique name of the Primary GIE to update
 * @param[in] handler pad-probe-handler to remove
 * @param[in] pad pad to remove the handler from; DSL_PAD_SINK | DSL_PAD SRC
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_INFER_RESULT otherwise
 */
DslReturnType dsl_tracker_pph_remove(const wchar_t* name, const wchar_t* handler, uint pad);

/**
 * @brief creates a new, uniquely named Optical Flow Visualizer (OFV) obj
 * @param[in] name unique name for the new OFV
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OFD_RESULT otherwise
 */
DslReturnType dsl_ofv_new(const wchar_t* name);


/**
 * @brief creates a new, uniquely named On-Screen Display (OSD) component
 * @param[in] name unique name for the new OSD
 * @param[in] text_enabled set to true to enable object labels display
 * @param[in] clock_enabled set to true to enable clock display
 * @param[in] bbox_enabled set to true to enable bounding box display
 * @param[in] mask_enabled set to true to enable mask display
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_new(const wchar_t* name, 
    boolean text_enabled, boolean clock_enabled, 
    boolean bbox_enabled, boolean mask_enabled);

/**
 * @brief returns the current text enabled setting for the named On-Screen Display.
 * @param[in] name name of the OSD to query.
 * @param[out] enabled true if text display is enabled, false otherwise.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise.
 */
DslReturnType dsl_osd_text_enabled_get(const wchar_t* name, boolean* enabled);

/**
 * @brief sets the text enabled setting for On-Screen-Display
 * @param[in] name name of the OSD to update.
 * @param[in] enabled set to true to enable text display, false otherwise.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_text_enabled_set(const wchar_t* name, boolean enabled);

/**
 * @brief returns the current clock enabled setting for the named On-Screen Display.
 * @param[in] name name of the OSD to query.
 * @param[out] enabled true if clock display is enabled, false otherwise.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise.
 */
DslReturnType dsl_osd_clock_enabled_get(const wchar_t* name, boolean* enabled);

/**
 * @brief sets the clock enabled setting for On-Screen-Display
 * @param[in] name name of the OSD to update.
 * @param[in] enabled set to true to enable clock display, false otherwise.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_clock_enabled_set(const wchar_t* name, boolean enabled);

/**
 * @brief returns the current X and Y offsets for On-Screen-Display clock
 * @param[in] name name of the OSD to query
 * @param[out] offset_x current offset in the X direction for the OSD clock in pixels
 * @param[out] offset_y current offset in the Y direction for the OSD clock in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_clock_offsets_get(const wchar_t* name, 
    uint* offset_x, uint* offset_y);

/**
 * @brief sets the X and Y offsets for the On-Screen-Display clock
 * @param[in] name name of the OSD to update
 * @param[in] offset_x new offset for the OSD clock in the X direction in pixels
 * @param[in] offset_y new offset for the OSD clock in the X direction in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_clock_offsets_set(const wchar_t* name, 
    uint offset_x, uint offset_y);

/**
 * @brief returns the font name and size for On-Screen-Display clock
 * @param[in] name name of the OSD to query
 * @param[out] font current font string for the OSD clock
 * @param[out] size current font size for the OSD clock
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_clock_font_get(const wchar_t* name, 
    const wchar_t** font, uint* size);

/**
 * @brief sets the font name and size for the On-Screen-Display clock
 * @param[in] name name of the OSD to update
 * @param[in] font new font string to use for the OSD clock
 * @param[in] size new size string to use for the OSD clock
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_clock_font_set(const wchar_t* name, 
    const wchar_t* font, uint size);

/**
 * @brief returns the font name and size for On-Screen-Display clock
 * @param[in] name name of the OSD to query
 * @param[out] red current red color value for the OSD clock
 * @param[out] gren current green color value for the OSD clock
 * @param[out] blue current blue color value for the OSD clock
 * @param[out] alpha current alpha color value for the OSD clock
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_clock_color_get(const wchar_t* name, 
    double* red, double* green, double* blue, double* alpha);

/**
 * @brief sets the font name and size for the On-Screen-Display clock
 * @param[in] name name of the OSD to update
 * @param[in] red new red color value for the OSD clock
 * @param[in] gren new green color value for the OSD clock
 * @param[in] blue new blue color value for the OSD clock
 * @param[out] alpha current alpha color value for the OSD clock
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_clock_color_set(const wchar_t* name, 
    double red, double green, double blue, double alpha);

/**
 * @brief returns the current bbox enabled setting for the named On-Screen Display.
 * @param[in] name name of the OSD to query.
 * @param[out] enabled true if bbox display is enabled, false otherwise.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise.
 */
DslReturnType dsl_osd_bbox_enabled_get(const wchar_t* name, boolean* enabled);

/**
 * @brief sets the bbox enabled setting for On-Screen-Display
 * @param[in] name name of the OSD to update.
 * @param[in] enabled set to true to enable bbox display, false otherwise.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_bbox_enabled_set(const wchar_t* name, boolean enabled);

/**
 * @brief returns the current mask enabled setting for the named On-Screen Display.
 * @param[in] name name of the OSD to query.
 * @param[out] enabled true if mask display is enabled, false otherwise.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise.
 */
DslReturnType dsl_osd_mask_enabled_get(const wchar_t* name, boolean* enabled);

/**
 * @brief sets the mask enabled setting for On-Screen-Display
 * @param[in] name name of the OSD to update.
 * @param[in] enabled set to true to enable mask display, false otherwise.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_mask_enabled_set(const wchar_t* name, boolean enabled);

/**
 * @brief returns the current process mode setting for the named On-Screen Display.
 * @param[in] name name of the OSD to query.
 * @param[out] mode one of 
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise.
 */
DslReturnType dsl_osd_text_enabled_get(const wchar_t* name, boolean* enabled);

/**
 * @brief sets the text enabled setting for On-Screen-Display
 * @param[in] name name of the OSD to update.
 * @param[in] enabled set to true to enable text display, false otherwise.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_text_enabled_set(const wchar_t* name, boolean enabled);

/**
 * @brief Adds a pad-probe-handler to be called to process each frame buffer.
 * An On-Screen-Display can have multiple Sink and Source pad-probe-handlers
 * @param[in] name unique name of the OSD to update
 * @param[in] handler callback function to process pad probe data
 * @param[in] pad pad to add the handler to; DSL_PAD_SINK | DSL_PAD SRC
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_pph_add(const wchar_t* name, 
    const wchar_t* handler, uint pad);

/**
 * @brief Removes a pad-probe-handler from the OSD
 * @param[in] name unique name of the OSD to update
 * @param[in] handler pad-probe-handler to remove
 * @param[in] pad pad to remove the handler from; DSL_PAD_SINK | DSL_PAD SRC
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_pph_remove(const wchar_t* name, 
    const wchar_t* handler, uint pad);

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
 * @param[in] offset_x upper left corner offset in the X direction in pixels
 * @param[in] offset_y upper left corner offset in the Y direction in pixels
 * @param[in] width width of the Ovelay Sink in pixels
 * @param[in] heigth height of the Overlay Sink in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT
 */
DslReturnType dsl_sink_overlay_new(const wchar_t* name, uint display_id,
    uint depth, uint offset_x, uint offset_y, uint width, uint height);

/**
 * @brief creates a new, uniquely named Window Sink component
 * @param[in] name unique component name for the new Overlay Sink
 * @param[in] offset_x upper left corner offset in the X direction in pixels
 * @param[in] offset_y upper left corner offset in the Y direction in pixels
 * @param[in] width width of the Window Sink in pixels
 * @param[in] heigth height of the Window Sink in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT
 */
DslReturnType dsl_sink_window_new(const wchar_t* name, 
    uint offset_x, uint offset_y, uint width, uint height);

/**
 * @brief Gets the current "force-aspect-ration" property setting for the 
 * named Window Sink
 * @param[in] name unique name of the Window Sink to query
 * @param[out] force true if the apect ratio is forced, false otherwise
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT
 */
DslReturnType dsl_sink_window_force_aspect_ratio_get(const wchar_t* name, 
    boolean* force);

/**
 * @brief Sets the "force-aspect-ration" property for the named Window Sink
 * @param[in] name unique name of the Window Sink to update
 * @param[in] force set to true to force the apect ratio, false otherwise
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT
 */
DslReturnType dsl_sink_window_force_aspect_ratio_set(const wchar_t* name, 
    boolean force);

/**
 * @brief returns the current X and Y offsets for the Render Sink
 * @param[in] name name of the Render Sink to query - of type Overlay or Window
 * @param[out] offset_x current offset in the X direction for the Render Sink in pixels
 * @param[out] offset_y current offset in the Y direction for the Render Sink in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT otherwise
 */
DslReturnType dsl_sink_render_offsets_get(const wchar_t* name, 
    uint* offset_x, uint* offset_y);

/**
 * @brief sets the X and Y offsets for the On-Screen-Display clock
 * @param[in] name name of the Render Sink to update - of type Overlay or Window
 * @param[in] offset_x new offset for the Render Sink in the X direction in pixels
 * @param[in] offset_y new offset for the Render Sink in the Y direction in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT otherwise
 */
DslReturnType dsl_sink_render_offsets_set(const wchar_t* name, 
    uint offset_x, uint offset_y);
    
/**
 * @brief Returns the dimensions, width and height, in use by the Render Sink
 * The Render Sink can be of type Window Sink or Overlay Sink
 * @param[in] name name of the Render Sink to query
 * @param[out] width current width of the video recording in pixels
 * @param[out] height current height of the video recording in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TILER_RESULT
 */
DslReturnType dsl_sink_render_dimensions_get(const wchar_t* name, 
    uint* width, uint* height);

/**
 * @brief Sets the dimensions, width and height, for the Render Sink
 * @param[in] name name of the Render Sink to update
 * @param[in] width width to set the video recording in pixels
 * @param[in] height height to set the video in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT otherwise
 */
DslReturnType dsl_sink_render_dimensions_set(const wchar_t* name, 
    uint width, uint height);

/**
 * @brief Resets the Render Sink causing it to close it's Rendering surface.
 * The Sink can only be reset when in a state of NULL or READY. 
 * A new surface will be created on Pipeline play, Overlay or Window.
 * @param[in] name unique name of the Render Sink to reset
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT otherwise
 */
DslReturnType dsl_sink_render_reset(const wchar_t* name);

/**
 * @brief creates a new, uniquely named File Sink component
 * @param[in] name unique component name for the new File Sink
 * @param[in] file_path absolute or relative file path including extension
 * @param[in] codec one of DSL_CODEC_H264, DSL_CODEC_H265, DSL_CODEC_MPEG4
 * @param[in] container one of DSL_MUXER_MPEG4 or DSL_MUXER_MK4
 * @param[in] bitrate in bits per second - H264 and H265 only
 * @param[in] interval iframe interval to encode at
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_file_new(const wchar_t* name, const wchar_t* file_path, 
     uint codec, uint container, uint bitrate, uint interval);

/**
 * @brief creates a new, uniquely named File Record component
 * @param[in] name unique component name for the new Record Sink
 * @param[in] outdir absolute or relative path to the recording output dir.
 * @param[in] codec one of DSL_CODEC_H264, DSL_CODEC_H265, DSL_CODEC_MPEG4
 * @param[in] container one of DSL_MUXER_MPEG4 or DSL_MUXER_MK4
 * @param[in] bitrate in bits per second - H264 and H265 only
 * @param[in] interval iframe interval to encode at
 * @param[in] client_listener client callback for notifications of recording
 * events, DSL_RECORDING_EVENT_START and DSL_RECORDING_EVENT_STOP.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_record_new(const wchar_t* name, const wchar_t* outdir, uint codec, 
    uint container, uint bitrate, uint interval, 
    dsl_record_client_listener_cb client_listener);
     
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
 * @param[in] name unique name of the Record Sink to stop
 * @param[in] sync if set to true this call will block until the 
 * stop operation has completed or timeout DSL_RECORDING_STOP_WAIT_TIMEOUT.
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_record_session_stop(const wchar_t* name, boolean sync);

/**
 * @brief returns the current output directory in use by the named Sink
 * @param[in] name name of the Record Sink to query
 * @param[out] outdir current output directory set for the Record Sink
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_TAP_RESULT on failure
  */
DslReturnType dsl_sink_record_outdir_get(const wchar_t* name, const wchar_t** outdir);

/**
 * @brief Sets the video recording output directory for the named sink.
 * The directory must exist prior to calling or the services will fail
 * @param[in] name name of the Record Sink to update
 * @param[in] outdir new output directory to use by the Record Sink
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_record_outdir_set(const wchar_t* name, const wchar_t* outdir);

/**
 * @brief returns the video recording container type for the named Sink
 * A fixed size cache is created when the Pipeline is linked and played. 
 * The default cache size is set to DSL_DEFAULT_VIDEO_RECORD_CACHE_IN_SEC
 * @param[in] name name of the Record Tap to query
 * @param[out] container current setting, one of DSL_MUXER_MPEG4 or DSL_MUXER_MK4
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_TAP_RESULT on failure
  */
DslReturnType dsl_sink_record_container_get(const wchar_t* name, uint* container);

/**
 * @brief Sets the video recording container type for the named Sink
 * @param[in] name name of the Record Sink to update
 * @param[in] container new setting, one of DSL_MUXER_MPEG4 or DSL_MUXER_MK4
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_TAP_RESULT on failure
 */
DslReturnType dsl_sink_record_container_set(const wchar_t* name,  uint container);

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
 * @brief Adds a Video Player, Render or RTSP type, to a named Record Sink.
 * Once added, each recorded video's file_path will be added (or queued) with
 * the Video Player to be played according to the Players settings. 
 * @param[in] name unique name of the Record Sink to update
 * @param[in] player unique name of the Video Player to add
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_sink_record_video_player_add(const wchar_t* name, 
    const wchar_t* player);

        
/**
 * @brief Removes a Video Player, Render or RTSP type, from a named Record Sink.
 * @param[in] name unique name of the Record Sink to update
 * @param[in] player unique name of the Video Player to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_ODE_ACTION_RESULT otherwise.
 */
DslReturnType dsl_sink_record_video_player_remove(const wchar_t* name, 
    const wchar_t* player);

/**
 * @brief Adds a SMTP mailer to a named Record Sink. Once added, the Sink will 
 * use the Mailer to send out the file_path and details of each saved recording 
 * according to the Mailer's settings.
 * @param[in] name unique name of the Record Sink to update
 * @param[in] mailer unique name of the Mailer to add
 * @param[in] subject subject line to use for all outgoing mail
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT otherwise.
 */
DslReturnType dsl_sink_record_mailer_add(const wchar_t* name, 
    const wchar_t* mailer, const wchar_t* subject);
    
/**
 * @brief Removes a named Mailer from a named Record Sink.
 * @param[in] name unique name of the Record Sink to update
 * @param[in] mailer unique name of the Mailer to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT otherwise.
 */
DslReturnType dsl_sink_record_mailer_remove(const wchar_t* name, 
    const wchar_t* mailer);
    
/**
 * @brief gets the current codec, bitrate, and interval settings for the named Encode Sink
 * @param[in] name unique name of the Encode Sink to query
 * @param[out] codec current Codec either DSL_CODEC_H264 DSL_CODEC_H265
 * @param[out] bitrate current encoder bitrate in bits/sec for the named Encode Sink
 * @param[out] interval current encoder frame interval value
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_encode_settings_get(const wchar_t* name,
    uint* codec, uint* bitrate, uint* interval);

/**
 * @brief sets new codec, bitrate, and interval settings for the named Encode Sink
 * @param[in] name unique name of the Encode Sink to update
 * @param[in] codec new codec either DSL_CODEC_H264 DSL_CODEC_H265
 * @param[in] bitrate new encoder bitrate in bits/sec for the named Encode Sink
 * @param[in] interval new encoder frame interval value to use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_encode_settings_set(const wchar_t* name, 
    uint codec, uint bitrate, uint interval);

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
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_rtsp_server_settings_get(const wchar_t* name,
    uint* udpPort, uint* rtspPort);

/**
 * @brief creates a new, uniquely named WebRTC Sink component
 * @param[in] name unique component name for the new WebRTC Sink
 * @param[in] stun_server STUN server to use of the form stun://hostname:port.
 * Set to NULL to omit if using TURN server(s)
 * @param[in] turn_server TURN server(s) to use of the form 
 * turn(s)://username:password@host:port. Set to NULL to omit if using a STUN server
 * @param[in] codec either DSL_CODEC_H264 DSL_CODEC_H265
 * @param[in] bitrate in bits per second
 * @param[in] interval frame interval to encode at
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 * ** IMPORTANT: the WebRTC Sink implementation requires DS 1.18.0 or later
 */
DslReturnType dsl_sink_webrtc_new(const wchar_t* name, const wchar_t* stun_server, 
    const wchar_t* turn_server, uint codec, uint bitrate, uint interval);

/**
 * @brief Closes a uniquely named WebRTC Sink component's Websocket connection
 * if currently connected.
 */
DslReturnType dsl_sink_webrtc_connection_close(const wchar_t* name);

/**
 * @brief Queries a uniquely named WebRTC Sink component for its current
 * STUN and TURN servers in use.
 * @param[in] name unique component name for the new WebRTC Sink
 * @param[out] stun_server STUN server in use of the form stun://hostname:port
 * @param[out] turn_server TURN server(s) in use of the form 
 * turn(s)://username:password@host:port. "
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_SINK_RESULT on failure
 * ** IMPORTANT: the WebRTC Sink implementation requires DS 1.18.0 or later
 */
DslReturnType dsl_sink_webrtc_servers_get(const wchar_t* name, 
    const wchar_t** stun_server, const wchar_t** turn_server);
     
/**
 * @brief Updates a uniquely named WebRTC Sink component with new
 * STUN and TURN servers to use
 * @param[in] name unique name of the WebRTC Sink to update
 * @param[in] stun_server STUN server to use of the form stun://hostname:port.
 * Set to NULL to omit if using TURN server(s)
 * @param[in] turn_server TURN server(s) to use of the form 
 * turn(s)://username:password@host:port. Set to NULL to omit if using a STUN server
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_SINK_RESULT on failure
 * ** IMPORTANT: the WebRTC Sink implementation requires DS 1.18.0 or later
 */
DslReturnType dsl_sink_webrtc_servers_set(const wchar_t* name, 
    const wchar_t* stun_server, const wchar_t* turn_server);

/**
 * @brief Adds a callback to a named WebRTC Sink to be called on every change
 * of Websocket connection state.
 * @param[in] name name of the WebRTC Sink to add the callback to
 * @param[in] listener pointer to the client's function to call on state change.
 * @param[in] client_data opaque pointer to client data passed to the listener function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT otherwise.
 */
DslReturnType dsl_sink_webrtc_client_listener_add(const wchar_t* name, 
    dsl_sink_webrtc_client_listener_cb listener, void* client_data);

/**
 * @brief Removes a callback previously added with dsl_sink_webrtc_client_listener_add
 * @param[in] name name of the WebRTC Sink to update
 * @param[in] listener pointer to the client's listener function to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT otherwise.
 */
DslReturnType dsl_sink_webrtc_client_listener_remove(const wchar_t* name, 
    dsl_sink_webrtc_client_listener_cb listener);

/**
 * @brief Adds a new Websocket Path to be handled by the Websocket Server
 * Note: the server is created with one default Path = "/ws". paths must be added when
 * the Server is in a non-listing state. 
 * @param[in] path the new path to add to the Websocket Server
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_WEBSOCKET_SERVER_RESULT otherwise.
 */
DslReturnType dsl_websocket_server_path_add(const wchar_t* path);

/**
 * @brief Starts the Websocket Server listening on a specified Websocket port number
 * @param[in] port_number the Websocket port number to start listing on
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_WEBSOCKET_SERVER_RESULT otherwise.
 */
DslReturnType dsl_websocket_server_listening_start(uint port_number);

/**
 * @brief Stops the Websocket Server from listening on its current Websocket port number
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_WEBSOCKET_SERVER_RESULT otherwise.
 */
DslReturnType dsl_websocket_server_listening_stop();

/**
 * @brief Gets the current listening state for the Websocket Server
 * @param[out] is_listening true if the Server is listening on a Websocket Port
 * @param[out] port_number the Websocket port number the Server is listening on, or 0
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_WEBSOCKET_SERVER_RESULT otherwise.
 */
DslReturnType dsl_websocket_server_listening_state_get(boolean* is_listening,
    uint* port_number);

/**
 * @brief Adds a callback to the Websocket Server Manager (singleton) to be called 
 * on every incoming Websocket opened event.
 * @param[in] listener pointer to the client's function to call on state change.
 * @param[in] client_data opaque pointer to client data passed to the listener function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_WEBSOCKET_SERVER_RESULT otherwise.
 */
DslReturnType dsl_websocket_server_client_listener_add( 
    dsl_websocket_server_client_listener_cb listener, void* client_data);

/**
 * @brief Removes a callback previously added with dsl_websocket_server_client_listener_add
 * @param[in] listener pointer to the client's listener function to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_WEBSOCKET_SERVER_RESULT otherwise.
 */
DslReturnType dsl_websocket_server_client_listener_remove(
    dsl_websocket_server_client_listener_cb listener);

/**
 * @brief Creates a new, uniquely named Message Sink.
 * @param[in] name unique component name for the new Message Sink.
 * @param[in] converter_config_file absolute or relate path to a message-converter
 * configuration file of type text or csv.
 * @param[in] payload_type one of the DSL_MSG_PAYLOAD constants.
 * @param[in] broker_config_file absolute or relate path to a message-broker 
 * configuration file required by nvds_msgapi_* interface.
 * @param[in] protocol_lib absolute or relative path to the protocol adapter
 * that implements the nvds_msgapi_* interface.
 * @param[in] connection_string end point for communication with server of
 * the format <name>;<port>;<specifier>
 * @param[in] topic (optional) message topic name.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_message_new(const wchar_t* name, 
    const wchar_t* converter_config_file, uint payload_type, 
    const wchar_t* broker_config_file, const wchar_t* protocol_lib, 
    const wchar_t* connection_string, const wchar_t* topic);

///**
// * @brief Gets the current message meta-type filter in use by the named Message Sink.
// * @param name unique name of the Message Sink to query.
// * @param[out] meta_type the current meta-type filter in use, 
// * default = NVDS_EVENT_MSG_META.
// * @return DSL_RESULT_SUCCESS on successful query,  on of DSL_RESULT_SINK_RESULT 
// * on failure.
// */
//DslReturnType dsl_sink_message_meta_type_get(const wchar_t* name,
//    uint* meta_type);
//    
///**
// * @brief Sets the current message meta-type filter for the named Message Sink to use.
// * @param name unique name of the Message Sink to update.
// * @param[in] meta_type the new meta-type filter to use, must be equal to or
// * greater than NVDS_START_USER_META.
// * @return DSL_RESULT_SUCCESS on successful update,  on of DSL_RESULT_SINK_RESULT 
// * on failure.
// */
//DslReturnType dsl_sink_message_meta_type_set(const wchar_t* name,
//    uint meta_type);

/**
 * @brief Gets the current message converter settings for the named Message Sink.
 * @param[out] converter_config_file absolute file-path to the current
 * message converter config file in use.
 * @param[out] payload_type current payload type, one of the DSL_MSG_PAYLOAD constants.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_message_converter_settings_get(const wchar_t* name, 
    const wchar_t** converter_config_file, uint* payload_type);
    
/**
 * @brief Sets the current message converter settings for the named Message Sink.
 * @param[in] converter_config_file absolute or relate file-path to a new
 * message converter config file to use.
 * @param[in] payload_type new payload type to use, one of the DSL_MSG_PAYLOAD constants.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_message_converter_settings_set(const wchar_t* name, 
    const wchar_t* converter_config_file, uint payload_type);

/**
 * @brief Gets the current message broker settings for the named Message Sink.
 * @param[out] broker_config_file absolute file-path to the current message
 * broker config file in use.
 * @param[out] protocol_lib absolute file-path to the current protocol adapter
 * library in use.
 * @param[out] connection_string current connection string in use.
 * @param[out] topic (optional) message topic current in use.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_message_broker_settings_get(const wchar_t* name, 
    const wchar_t** broker_config_file, const wchar_t** protocol_lib,
    const wchar_t** connection_string, const wchar_t** topic);

/**
 * @brief Sets the message broker settings for the named Message Sink.
 * @param[in] broker_config_file absolute or relative file-path to 
 * a new message broker config file to use.
 * @param[out] protocol_lib absolute file-path to a new protocol adapter
 * library to use.
 * @param[in] connection_string new connection string in use.
 * @param[in] topic (optional) new message topic to use.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_message_broker_settings_set(const wchar_t* name, 
    const wchar_t* broker_config_file, const wchar_t* protocol_lib,
    const wchar_t* connection_string, const wchar_t* topic);

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
 * @brief Gets the current settings for the "sync" attribute for the named Sink
 * @param[in] name unique name of the Sink to query
 * @param[out] enabled the current setting for the Sink's "sync" attribute
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_SINK_RESULT otherwise
 */
DslReturnType dsl_sink_sync_enabled_get(const wchar_t* name, boolean* enabled);

/**
 * @brief Sets the "sync" attribute for the named Sink
 * @param[in] name unique name of the Sink to update
 * @param[in] enabled set to true to enable the Sink's "sync" attribute, false otherwise.
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_SINK_RESULT otherwise
 */
DslReturnType dsl_sink_sync_enabled_set(const wchar_t* name, boolean enabled);

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
 * @brief Queries a Component for its current NVIDIA buffer memory type.
 * @param[in] name name of the Component to query.
 * @param[out] type one of the DSL_NVBUF_MEM constant values.
 * @return DSL_RESULT_SUCCESS on successful query, one of DSL_RESULT_COMPONENT_RESULT on failure. 
 */
DslReturnType dsl_component_nvbuf_mem_type_get(const wchar_t* name, 
    uint* type);

/**
 * @brief Updates a Component with a new NVIDIA memory type to use.
 * @param[in] name name of the Component to update.
 * @param[in] type one of the DSL_NVBUF_MEM constant values.
 * @return DSL_RESULT_SUCCESS on successful update, one of DSL_RESULT_COMPONENT_RESULT on failure. 
 */
DslReturnType dsl_component_nvbuf_mem_type_set(const wchar_t* name, 
    uint type);

/**
 * @brief Updates a list of Components with a new NVIDIA memory type to use.
 * @param[in] names a null terminated list of component names to update.
 * @param[in] type one of the DSL_NVBUF_MEM constant values.
 * @return DSL_RESULT_SUCCESS on successful update, one of DSL_RESULT_COMPONENT_RESULT on failure. 
 */
DslReturnType dsl_component_nvbuf_mem_type_set_many(const wchar_t** names, 
    uint type);
    
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
 * @param[in] name unique name for the new Pipeline
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT
 */
DslReturnType dsl_pipeline_new(const wchar_t* name);

/**
 * @brief creates a new Pipeline for each name in the names array
 * @param[in] names a NULL terminated array of unique Pipeline names
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT
 */
DslReturnType dsl_pipeline_new_many(const wchar_t** names);

/**
 * @brief creates a new Pipeline and adds a list of components
 * @param[in] name name of the pipeline to update
 * @param[in] components NULL terminated array of component names to add
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT
 */
DslReturnType dsl_pipeline_new_component_add_many(const wchar_t* name, 
    const wchar_t** components);

/**
 * @brief deletes a Pipeline object by name.
 * @param[in] name unique name of the Pipeline to delete.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT otherwise.
 * @info any/all components owned by the pipeline move
 * to a state of not-in-use.
 */
DslReturnType dsl_pipeline_delete(const wchar_t* name);

/**
 * @brief deletes a NULL terminated list of pipelines
 * @param[in] names NULL terminated list of names to delete
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT
 * @info any/all components owned by the pipelines move
 * to a state of not-in-use.
 */
DslReturnType dsl_pipeline_delete_many(const wchar_t** names);

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
 * @param[in] name name of the pipeline to update
 * @param[in] component component names to add
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT
 */
DslReturnType dsl_pipeline_component_add(const wchar_t* name, 
    const wchar_t* component);

/**
 * @brief adds a list of components to a Pipeline 
 * @param[in] name name of the pipeline to update
 * @param[in] components NULL terminated array of component names to add
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT
 */
DslReturnType dsl_pipeline_component_add_many(const wchar_t* name, 
    const wchar_t** components);

/**
 * @brief removes a Component from a Pipeline
 * @param[in] name name of the Pipepline to update
 * @param[in] component name of the Component to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT
 */
DslReturnType dsl_pipeline_component_remove(const wchar_t* name, 
    const wchar_t* component);

/**
 * @brief removes a list of Components from a Pipeline
 * @param[in] name name of the Pipeline to update
 * @param[in] components NULL terminated array of component names to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT
 */
DslReturnType dsl_pipeline_component_remove_many(const wchar_t* name, 
    const wchar_t** components);

/**
 * @brief Queries a Pipeline's stream-muxer for its current NVIDIA buffer memory type.
 * @param[in] name name of the pipeline to query.
 * @param[out] type one of the DSL_NVBUF_MEM constant values.
 * @return DSL_RESULT_SUCCESS on successful query, one of 
 * DSL_RESULT_PIPELINE_RESULT on failure. 
 */
DslReturnType dsl_pipeline_streammux_nvbuf_mem_type_get(const wchar_t* name, 
    uint* type);

/**
 * @brief Updates a Pipeline's stream-muxer with a new NVIDIA memory type to use
 * @param[in] name name of the pipeline to update.
 * @param[in] type one of the DSL_NVBUF_MEM constant values.
 * @return DSL_RESULT_SUCCESS on successful update, one of 
 * DSL_RESULT_PIPELINE_RESULT on failure. 
 */
DslReturnType dsl_pipeline_streammux_nvbuf_mem_type_set(const wchar_t* name, 
    uint type);
    
/**
 * @brief Queryies the named Pipeline's stream-muxer for its current batch properties
 * @param[in] name name of the pipeline to query
 * @param[out] batchSize the current batch size in use
 * @param[out] batchTimeout the current batch timeout in use
 * @return DSL_RESULT_SUCCESS on successful query, one of 
 * DSL_RESULT_PIPELINE_RESULT on failure. 
 */
DslReturnType dsl_pipeline_streammux_batch_properties_get(const wchar_t* name, 
    uint* batchSize, uint* batchTimeout);

/**
 * @brief Updates the named Pipeline's batch-size and batch-push-timeout properties
 * @param[in] name name of the pipeline to update
 * @return DSL_RESULT_SUCCESS on successful update, one of 
 * DSL_RESULT_PIPELINE_RESULT on failure. 
 */
DslReturnType dsl_pipeline_streammux_batch_properties_set(const wchar_t* name, 
    uint batchSize, uint batchTimeout);

/**
 * @brief Queries the named Pipeline's stream-muxer for its current output dimensions.
 * @param[in] name name of the pipeline to query
 * @return DSL_RESULT_SUCCESS on successful query, one of 
 * DSL_RESULT_PIPELINE_RESULT on failure. 
 */
DslReturnType dsl_pipeline_streammux_dimensions_get(const wchar_t* name, 
    uint* width, uint* height);

/**
 * @brief updates the named Pipeline's stream-muxer output dimensions.
 * @param[in] name name of the pipeline to update
  * @return DSL_RESULT_SUCCESS on successful update, one of 
  * DSL_RESULT_PIPELINE_RESULT on failure. 
*/
DslReturnType dsl_pipeline_streammux_dimensions_set(const wchar_t* name, 
    uint width, uint height);

/**
 * @brief returns the current setting - enabled/disabled - for the Streammux padding
 * property for the named Pipeline.
 * @param[in] name name of the Display to query
 * @param[out] enable true if the aspect ration is fixed, false if not
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT
 */
DslReturnType dsl_pipeline_streammux_padding_get(const wchar_t* name, 
    boolean* enabled);

/**
 * @brief updates the current setting - enabled/disabled - for Streammux padding
 * property for the name Pipeline.
 * @param[in] name name of the Pipeline to update
 * @param[out] enable set true to fix the aspect ratio, false to disable
 * @return DSL_RESULT_SUCCESS on successful update, one of 
 * DSL_RESULT_PIPELINE_RESULT on failure. 
 */
DslReturnType dsl_pipeline_streammux_padding_set(const wchar_t* name, 
    boolean enabled);

/**
 * @brief returns the current num-surfaces-per-frame stream-muxer setting 
 * for the named Pipeline.
 * @param[in] name name of the Pipeline to query.
 * @param[out] num number of surfaces per frame [1..4].
 * @return DSL_RESULT_SUCCESS on successful query, one of 
 * DSL_RESULT_PIPELINE_RESULT on failure. 
 */
DslReturnType dsl_pipeline_streammux_num_surfaces_per_frame_get(
    const wchar_t* name, uint* num);

/**
 * @brief sets the current num-surfaces-per-frame stream-muxer setting 
 * for the named Pipeline.
 * @param[in] name name of the Pipeline to update
 * @param[in] num number of surfaces per frame [1..4]
 * @return DSL_RESULT_SUCCESS on successful update, one of 
 * DSL_RESULT_PIPELINE_RESULT on failure. 
 */
DslReturnType dsl_pipeline_streammux_num_surfaces_per_frame_set(
    const wchar_t* name, uint num);

/**
 * @brief adds a named Tiler to a named Pipeline's Stream-Muxer output.
 * The Stream-Muxer can have at most one Tiler.
 * @param[in] name name of the Pipeline to update.
 * @param[in] tiler name of the Tiler to add.
 * @return DSL_RESULT_SUCCESS on successful update, one of 
 * DSL_RESULT_PIPELINE_RESULT on failure. 
 */
DslReturnType dsl_pipeline_streammux_tiler_add(const wchar_t* name, 
    const wchar_t* tiler);

/**
 * @brief removes a Tiler from a named Pipeline's Stream-Muxer output,
 * previously added with dsl_pipeline_streammux_tiler_add
 * @param[in] name name of the Pipeline to update
 * @return DSL_RESULT_SUCCESS on successful update, one of 
 * DSL_RESULT_PIPELINE_RESULT on failure. 
 */
DslReturnType dsl_pipeline_streammux_tiler_remove(const wchar_t* name);

/**
 * @brief pauses a Pipeline if in a state of playing
 * @param[in] name unique name of the Pipeline to pause.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_pause(const wchar_t* name);

/**
 * @brief plays a Pipeline if in a state of paused
 * @param[in] name unique name of the Pipeline to play.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_play(const wchar_t* name);

/**
 * @brief Stops a Pipeline if in a state of paused or playing
 * @param[in] name unique name of the Pipeline to stop.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_stop(const wchar_t* name);

/**
 * @brief gets the current state of a Pipeline
 * @param[in] name unique name of the Pipeline to query
 * @param[out] state one of the DSL_STATE_* values representing the current state
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_state_get(const wchar_t* name, uint* state);

/**
 * @brief gets the type of source(s) in use, live, or non-live 
 * @param name unique name of the Pipeline to query
 * @param is_live true if the Pipeline's sources are live, false otherwise
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_is_live(const wchar_t* name, boolean* is_live);

/**
 * @brief dumps a Pipeline's graph to dot file.
 * @param[in] name unique name of the Pipeline to dump
 * @param[in] filename name of the file without extention.
 * The caller is responsible for providing a correctly formated filename
 * The diretory location is specified by the GStreamer debug 
 * environment variable GST_DEBUG_DUMP_DOT_DIR
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */ 
DslReturnType dsl_pipeline_dump_to_dot(const wchar_t* name, wchar_t* filename);

/**
 * @brief dumps a Pipeline's graph to dot file prefixed
 * with the current timestamp.  
 * @param[in] name unique name of the Pipeline to dump
 * @param[in] filename name of the file without extention.
 * The caller is responsible for providing a correctly formated filename
 * The diretory location is specified by the GStreamer debug 
 * environment variable GST_DEBUG_DUMP_DOT_DIR
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */ 
DslReturnType dsl_pipeline_dump_to_dot_with_ts(const wchar_t* name, wchar_t* filename);

/**
 * @brief adds a callback to be notified on End of Stream (EOS)
 * @param[in] name name of the pipeline to update
 * @param[in] listener pointer to the client's function to call on EOS
 * @param[in] client_data opaque pointer to client data passed into the listener function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_eos_listener_add(const wchar_t* name, 
    dsl_eos_listener_cb listener, void* client_data);

/**
 * @brief removes a callback previously added with dsl_pipeline_eos_listener_add
 * @param[in] name name of the pipeline to update
 * @param[in] listener pointer to the client's function to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_eos_listener_remove(const wchar_t* name, 
    dsl_eos_listener_cb listener);

/**
 * @brief Adds a callback to be notified on the event of an error message received by
 * the Pipeline's bus-watcher. The callback is called from the mainloop context allowing
 * clients to change the state of, and components within, the Pipeline
 * @param[in] name name of the pipeline to update
 * @param[in] handler pointer to the client's callback function to add
 * @param[in] client_data opaque pointer to client data passed back to the handler function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_error_message_handler_add(const wchar_t* name, 
    dsl_error_message_handler_cb handler, void* client_data);

/**
 * @brief Removes a callback previously added with dsl_pipeline_error_message_handler_add
 * @param[in] name name of the pipeline to update
 * @param[in] handler pointer to the client's callback function to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_error_message_handler_remove(const wchar_t* name, 
    dsl_error_message_handler_cb handler);

/**
 * @brief Gets the last error message received by the Pipeline's bus watcher
 * @param[in] name name of the pipeline to query
 * @param[out] source name of the GST object that was the source of the error message
 * @param[out] message error message sent from the source object
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_error_message_last_get(const wchar_t* name, 
    const wchar_t** source, const wchar_t** message);

/**
 * @brief adds a callback to be notified on change of Pipeline state
 * @param[in] name name of the pipeline to update
 * @param[in] listener pointer to the client's function to call on state change
 * @param[in] client_data opaque pointer to client data passed into the listener function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_state_change_listener_add(const wchar_t* name, 
    dsl_state_change_listener_cb listener, void* client_data);

/**
 * @brief removes a callback previously added with dsl_pipeline_state_change_listener_add
 * @param[in] name name of the pipeline to update
 * @param[in] listener pointer to the client's function to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_state_change_listener_remove(const wchar_t* name, 
    dsl_state_change_listener_cb listener);

/**
 * @brief gets the Pipeline's current XWindow handle. The handle will be NULL until one
 * is created on Pipeline play, or provided prior to play by calling xwindow handle set.
 * @param[in] name name of the Pipeline to query
 * @param[out] xwindow XWindow handle currently in use. NULL if none 
 * @return DSL_RESULT_SUCCESS on successful query, DSL_RESULT_PIPELINE_RESULT otherwise.
 */
DslReturnType dsl_pipeline_xwindow_handle_get(const wchar_t* name, uint64_t* xwindow);

/**
 * @brief gets the Pipeline's current XWindow handle. The handle will be NULL until one
 * is created on Pipeline play, or provided prior to play by calling xwindow handle set.
 * @param[in] name name of the Pipeline to update
 * @param[in] xwindow XWindow handle to use on Pipeline play. Requires a Window Sink
 * @return DSL_RESULT_SUCCESS on successful update, DSL_RESULT_PIPELINE_RESULT otherwise.
 */
DslReturnType dsl_pipeline_xwindow_handle_set(const wchar_t* name, uint64_t window);

/**
 * @brief clears the Pipeline's XWindow
 * @param[in] name name of the pipeline to update
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT otherwise.
 */
DslReturnType dsl_pipeline_xwindow_clear(const wchar_t* name);

/**
 * @brief destroys the Pipeline's XWindow if one exists and was not provided by the
 * client with an earlier call to dsl_pipeline_xwindow_handle_set
 * @param[in] name name of the pipeline to update
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT otherwise.
 */
DslReturnType dsl_pipeline_xwindow_destroy(const wchar_t* name);

/**
 * @brief gets the current Pipeline XWindow Offsets. X and Y offsets will return 0
 * prior to window creation which occurs when the Pipeline is played. 
 * @param[in] name name of the pipeline to query
 * @param[out] x_offset offset in the x direction of the XWindow in pixels
 * @param[out] x_offset offset in the Y direction of the XWindow in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT otherwise.
 */
DslReturnType dsl_pipeline_xwindow_offsets_get(const wchar_t* name, 
    uint* x_offset, uint* y_offset);

/**
 * @brief gets the current Pipeline XWindow dimensions. 
 * @param[in] name name of the pipeline to query
 * @param[out] width width of the XWindow in pixels
 * @param[out] heigth height of the Window in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT otherwise.
 */
DslReturnType dsl_pipeline_xwindow_dimensions_get(const wchar_t* name, 
    uint* width, uint* height);

/**
 * @brief gets the current full-screen-enabled setting for the Pipeline's XWindow
 * @param[in] name name of the pipeline to query
 * @param[out] enabled true if full-screen-mode is currently enabled, false otherwise 
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT otherwise.
 */
DslReturnType dsl_pipeline_xwindow_fullscreen_enabled_get(const wchar_t* name, 
    boolean* enabled);

/**
 * @brief sets the full-screen-enabled setting for the Pipeline's XWindow
 * @param[in] name name of the pipeline to update
 * @param[in] enabled if true, sets the XWindow to full-screen on creation.
 * The service will fail if called after the XWindow has been created.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT otherwise.
 */
DslReturnType dsl_pipeline_xwindow_fullscreen_enabled_set(const wchar_t* name, 
    boolean enabled);

/**
 * @brief adds a callback to be notified on XWindow KeyRelease Event
 * @param[in] name name of the pipeline to update
 * @param[in] handler pointer to the client's function to handle XWindow key events.
 * @param[in] client_data opaque pointer to client data passed into the handler function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_xwindow_key_event_handler_add(const wchar_t* name, 
    dsl_xwindow_key_event_handler_cb handler, void* client_data);

/**
 * @brief removes a callback previously added with dsl_pipeline_xwindow_key_event_handler_add
 * @param[in] name name of the pipeline to update
 * @param[in] handler pointer to the client's function to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_xwindow_key_event_handler_remove(const wchar_t* name, 
    dsl_xwindow_key_event_handler_cb handler);

/**
 * @brief adds a callback to be notified on XWindow ButtonPress Event
 * @param[in] name name of the pipeline to update
 * @param[in] handler pointer to the client's function to call to handle XWindow button events.
 * @param[in] client_data opaque pointer to client data passed into the handler function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_xwindow_button_event_handler_add(const wchar_t* name, 
    dsl_xwindow_button_event_handler_cb handler, void* client_data);

/**
 * @brief removes a callback previously added with dsl_pipeline_xwindow_button_event_handler_add
 * @param[in] name name of the pipeline to update
 * @param[in] handler pointer to the client's function to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_xwindow_button_event_handler_remove(const wchar_t* name, 
    dsl_xwindow_button_event_handler_cb handler);

/**
 * @brief adds a callback to be notified on XWindow Delete Message Event
 * @param[in] name name of the pipeline to update
 * @param[in] handler pointer to the client's function to call to handle XWindow Delete event.
 * @param[in] client_data opaque pointer to client data passed into the handler function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_xwindow_delete_event_handler_add(const wchar_t* name, 
    dsl_xwindow_delete_event_handler_cb handler, void* client_data);

/**
 * @brief removes a callback previously added with dsl_pipeline_xwindow_delete_event_handler_add
 * @param[in] name name of the pipeline to update
 * @param[in] handler pointer to the client's function to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_xwindow_delete_event_handler_remove(const wchar_t* name, 
    dsl_xwindow_delete_event_handler_cb handler);

/**
 * @brief Creates a new main-context and main-loop for a named Pipeline. This service
 * must be called prior to calling dsl_pipeline_play and dsl_pipeline_main_loop_run.
 * @param name name of the pipeline to update. 
 */
DslReturnType dsl_pipeline_main_loop_new(const wchar_t* name);
    
/**
 * @brief Runs and joins a Pipeline's main-loop that was previously created 
 * with a call to dsl_pipeline_main_loop_new.
 * Note: this call will block until dsl_pipeline_main_loop_quit is called.
 * @param name name of the Pipeline to update.
 */
DslReturnType dsl_pipeline_main_loop_run(const wchar_t* name);

/**
 * @brief Quits a Pipeline's running main-loop allowing the thread blocked
 * on the call to dsl_pipeline_main_loop_run to return.
 * @param name name of the Pipeline to update.
 */
DslReturnType dsl_pipeline_main_loop_quit(const wchar_t* name);

/**
 * @brief Deletes a Pipeline's main-context and main-loop previously created
 * by calling dsl_pipeline_main_loop_new. 
 * @param name name of the Pipeline to update. 
 */
DslReturnType dsl_pipeline_main_loop_delete(const wchar_t* name);
/**
 * @brief Creates a new, uniquely named Player
 * @param[in] name unique name for the new Player
 * @parma[in] file_source name of the file source to use for the Player
 * @parma[in] sink name of the sink to use for the Player
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT
 */
DslReturnType dsl_player_new(const wchar_t* name,
    const wchar_t* file_source, const wchar_t* sink);

/**
 * @brief Creates a new, uniquely named Video Render Player
 * @param[in] name unique name for the new Player
 * @param[in] file_path absolute or relative path to the file to render
 * @param[in] render_type one of DSL_RENDER_TYPE_OVERLAY or DSL_RENDER_TYPE_WINDOW
 * @param[in] offset_x offset in the X direction for the Render Sink in units of pixels
 * @param[in] offset_y offset in the Y direction for the Render Sink in units of pixels
 * @param[in] zoom digital zoom factor in units of %
 * @param[in] repeat_enabled set to true to auto-repeat on EOS
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT
 */
DslReturnType dsl_player_render_video_new(const wchar_t* name,  const wchar_t* file_path, 
   uint render_type, uint offset_x, uint offset_y, uint zoom, boolean repeat_enabled);

/**
 * @brief Creates a new, uniquely named Image Render Player
 * @param[in] name unique name for the new Player
 * @param[in] file_path absolute or relative path to the image to render
 * @param[in] render_type one of DSL_RENDER_TYPE_OVERLAY or DSL_RENDER_TYPE_WINDOW
 * @param[in] offset_x offset in the X direction for the Render Sink in units of pixels
 * @param[in] offset_y offset in the Y direction for the Render Sink in units of pixels
 * @param[in] zoom digital zoom factor in units of %
 * @param[in] timeout will generate an EOS event on timeout in units of seconds, 0 = no timeout.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT
 */
DslReturnType dsl_player_render_image_new(const wchar_t* name, const wchar_t* file_path,
    uint render_type, uint offset_x, uint offset_y, uint zoom, uint timeout);

/**
 * @brief Gets the current file path in use by the named Image or File Render Player
 * @param[in] name name of the Player to query
 * @param[out] file_path in use by the Render Player
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT otherwise.
 */
DslReturnType dsl_player_render_file_path_get(const wchar_t* name, 
    const wchar_t** file_path);
    
/**
 * @brief Sets the current file path to use for the named Image or File Render Player
 * @param[in] name name of the Render Player to update
 * @param[in] file_path file path for the Render Player to use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT otherwise.
 */
DslReturnType dsl_player_render_file_path_set(const wchar_t* name, 
    const wchar_t* file_path);
    
/**
 * @brief Queues a file path to be played, in turn, on EOS Termination by the  
 * named Image or File Render Player.
 * @param[in] name name of the Render Player to update
 * @param[in] file_path file path for the Render Player to queue
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT otherwise.
 */
DslReturnType dsl_player_render_file_path_queue(const wchar_t* name, 
    const wchar_t* file_path);

/**
 * @brief returns the current X and Y offsets for the Render Player
 * @param[in] name name of the Render Player to query
 * @param[out] offset_x current offset in the X direction for the Render Player in pixels
 * @param[out] offset_y current offset in the Y direction for the Render Player in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT otherwise.
 */
DslReturnType dsl_player_render_offsets_get(const wchar_t* name, 
    uint* offset_x, uint* offset_y);

/**
 * @brief Sets the X and Y offsets for the Render Player
 * @param[in] name name of the Render Player to update
 * @param[in] offset_x new offset for the Render Player in the X direction in pixels
 * @param[in] offset_y new offset for the Render Player in the X direction in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT otherwise.
 */
DslReturnType dsl_player_render_offsets_set(const wchar_t* name, 
    uint offset_x, uint offset_y);

/**
 * @brief returns the current X and Y offsets for the Render Player
 * @param[in] name name of the Render Player to query
 * @param[out] offset_x current offset in the X direction for the Render Player in pixels
 * @param[out] offset_y current offset in the Y direction for the Render Player in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT otherwise.
 */
DslReturnType dsl_player_render_dimensions_get(const wchar_t* name, 
    uint* width, uint* height);

/**
 * @brief Gets the current zoom setting in use by the named Image or File Render Player
 * @param[in] name name of the Player to query
 * @param[out] zoom zoom setting in use by the Render Player, in unit of %
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT otherwise
 */
DslReturnType dsl_player_render_zoom_get(const wchar_t* name, uint* zoom);
    
/**
 * @brief Sets the zoom setting to use for the named Image or File Render Player
 * @param[in] name name of the Render Player to update
 * @param[in] zoom zoom setting for the Render Player to use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT otherwise.
 */
DslReturnType dsl_player_render_zoom_set(const wchar_t* name, uint zoom);

/**
 * @brief Resets the Render Player causing it to close it's Rendering surface.
 * The Player can only be reset when in a state READY, i.e. after Stop or EOS. 
 * A new surface (Overlay or Window) will be created on next call to dsl_player_play.
 * @param[in] name unique name of the Render Player to udate
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT otherwise
 */
DslReturnType dsl_player_render_reset(const wchar_t* name);

/**
 * @brief Gets the current timeout setting in use by the named Image Render Player
 * @param[in] name name of the Player to query
 * @param[out] timeout timeout setting in use by the Render Player, in unit of seconds.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT otherwise.
 */
DslReturnType dsl_player_render_image_timeout_get(const wchar_t* name, uint* timeout);
    
/**
 * @brief Sets the timeout setting to use for the named Image Render Player
 * @param[in] name name of the Render Player to update
 * @param[in] timeout timeout setting for the Render Player to use, in uints of seconds.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT otherwise.
 */
DslReturnType dsl_player_render_image_timeout_set(const wchar_t* name, uint timeout);
        
/**
 * @brief Gets the current repeat_enabled setting in use by the named video Render Player
 * @param[in] name name of the Player to query
 * @param[out] timeout_enabled enabled if true, disabled otherwise
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT otherwise.
 */
DslReturnType dsl_player_render_video_repeat_enabled_get(const wchar_t* name, 
    boolean* repeat_enabled);
    
/**
 * @brief Sets the repeat enabled setting to use for the named video Render Player
 * @param[in] name name of the Render Player to update
 * @param[in] repeat_enabled set to true to enable, false otherwise.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT otherwise.
 */
DslReturnType dsl_player_render_video_repeat_enabled_set(const wchar_t* name, 
    boolean repeat_enabled);
   
/**
 * @brief Adds a callback to be notified on Player Termination Event.
 * Termination can be the result of EOS, image timeout, or XWindow deletion.
 * @param[in] name name of the player to update
 * @param[in] listener pointer to the client's function to call on Termination event.
 * @param[in] client_data opaque pointer to client data passed to the listener function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT otherwise.
 */
DslReturnType dsl_player_termination_event_listener_add(const wchar_t* name, 
    dsl_player_termination_event_listener_cb listener, void* client_data);

/**
 * @brief Removes a callback previously added with dsl_player_termination_event_listener_add
 * @param[in] name name of the player to update
 * @param[in] listener pointer to the client's listener function to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT otherwise.
 */
DslReturnType dsl_player_termination_event_listener_remove(const wchar_t* name, 
    dsl_player_termination_event_listener_cb listener);

/**
 * @brief gets the Player's current XWindow handle. The handle will be NULL until one
 * is created on Player play, or provided prior to play by calling xwindow handle set.
 * @param[in] name name of the Player to query
 * @param[out] xwindow XWindow handle currently in use. NULL if none 
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT otherwise.
 */
DslReturnType dsl_player_xwindow_handle_get(const wchar_t* name, uint64_t* xwindow);

/**
 * @brief gets the Players's current XWindow handle. The handle will be NULL until one
 * is created on Player play, or provided prior to play by calling xwindow handle set.
 * @param[in] name name of the Player to update
 * @param[in] xwindow XWindow handle to use on Player play. Requires a Window Sink
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT otherwise.
 */
DslReturnType dsl_player_xwindow_handle_set(const wchar_t* name, uint64_t window);

/**
 * @brief adds a callback to be notified on XWindow KeyRelease Event
 * @param[in] name name of the pipeline to update
 * @param[in] handler pointer to the client's function to handle XWindow key events.
 * @param[in] client_data opaque pointer to client data passed into the handler function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT otherwise.
 */
DslReturnType dsl_player_xwindow_key_event_handler_add(const wchar_t* name, 
    dsl_xwindow_key_event_handler_cb handler, void* client_data);

/**
 * @brief removes a callback previously added with dsl_pipeline_xwindow_key_event_handler_add
 * @param[in] name name of the pipeline to update
 * @param[in] handler pointer to the client's function to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT otherwise.
 */
DslReturnType dsl_player_xwindow_key_event_handler_remove(const wchar_t* name, 
    dsl_xwindow_key_event_handler_cb handler);

/**
 * @brief Plays a Player if in a state of NULL or Paused
 * @param[in] name unique name of the Player to play.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT on failure.
 */
DslReturnType dsl_player_play(const wchar_t* name);

/**
 * @brief Pauses a Player if in a state of Playing
 * @param[in] name unique name of the Player to pause.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT.
 */
DslReturnType dsl_player_pause(const wchar_t* name);

/**
 * @brief Stops a Player if in a state of Paused or Playing
 * @param[in] name unique name of the Player to stop.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT on failure.
 */
DslReturnType dsl_player_stop(const wchar_t* name);

/**
 * @brief Stops a Player and plays the next queued file
 * @param name unique name of the Render Player to play next
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT on failure.
 */
DslReturnType dsl_player_render_next(const wchar_t* name);

/**
 * @brief gets the current state of a Player
 * @param[in] name unique name of the player to query
 * @param[out] state one of the DSL_STATE_* values representing the current state
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT otherwise.
 */
DslReturnType dsl_player_state_get(const wchar_t* name, uint* state);

/**
 * @brief Queries DSL to determine if a uniquely named Player Object exists 
 * @param name of the Player to determine if exists
 * @return true if the named Player exists, false otherwise
 */
boolean dsl_player_exists(const wchar_t* name);

/**
 * @brief Deletes a Player object by name.
 * @param[in] name unique name of the Player to delete.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT otherwise.
 * @info the Source and Sink components owned by the player move
 * to a state of "not-in-use".
 */
DslReturnType dsl_player_delete(const wchar_t* name);

/**
 * @brief Deletes all media players in memory
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT
 * @info the source and sink components owned by the players move
 * to a state of not-in-use.
 */
DslReturnType dsl_player_delete_all();

/**
 * @brief Returns the current number of Players in memory
 * @return size of the list of Players
 */
uint dsl_player_list_size();

/**
 * @brief Creates a uniquely named SMTP Mailer 
 * @param[in] name unique name for the new Mailer
 * @return DSL_RESULT_SUCCESS on success, one of DSL_MAILER_RESULT otherwise
 */
DslReturnType dsl_mailer_new(const wchar_t* name);

/**
 * @brief Gets the current Enabled state for the named SMTP Mailer
 * @param[in] name unique name of the Mailer to query
 * @return DSL_RESULT_SUCCESS on success, one of DSL_MAILER_RESULT otherwise
 */
DslReturnType dsl_mailer_enabled_get(const wchar_t* name, boolean* enabled);

/**
 * @brief Sets the Enabled state of the named SMTP Mailer
 * Disabling the Mailer will block all subsequent emails from being queued for sending.
 * @param[in] name unique name of the Mailer to update
 * @param[in] enabled set to true to enable, false to disabled
 * @return DSL_RESULT_SUCCESS on success, one of DSL_MAILER_RESULT otherwise
 */
DslReturnType dsl_mailer_enabled_set(const wchar_t* name, boolean enabled);

/**
 * @brief Sets the user credentials for the Mailer's SMTP host for all subsequent emails
 * @param[in] name unique name of the Mailer to update
 * @param[in] username username to use
 * @param[in] password password to use
 * @return DSL_RESULT_SUCCESS on success, one of DSL_MAILER_RESULT otherwise
 */
DslReturnType dsl_mailer_credentials_set(const wchar_t* name, 
    const wchar_t* username, const wchar_t* password);

/**
 * @brief Gets the current SMTP server URL setting for the named Mailer
 * @param[in] name unique name of the Mailer to query
 * @param[out] server_url current server URL in use
 * @return DSL_RESULT_SUCCESS on success, one of DSL_MAILER_RESULT otherwise
 */
DslReturnType dsl_mailer_server_url_get(const wchar_t* name,
    const wchar_t** server_url);

/**
 * @brief Sets the SMTP server URL for the Mailer to use for all 
 * subsequence email sent out
 * @param[in] name unique name of the Mailer to update
 * @param[in] server_url to use 
 * @return DSL_RESULT_SUCCESS on success, one of DSL_MAILER_RESULT otherwise
 */
DslReturnType dsl_mailer_server_url_set(const wchar_t* name,
    const wchar_t* server_url);

/**
 * @brief Gets the current From address for the named Mailer
 * @param[in] name unique name of the Mailer to query
 * @param[out] display_name current From address display name
 * @param[out] address current From address
 * @return DSL_RESULT_SUCCESS on success, one of DSL_MAILER_RESULT otherwise
 */
DslReturnType dsl_mailer_address_from_get(const wchar_t* name,
    const wchar_t** display_name, const wchar_t** address);

/**
 * @brief Sets the current From address for the Mailer to use 
 * for all subsequent emails
 * @param[in] name unique name of the Mailer to update
 * @param[in] display_name new From address display name to use
 * @param[in] address new From address
 * @return DSL_RESULT_SUCCESS on success, one of DSL_MAILER_RESULT otherwise
 */
DslReturnType dsl_mailer_address_from_set(const wchar_t* name,
    const wchar_t* display_name, const wchar_t* address);

/**
 * @brief Returns the current SMTP SSL enabled setting in use by the named Mailer
 * The setting is enabled by default
 * @param[in] name unique name of the Mailer to query
 * @param[out] enabled true if SSL is enabled, false otherwise 
 * @return DSL_RESULT_SUCCESS on success, one of DSL_MAILER_RESULT otherwise
 */
DslReturnType dsl_mailer_ssl_enabled_get(const wchar_t* name, boolean* enabled);

/**
 * @brief Sets the SMTP SSL enabled setting for the named Mailer
 * @param[in] name unique name of the Mailer to update
 * @param[in] enabled set to true to enable, false otherwise
 * @return DSL_RESULT_SUCCESS on success, one of DSL_MAILER_RESULT otherwise
*/
DslReturnType dsl_mailer_ssl_enabled_set(const wchar_t* name, boolean enabled);

/**
 * @brief Adds a new email address to the To list of the named Mailer
 * @param[in] name unique name of the Mailer to update
 * @param[in] display_name display name for the To address
 * @param[in] address qualifed email To address
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT otherwise.
 */
DslReturnType dsl_mailer_address_to_add(const wchar_t* name,
    const wchar_t* display_name, const wchar_t* address);

/**
 * @brief Removes all current To addresses from the named Mailer
 * @param[in] name unique name of the Mailer to update
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT otherwise.
 */
DslReturnType dsl_mailer_address_to_remove_all(const wchar_t* name);

/**
 * @brief Adds a new email address to the Cc list of the name Mailer
 * @param[in] name unique name of the Mailer to update
 * @param[in] display_name display name for the Cc address
 * @param[in] address qualifed email Cc address
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT otherwise.
 */
DslReturnType dsl_mailer_address_cc_add(const wchar_t* name,
    const wchar_t* display_name, const wchar_t* address);

/**
 * @brief Removes all current CC addresses from the named Mailer
 * @param[in] name unique name of the Mailer to update
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT otherwise.
 */
DslReturnType dsl_mailer_address_cc_remove_all(const wchar_t* name);

/**
 * @brief Sends a test message using the current SMTP
 * settings and email addresses (From, To, Cc)
 * @param[in] name unique name of the Mailer to test
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT otherwise.
 */
DslReturnType dsl_mailer_test_message_send(const wchar_t* name);

/**
 * @brief Deletes a SMTP Mailer Object by name.
 * @param[in] name unique name of the Mailer to delete.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT otherwise.
 */
DslReturnType dsl_mailer_delete(const wchar_t* name);

/**
 * @brief Deletes all SMTP Mailers in memory
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PLAYER_RESULT
 */
DslReturnType dsl_mailer_delete_all();

/**
 * @brief Queries DSL to determine if a uniquely named Mailer Object exists 
 * @param name of the Mailer to check for existence
 * @return true if the named Mailer exists, false otherwise
 */
boolean dsl_mailer_exists(const wchar_t* name);

/**
 * @brief Returns the current number of Mailers in memeory
 * @return size of the list of Mailers
 */
uint dsl_mailer_list_size();

/**
 * @brief Creates a uniquely name Message Broker.
 * @param name unique name for the new Message Broker
 * @param[in] broker_config_file absolute or relate path to a message-broker 
 * configuration file required by nvds_msgapi_* interface.
 * @param[in] protocol_lib absolute or relative path to the protocol adapter
 * that implements the nvds_msgapi_* interface.
 * @param[in] connection_string end point for communication with server of
 * the format <name>;<port>;<specifier>
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_BROKER_RESULT otherwise.
 */
DslReturnType dsl_message_broker_new(const wchar_t* name, 
    const wchar_t* broker_config_file, const wchar_t* protocol_lib, 
    const wchar_t* connection_string);

/**
 * @brief Gets the current broker settings for the named Message Broker.
 * @param[out] broker_config_file absolute file-path to the current message
 * broker config file in use.
 * @param[out] protocol_lib absolute file-path to the current protocol adapter
 * library in use.
 * @param[out] connection_string current connection string in use.
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_BROKER_RESULT otherwise.
 */
DslReturnType dsl_message_broker_settings_get(const wchar_t* name, 
    const wchar_t** broker_config_file, const wchar_t** protocol_lib,
    const wchar_t** connection_string);

/**
 * @brief Sets the broker settings for the named Message Broker.
 * @param[in] broker_config_file absolute or relative file-path to 
 * a new message broker config file to use.
 * @param[out] protocol_lib absolute file-path to a new protocol adapter
 * library to use.
 * @param[in] connection_string new connection string in use.
 * @param[in] topic (optional) new message topic to use.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_message_broker_settings_set(const wchar_t* name, 
    const wchar_t* broker_config_file, const wchar_t* protocol_lib,
    const wchar_t* connection_string);

/**
 * @brief Connects a uniquely name Iot Message Broker with an IoT hub.
 * @param[in] name unique name of the Message Broker to connect.
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_BROKER_RESULT otherwise.
 */
DslReturnType dsl_message_broker_connect(const wchar_t* name);

/**
 * @brief Disconnects a uniquely named Message Broker from an IoT hub.
 * @param[in] name unique name of the Message Broker to disconnect.
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_BROKER_RESULT otherwise.
 */
DslReturnType dsl_message_broker_disconnect(const wchar_t* name);

/**
 * @brief Returns the current connected state for the named Message Broker.
 * @param[in] name unique name of the Message Broker to query.
 * @param[out] connected true if connected, false otherwise.
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_BROKER_RESULT otherwise.
 */
DslReturnType dsl_message_broker_is_connected(const wchar_t* name, boolean* connected);

/**
 * @brief Sends an asynchronous message to a connected end-point.
 * @param name name of the Message Broker to send the message.
 * @param topic topic for the message to send.
 * @param message payload of the message to send
 * @param size of the message payload in bytes.
 * @param result callback to be invoked to provide an asynchronous result
 * of the send operation.
 * @param[in] client_data opaque pointer to client data to be passed backed to 
 * the hanlder function when called.
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_BROKER_RESULT otherwise.
 */
DslReturnType dsl_message_broker_message_send_async(const wchar_t* name,
    const wchar_t* topic, void* message, size_t size, 
    dsl_message_broker_send_result_listener_cb result_listener, void* user_data);

/**
 * @brief Adds a client subscriber callback function to a named Message Broker.
 * Once added, the client will be called with each message received for a given
 * set of topics.
 * @param[in] name unique name of the Message Broker to update.
 * @param[in] subscriber subscriber function to add to the named Message Broker.
 * @param[in] topics topics for the subscriber to register. 
 * @param[in] client_data opaque pointer to client data to be passed backed to 
 * the hanlder function when called.
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_BROKER_RESULT otherwise.
 */
DslReturnType dsl_message_broker_subscriber_add(const wchar_t* name,
    dsl_message_broker_subscriber_cb subscriber, const wchar_t** topics,
    void* client_data);
    
/**
 * @brief Removes a client handler previously added with a call to
 * dsl_message_broker_subscriber_add.
 * @param[in] name unique name of the Message Broker to update.
 * @param[in] subscriber subscriber function to remove from the named Message Broker.
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_BROKER_RESULT otherwise.
 */
DslReturnType dsl_message_broker_subscriber_remove(const wchar_t* name,
    dsl_message_broker_subscriber_cb subscriber);

/**
 * @brief Adds a client handler callback function to a named Message Broker.
 * Once added, the client will be called on each messaging error that occurs.
 * @param[in] name unique name of the Message Broker to update.
 * @param[in] handler error handler function to add to the named Message Broker.
 * @param[in] client_data opaque pointer to client data to be passed backed to 
 * the handler function when called.
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_BROKER_RESULT otherwise.
 */
DslReturnType dsl_message_broker_connection_listener_add(const wchar_t* name,
    dsl_message_broker_connection_listener_cb handler, void* client_data);
        
/**
 * @brief Removes a client handler previously added with a call to
 * dsl_message_broker_connection_listener_add.
 * @param[in] name unique name of the Message Broker to update.
 * @param[in] handler handler function to remove from the named Message Broker.
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_BROKER_RESULT otherwise.
 */
DslReturnType dsl_message_broker_connection_listener_remove(const wchar_t* name,
    dsl_message_broker_connection_listener_cb handler);

/**
 * @brief deletes a uniquely named Message Broker.
 * @param[in] name unique name of the Message Broker to delete.
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_BROKER_RESULT otherwise.
 */
DslReturnType dsl_message_broker_delete(const wchar_t* name);

/**
 * @brief Deletes all Message Brokers.
 * @return DSL_RESULT_SUCCESS on success, one of DSL_RESULT_BROKER_RESULT otherwise.
 */
DslReturnType dsl_message_broker_delete_all();

/**
 * @brief Returns the size of the list of Message Brokers
 * @return the number of Display Types in the list
 */
uint dsl_message_broker_list_size();

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
 * @brief Releases/deletes all DSL/GST resources
 */
void dsl_delete_all();

/**
 * @brief Returns the current version of DSL
 * @return string representation of the current release
 */
const wchar_t* dsl_info_version_get();

/**
 * @brief Gets the GPU type for a specified GPU Id.
 * @param[in] gpu_id id of the GPU to query.
 * @return one of the DSL_GPU_TYPE constant values.
 */ 
uint dsl_info_gpu_type_get(uint gpu_id);

/**
 * @brief Gets the current setting for where stdout is directed.
 * @param[out] file_path absolute file path specification or "console".
 * @return true on successful query, one DSL_RESULT otherwise.
 */
DslReturnType dsl_info_stdout_get(const wchar_t** file_path);

/**
 * @brief Redirects all data streamed to stdout including debug logs by default. 
 * The current log file, if one is active, will be saved. The file can be opened for 
 * append or truncated if found. 
 * @param[in] file_path absolute or relative file path specification
 * @param[in] mode either DSL_WRITE_MODE_APPEND or DSL_WRITE_MODE_TRUNCATE
 * @return true on successful query, one DSL_RESULT otherwise
 * @note this service appends a ".log" extension to file_path. 
 */
DslReturnType dsl_info_stdout_redirect(const wchar_t* file_path, uint mode);

/**
 * @brief Redirects all data streamed to stdout including debug logs by default.
 * The current log file, if one is active, will be saved. 
 * @param[in] file_path absolute or relative file path specification
 * @param[in] mode either DSL_WRITE_MODE_APPEND or DSL_WRITE_MODE_TRUNCATE
 * @return true on successful update, one DSL_RESULT otherwise.
 * @note this service appends a timestamp with the format %Y%m%d-%H%M%S.
 * and ".log" extension to file_path. If redirecting to a new file from an existing file.
 */
DslReturnType dsl_info_stdout_redirect_with_ts(const wchar_t* file_path);

/**
 * @brief Restores the std::cout rdbuf from redirection
 * @return true on successful update, one DSL_RESULT otherwise.
 */
DslReturnType dsl_info_stdout_restore();

/**
 * @brief Gets the current GST debug log level if set with the environment
 * variable GST_DEBUG or with a call to dsl_info_log_level_set
 * @parma[out] level current level string defining one or more debug group/level pairs.
 * prefixed with optional global default. Empty string if undefined.
 * @return true on successful query, one of DSL_RESULT otherwise
*/
DslReturnType dsl_info_log_level_get(const wchar_t** level);

/**
 * @brief Sets the GST debug log level. The call will override the currnet 
 * value of the GST_DEBUG environment variable.
 * @param[in] level new level (string) defining one or more debug group/level pairs
 * prefixed with optional global default. eg. export GST_DEBUG=1,DSL:4
 * @return true on successful update, one of DSL_RESULT otherwise
 */
DslReturnType dsl_info_log_level_set(const wchar_t* level);

/**
 * @brief Gets the current GST debug log file if set with the environment
 * variable GST_DEBUG_FILE or with a call to dsl_info_log_file_set or 
 * dsl_info_log_file_set_with_ts. 
 * @param[out] file_path the complete file path/name specification with 
 * (optional) timestamp and file extension. Empty string if undefined.
 * @return true on successful query, one of DSL_RESULT otherwise.
 */
DslReturnType dsl_info_log_file_get(const wchar_t** file_path);

/**
 * @brief Sets the GST debug log file. The call will override the current value of 
 * the DSL_DEBUG_FILE environment variable.  The current  running log file, if one 
 * will be saved. The file can be opened for append or truncated if found. 
 * @param[in] file_path relative or absolute file path without an extension.
 * @param[in] mode either DSL_WRITE_MODE_APPEND or DSL_WRITE_MODE_TRUNCATE.
 * @return true on successful update, one of DSL_RESULT otherwise.
 * @note this service appends a ".log" extension to file_path. The current
 * log file in use, if one exists, will be closed.
*/ 
DslReturnType dsl_info_log_file_set(const wchar_t* file_path, uint mode);

/**
 * @brief Sets the GST debug log file. The call will override the current
 * value of the DSL_DEBUG_FILE environment variable. The current  running
 * log file, if one is active, will be saved.
 * @param[in] file_path relative or absolute file path to persist GST Logs.
 * @return true on successful update, one of DSL_RESULT otherwise
 * @note this service appends a timestamp with the format %Y%m%d-%H%M%S 
 * and ".log" extension to file_path.
 */
DslReturnType dsl_info_log_file_set_with_ts(const wchar_t* file_path);

/**
 * @brief Restores the original default log function which will write
 * logs to GST_DEBUG_FILE if set, or stdio otherwise.
 * @return true on successful update, one of DSL_RESULT otherwise
 */
DslReturnType dsl_info_log_function_restore();


EXTERN_C_END

#endif /* _DSL_API_H */
