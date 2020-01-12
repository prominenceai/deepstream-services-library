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

/**
 * Component API Return Values
 */
#define DSL_RESULT_COMPONENT_RESULT                                 0x00010000
#define DSL_RESULT_COMPONENT_NAME_NOT_UNIQUE                        0x00010001
#define DSL_RESULT_COMPONENT_NAME_NOT_FOUND                         0x00010002
#define DSL_RESULT_COMPONENT_NAME_BAD_FORMAT                        0x00010003
#define DSL_RESULT_COMPONENT_IN_USE                                 0x00010004
#define DSL_RESULT_COMPONENT_NOT_USED_BY_PIPELINE                   0x00010005

/**
 * Source API Return Values
 */
#define DSL_RESULT_SOURCE_RESULT                                    0x00020000
#define DSL_RESULT_SOURCE_NAME_NOT_UNIQUE                           0x00020001
#define DSL_RESULT_SOURCE_NAME_NOT_FOUND                            0x00020002
#define DSL_RESULT_SOURCE_NAME_BAD_FORMAT                           0x00020003
#define DSL_RESULT_SOURCE_THREW_EXCEPTION                           0x00020004
#define DSL_RESULT_SOURCE_FILE_NOT_FOUND                            0x00020005
#define DSL_RESULT_SOURCE_NOT_IN_USE                                0x00020006
#define DSL_RESULT_SOURCE_NOT_IN_PLAY                               0x00020007
#define DSL_RESULT_SOURCE_NOT_IN_PAUSE                              0x00020008
#define DSL_RESULT_SOURCE_FAILED_TO_CHANGE_STATE                    0x00020009
#define DSL_RESULT_SOURCE_CODEC_PARSER_INVALID                      0x0002000A
#define DSL_RESULT_SOURCE_SINK_ADD_FAILED                           0x0002000B
#define DSL_RESULT_SOURCE_SINK_REMOVE_FAILED                        0x0002000C

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

/**
 * Display API Return Values
 */
#define DSL_RESULT_TILER_RESULT                                   0x00070000
#define DSL_RESULT_TILER_NAME_NOT_UNIQUE                          0x00070001
#define DSL_RESULT_TILER_NAME_NOT_FOUND                           0x00070002
#define DSL_RESULT_TILER_NAME_BAD_FORMAT                          0x00070003
#define DSL_RESULT_TILER_THREW_EXCEPTION                          0x00070004
#define DSL_RESULT_TILER_IS_IN_USE                                0x00070005
#define DSL_RESULT_TILER_SET_FAILED                               0x00070006
#define DSL_RESULT_TILER_HANDLER_ADD_FAILED                       0x00070007
#define DSL_RESULT_TILER_HANDLER_REMOVE_FAILED                    0x00070008
#define DSL_RESULT_TILER_PAD_TYPE_INVALID                         0x00070009

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

#define DSL_CUDADEC_MEMTYPE_DEVICE                                  0
#define DSL_CUDADEC_MEMTYPE_PINNED                                  1
#define DSL_CUDADEC_MEMTYPE_UNIFIED                                 2

#define DSL_SOURCE_CODEC_PARSER_H264                                0
#define DSL_SOURCE_CODEC_PARSER_H265                                1

#define DSL_CODEC_H264                                              0
#define DSL_CODEC_H265                                              1
#define DSL_CODEC_MPEG4                                             2

#define DSL_MUXER_MPEG4                                             0
#define DSL_MUXER_MK4                                               1

#define DSL_STATE_NULL                                              1
#define DSL_STATE_READY                                             2
#define DSL_STATE_PAUSED                                            3
#define DSL_STATE_PLAYING                                           4
#define DSL_STATE_IN_TRANSITION                                     5

#define DSL_PAD_SINK                                                0
#define DSL_PAD_SRC                                                 1

#define DSL_RTP_TCP                                                 0x04
#define DSL_RTP_ALL                                                 0x07

/**
 * @brief DSL_DEFAULT values initialized on first call to DSL
 */
//TODO move to new defaults schema
#define DSL_DEFAULT_SOURCE_IN_USE_MAX                               8 
#define DSL_DEFAULT_STREAMMUX_WIDTH                                 1920
#define DSL_DEFAULT_STREAMMUX_HEIGHT                                1080


EXTERN_C_BEGIN

typedef uint DslReturnType;
typedef uint boolean;

/**
 * @brief callback typedef for a client batch meta handler function. Once added to a Component, 
 * the function will be called when the component receives a batch meta buffer.
 * @param[in] batch_meta pointer to a Batch Meta structure to process
 * @param[in] user_data opaque pointer to client's user data
 */
typedef boolean (*dsl_batch_meta_handler_cb)(void* batch_meta, void* user_data);

/**
 * @brief callback typedef for a client listener function. Once added to a Pipeline, 
 * the function will be called when the Pipeline changes state.
 * @param[in] prev_state one of DSL_PIPELINE_STATE constants for the previous pipeline state
 * @param[in] curr_state one of DSL_PIPELINE_STATE constants for the current pipeline state
 * @param[in] user_data opaque pointer to client's data
 */
typedef void (*dsl_state_change_listener_cb)(uint prev_state, uint curr_state, void* user_data);

/**
 * @brief callback typedef for a client listener function. Once added to a Pipeline, 
 * the function will be called on receipt of EOS message from the Pipeline bus.
 * @param[in] user_data opaque pointer to client's data
 */
typedef void (*dsl_eos_listener_cb)(void* user_data);

/**
 * @brief callback typedef for a client XWindow KeyRelease event handler function. Once added to a Pipeline, 
 * the function will be called when the Pipeline receives XWindow KeyRelease events.
 * @param[in] key UNICODE key string for the key pressed
 * @param[in] user_data opaque pointer to client's user data
 */
typedef void (*dsl_xwindow_key_event_handler_cb)(const wchar_t* key, void* user_data);

/**
 * @brief callback typedef for a client XWindow ButtonPress event handler function. Once added to a Pipeline, 
 * the function will be called when the Pipeline receives XWindow ButtonPress events.
 * @param[in] xpos from the top left corner of the window
 * @param[in] ypos from the top left corner of the window
 * @param[in] user_data opaque pointer to client's user data
 */
typedef void (*dsl_xwindow_button_event_handler_cb)(uint xpos, uint ypos, void* user_data);

/**
 * @brief callback typedef for a client XWindow Delete Message event handler function. Once added to a Pipeline, 
 * the function will be called when the Pipeline receives XWindow Delete Message event.
 * @param[in] user_data opaque pointer to client's user data
 */
typedef void (*dsl_xwindow_delete_event_handler_cb)(void* user_data);

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
 * @param[in] 
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_rtsp_new(const wchar_t* name, const wchar_t* uri, uint protocol,
    uint cudadec_mem_type, uint intra_decode, uint drop_frame_interval);

/**
 * @brief returns the frame rate of the name source as a fraction
 * Camera sources will return the value used on source creation
 * URL and RTPS sources will return 0 until prior entering a state of play
 * @param name unique name of the source to query
 * @param width of the source in pixels
 * @param height of the source in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_dimensions_get(const wchar_t* name, uint* width, uint* height);

/**
 * @brief returns the frame rate of the named source as a fraction
 * Camera sources will return the value used on source creation
 * URL and RTPS sources will return 0 until prior entering a state of play
 * @param name unique name of the source to query
 * @param fps_n frames per second numerator
 * @param fps_d frames per second denominator
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_frame_rate_get(const wchar_t* name, uint* fps_n, uint* fps_d);

/**
 * @brief adds a named Sink object to a named Source object
 * @param source name of the Source object update
 * @param sink name of the Sink object to add
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_sink_add(const wchar_t* source, const wchar_t* sink);

/**
 * @brief removes a named Sink object from a named Source object
 * @param source name of the Source object update
 * @param sink name of the Sink object to add
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_sink_remove(const wchar_t* source, const wchar_t* sink);

/**
 * @brief pauses a single Source object if the Source is 
 * currently in a state of in-use and Playing..
 * @param name the name of Source component to pause
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_pause(const wchar_t* name);

/**
 * @brief resumes a single Source object if the Source is 
 * currently in a state of in-use and Paused..
 * @param name the name of Source component to resume
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
 */
DslReturnType dsl_source_resume(const wchar_t* name);

/**
 * @brief returns whether the source stream is live or not
 * @param name the name of Source component to query
 * @return True if the source's stream is live
 */
boolean dsl_source_is_live(const wchar_t* name);

/**
 * @brief returns the number of sources currently in use by 
 * all of the Pipelines in memeroy. 
 * @return number of Sources in use
 */
uint dsl_source_get_num_in_use();  

/**
 * @brief Get the maximum number of in-memory sources 
 * that can be in use at any time. The maximum number is 
 * limited by Hardware, see dsl_source_set_num_in_use_max() 
 * @return the current max sources in use setting.
 */
uint dsl_source_get_num_in_use_max();  

/**
 * @brief Sets the maximum number of in-memory sources 
 * that can be in use at any time. The function overrides 
 * the default value on first call. The maximum number is 
 * limited by Hardware. The caller must ensure to set the 
 * number correctly, based on the TEGRA platform in use.
 */
void dsl_source_set_num_in_use_max(uint max);  

/**
 * @brief creates a new, uniquely named Primary GIE object
 * @param[in] name unique name for the new GIE object
 * @param[in] infer_config_file pathspec of the Infer Config file to use
 * @param[in] model_engine_file pathspec of the Model Engine file to use
 * @param[in] interval
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_GIE_RESULT otherwise.
 */
DslReturnType dsl_gie_primary_new(const wchar_t* name, const wchar_t* infer_config_file,
    const wchar_t* model_engine_file, uint interval);

/**
 * @brief Adds a batch meta handler callback function to be called to process each buffer.
 * A Primary GIE can have at most one Sink and Source batch meta handler each
 * @param name unique name of the Primary GIE to update
 * @param pad pad to add the handler to; DSL_PAD_SINK | DSL_PAD SRC
 * @param handler callback function to process batch meta data
 * @param user_data opaque pointer to clients user data passed in to each callback call.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_GIE_RESULT otherwise
 */
DslReturnType dsl_gie_primary_batch_meta_handler_add(const wchar_t* name, uint type, 
    dsl_batch_meta_handler_cb handler, void* user_data);

/**
 * @brief Removes a batch meta handler callback function from the Primary GIE
 * @param name unique name of the Primary GIE to update
 * @param pad pad to remove the handler from; DSL_PAD_SINK | DSL_PAD SRC
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_GIE_RESULT otherwise
 */
DslReturnType dsl_gie_primary_batch_meta_handler_remove(const wchar_t* name, uint pad);

/**
 * @brief creates a new, uniquely named Secondary GIE object
 * @param[in] name unique name for the new GIE object
 * @param[in] infer_config_file pathspec of the Infer Config file to use
 * @param[in] model_engine_file pathspec of the Model Engine file to use
 * @param[in] infer_on_gie_name name of the Primary or Secondary GIE to infer on
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_GIE_RESULT otherwise.
 */
DslReturnType dsl_gie_secondary_new(const wchar_t* name, const wchar_t* infer_config_file,
    const wchar_t* model_engine_file, const wchar_t* infer_on_gie_name);
//
//DslReturnType dsl_gie_infer_config_file_get(const wchar_t* name, const wchar_t** infer_config_file);
//DslReturnType dsl_gie_infer_config_file_set(const wchar_t* name, const wchar_t* infer_config_file);
//
//DslReturnType dsl_gie_model_engine_file_get(const wchar_t* name, const wchar_t** model_engine_file);
//DslReturnType dsl_gie_model_engine_file_set(const wchar_t* name, const wchar_t* model_engine_file);
//
//DslReturnType dsl_gie_interval_get(const wchar_t* name, uint* interval);
//DslReturnType dsl_gie_interval_set(const wchar_t* name, uint interval);

/**
 * @brief creates a new, uniquely named KTL Tracker object
 * @param name unique name for the new Tracker
 * @param max_width maximum frame width of the input transform buffer
 * @param max_height maximum_frame height of the input tranform buffer
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_ktl_new(const wchar_t* name, uint max_width, uint max_height);

/**
 * @brief creates a new, uniquely named IOU Tracker object
 * @param name unique name for the new Tracker
 * @param config_file fully qualified pathspec to the IOU Lib config text file
 * @param max_width maximum frame width of the input transform buffer
 * @param max_height maximum_frame height of the input tranform buffer
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_iou_new(const wchar_t* name, const wchar_t* config_file, uint max_width, uint max_height);

/**
 * @brief returns the current maximum frame width and height settings for the named IOU Tracker object
 * @param name unique name of the Tracker to query
 * @param max_width maximum frame width of the input transform buffer
 * @param max_height maximum_frame height of the input tranform buffer
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_max_dimensions_get(const wchar_t* name, uint* max_width, uint* max_height);

/**
 * @brief sets the maximum frame width and height settings for the named IOU Tracker object
 * @param name unique name of the Tracker to update
 * @param max_width new maximum frame width of the input transform buffer
 * @param max_height new maximum_frame height of the input tranform buffer
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_max_dimensions_set(const wchar_t* name, uint max_width, uint max_height);

/**
 * @brief returns the current config file in use by the named IOU Tracker object
 * @param name unique name of the Tracker to query
 * @param config_file absolute or relative pathspec to the new config file to use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_iou_config_file_get(const wchar_t* name, const wchar_t** config_file);

/**
 * @brief Add a batch meta handler callback function to be called to process each frame buffer.
 * A Tracker can have at most one Sink and Source batch meta handler each
 * @param name unique name of the Tracker to update
 * @param pad pad to add the handler to; DSL_PAD_SINK | DSL_PAD SRC
 * @param handler callback function to process batch meta data
 * @param user_data opaque pointer to clients user data passed in to each callback call.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_batch_meta_handler_add(const wchar_t* name, uint pad, 
    dsl_batch_meta_handler_cb handler, void* user_data);

/**
 * @brief Removes a batch meta handler callback function from the Tracker
 * @param name unique name of the Tracker to update
 * @param pad pad to remove the handler from; DSL_PAD_SINK | DSL_PAD SRC
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_batch_meta_handler_remove(const wchar_t* name, uint pad);

/**
 * @brief sets the config file to use by named IOU Tracker object
 * @param name unique name of the Tracker to Update
 * @param config_file absolute or relative pathspec to the new config file to use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TRACKER_RESULT otherwise
 */
DslReturnType dsl_tracker_iou_config_file_set(const wchar_t* name, const wchar_t* config_file);

/**
 * @brief creates a new, uniquely named OSD obj
 * @param[in] name unique name for the new Sink
 * @param[in] is_clock_enabled true if clock is visible
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_new(const wchar_t* name, boolean is_clock_enabled);

/**
 * @brief Adds a batch meta handler callback function to be called to process each frame buffer.
 * An On-Screen-Display can have at most one Sink and Source batch meta handler each
 * @param name unique name of the OSD to update
 * @param pad pad to add the handler to; DSL_PAD_SINK | DSL_PAD SRC
 * @param handler callback function to process batch meta data
 * @param user_data opaque pointer to clients user data passed in to each callback call.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_batch_meta_handler_add(const wchar_t* name, uint type, 
    dsl_batch_meta_handler_cb handler, void* user_data);

/**
 * @brief Removes a batch meta handler callback function from the OSD
 * @param name unique name of the OSD to update
 * @param pad pad to remove the handler from; DSL_PAD_SINK | DSL_PAD SRC
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_OSD_RESULT otherwise
 */
DslReturnType dsl_osd_batch_meta_handler_remove(const wchar_t* name, uint pad);

/**
 * @brief creates a new, uniquely named Display obj
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
 * @brief Adds a batch meta handler callback function to be called to process each frame buffer.
 * A Tiled Display can have at most one Sink and Source batch meta handler each
 * @param name unique name of the Tiled Display to update
 * @param pad pad to add the handler to; DSL_PAD_SINK | DSL_PAD SRC
 * @param handler callback function to process batch meta data
 * @param user_data opaque pointer to clients user data passed in to each callback call.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TILER_RESULT otherwise
 */
DslReturnType dsl_tiler_batch_meta_handler_add(const wchar_t* name, uint type, 
    dsl_batch_meta_handler_cb handler, void* user_data);

/**
 * @brief Removes a batch meta handler callback function from the Tiled Display
 * @param name unique name of the Tiled Dislplay to update
 * @param pad pad to remove the handler from; DSL_PAD_SINK | DSL_PAD SRC
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_TILER_RESULT otherwise
 */
DslReturnType dsl_tiler_batch_meta_handler_remove(const wchar_t* name, uint pad);

/**
 * @brief creates a new, uniquely named Ovelay Sink component
 * @param[in] name unique component name for the new Overlay Sink
 * @param[in] offsetX upper left corner offset in the X direction in pixels
 * @param[in] offsetY upper left corner offset in the Y direction in pixels
 * @param[in] width width of the Ovelay Sink in pixels
 * @param[in] heigth height of the Overlay Sink in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT
 */
DslReturnType dsl_sink_overlay_new(const wchar_t* name, 
    uint offsetX, uint offsetY, uint width, uint height);

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
 * @param name unique component name for the new File Sink
 * @param filepath absolute or relative file path including extension
 * @param codec one of DSL_CODEC_H264, DSL_CODEC_H265, DSL_CODEC_MPEG4
 * @param muxer one of DSL_MUXER_MPEG4 or DSL_MUXER_MK4
 * @param bit_rate in bits per second - H264 and H265 only
 * @param interval iframe interval to wite out at
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_file_new(const wchar_t* name, const wchar_t* filepath, 
     uint codec, uint muxer, uint bit_rate, uint interval);

/**
 * @brief gets the current bit_rate and interval settings for the named File Sink
 * @param name unique name of the File Sink to query
 * @param bit_rate current Encoder bit-rate in bits/sec for the named File Sink
 * @param interval current Encoder iframe interval value
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_file_encoder_settings_get(const wchar_t* name,
    uint* bit_rate, uint* interval);

/**
 * @brief sets new bit_rate and interval settings for the named File Sink
 * @param name unique name of the File Sink to update
 * @param bit_rate new Encoder bit-rate in bits/sec for the named File Sink
 * @param interval new Encoder iframe interval value to use
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SINK_RESULT on failure
 */
DslReturnType dsl_sink_file_encoder_settings_set(const wchar_t* name,
    uint bit_rate, uint interval);

/**
 * @brief deletes a Component object by name
 * @param[in] component name of the Component object to delete
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_COMPONENT_RESULT
 * @info the function checks that the component is not 
 * owned by a pipeline before deleting, and returns
 * DSL_RESULT_COMPONENT_IN_USE as failure
 */
DslReturnType dsl_component_delete(const wchar_t* component);

/**
 * @brief deletes a NULL terminated list of components
 * @param components NULL terminated list of names to delete
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_COMPONENT_RESULT
 */
DslReturnType dsl_component_delete_many(const wchar_t** components);

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
 * @brief creates a new, uniquely named Pipeline
 * @param[in] pipeline unique name for the new Pipeline
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT
 */
DslReturnType dsl_pipeline_new(const wchar_t* pipeline);

/**
 * @brief creates a new Pipeline for each name pipelines array
 * @param pipelines a NULL terminated array of unique Pipeline names
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT
 */
DslReturnType dsl_pipeline_new_many(const wchar_t** pipelines);

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
 * @param pipelines NULL terminated list of names to delete
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
 * @param[in] components NULL terminated array of component names to add
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
 * @return DSL_RESULT_SUCCESS on success, 
 */
DslReturnType dsl_pipeline_streammux_dimensions_get(const wchar_t* pipeline, 
    uint* width, uint* height);

/**
 * @brief 
 * @param[in] pipeline name of the pipeline to update
 * @return DSL_RESULT_SUCCESS on success, 
 */
DslReturnType dsl_pipeline_streammux_dimensions_set(const wchar_t* pipeline, 
    uint width, uint height);

/**
 * @brief gets the current Pipeline XWindow dimensions
 * @param[in] pipeline name of the pipeline to query
 * @param[in] width width of the XWindow in pixels
 * @param[in] heigth height of the Window in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT otherwise.
 */
DslReturnType dsl_pipeline_xwindow_dimensions_get(const wchar_t* pipeline, 
    uint* width, uint* height);

/**
 * @brief Sets the Pipeline XWindow dimensions to used on XWindow creation
 * If not explicitely set, the Pipeline will use the Tiled Display's dimensions if one exists.
 * @param[in] pipeline name of the pipeline to update
 * @param[in] width width of the XWindow in pixels
 * @param[in] heigth height of the Window in pixels
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT otherwise.
 */
DslReturnType dsl_pipeline_xwindow_dimensions_set(const wchar_t* pipeline, 
    uint width, uint height);

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
 * @return DSL_RESULT_PIPELINE_PAUSED | DSL_RESULT_PIPELINE_PLAYING
 */
DslReturnType dsl_pipeline_get_state(const wchar_t* pipeline);

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
 * @param[in] userdata opaque pointer to client data passed into the listner function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_eos_listener_add(const wchar_t* pipeline, 
    dsl_eos_listener_cb listener, void* userdata);

/**
 * @brief removes a callback previously added with dsl_pipeline_eos_listener_add
 * @param[in] pipeline name of the pipeline to update
 * @param[in] listener pointer to the client's function to remove
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_eos_listener_remove(const wchar_t* pipeline, 
    dsl_eos_listener_cb listener);

/**
 * @brief adds a callback to be notified on change of Pipeline state
 * @param[in] pipeline name of the pipeline to update
 * @param[in] listener pointer to the client's function to call on state change
 * @param[in] userdata opaque pointer to client data passed into the listner function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_state_change_listener_add(const wchar_t* pipeline, 
    dsl_state_change_listener_cb listener, void* userdata);

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
 * @param[in] user_data opaque pointer to client data passed into the handler function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_xwindow_key_event_handler_add(const wchar_t* pipeline, 
    dsl_xwindow_key_event_handler_cb handler, void* user_data);

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
 * @param[in] user_data opaque pointer to client data passed into the handler function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_xwindow_button_event_handler_add(const wchar_t* pipeline, 
    dsl_xwindow_button_event_handler_cb handler, void* user_data);

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
 * @param[in] user_data opaque pointer to client data passed into the handler function.
 * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_PIPELINE_RESULT on failure.
 */
DslReturnType dsl_pipeline_xwindow_delete_event_handler_add(const wchar_t* pipeline, 
    dsl_xwindow_delete_event_handler_cb handler, void* user_data);

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

EXTERN_C_END

#endif /* _DSL_API_H */
