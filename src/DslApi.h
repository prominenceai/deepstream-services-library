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

#include "DslServices.h"

#define DSL_FALSE                                                   0
#define DSL_TRUE                                                    1

#define DSL_RESULT_SUCCESS                                          0x00000000
#define DSL_RESULT_API_NOT_IMPLEMENTED                              0x00000001

/**
 * Clock Object Return Values
 */
#define DSL_RESULT_COMPONENT_RESULT                                 0x00010000
#define DSL_RESULT_COMPONENT_NAME_NOT_UNIQUE                        0x00010001
#define DSL_RESULT_COMPONENT_NAME_NOT_FOUND                         0x00010010
#define DSL_RESULT_COMPONENT_NAME_BAD_FORMAT                        0x00010011

/**
 * Source Object Return Values
 */
#define DSL_RESULT_SOURCE_RESULT                                    0x00100000
#define DSL_RESULT_SOURCE_NAME_NOT_UNIQUE                           0x00100001
#define DSL_RESULT_SOURCE_NAME_NOT_FOUND                            0x00100010
#define DSL_RESULT_SOURCE_NAME_BAD_FORMAT                           0x00100011
#define DSL_RESULT_SOURCE_NEW_EXCEPTION                             0x00100100
#define DSL_RESULT_SOURCE_STREAM_FILE_NOT_FOUND                     0x00100101

/**
 * StreamMux Object Return Values
 */
#define DSL_RESULT_STREAMMUX_RESULT                                 0x00110000
#define DSL_RESULT_STREAMMUX_NAME_NOT_UNIQUE                        0x00110001
#define DSL_RESULT_STREAMMUX_NAME_NOT_FOUND                         0x00110010
#define DSL_RESULT_STREAMMUX_NAME_BAD_FORMAT                        0x00110011
#define DSL_RESULT_STREAMMUX_NEW_EXCEPTION                          0x00110100

/**
 * Sink Object Return Values
 */
#define DSL_RESULT_SINK_RESULT                                      0x01000000
#define DSL_RESULT_SINK_NAME_NOT_UNIQUE                             0x01000001
#define DSL_RESULT_SINK_NAME_NOT_FOUND                              0x01000010
#define DSL_RESULT_SINK_NAME_BAD_FORMAT                             0x01000011
#define DSL_RESULT_SINK_NEW_EXCEPTION                               0x01000100

/**
 * OSD Object Return Values
 */
#define DSL_RESULT_OSD_RESULT                                       0x01010000
#define DSL_RESULT_OSD_NAME_NOT_UNIQUE                              0x01010001
#define DSL_RESULT_OSD_NAME_NOT_FOUND                               0x01010010
#define DSL_RESULT_OSD_NAME_BAD_FORMAT                              0x01010011
#define DSL_RESULT_OSD_NEW_EXCEPTION                                0x01010100

/**
 * GIE Object Return Values
 */
#define DSL_RESULT_GIE_RESULT                                       0x01100000
#define DSL_RESULT_GIE_NAME_NOT_UNIQUE                              0x01100001
#define DSL_RESULT_GIE_NAME_NOT_FOUND                               0x01100010
#define DSL_RESULT_GIE_NAME_BAD_FORMAT                              0x01100011
#define DSL_RESULT_GIE_CONFIG_FILE_NOT_FOUND                        0x01100100
#define DSL_RESULT_GIE_MODEL_FILE_NOT_FOUND                         0x01100100
#define DSL_RESULT_GIE_NEW_EXCEPTION                                0x01100100

/**
 * Display Object Return Values
 */
#define DSL_RESULT_DISPLAY_RESULT                                   0x10000000
#define DSL_RESULT_DISPLAY_NAME_NOT_UNIQUE                          0x10000001
#define DSL_RESULT_DISPLAY_NAME_NOT_FOUND                           0x10000010
#define DSL_RESULT_DISPLAY_NAME_BAD_FORMAT                          0x10000011
#define DSL_RESULT_DISPLAY_NEW_EXCEPTION                            0x10000100

/**
 * Pipeline Object Return Values
 */
#define DSL_RESULT_PIPELINE_RESULT                                  0x11000000
#define DSL_RESULT_PIPELINE_NAME_NOT_UNIQUE                         0x11000001
#define DSL_RESULT_PIPELINE_NAME_NOT_FOUND                          0x11000010
#define DSL_RESULT_PIPELINE_NAME_BAD_FORMAT                         0x11000011
#define DSL_RESULT_PIPELINE_STATE_PAUSED                            0x11000100
#define DSL_RESULT_PIPELINE_STATE_RUNNING                           0x11000101
#define DSL_RESULT_PIPELINE_NEW_EXCEPTION                           0x11000110
#define DSL_RESULT_PIPELINE_COMPONENT_ADD_FAILED                    0x11000111
#define DSL_RESULT_PIPELINE_STREAMMUX_SETUP_FAILED                  0x11001000
#define DSL_RESULT_PIPELINE_FAILED_TO_PLAY                          0x11001001
#define DSL_RESULT_PIPELINE_FAILED_TO_PAUSE                         0x11001010

#define DSL_CUDADEC_MEMTYPE_DEVICE                                  0
#define DSL_CUDADEC_MEMTYPE_PINNED                                  1
#define DSL_CUDADEC_MEMTYPE_UNIFIED                                 2

/**
 * @brief creates a new, uniquely named CSI Camera Source obj
 * @param[in] source unique name for the new Source
 * @param[in] width width of the source in pixels
 * @param[in] height height of the source in pixels
 * @param[in] fps-n frames/second fraction numerator
 * @param[in] fps-d frames/second fraction denominator
 * @return DSL_RESULT_SOURCE_RESULT
 */
DslReturnType dsl_source_csi_new(const char* source,
    guint width, guint height, guint fps_n, guint fps_d);

/**
 * @brief creates a new, uniquely named URI Source obj
 * @param[in] uri Unique Resource Identifier (file or live)
 * @param[in] cudadec_mem_type, use DSL_CUDADEC_MEMORY_TYPE_<type>
 * @return DSL_RESULT_SOURCE_RESULT
 */
DslReturnType dsl_source_uri_new(const char* source, 
    const char* uri, guint cudadec_mem_type, guint intra_decode);

/**
 * @brief creates a new, uniquely named Sink obj
 * @param[in] sink unique name for the new Sink
 * @param[in] displayId
 * @param[in] overlatId
 * @param[in] offsetX
 * @param[in] offsetY
 * @param[in] width width of the Sink
 * @param[in] heigth height of the Sink
 * @return DSL_RESULT_SINK_RESULT
 */
DslReturnType dsl_sink_new(const char* sink, guint displayId, 
    guint overlayId, guint offsetX, guint offsetY, guint width, guint height);

/**
 * @brief creates a new, uniquely named OSD obj
 * @param[in] osd unique name for the new Sink
 * @param[in] isClockEnabled true if clock is visible
 * @return DSL_RESULT_SINK_RESULT
 */
DslReturnType dsl_osd_new(const char* osd, gboolean isClockEnabled);

/**
 * @brief creates a new, uniquely named GIE object
 * @param[in] gie unique name for the new GIE object
 * @param[in] inferConfigFile name of the Infer Config file to use
 * @param[in] batchSize
 * @param[in] interval
 * @param[in] uniqueId
 * @param[in] gpuId
 * @param[in] modelEngineFile name of the Model Engine file to use
 * @param[in] rawOutputDir
 * @return DSL_RESULT_GIE_RESULT
 */
DslReturnType dsl_gie_new(const char* gie, const char* inferConfigFile, 
    guint batchSize, guint interval, guint uniqueId, guint gpuId, 
    const char* modelEngineFile, const char* rawOutputDir);

/**
 * @brief creates a new, uniquely named Display obj
 * @param[in] display unique name for the new Display
 * @param[in] rows number of horizotal display rows
 * @param[in] columns number of vertical display columns
 * @param[in] width width of each column in pixals
 * @param[in] height height of each row in pix  als
 * @return DSL_RESULT_DISPLAY_RESULT
 */
DslReturnType dsl_display_new(const char* display, 
    guint rows, guint columns, guint width, guint height);

/**
 * @brief deletes a Component object by name
 * @param[in] name of the Component object to delete
 * @return DSL_RESULT_COMPONENT_RESULT
 */
DslReturnType dsl_component_delete(const char* component);

/**
 * @brief returns the current number of components
 * @return size of the list of components
 */
uint dsl_component_list_size();

/**
 * @brief returns the list of components
 * @return a NULL terminated array of char* component names
 */
const char** dsl_component_list_all();

/**
 * @brief creates a new, uniquely named Pipeline
 * @param[in] pipeline unique name for the new Pipeline
 * @return DSL_RESULT_PIPELINE_RESULT
 */
DslReturnType dsl_pipeline_new(const char* pipeline);

/**
 * @brief deletes a Pipeline object by name.
 * Does NOT delete the Pipeline object used to create the pipeline
 * @param[in] pipeline unique name of the Pipeline to delete.
 * @return DSL_RESULT_PIPELINE_RESULT.
 */
DslReturnType dsl_pipeline_delete(const char* pipeline);

/**
 * @brief adds a list of components to a Pipeline 
 * @param[in] pipeline name of the pipepline to update
 * @param[in] components NULL terminated array of component names to add
 * @return DSL_RESULT_PIPELINE_RESULT
 */
DslReturnType dsl_pipeline_components_add(const char* pipeline, 
    const char** components);

/**
 * @brief removes a list of names components from a Pipeline
 * @param[in] pipeline name of the pipepline to update
 * @param[in] components NULL terminated array of component names to remove
 * @return DSL_RESULT_PIPELINE_RESULT
 */
DslReturnType dsl_pipeline_components_remove(const char* pipeline, 
    const char** components);

/**
 * @brief 
 * @param[in] pipeline name of the pipepline to update
 * @return 
 */
DslReturnType dsl_pipeline_streammux_properties_set(const char* pipeline,
    gboolean areSourcesLive, guint batchSize, guint batchTimeout, guint width, guint height);

/**
 * @brief pauses a Pipeline if in a state of playing
 * @param[in] pipeline unique name of the Pipeline to pause.
 * @return DSL_RESULT.
 */
DslReturnType dsl_pipeline_pause(const char* pipeline);

/**
 * @brief plays a Pipeline if in a state of paused
 * @param[in] pipeline unique name of the Pipeline to play.
 * @return DSL_RESULT_PIPELINE_RESULT.
 */
DslReturnType dsl_pipeline_play(const char* pipeline);

/**
 * @brief gets the current state of a Pipeline
 * @param[in] pipeline unique name of the Pipeline to query
 * @return DSL_RESULT_PIPELINE_PAUSED | DSL_RESULT_PIPELINE_PLAYING
 */
DslReturnType dsl_pipeline_get_state(const char* pipeline);

/**
 * @brief entry point to the GST Main Loop
 * Note: This is a blocking call - executes an endless loop
 */
void dsl_main_loop_run();

#endif // _DSL_API_H
