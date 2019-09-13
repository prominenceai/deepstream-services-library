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

#ifndef _DSD_API_H
#define _DSD_API_H

#define DSD_RESULT_SUCCESS                                          0x00000000
#define DSD_RESULT_API_NOT_IMPLEMENTED                              0x00000001

/**
 * Clock Object Return Values
 */
#define DSD_RESULT_CLOCK_RESULT                                     0x00010000
#define DSD_RESULT_CLOCK_NAME_NOT_UNIQUE                            0x00010001
#define DSD_RESULT_CLOCK_NAME_NOT_FOUND                             0x00010010
#define DSD_RESULT_CLOCK_NAME_BAD_FORMAT                            0x00010011

/**
 * Source Object Return Values
 */
#define DSD_RESULT_SOURCE_RESULT                                    0x00100000
#define DSD_RESULT_SOURCE_NAME_NOT_UNIQUE                           0x00100001
#define DSD_RESULT_SOURCE_NAME_NOT_FOUND                            0x00100010
#define DSD_RESULT_SOURCE_NAME_BAD_FORMAT                           0x00100011
#define DSD_RESULT_SOURCE_NEW_EXCEPTION                             0x00100100

/**
 * StreamMux Object Return Values
 */
#define DSD_RESULT_STREAMMUX_RESULT                                 0x00010000
#define DSD_RESULT_STREAMMUX_NAME_NOT_UNIQUE                        0x00010001
#define DSD_RESULT_STREAMMUX_NAME_NOT_FOUND                         0x00010010
#define DSD_RESULT_STREAMMUX_NAME_BAD_FORMAT                        0x00010011

/**
 * Sink Object Return Values
 */
#define DSD_RESULT_SINK_RESULT                                      0x00100000
#define DSD_RESULT_SINK_NAME_NOT_UNIQUE                             0x00100001
#define DSD_RESULT_SINK_NAME_NOT_FOUND                              0x00100010
#define DSD_RESULT_SINK_NAME_BAD_FORMAT                             0x00100011

/**
 * OSD Object Return Values
 */
#define DSD_RESULT_OSD_RESULT                                       0x00100000
#define DSD_RESULT_OSD_NAME_NOT_UNIQUE                              0x00100001
#define DSD_RESULT_OSD_NAME_NOT_FOUND                               0x00100010
#define DSD_RESULT_OSD_NAME_BAD_FORMAT                              0x00100011

/**
 * GIE Object Return Values
 */
#define DSD_RESULT_GIE_RESULT                                       0x00100000
#define DSD_RESULT_GIE_NAME_NOT_UNIQUE                              0x00100001
#define DSD_RESULT_GIE_NAME_NOT_FOUND                               0x00100010
#define DSD_RESULT_GIE_NAME_BAD_FORMAT                              0x00100011

/**
 * Display Object Return Values
 */
#define DSD_RESULT_DISPLAY_RESULT                                   0x00100000
#define DSD_RESULT_DISPLAY_NAME_NOT_UNIQUE                          0x00100001
#define DSD_RESULT_DISPLAY_NAME_NOT_FOUND                           0x00100010
#define DSD_RESULT_DISPLAY_NAME_BAD_FORMAT                          0x00100011

/**
 * Config Object Return Values
 */
#define DSD_RESULT_CONFIG_RESULT                                    0x00100000
#define DSD_RESULT_CONFIG_NAME_NOT_UNIQUE                           0x00100001
#define DSD_RESULT_CONFIG_NAME_NOT_FOUND                            0x00100010
#define DSD_RESULT_CONFIG_NAME_BAD_FORMAT                           0x00100011
#define DSD_RESULT_CONFIG_FILE_NOT_FOUND                            0x00100100
#define DSD_RESULT_CONFIG_FILE_EXISTS                               0x00100101
#define DSD_RESULT_CONFIG_MAX_SOURCES_REACHED                       0x00100110

/**
 * Pipeline Object Return Values
 */
#define DSD_RESULT_PIPELINE_RESULT                                  0x00110000
#define DSD_RESULT_PIPELINE_NAME_NOT_UNIQUE                         0x00110001
#define DSD_RESULT_PIPELINE_NAME_NOT_FOUND                          0x00110010
#define DSD_RESULT_PIPELINE_NAME_BAD_FORMAT                         0x00110011
#define DSD_RESULT_PIPELINE_STATE_PAUSED                            0x00110100
#define DSD_RESULT_PIPELINE_STATE_RUNNING                           0x00110101

#define DSD_SOURCE_TYPE_CAMERA_V4L2                                 1
#define DSD_SOURCE_TYPE_URI                                         2
#define DSD_SOURCE_TYPE_MULTI_URI                                   3
#define DSD_SOURCE_TYPE_RTSP                                        4
#define DSD_SOURCE_TYPE_CSI                                         5

#define DSD_SINK_TYPE_FAKE                                          1
#define DSD_SINK_TYPE_EGL                                           2
#define DSD_SINK_TYPE_FILE                                          3
#define DSD_SINK_TYPE_RTSP                                          4
#define DSD_SINK_TYPE_CSI                                           5

typedef int DsdReturnType;

/**
 * @brief creates a new, uniquely named Source obj
 * @param source unique name for the new Config
 * @param type value of DSD_SOURCE_TYPE
 * @param width width of the source in pixels
 * @param height height of the source in pixels
 * @param fps-n
 * @param fps-d
 * @return DSD_RESULT_SOURCE_RESULT
 */
#define DSD_SOURCE_NEW(source, type, width, height, fps_n, fps_d)

/**
 * @brief deletes a Source object by name
 * @param source name of the Source object to delete
 * @return DSD_RESULT_SOURCE_RESULT
 */
#define DSD_SOURCE_DELETE(source)

/**
 * @brief creates a new, uniquely named Streammux obj
 * @param streammux unique name for the new Streammux obj
 * @param live DSD_TRUE | DSD_FLASE
 * @param batchSize
 * @param batchTimeout
 * @param width width of the muxer output
 * @param heigth height of the muxer output
 * @return DSD_RESULT_STREAMMUX_RESULT
 */
#define DSD_STREAMMUX_NEW(streammux, live, batchSize, batchTimeout, width, height)

/**
 * @brief deletes a Source object by name
 * @param source name of the Source object to delete
 * @return DSD_RESULT_STREAMMUX_RESULT
 */
#define DSD_STREAMMUX_DELETE(streammux)

/**
 * @brief creates a new, uniquely named Display obj
 * @param display unique name for the new Display
 * @param rows number of horizotal display rows
 * @param columns number of vertical display columns
 * @param width width of each column in pixals
 * @param height height of each row in pix  als
 * @return DSD_RESULT_DISPLAY_RESULT
 */
#define DSD_DISPLAY_NEW(display, rows, columns, width, height)

/**
 * @brief deletes a Display object by name
 * @param display name of the Display object to delete
 * @return DSD_RESULT_DISPLAY_RESULT
 */
#define DSD_DISPLAY_DELETE(display)

/**
 * @brief creates a new, uniquely named GIE object
 * @param gie unique name for the new GIE object
 * @param model full pathspec to the model config file
 * @param infer full pathspec to the inference config file
 * @param batchSize
 * @param boarder-box colors 1..4
 * @param height height of each row in pix  als
 * @return DSD_RESULT_DISPLAY_RESULT
 */
#define DSD_GIE_NEW(gie, model, infer, batchSize, bc1, bc2, bc3, bc4)

/**
 * @brief deletes a GIE object by name
 * @param display name of the Display object to delete
 * @return DSD_RESULT_DISPLAY_RESULT
 */
#define DSD_GIE_DELETE(gie)

/**
 * @brief creates a new, uniquely named Config obj
 * @param config unique name for the new Config
 * @return DSD_RESULT_CONFIG_RESULT
 */
#define DSD_CONFIG_NEW(config)

/**
 * @brief deletes a Config object by name
 * @param config name of the Config to delete
 * @return DSD_RESULT_CONFIG_RESULT
 */
#define DSD_CONFIG_DELETE(config)

/**
 * @brief creates a new, uniquely named Config obj
 * @param config unique name for the new Config
 * @return DSD_RESULT
 */
#define DSD_CONFIG_NEW(config)

/**
 * @brief deletes a Config object by name
 * @param config unique name of the Config to delete
 * @return DSD_RESULT
 */
#define DSD_CONFIG_DELETE(config)

/**
 * @brief loads a deepstream configuration from text file
 * @param config unique name of the Config object
 * @param file full file-path specification of the configuration to load
 * @return DSD_RESULT
 */
#define DSD_CONFIG_FILE_LOAD(config, file)

/**
 * @brief saves a deepstream configuration to text file
 * @param config unique name of the Config object
 * @param file full path-spec of the configuration file to save to
 * @return DSD_RESULT
 */
#define DSD_CONFIG_FILE_SAVE(config, file)

/**
 * @brief saves a deepstream configuration to text file
 * Will overwrite any existing file of the same name
 * @param config unique name of the Config object
 * @param file full path-spec of the configuration file to save to
 * @return DSD_RESULT
 */
#define DSD_CONFIG_FILE_OVERWRITE(config, file)

/**
 * @brief adds a Source object to a Config object
 * @param config name of the Config object to update
 * @param source name of the Source object to add
 * @return DSD_RESULT_CONFIG_RESULT
 */
#define DSD_CONFIG_SOURCE_ADD(config, source)

/**
 * @brief removes a Source object from a Config object
 * @param config name of the Config object to update
 * @param source name of the Source object to remove
 * @return DSD_RESULT_CONFIG_RESULT
 */
#define DSD_CONFIG_SOURCE_REMOVE(config, source)

/**
 * @brief adds an OSD object to a Config object
 * @param config name of the Config object to update
 * @param osd name of the OSD object to add
 * @return DSD_RESULT_CONFIG_RESULT
 */
#define DSD_CONFIG_OSD_ADD(config, osd)

/**
 * @brief removes an OSD object from a Config object
 * @param config name of the Config object to update
 * @param source name of the OSD object to remove
 * @return DSD_RESULT_CONFIG_RESULT
 */
#define DSD_CONFIG_OSD_REMOVE(config, source)

/**
 * @brief adds a GIE object to a Config object
 * @param config name of the Config object to update
 * @param osd name of the OSD object to add
 * @return DSD_RESULT_CONFIG_RESULT
 */
#define DSD_CONFIG_GIE_ADD(config, osd)

/**
 * @brief removes an OSD object from a Config object
 * @param config name of the Config object to unsource
 * @param source name of the OSD object to remove
 * @return DSD_RESULT_CONFIG_RESULT
 */
#define DSD_CONFIG_GIE_REMOVE(config, source)

/**
 * @brief creates a new, uniquely named Pipeline
 * @param pipeline unique name for the new Pipeline
 * @param config unique name of the configuration object to use
 * @return DSD_RESULT_PIPELINE_RESULT
 */
#define DSD_PIPELINE_NEW(pipeline, config)

/**
 * @brief deletes a Pipeline object by name.
 * Does NOT delete the Config object used to create the pipeline
 * @param pipeline unique name of the Pipeline to delete.
 * @return DSD_RESULT_PIPELINE_RESULT.
 */
#define DSD_PIPELINE_DELETE(pipeline)

/**
 * @brief pauses a Pipeline if in a state of playing
 * @param pipeline unique name of the Pipeline to pause.
 * @return DSD_RESULT.
 */
#define DSD_PIPELINE_PAUSE(pipeline)

/**
 * @brief plays a Pipeline if in a state of paused
 * @param pipeline unique name of the Pipeline to play.
 * @return DSD_RESULT_PIPELINE_RESULT.
 */
#define DSD_PIPELINE_PLAY(pipeline)

/**
 * @brief gets the current state of a Pipeline
 * @param pipeline unique name of the Pipeline to query
 * @return DSD_RESULT_PIPELINE_PAUSED | DSD_RESULT_PIPELINE_PLAYING
 */
#define DSD_PIPELINE_GET_STATE(pipeline)

#endif // _DSD_API_H
