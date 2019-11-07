"""
The MIT License

Copyright (c) 2019-Present, Michael Patrick

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

#######################################################################################

Name:            ds_test1.py

Description:    This program attempts to duplicate deepstream-test1-app that is
                provided by Nvida

#######################################################################################


"""
from __future__ import print_function
import ctypes
import os
import sys
import ds_consts as k

DSL = ctypes.CDLL("dsl-lib.so")

#####################################################
#
#   Function: create_pipe()
#
#   Description:  This trys to create a pipeline.  It
#   takes as input the name to use for the pipeline.
#
#   Input: pipe_name
#
#####################################################
def create_pipe(pipe_name):
    """ create a pipe with this name """
    if (DSL.dsl_pipeline_new(pipe_name)) == 0:
        print("Pipeline created successfully")
    else:
        print("Pipeline creation error")


#####################################################
#
#   Function: create_source()
#
#   Description:    This creates a source to use as
#   an input to the pipeline.  The function has two
#   inputs, a source ID and a URI
#
#   Input:  src_name
#   Input:  uri
#
#####################################################
def create_source(src_name, uri):
    """ create a source with this name and location """
    src = DSL.dsl_source_uri_new(src_name, uri, 0, 0)

    if src == k.DSL_RESULT_SUCCESS:
        print("Source created")

    elif src == k.DSL_RESULT_SOURCE_NAME_NOT_UNIQUE:
        print("Source Creation error: Name not unique")

    elif src == k.DSL_RESULT_SOURCE_NAME_NOT_FOUND:
        print("Source Creation error: Name not found")
        print("Source Creation error: Name nbad format")

    elif src == k.DSL_RESULT_SOURCE_NEW_EXCEPTION:
        print("Source Creation error: New exception")

    elif src == k.DSL_RESULT_SOURCE_STREAM_FILE_NOT_FOUND:
        print("Source Creation error: stream file not found")

    else:
        print("Source creation error: Unknown error: ", src)

#####################################################
#
#   Function: create_sink()
#
#   Description:    This creates a sink to use as
#   an output from the pipeline.  The function has
#   three inputs, a source ID, width  and height
#
#   Input:  display_name
#   Input:  width
#   Input:  height
#
#####################################################
def create_sink(display_name, width, height):
    """create a sink with the display name width and height  """
    src = DSL.dsl_display_new(display_name, width, height)

    if src == k.DSL_RESULT_SUCCESS:
        print("Sink created")

    elif src == k.DSL_RESULT_SINK_NAME_NOT_UNIQUE:
        print("Sink Creation error: Name not unique")

    elif src == k.DSL_RESULT_SINK_NAME_NOT_FOUND:
        print("Sink Creation error: Name not found")

    elif src == k.DSL_RESULT_SINK_NAME_BAD_FORMAT:
        print("Sink Creation error: Bad format")

    elif src == k.DSL_RESULT_SINK_NEW_EXCEPTION:
        print("Sink Creation error: New exception")
    else:
        print("Sink creation error: Unknown error: ", src)

#####################################################
#
#   Function: play_pipeline()
#
#   Description:    This plays a pipeline
#
#   Input:  pipeline_name
#
#   Return: return the pipeline play status
#
#####################################################
def play_pipeline(pipeline_name):
    """start the pipeline playing """
    com = DSL.dsl_pipeline_play(pipeline_name)

    if com == k.DSL_RESULT_SUCCESS:
        print("Playing...")

    elif com == k.DSL_RESULT_PIPELINE_NAME_NOT_FOUND:
        print("Pipe error: Name not found")

    elif com == k.DSL_RESULT_PIPELINE_FAILED_TO_PLAY:
        print("Pipe error: Failed to play")
    else:
        print("Pipe error: Unknown error", com)

    return com
#####################################################
#
#   Function: add_component()
#
#   Description:  This function adds a component to
#   the pipeline.  It takes two inputs a pipe_name and
#   the component source id
#
#   Input:  pipe_name
#   Input:  source
#
#####################################################
def add_component(pipe_name, source):
    """add a component to the pipeline """
    com1 = DSL.dsl_pipeline_component_add(pipe_name, source)
    if com1 == k.DSL_RESULT_SUCCESS:
        print("Component added")
    elif com1 == k.DSL_RESULT_COMPONENT_NAME_NOT_UNIQUE:
        print("Component Creation error: Name not unique")

    elif com1 == k.DSL_RESULT_COMPONENT_NAME_NOT_FOUND:
        print("Component Creation error: Name not found")

    elif com1 == k.DSL_RESULT_COMPONENT_NAME_BAD_FORMAT:
        print("Component Creation error: Name bad format")

    elif com1 == k.DSL_RESULT_COMPONENT_IN_USE:
        print("Component Creation error: New exception")

    elif com1 == k.DSL_RESULT_COMPONENT_NOT_USED_BY_PIPELINE:
        print("Component Creation error: stream file not found")
    else:
        print("Component creation error: Unknown error: ", com1)

#####################################################
#
#   Function: remove_component()
#
#   Description:  This function removes a component to
#   the pipeline.  It takes two inputs a pipe_name and
#   the component source id
#
#   Input:  pipe_name
#   Input:  source
#
#####################################################
def remove_component(pipe_name, source):
    """remove a component from the pipeline """
    com1 = DSL.dsl_pipeline_component_remove(pipe_name, source)
    if com1 == k.DSL_RESULT_SUCCESS:
        print("Component added")
    elif com1 == k.DSL_RESULT_COMPONENT_NAME_NOT_UNIQUE:
        print("Component Removal error: Name not unique")

    elif com1 == k.DSL_RESULT_COMPONENT_NAME_NOT_FOUND:
        print("Component Removal error: Name not found")

    elif com1 == k.DSL_RESULT_COMPONENT_NAME_BAD_FORMAT:
        print("Component Removal error: Name bad format")

    elif com1 == k.DSL_RESULT_COMPONENT_IN_USE:
        print("Component Removal error: New exception")

    elif com1 == k.DSL_RESULT_COMPONENT_NOT_USED_BY_PIPELINE:
        print("Component Removal error: stream file not found")
    else:
        print("Component Removal error: Unknown error: ", com1)


#####################################################
#
#   Function: delete_pipe()
#
#   Description:  This function deletes a pipe.  It
#   takes as input the pipe name to delete.
#
#   Input:  pipe_name
#
#####################################################
def delete_pipe(pipe_name):
    """delete the pipeline with the name provided """
    if (DSL.dsl_pipeline_delete(pipe_name)) == k.DSL_RESULT_SUCCESS:
        print("Pipeline deleted successfully")
    else:
        print("Pipeline not deleted successfully")

#####################################################
#
#   Function: run_main_loop()
#
#   Description:  This function executes a main
#                 loop.
#
#
#####################################################
def run_main_loop():
    """run the pipeline until source finished or error """
    print("Running .....")
    DSL.dsl_main_loop_run()

#####################################################
#
#   Function: test1()
#
#   Description:  This test creates a pipeline, adds
#   a video file as a URI and a sink.  It then plays
#   the video and when it ends, it tears down the
#   pipeline
#
#####################################################

def test1():
    """Execute test """
    uri = os.path.abspath(sys.argv[1])

    #########################################
    #   Create source
    #########################################

    create_source("video1", uri)

    #########################################
    #   Create sink
    #########################################

    create_sink("display1", 1024, 768)

    #########################################
    #   Create pipeline1
    #########################################

    create_pipe("pipeline1")

    #########################################
    # Add video1 to pipeline
    #########################################

    add_component("pipeline1", "video1")

    #########################################
    # Add display1 to pipeline
    #########################################

    add_component("pipeline1", "display1")

    #########################################
    # Play video
    #########################################

    result = play_pipeline("pipeline1")

    #########################################
    # Run main loop
    #########################################

    # If pipeline playing then run the main loop

    if result == k.DSL_RESULT_SUCCESS:
        run_main_loop()
    else:
        print("Main loop not running")

    #########################################
    # Remove video1 from pipeline
    #########################################

    remove_component("pipeline1", "video1")

    #########################################
    # Remove display1 from pipeline
    #########################################

    remove_component("pipeline1", "display1")

    #########################################
    # Delete pipeline1
    #########################################

    delete_pipe("pipeline1")

def main():
    """Parse the command line parameters and run test """
    if len(sys.argv) < 2:
        print("")
        print("#################################################################")
        print("#")
        print("#    Error: Missing source file name.")
        print("#    Calling sequence: python3 ds_test1.py <Video source file>")
        print("#")
        print("##################################################################")
    else:
        test1()

if __name__ == '__main__':
    main()
