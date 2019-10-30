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
import ctypes
import ds_consts as k

dsl = ctypes.CDLL("dsl-lib.so")

uri = "/home/sscsmatrix/Projects/dsl-python/sample_720p.h264"

#####################################################
#
#   Function: createPipe()
#
#   Description:  This trys to create a pipeline.  It
#   takes as input the name to use for the pipeline.
#
#   Input: pipeName
#
#####################################################
def createPipe(pipeName):
    if (dsl.dsl_pipeline_new(pipeName)) == 0:
        print("Pipeline created successfully")
    else:
        print("Pipeline creation error")


#####################################################
#
#   Function: createSource()
#
#   Description:    This creates a source to use as
#   an input to the pipeline.  The function has two
#   inputs, a source ID and a URI
#
#   Input:  srcName
#   Input:  uri
#
#####################################################
def createSource(srcName, uri):

    src = dsl.dsl_source_uri_new(srcName,uri,0,0)

    if src == 0:
        print("Source created")
   
    elif src == k.DSL_RESULT_SOURCE_NAME_NOT_UNIQUE:
        print("Source Creation error: Name not unique")

    elif src == k.DSL_RESULT_SOURCE_NAME_NOT_FOUND:
        print("Source Creation error: Name not found")

    elif src == k.DSL_RESULT_SOURCE_NAME_BAD_FORMAT:
        print("Source Creation error: Name nbad format")

    elif src == k.DSL_RESULT_SOURCE_NEW_EXCEPTION:
        print("Source Creation error: New exception")

    elif src == k.DSL_RESULT_SOURCE_STREAM_FILE_NOT_FOUND:
        print("Source Creation error: stream file not found")
        
    else:
        print("Source creation error: Unknown error: ",src)

#####################################################
#
#   Function: addComponent()
#
#   Description:  This function adds a component to
#   the pipeline.  It takes two inputs a pipeName and
#   the component source id
#
#   Input:  pipeName
#   Input:  source
#
#####################################################
def addComponent(pipeName,source):
    
    com1 = dsl.dsl_pipeline_component_add(pipeName,source)
    if com1 == 0:
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
        print("Component creation error: Unknown error: ",src)

#####################################################
#
#   Function: deletePipe()
#
#   Description:  This function deletes a pipe.  It
#   takes as input the pipe name to delete.
#
#   Input:  pipeName
#
#####################################################
def deletePipe(pipeName):

    if (dsl.dsl_pipeline_delete(pipeName)) == 0: 
        print("Pipeline deleted successfully")
    else:
        print("Pipeline not deleted successfully")


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

    #########################################
    #   Create pipeline1
    #########################################
    
    createPipe("pipeline1")
    
    #########################################
    #   Create source
    #########################################
    
    createSource("video1",uri)
 
    #########################################
    # Add video1 to pipeline
    #########################################
    
    addComponent("pipeline1","video1")
    
    #########################################
    # Delete pipeline1
    #########################################
   
    deletePipe("pipeline1")

def main():
    test1()

if __name__ == '__main__':
	main()


