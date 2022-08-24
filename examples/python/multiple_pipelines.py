################################################################################
# The MIT License
#
# Copyright (c) 2021, Prominence AI, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

#!/usr/bin/env python

import sys
sys.path.insert(0, "../../")
import time

from dsl import *
from time import sleep
import threading


#-------------------------------------------------------------------------------------------
#
# This script demonstrates the running multple Pipelines, each in their own thread, 
# and each with their own main-context and main-loop.
#
# After creating and starting each Pipelines, the script joins each of the threads
# waiting for them to complete - either by EOS message, 'Q' key, or Delete Window

# File path used for all File Sources
file_path = '/opt/nvidia/deepstream/deepstream/samples/streams/sample_qHD.mp4'

# Filespecs for the Primary Triton Inference Server (PTIS)
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-trtis/config_infer_plan_engine_primary.txt'

# Window Sink Dimensions
sink_width = 1280
sink_height = 720

# global used to keep track of how many Pipelines are currently running.

g_num_active_pipelines = 0

##     
# Objects of this class will be used as "client_data" for all callback notifications.    
# defines a class of all component names associated with a single Pipeline.     
# The names are derived from the unique Pipeline Id
##    
class ComponentNames:    
    def __init__(self, id):    
        self.pipeline = 'pipeline-' + str(id)
        self.source = 'source-' + str(id)
        self.sink = 'window-sink-' + str(id)

## 
# Function to be called on XWindow KeyRelease event
## 
def xwindow_key_event_handler(key_string, client_data):

    # cast the C void* client_data back to a py_object pointer and deref
    components = cast(client_data, POINTER(py_object)).contents.value

    print('key released = ', key_string, 'for Pipeline', components.pipeline)
    if key_string.upper() == 'P':
        dsl_pipeline_pause(components.pipeline)
    elif key_string.upper() == 'S':
        dsl_pipeline_stop(components.pipeline)
    elif key_string.upper() == 'R':
        dsl_pipeline_play(components.pipeline)
    elif key_string.upper() == 'Q' or key_string == '' or key_string == '':
        dsl_pipeline_main_loop_quit(components.pipeline)
 
## 
# Function to be called on XWindow Delete event
## 
def xwindow_delete_event_handler(client_data):

    # cast the C void* client_data back to a py_object pointer and deref
    components = cast(client_data, POINTER(py_object)).contents.value

    print('Pipeline EOS event received for', components.pipeline)
    dsl_pipeline_main_loop_quit(components.pipeline)
    

# Function to be called on End-of-Stream (EOS) event
def eos_event_listener(client_data):

    # cast the C void* client_data back to a py_object pointer and deref
    components = cast(client_data, POINTER(py_object)).contents.value

    print('Pipeline EOS event received for', components.pipeline)
    dsl_pipeline_main_loop_quit(components.pipeline)

## 
# Function to be called on every change of Pipeline state
## 
def state_change_listener(old_state, new_state, client_data):

    # cast the C void* client_data back to a py_object pointer and deref
    components = cast(client_data, POINTER(py_object)).contents.value

    print('previous state = ', old_state, ', new state = ', new_state)
    if new_state == DSL_STATE_PLAYING:
        dsl_pipeline_dump_to_dot(components.pipeline, "state-playing")

def create_pipeline(client_data):

    # New File Source using the same URI for all Piplines
    retval = dsl_source_file_new(client_data.source,
        file_path, False);
    if (retval != DSL_RETURN_SUCCESS):    
        return retval    

    # New Window Sink using the global dimensions
    retval = dsl_sink_window_new(client_data.sink,
        0, 0, sink_width, sink_height)
    if (retval != DSL_RETURN_SUCCESS):    
        return retval    

    retval = dsl_pipeline_new_component_add_many(client_data.pipeline,
        components=[client_data.source, client_data.sink, None]);
    if (retval != DSL_RETURN_SUCCESS):    
        return retval    

    # Add the XWindow event handler functions defined above
    retval = dsl_pipeline_xwindow_key_event_handler_add(client_data.pipeline, 
        xwindow_key_event_handler, client_data)
    if (retval != DSL_RETURN_SUCCESS):    
        return retval    
    retval = dsl_pipeline_xwindow_delete_event_handler_add(client_data.pipeline, 
        xwindow_delete_event_handler, client_data);
    if (retval != DSL_RETURN_SUCCESS):    
        return retval    

    # Add the listener callback functions defined above
    retval = dsl_pipeline_state_change_listener_add(client_data.pipeline, 
        state_change_listener, client_data)
    if (retval != DSL_RETURN_SUCCESS):    
        return retval    
    retval = dsl_pipeline_eos_listener_add(client_data.pipeline, 
        eos_event_listener, client_data)
    if (retval != DSL_RETURN_SUCCESS):    
        return retval    
    
    # Call on the Pipeline to create its own main-context and main-loop that
    # will be set as the default main-context for the main_loop_thread_func
    # defined below once it is run. 
    retval = dsl_pipeline_main_loop_new(client_data.pipeline)

    return retval

def delete_pipeline(client_data):
    global g_num_active_pipelines

    print('stoping and deleting Pipeline', client_data.pipeline)
        
    # Stop the pipeline
    retval = dsl_pipeline_stop(client_data.pipeline)
    if (retval != DSL_RETURN_SUCCESS):    
        return retval    

    print('deleting Pipeline', client_data.pipeline)

    # Delete the Pipeline first, then the components. 
    retval = dsl_pipeline_delete(client_data.pipeline);
    if (retval != DSL_RETURN_SUCCESS):    
        return retval    

    # Now safe to delete all components for this Pipeline
    retval = dsl_component_delete_many(
        components=[client_data.source, client_data.sink, None])
    if (retval != DSL_RETURN_SUCCESS):    
        return retval    
        
    g_num_active_pipelines -= 1;
    
    if not g_num_active_pipelines:
        dsl_main_loop_quit()
        
    return retval

#
# Thread function to start and wait on the main-loop
#
def main_loop_thread_func(client_data):
    global g_num_active_pipelines

    # Play the pipeline
    retval = dsl_pipeline_play(client_data.pipeline);
    if (retval != DSL_RETURN_SUCCESS):
        return

    g_num_active_pipelines += 1

    # blocking call
    dsl_pipeline_main_loop_run(client_data.pipeline)
    
    delete_pipeline(client_data)


def main(args):    

    # Since we're not using args, we can Let DSL initialize GST on first call    
    while True:    
    
        components_1 = ComponentNames(1)
        components_2 = ComponentNames(2)
        components_3 = ComponentNames(3)
        
        # Create the first Pipeline and sleep for a second to seperate 
        # the start time with the next Pipeline.
        retval = create_pipeline(components_1)
        if (retval != DSL_RETURN_SUCCESS):    
            break    

        # Start the Pipeline with its own main-context and main-loop in a 
        # seperate thread. 
        main_loop_thread_1 = threading.Thread(
            target=main_loop_thread_func, args=(components_1,))
        main_loop_thread_1.start()
        
        sleep(1)
        
        # Create the second Pipeline and sleep for a second to seperate 
        # the start time with the next Pipeline.
        retval = create_pipeline(components_2)
        if (retval != DSL_RETURN_SUCCESS):    
            break    

        # Start the second Pipeline with its own main-context and main-loop in a 
        # seperate thread. 
        main_loop_thread_2 = threading.Thread(
            target=main_loop_thread_func, args=(components_2,))
        main_loop_thread_2.start()
        
        sleep(1)
        
        # Create the third Pipeline and sleep for a second to seperate 
        # the start time with the next Pipeline.
        retval = create_pipeline(components_3)
        if (retval != DSL_RETURN_SUCCESS):
            break    

        # Start the third Pipeline with its own main-context and main-loop in a 
        # seperate thread. 
        main_loop_thread_3 = threading.Thread(
            target=main_loop_thread_func, args=(components_3,))
        main_loop_thread_3.start()
        
        # join each of the three threads, order does not matter.
        main_loop_thread_1.join()
        main_loop_thread_2.join()
        main_loop_thread_3.join()
        
        break;
        
    # Print out the final result
    print(dsl_return_value_to_string(retval))

    # Cleanup all DSL/GST resources
    dsl_delete_all()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
