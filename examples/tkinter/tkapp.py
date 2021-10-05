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
from dsl import *	

import tkinter as tk
import tkinter.messagebox
import pyds
import os
import threading

from tkapp_window import AppWindow

# RTSP Source URI	
rtsp_uri = 'rtsp://user:pwd@192.168.1.64:554/Streaming/Channels/101'	

TILER_WIDTH = DSL_DEFAULT_STREAMMUX_WIDTH	
TILER_HEIGHT = DSL_DEFAULT_STREAMMUX_HEIGHT	
WINDOW_WIDTH = DSL_DEFAULT_STREAMMUX_WIDTH	
WINDOW_HEIGHT = DSL_DEFAULT_STREAMMUX_HEIGHT	

# Filespecs for the Primary GIE	
primary_infer_config_file = '../../test/configs/config_infer_primary_nano.txt'	
primary_model_engine_file = '../../test/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine'	

##
# MetaData class for creating a display type. A single Object
# that will be accessed by multiple threads. Pipeline and TK APP
##
class MetaData():
    def __init__(self):
        self.mutex = threading.Lock()

##
# Thread to block on the DSL/GStreamer Main Loop
##
def thread_loop():
    try:
        dsl_main_loop_run()
    except:
        pass

##
# tkinter app class
##
class App(tk.Tk):

    ##
    # Ctor for this Tkinter Application
    ##
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        
        # decrease the niceness by 1 to increase the priority
        # this operation requires admin privileges, run as
        # $ sudo python3 dsl_tk_app.py
        os.nice(-1)

        # we need a way to close the application in fullscreen mode
        self.bind('<Escape>', self.quit)
        
        # if either the window width or height match the screen size, go to fullscreen
        if WINDOW_WIDTH == self.winfo_screenwidth() or \
            WINDOW_HEIGHT == self.winfo_screenheight():
            self.attributes('-fullscreen', True)
        
        # Create the App window to be shared with the Pipeline's Window Sink
        self.window_frame = AppWindow(self, bg='black',
            width=WINDOW_WIDTH, height=WINDOW_HEIGHT, bd=0)
        self.window_frame.pack()
        
        # MetaData object for creating display-types.
        self.meta_data = MetaData()
        
        # Build the Pipeline using the path specs and sizes defined above.
        self.retval = self.build_pipeline()
        if self.retval != DSL_RETURN_SUCCESS:
            self.quit(None)
        
        # Once built, the Pipeline can now be played
        self.retval = self.play_pipeline()
        if self.retval != DSL_RETURN_SUCCESS:
            self.quit(None)

        self.dsl_main_loop = threading.Thread(target=thread_loop, daemon=True)
        self.dsl_main_loop.start()

    ##
    # Dtor for this Tkinter Application
    ##
    def __del__(self):
        
        # If the DSL/GST main-loop was started, then stop and join thread shutdown.
        if self.dsl_main_loop:
            dsl_main_loop_quit()
            self.dsl_main_loop.join()
            
        # Cleanup all DSL/GST resources
        dsl_delete_all()
        
    ##
    # override the tk.quit so that all DSL/GST resources can be released
    ##
    def quit(self, event):
    
        if tk.messagebox.askquestion(
            message='Are you sure you want to exit?') == 'no':
            return
    
        # Explicity call the destructor as tk.quit will stop the main-loop and exit.
        self.__del__()
        super().quit()

    def build_pipeline(self):
        # For each camera, create a new RTSP Source for the specific RTSP URI	
        retval = dsl_source_rtsp_new('rtsp-source',
            uri = rtsp_uri, 	
            protocol = DSL_RTP_ALL, 	
            cudadec_mem_type = DSL_CUDADEC_MEMTYPE_DEVICE, 	
            intra_decode = False, 	
            drop_frame_interval = 0, 	
            latency = 100,
            timeout = 2)	
        if (retval != DSL_RETURN_SUCCESS):	
            return retval

        # New Primary GIE using the filespecs above, with interval and Id	
        retval = dsl_infer_gie_primary_new('primary-gie', 
            primary_infer_config_file, primary_model_engine_file, 2)	
        if retval != DSL_RETURN_SUCCESS:	
            return retval

        # New KTL Tracker, setting max width and height of input frame	
        retval = dsl_tracker_ktl_new('ktl-tracker', 480, 272)	
        if retval != DSL_RETURN_SUCCESS:	
            return retval

        # New Tiled Display, setting width and height, use default cols/rows set by source count	
        retval = dsl_tiler_new('tiler', TILER_WIDTH, TILER_HEIGHT)	
        if retval != DSL_RETURN_SUCCESS:	
            return retval

        # New OSD with clock and labels enabled... using default values.
        retval = dsl_osd_new('on-screen-display', True, True)
        if retval != DSL_RETURN_SUCCESS:	
            return retval

        # New Custom Pad Probe Handler to call Nvidia's example callback for handling the Batched Meta Data
        retval = dsl_pph_custom_new('custom-pph', 
            client_handler=self.osd_sink_pad_buffer_probe, client_data=self.meta_data)

        # Add the custom PPH to the Sink pad of the OSD
        retval = dsl_osd_pph_add('on-screen-display', handler='custom-pph', pad=DSL_PAD_SINK)
        if retval != DSL_RETURN_SUCCESS:
            return retval

        # New Overlay Sink, 0 x/y offsets and same dimensions as Tiled Display	
        retval = dsl_sink_window_new('window-sink', 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)	
        if retval != DSL_RETURN_SUCCESS:	
            return retval
            
        # Fix the aspect ration for the Window sink
        retval = dsl_sink_window_force_aspect_ratio_set('window-sink', force=True)
        if retval != DSL_RETURN_SUCCESS:	
            return retval

        # Add all the components to our pipeline	
        retval = dsl_pipeline_new_component_add_many('pipeline', 	
            ['rtsp-source', 'primary-gie', 'ktl-tracker', 'on-screen-display', 'window-sink', None])
        if retval != DSL_RETURN_SUCCESS:	
            return retval
            
        # Setup the Pipeline's XWindow handle to use application window. 
        return dsl_pipeline_xwindow_handle_set('pipeline', self.window_frame.winfo_id())

    def play_pipeline(self):
        return dsl_pipeline_play('pipeline')
        
    def osd_sink_pad_buffer_probe(self, gst_buffer, client_data):
    
        # cast the C void* client_data back to a py_object pointer and deref
        meta_data = cast(client_data, POINTER(py_object)).contents.value
        meta_data.mutex.acquire()
        
        frame_number=0

        # Retrieve batch metadata from the gst_buffer
        # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
        # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
                # The casting is done by pyds.NvDsFrameMeta.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
        
                #pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        
            try:
                l_frame=l_frame.next
            except StopIteration:
                break

        meta_data.mutex.release()
        return True
    
# Create the App and run the mainloop
app = App()
app.mainloop()
print('exit')