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
from dsl import *	

import tkinter as tk
import tkinter.messagebox
import pyds
import os
import threading

from tkapp_window import MetaData, AppWindow
from config import *

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
        
        # go to fullscreen regardless of Window width and height
        self.attributes('-fullscreen', True)

        # Create the App window to be shared with the Pipeline's Window Sink
        self.app_window_frame = AppWindow(self, bg='black',
            width=self.winfo_screenwidth(), height=self.winfo_screenheight(), bd=0)
        self.app_window_frame.pack()
        
        # Build the Pipeline using the path specs and sizes defined above.
        self.retval = self.build_pipeline()
        if self.retval != DSL_RETURN_SUCCESS:
            self.quit(None)

    ##
    # Dtor for this Tkinter Application
    ##
    def __del__(self):
        
        # Cleanup all DSL/GST resources
        dsl_delete_all()
        
    ##
    # override the tk.quit so that all DSL/GST resources can be released
    ##
    def quit(self, event):
    
        if event is not None:
            if tk.messagebox.askquestion(
                message='Are you sure you want to exit?') == 'no':
                return
    
        # Explicity call the destructor as tk.quit will stop the main-loop and exit.
        self.__del__()
        super().quit()

    def build_pipeline(self):
        # For each camera, create a new RTSP Source for the specific RTSP URI	
        retval = dsl_source_rtsp_new(SOURCE_1,
            uri = rtsp_uri, 	
            protocol = DSL_RTP_ALL, 	
            intra_decode = False, 	
            drop_frame_interval = 0, 	
            latency = 100,
            timeout = 2)	
        if (retval != DSL_RETURN_SUCCESS):	
            return retval

        # New Primary GIE using the filespecs above, with interval and Id	
        retval = dsl_infer_gie_primary_new(PGIE, 
            primary_infer_config_file, primary_model_engine_file, 2)	
        if retval != DSL_RETURN_SUCCESS:	
            return retval

        # New KTL Tracker, setting max width and height of input frame	
        retval = dsl_tracker_ktl_new(TRACKER, 480, 272)	
        if retval != DSL_RETURN_SUCCESS:	
            return retval

        # New Tiled Display, setting width and height, use default cols/rows set by source count	
        retval = dsl_tiler_new(TILER, TILER_WIDTH, TILER_HEIGHT)	
        if retval != DSL_RETURN_SUCCESS:	
            return retval

        # New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new('on-screen-display', 
            text_enabled=True, clock_enabled=True, bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            return retval

        # New Custom Pad Probe Handler to draw the active display-type
        retval = dsl_pph_custom_new(DISPLAY_TYPE_PPH, 
            client_handler=self.overlay_display_type, client_data=self.app_window_frame.meta_data)

        # Add the custom PPH to the Sink pad of the OSD
        retval = dsl_osd_pph_add(OSD, handler=DISPLAY_TYPE_PPH, 
            pad=DSL_PAD_SINK)
        if retval != DSL_RETURN_SUCCESS:
            return retval

        # New Overlay Sink, 0 x/y offsets and same dimensions as Tiled Display	
        retval = dsl_sink_window_new(WINDOW_SINK, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)	
        if retval != DSL_RETURN_SUCCESS:	
            return retval
            
        # Fix the aspect ratio for the Window sink
        retval = dsl_sink_window_force_aspect_ratio_set(WINDOW_SINK, force=True)
        if retval != DSL_RETURN_SUCCESS:	
            return retval

        # Add all the components to our pipeline	
        retval = dsl_pipeline_new_component_add_many(PIPELINE, 	
            [SOURCE_1, PGIE, TRACKER, TILER, OSD, WINDOW_SINK, None])
        if retval != DSL_RETURN_SUCCESS:	
            return retval
            
        # Set the Stream-muxer dimensions using config settings
        retval = dsl_pipeline_streammux_dimensions_set(PIPELINE,
            width=STREAMMUX_WIDTH, height=STREAMMUX_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:	
            return retval
        
        # Setup the Pipeline's XWindow handle to use application's sink window frame. 
        return dsl_pipeline_xwindow_handle_set(PIPELINE, 
            self.app_window_frame.get_sink_window())
        
    ##
    # Custom Pad Probe Handler to overlay the active display-type
    def overlay_display_type(self, gst_buffer, client_data):
    
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
        
            if meta_data.active_display_type is not None:
                display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)

                display_meta.num_lines = 0
                display_meta.num_rects = 0
                
                if 'Line' in meta_data.active_display_type:
                    line = meta_data.active_display_type['Line']
                    py_nvosd_line_params = display_meta.line_params[display_meta.num_lines]
                    display_meta.num_lines +=1
                    
                    py_nvosd_line_params = line.copy(py_nvosd_line_params)

                elif 'Polygon' in meta_data.active_display_type:
                    
                    lines = meta_data.active_display_type['Polygon']
                    
                    for line in lines:
                        if line:
                            py_nvosd_line_params = display_meta.line_params[display_meta.num_lines]
                            display_meta.num_lines +=1
                            
                            py_nvosd_line_params = line.copy(py_nvosd_line_params)

                pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        
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
