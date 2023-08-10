################################################################################
# The MIT License
#
# Copyright (c) 2019-2023, Prominence AI, Inc.
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
import time

from dsl import *
import importlib

   
     #retval = dsl_branch_new_component_add_many('branch'+str(i), ['on-screen-display'+str(i), 'fake'+str(i), None])                  

def addDemux(name,i):
    
    
    while(i>0):
        print(i)
#        dsl_osd_new('on-screen-display'+str(name)+str(i),text_enabled=False, clock_enabled=False, bbox_enabled=False, mask_enabled=False)  
        dsl_sink_window_new('w'+str(name)+str(i),0,0,500,500)
        dsl_sink_sync_enabled_set('w'+str(name)+str(i), False)
        dsl_sink_qos_enabled_set('w'+str(name)+str(i), False)
#        dsl_branch_new_component_add_many('branch'+str(name)+str(i), ['on-screen-display'+str(name)+str(i), 'w'+str(name)+str(i),None])
#        dsl_tee_branch_add(name,'branch'+str(name)+str(i))  
        dsl_tee_branch_add(name,'w'+str(name)+str(i))  
        i=i-1  
       
        
def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        #URI='rtsp://192.168.101.191:8554/65/65'
        i=1
        dsl_pipeline_new('pipeline1')
      
        while(i<2):
             #dsl_source_rtsp_new('uri-source-'+str(i), URI,DSL_RTP_TCP,1,2,0,0) 
             retval = dsl_source_uri_new('uri-source-'+str(i), '/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4', False, 0, 0)
             retval = dsl_pipeline_component_add_many('pipeline1', ['uri-source-'+str(i),  None])
             i=i+1
       
        retval = dsl_tee_splitter_new('splitter')     
        dsl_branch_new('branchA')
        dsl_branch_new('branchB')
        
        retval=dsl_tee_demuxer_new('demuxer1',50)
        retval=dsl_tee_demuxer_new('demuxer2',50)   
        
        addDemux('demuxer1',1)
        addDemux('demuxer2',1)
       
        retval = dsl_branch_component_add_many('branchA', [
             'demuxer1', None])
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_branch_component_add_many('branchB', [
             'demuxer2', None])
        if retval != DSL_RETURN_SUCCESS:
            break
        
        retVal = dsl_tee_branch_add_many('splitter', [
         'branchA',
         'branchB',
         None])
        if retval != DSL_RETURN_SUCCESS:
            break
         
        retval = dsl_pipeline_component_add_many('pipeline1', 
            ['splitter',None])
        if retval != DSL_RETURN_SUCCESS:
            break     

        retval = dsl_pipeline_play('pipeline1')
        
        #addNew()
        
        if retval != DSL_RETURN_SUCCESS:
            break

        dsl_main_loop_run()
        retval = DSL_RETURN_SUCCESS
        break

    # Print out the final result
    print(dsl_return_value_to_string(retval))

    dsl_delete_all()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
