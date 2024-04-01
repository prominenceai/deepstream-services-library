

################################################################################
# The MIT License
#
# Copyright (c) 2019-2024, Prominence AI, Inc.
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

def main(args):

    while True:

        retval = dsl_gst_element_new('my-queue-1', "queue")
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_gst_element_new('my-queue-2', "queue")
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_gst_element_new('my-identity', "identity")
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_gst_element_new('my-v4l2sink', "v4l2sink")
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # ---------------------------------------------------------------------------_    
        # boolean values
        retval, value = dsl_gst_element_property_boolean_get('my-queue-1', 
            'flush-on-eos')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_gst_element_property_boolean_set('my-queue-1', 
            'flush-on-eos', True)
        if retval != DSL_RETURN_SUCCESS:
            break
        print('flush-on-eos =', value)
            
        # ---------------------------------------------------------------------------_    
        # float values
        retval, value = dsl_gst_element_property_float_get('my-identity', 
            'drop-probability')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_gst_element_property_float_set('my-identity', 
            'drop-probability', 0.11)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # ---------------------------------------------------------------------------_    
        # int values
        retval, value = dsl_gst_element_property_int_get('my-identity', 
            'datarate')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_gst_element_property_int_set('my-identity', 
            'datarate', 1234)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # ---------------------------------------------------------------------------_    
        # uint values
        retval, value = dsl_gst_element_property_uint_get('my-queue-1', 
            'max-size-buffers')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_gst_element_property_uint_set('my-queue-1', 
            'max-size-buffers', 1234)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # ---------------------------------------------------------------------------_    
        # int64 values
        retval, value = dsl_gst_element_property_int64_get('my-identity', 
            'ts-offset')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_gst_element_property_int64_set('my-identity', 
            'ts-offset', 1234)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # ---------------------------------------------------------------------------_    
        # uint64 values
        retval, value = dsl_gst_element_property_uint64_get('my-identity', 
            'ts-offset')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_gst_element_property_uint64_set('my-identity', 
            'ts-offset', 1234)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # ---------------------------------------------------------------------------_    
        # string values
        retval, value = dsl_gst_element_property_string_get('my-v4l2sink', 
            'device')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_gst_element_property_string_set('my-v4l2sink', 
            'device', '/dev/video1')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # ---------------------------------------------------------------------------_    
        # GST Bin
        retval = dsl_gst_bin_new('my-bin')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        retval = dsl_gst_bin_element_add('my-bin', 'my-queue-1')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        retval = dsl_gst_bin_element_add_many('my-bin', 
            ['my-identity', 'my-queue-2', None])
        if retval != DSL_RETURN_SUCCESS:
            break
            
        retval = dsl_gst_bin_element_remove('my-bin', 'my-queue-1')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        retval = dsl_gst_bin_element_remove_many('my-bin', 
            ['my-identity', 'my-queue-2', None])
        if retval != DSL_RETURN_SUCCESS:
            break
            
        retval = dsl_component_delete('my-bin')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        retval = dsl_gst_bin_new_element_add_many('my-bin', 
            ['my-identity', 'my-queue-2', None])
        if retval != DSL_RETURN_SUCCESS:
            break
            
        break

    # Print out the final result
    print(dsl_return_value_to_string(retval))

    dsl_delete_all()

if __name__ == '__main__':
    sys.exit(main(sys.argv))