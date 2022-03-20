################################################################################
# The MIT License
#
# Copyright (c) 2022, Prominence AI, Inc.
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

import threading
import time

################################################################################
# This example demonstrates the use of the Module Client Azure protocol for
# bi-directional messaging - CAUTION: this script is a work in progress (WIP).
################################################################################
protocol_lib = \
    '/opt/nvidia/deepstream/deepstream/lib/libnvds_azure_edge_proto.so'
broker_config_file = \
    '/opt/nvidia/deepstream/deepstream/sources/libs/azure_protocol_adaptor/module_client/cfg_azure.txt'

# Connection string must be defined in /etc/iotedge/config.yaml
connection_string = None

## 
# Function to be called on connection failure 
## 
def message_broker_connection_listener(client_data, status):
    print('Connection listener called with status:', status)

## 
# Function to be called on send-async result
## 
def message_broker_send_result_listener(client_data, status):
    print('Result listener called with status:', status)

## 
# Thread loop function to periodically send a pre-canned message
## 
def thread_function(name):

    # Simple message to send to the server
    unicode_message = "Hello remote server - edge device is messaging"
    
    while message_limit:
        retval = dsl_message_broker_message_send_async('message-broker',
            topic = "/dsl/message",
            message = unicode_message.encode('ascii'),
            size = len(unicode_message),
            response_listener = message_broker_send_result_listener,
            client_data = None)
        print('dsl_message_broker_message_send_async returned ', 
            dsl_return_value_to_string(retval))
        time.sleep(5)
        

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:
    
        # Direct debug logs and stdout to the mapped /tmp/.dsl/ folder which is 
        # accessible from outside of the running Docker container.
        retval = dsl_info_log_file_set_with_ts('/tmp/.dsl/debug')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_info_stdout_redirect_with_ts('/tmp/.dsl/stdout')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Set the debug log level with a default level of 1 (ERROR) for all
        # Gstreamer objects and plugins, and with DSL's level set to 4 (INFO).
        retval = dsl_info_log_level_set('1,DSL:4')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a new Message Broker with the specs defined above
        retval = dsl_message_broker_new('message-broker', 
            broker_config_file = broker_config_file,
            protocol_lib = protocol_lib,
            connection_string = connection_string)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Add a connection listener to be notified on connection failure.
        retval = dsl_message_broker_connection_listener_add('message-broker',
            message_broker_connection_listener, None)    

        # Connect to the remote server
        retval = dsl_message_broker_connect('message-broker')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Start the messaging thread to send a number of canned messages.
        send_thread = threading.Thread(target=thread_function, args=(1,))
        
        send_thread.start()
        send_thread.join()
        
        retval = DSL_RETURN_SUCCESS
        break

    # Print out the final result
    print(dsl_return_value_to_string(retval))
    

    # Cleanup all DSL/GST resources
    dsl_delete_all()
    
if __name__ == '__main__':
    sys.exit(main(sys.argv))
