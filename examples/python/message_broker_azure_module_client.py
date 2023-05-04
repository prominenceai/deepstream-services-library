################################################################################
# The MIT License
#
# Copyright (c) 2022-2023, Prominence AI, Inc.
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

import datetime
import threading
import time
import logging

#####################################################################################
# This example demonstrates the use of the Module Client Azure protocol for
# bi-directional messaging.  The script must be run from a deployed Docker container
#
# See https://github.com/prominenceai//deepstream-services-library/proto-lib-azure.md
#####################################################################################
protocol_lib = \
    '/opt/nvidia/deepstream/deepstream/lib/libnvds_azure_edge_proto.so'
broker_config_file = \
    '/opt/nvidia/deepstream/deepstream/sources/libs/azure_protocol_adaptor/module_client/cfg_azure.txt'

# Connection string must be defined in /etc/iotedge/config.toml
connection_string = None

# Topics used for sending and subscribing to messages
DEFAULT_TOPIC_1 = '/dsl/example-topic-1'
DEFAULT_TOPIC_2 = '/dsl/example-topic-2'

## 
# Function to be called on connection failure 
## 
def message_broker_connection_listener(client_data, status):
    logging.info('Connection listener called with status: {}'.format(status))

## 
# Function #1 to be called on send-async result
## 
def message_broker_send_result_listener_1(client_data, status):
    logging.info('Result listener_1 called with status: {}'.format(status))

## 
# Function #2 to be called on send-async result
## 
def message_broker_send_result_listener_2(client_data, status):
    logging.info('Result listener_2 called with status: {}'.format(status))

## 
# Function to be called on incomming message received
## 
def message_broker_subscriber(client_data, status, message, length, topic):

    logging.info('Incomming message received with status: {}'.format(status))
    logging.info('  topic: {}'.format(topic))
    logging.info('  length: {}'.format(length))
    
    # NOTE: need to determine how to cast the `void* message` with `length` 
    # back to a ascii string. This example is still a work in progress (WIP)
    # logging.info('  payload: {}'.format(payload.decode('utf-8')))

## 
# Thread loop function to periodically send a pre-canned message
## 
def thread_function_1(name):

    # Simple message to send to the server
    unicode_message = "Hello world - thread_function_1 is messaging"
    
    while 1:
        retval = dsl_message_broker_message_send_async('message-broker',
            topic = DEFAULT_TOPIC_1,
            message = unicode_message.encode('utf-8'),
            size = len(unicode_message),
            response_listener = message_broker_send_result_listener_1,
            client_data = None)
        logging.info('thread_function_1 - send async returned:{}'.
            format(dsl_return_value_to_string(retval)))
        time.sleep(5)
        
## 
# Thread loop function to periodically send a pre-canned message
## 
def thread_function_2(name):

    # Simple message to send to the server
    unicode_message = "Hello world - thread_function_2 is messaging"
    
    while 1:
        retval = dsl_message_broker_message_send_async('message-broker',
            topic = DEFAULT_TOPIC_2,
            message = unicode_message.encode('ascii'),
            size = len(unicode_message),
            response_listener = message_broker_send_result_listener_2,
            client_data = None)
        logging.info('thread_function_2 - send async returned:{}'.
            format(dsl_return_value_to_string(retval)))
        time.sleep(6)
        

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        # Direcect log statements to file in the /tmp/.dsl/ folder which is 
        # accessible from outside of the running Docker container.
        logging.basicConfig(
            filename='/tmp/.dsl/msg_app-{:%Y%m%d-%H%M%S}.log'.format(datetime.datetime.now()),
            filemode = 'w',
            format = '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt = '%H:%M:%S',
            level=logging.INFO)

        # Direct GStreamer debug logs to a file in the /tmp/.dsl/ folder .
        retval = dsl_info_log_file_set_with_ts('/tmp/.dsl/debug')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Set the debug log level with a default level of 1 (ERROR) for all
        # GStreamer objects and plugins, and with DSL's level set to 4 (INFO).
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
            message_broker_connection_listener, client_data=None)    
        if retval != DSL_RETURN_SUCCESS:
            break

        # Connect to the remote server
        retval = dsl_message_broker_connect('message-broker')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Add the subscriber callback function defined above. This a contrived
        # example that subscribes to its own messages -- loopback test
        retval = dsl_message_broker_subscriber_add('message-broker',
            message_broker_subscriber, 
            topics = [DEFAULT_TOPIC_1, DEFAULT_TOPIC_2, None], 
            client_data = None)
        if retval != DSL_RETURN_SUCCESS:
            break

            
        # Start the messaging threads and join
        send_thread_1 = threading.Thread(target=thread_function_1, args=(1,))
        send_thread_2 = threading.Thread(target=thread_function_2, args=(1,))
        
        send_thread_1.start()
        send_thread_2.start()
        send_thread_1.join()
        send_thread_2.join()
        
        retval = DSL_RETURN_SUCCESS
        break

    # Print out the final result
    logging.info('Final result = {}'.format(dsl_return_value_to_string(retval)))

    # Cleanup all DSL/GST resources
    dsl_delete_all()
    
if __name__ == '__main__':
    sys.exit(main(sys.argv))
