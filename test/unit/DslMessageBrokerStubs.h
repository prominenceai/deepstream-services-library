/*
The MIT License

Copyright (c)   2022, Prominence AI, Inc.

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

#ifndef __NV_MESSAGE_BROKER_STUBS_H__
#define __NV_MESSAGE_BROKER_STUBS_H__

#ifdef __cplusplus
extern "C"
{
#endif

#undef nv_msgbroker_connect
#define nv_msgbroker_connect _dsl_nv_msgbroker_connect

static NvMsgBrokerClientHandle _dsl_nv_msgbroker_connect(char *broker_conn_str, 
    char *broker_proto_lib, nv_msgbroker_connect_cb_t connect_cb, char *cfg);

#undef nv_msgbroker_send_async
#define nv_msgbroker_send_async _dsl_nv_msgbroker_send_async

static NvMsgBrokerErrorType _dsl_nv_msgbroker_send_async(NvMsgBrokerClientHandle h_ptr, 
    NvMsgBrokerClientMsg message, nv_msgbroker_send_cb_t cb, void *user_ctx);

#undef nv_msgbroker_subscribe
#define nv_msgbroker_subscribe _dsl_nv_msgbroker_subscribe

static NvMsgBrokerErrorType _dsl_nv_msgbroker_subscribe(NvMsgBrokerClientHandle h_ptr, 
    char ** topics, int num_topics,  nv_msgbroker_subscribe_cb_t cb, void *user_ctx);

#undef nv_msgbroker_disconnect
#define nv_msgbroker_disconnect _dsl_nv_msgbroker_disconnect

NvMsgBrokerErrorType _dsl_nv_msgbroker_disconnect(NvMsgBrokerClientHandle h_ptr);

static NvMsgBrokerClientHandle _dsl_nv_msgbroker_connect(char *broker_conn_str, 
    char *broker_proto_lib, nv_msgbroker_connect_cb_t connect_cb, char *cfg)
{
    LOG_INFO("_dsl_nv_msgbroker_connect called");
    
    return (NvMsgBrokerClientHandle)0x1234567812345678;
}

static NvMsgBrokerErrorType _dsl_nv_msgbroker_send_async(NvMsgBrokerClientHandle h_ptr, 
    NvMsgBrokerClientMsg message, nv_msgbroker_send_cb_t cb, void *user_ctx)
{
    LOG_INFO("_dsl_nv_msgbroker_connect called");
    LOG_INFO("  topic = " << message.topic);
    
    cb(user_ctx, NV_MSGBROKER_API_OK);

    return NV_MSGBROKER_API_OK;
}

static NvMsgBrokerErrorType _dsl_nv_msgbroker_subscribe(NvMsgBrokerClientHandle h_ptr, 
    char ** topics, int num_topics,  nv_msgbroker_subscribe_cb_t cb, void *user_ctx)
{
    LOG_INFO( "_dsl_nv_msgbroker_subscribe called");

    return NV_MSGBROKER_API_OK;
}

NvMsgBrokerErrorType _dsl_nv_msgbroker_disconnect(NvMsgBrokerClientHandle h_ptr)
{
    LOG_INFO("_dsl_nv_msgbroker_disconnect called");

    return NV_MSGBROKER_API_OK;
}

#ifdef __cplusplus
}
#endif

#endif
