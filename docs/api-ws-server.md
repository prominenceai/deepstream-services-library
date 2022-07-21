# WebSocket Server API Reference
The WebSocket Server connects remote clients with DSL Signaling Transceivers. The [WebRTC Sink](/docs/api-sink.md) is the only Signaling Transceiver at this time. The WebSocket Server is a singleton object initialized on any first use. Signaling Transceivers are automatically added to the WebSocket Server when the client application adds the Signaling Transceiver to a parent Pipeline or Branch. 

**IMPORTANT: The WebSocket Server and WebRTC Sink require GStreamer 1.18 or later - only available for ubuntu 20.04 or later**

Client Applications use the WebSocket Server to listen on a specified port for incoming WebSocket connections opened by remote HTML clients. When handling the connection, the WebSocket server iterates over its collection of Signaling Transceivers looking for the first available Transceiver to connect with. Clients can add a listener callback function to be called with the specific WebSocket path when the WebSocket is first opened (see use case 1. below). 

**Import notes:** 
* The WebSocket Server calls all client listeners functions prior to checking for the first available Signaling Transceiver. This allows the client to create and add the new Transceiver to a specific Pipeline based on the WebSocket path used by the remote HTML client.
* Client Applications can add client listener callback functions to each Signaling Transceiver to be notified on change of connection state.

### Adding Signal Transceivers
**Use case 1.** One or more Pipelines are created and set to a playing state prior to listening for incoming WebSocket connections. New WebRTC Sinks (Signaling Transceivers) are created and added to a running Pipeline when a remote HTML client opens a new connection.
![](/Images/websocket-server-calling-sequence-1.png)

**Use case 2.** The client application creates the Pipeline and WebRTC Sink prior to starting the WebSocket Server. The Pipeline is then set to a playing state when the remote client opens the WebSocket connection. The Pipeline is stopped when the WebRTC Sink calls its client listener in response to the remote client closing the WebSocket connection. 

![](/Images/websocket-server-calling-sequence-2.png)


## Relevant Examples
* [webrtc.html](/examples/webtrc-html/webrtc.html) - remote WebRTC Client
* [1file_webrtc_connect_post_play.py](/examples/python/1file_webrtc_connect_post_play.py)
* [1file_webrtc_connect_pre_play.py](/examples/python/1file_webrtc_connect_pre_play.py)

## WebSocket Server API
**Client Callback Typedefs**
* [dsl_websocket_server_client_listener_cb](#dsl_websocket_server_client_listener_cb)

**Methods**
* [dsl_websocket_server_path_add](#dsl_websocket_server_path_add)
* [dsl_websocket_server_listening_start](#dsl_websocket_server_listening_start)
* [dsl_websocket_server_listening_stop](#dsl_websocket_server_listening_stop)
* [dsl_websocket_server_listening_state_get](#dsl_websocket_server_listening_state_get)
* [dsl_websocket_server_client_listener_add](#dsl_websocket_server_client_listener_add)
* [dsl_websocket_server_client_listener_remove](#dsl_websocket_server_client_listener_remove)

---
## Return Values
The following return codes are used by the WebSocket Server API
```C
#define DSL_RESULT_WEBSOCKET_SERVER_THREW_EXCEPTION                 0x00700001
#define DSL_RESULT_WEBSOCKET_SERVER_SET_FAILED                      0x00700002
#define DSL_RESULT_WEBSOCKET_SERVER_CLIENT_LISTENER_ADD_FAILED      0x00700003
#define DSL_RESULT_WEBSOCKET_SERVER_CLIENT_LISTENER_REMOVE_FAILED   0x00700004
```

<br>

---

## Client Callback Typedefs
### *dsl_websocket_server_client_listener_cb*
```C++
typedef void (*dsl_websocket_server_client_listener_cb)(const wchar_t* path,
    void* client_data);
```
Callback typedef for a client listener. Functions of this type are added to the WebSocket Server by calling [dsl_websocket_server_client_listener_add](#dsl_websocket_server_client_listener_add). Once added, the function will be called on every incoming WebSocket connection until the client removes the listener by calling [dsl_websocket_server_client_listener_remove](#dsl_websocket_server_client_listener_remove).

Important note: The WebSocket Server will call all registered client listener callback functions prior to checking for any available Signaling Transceivers to connect with allowing the client to create and add a new transceiver in the callback function.

**Parameters**
* `path` - [in] the WebSocket path for the new incoming connection.
* `client_data` - [in] opaque pointer to client's user data, provided to the WebSocket server on callback add.

<br>

---
## Methods


### *dsl_websocket_server_path_add*
```C++
DslReturnType dsl_websocket_server_path_add(const wchar_t* path);
```
This service adds a new WebSocket path to the WebSocket Server. The server must be in a non-listening state when adding a new path. The new path must be prefixed with the `/ws/` root.

**Parameters**
* `path` - [in] new WebSocket path for the Server to handler

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_websocket_server_path_add('/ws/pipeline-1')
```

<br>

### *dsl_websocket_server_listening_start*
```C++
DslReturnType dsl_websocket_server_listening_start(uint port_number);
```
This service starts the WebSocket Server listening on a specified WebSocket port number

**Parameters**
* `port_number` - [in] the WebSocket port number to start listing on

**Returns**
* `DSL_RESULT_SUCCESS` on successful start. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_websocket_server_listening_start(DSL_WEBSOCKET_SERVER_DEFAULT_WEBSOCKET_PORT)
```

<br>

### *dsl_websocket_server_listening_stop*
```C++
DslReturnType dsl_websocket_server_listening_stop();
```
This service stops the WebSocket Server when listening on a specified port. Each connected Signaling Transceiver will be disconnected with each notifying their client listeners of the change in connection state.

**Returns**
* `DSL_RESULT_SUCCESS` on successful stop. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_websocket_server_listening_stop()
```

<br>

### *dsl_websocket_server_listening_state_get*
```C++
DslReturnType dsl_websocket_server_listening_state_get(boolean* is_listening,
    uint* port_number);
```
This service gets the current listening state and port_number for the WebSocket Server.

**Parameters**
* `is_listening` - [out] true if the Server is listening on a client specified port, false otherwise.
* `port_number` - [out] the WebSocket port number the Server is listening on, or 0.

**Returns**
* `DSL_RESULT_SUCCESS` on successful start. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, is_listening, port_number = dsl_websocket_server_listening_state_get()
```

<br>

### *dsl_websocket_server_client_listener_add*
```C++
DslReturnType dsl_websocket_server_client_listener_add(
    dsl_websocket_server_client_listener_cb listener, void* client_data);
```
This service adds a callback function of type [dsl_websocket_server_client_listener_cb](#dsl_websocket_server_client_listener_cb) to the WebSocket Server. The function will be called on all incoming socket connections. Multiple callback functions can be added to the WebSocket Server.

**Parameters**
* `listener` - [in] listener callback function to add.
* `client_data` - [in] opaque pointer to user data, returned to the listener on call back

**Returns**  `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_websocket_server_client_listener_add(client_listener_cb, None)
```

<br>

### *dsl_websocket_server_client_listener_remove*
```C++
DslReturnType dsl_websocket_server_client_listener_remove(
    dsl_websocket_server_client_listener_cb listener);
```
This service removes a callback function of type [dsl_websocket_server_client_listener_cb](#dsl_websocket_server_client_listener_cb) from the Websocket Server.

**Parameters**
* `listener` - [in] listener callback function to remove.

**Returns**  
* `DSL_RESULT_SUCCESS` on successful removal. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_websocket_server_client_listener_remove(client_listener_cb)
```

<br>

---

## API Reference
* [List of all Services](/docs/api-reference-list.md)
* [Pipeline](/docs/api-pipeline.md)
* [Player](/docs/api-player.md)
* [Source](/docs/api-source.md)
* [Tap](/docs/api-tap.md)
* [Dewarper](/docs/api-dewarper.md)
* [Preprocessor](/docs/api-preproc.md)
* [Inference Engine and Server](/docs/api-infer.md)
* [Tracker](/docs/api-tracker.md)
* [Segmentation Visualizer](/docs/api-segvisual.md)
* [Tiler](/docs/api-tiler.md)
* [Splitter and Demuxer](/docs/api-tee.md)
* [On-Screen Display](/docs/api-osd.md)
* [Sink](/docs/api-sink.md)
* [Pad Probe Handler](/docs/api-pad-probe-handler.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Accumulator](/docs/api-ode-accumulator.md)
* [ODE Acton](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [ODE Heat-Mapper](/docs/api-ode-heat-mapper.md)
* [Display Type](/docs/api-display-type.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [Mailer](/docs/api-mailer.md)
* **Websocket Server**
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)
