# WebRTC HTML Client Example

### [webrtc.html](/examples/webtrc-html/webrtc.html)
The HTML page implements a simple WebRTC Client script to demonstrate the use of DSL's [Websocket Server](/docs/api-ws-server.md) and [WebRTC Sink](/docs/api-sink.md) APIs. The client can be used with the following Python examples.
* [1file_webrtc_connect_post_play.py](/examples/python/1file_webrtc_connect_post_play.py)
* [1file_webrtc_connect_pre_play.py](/examples/python/1file_webrtc_connect_pre_play.py)

**Important Note: The current WebRTC implementation is based on the webrtcbin plugin only available with GStreamer 1.18 or later.**

The javascript implements two browser window functions:

#### window.onload
Opens a Websocket connection using the following configuration parameters
```java
var config = { 'iceServers': [{ 'urls': 'stun:stun.l.google.com:19302'}] }
var wsHost = "localhost"; 
var wsPort = 60001; 
var wsPath = "ws"; 
```
#### window.onbeforeunload
Closes the Websocket connection - before the browser window unloads - on window/tab close.

The script implements the following WebRTC signaling functions.
* `onLocalDescription`
* `onIncomingSDP`
* `onIncomingICE`
* `onTrack`
* `onIceCandidate`
* `onDataChannel`
