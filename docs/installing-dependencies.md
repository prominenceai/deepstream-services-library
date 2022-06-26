# Installing DSL Dependencies
The DeepStream Services Library (DSL) is built on the NVIDA® [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) and requires all SDK components to be installed and verified.

Please consult the [NVIDIA DeepStream Quick Start Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html) for complete Installation Instructions.

## Contents
* [Single Command Install](#single-command-install)
* [Alternative Step-by-Step Instructions](#alternative-step-by-step-instructions)
* [Optional Documentation and Debug Dependencies](#optional-documentation-and-debug-dependencies)

---

## Single Command Install
Copy and execute one of the following commands according to your installed version of GStreamer.

### Jetson Install - GStreamer 1.16
```bash
sudo apt update && sudo apt-get install \
    libgstrtspserver-1.0-dev \
    gstreamer1.0-rtsp \
    libapr1 \
    libapr1-dev \
    libaprutil1 \
    libaprutil1-dev \
    libgeos-dev \
    python3-opencv \
    libcurl4-openssl-dev
```    

### dGPU Install - GStreamer 1.18
```bash
sudo apt update && sudo apt-get install \
    libgstrtspserver-1.0-dev \
    gstreamer1.0-rtsp \
    libapr1 \
    libapr1-dev \
    libaprutil1 \
    libaprutil1-dev \
    libgeos-dev \
    python3-opencv \
    libcurl4-openssl-dev \
    libjson-glib-1.0-0 \
    libsoup-gnome2.4-dev  
```    

---
## Alternative Step-by-Step Instructions.
**Note** These steps are an alternative to the [Single Command Install](#single_command_install) above.
### GStreamer RTSP Server
The RTSP Server lib is required by the RTSP Sink Component
```
sudo apt update
sudo apt-get install libgstrtspserver-1.0-dev gstreamer1.0-rtsp
```

### Apache Runtime
The Apache Runtime is used by the GStreamer Window Sink requiring the following libraries to be installed
```
sudo apt-get install libapr1 libapr1-dev libaprutil1 libaprutil1-dev
```

### Geometry Engine, Open Source (GEOS)
DSL uses the [GEOS](https://trac.osgeo.org/geos) C Library `libgeos-dev` - specifically, a set of spatial predicate functions for determining if geometries - points, lines, polygons - touch, cross, overlap, etc.
```
sudo apt install libgeos-dev
```

### Open Computer Vision
opencv4 is used to convert raw video frames to JPEG image files.

```
sudo apt install python3-opencv
```

### Lib cURL
libcurl provides Secure Socket Layer (SSL) protocol services.  
```
sudo apt install libcurl4-openssl-dev
```

### GLib JSON - GStreamer 1.18
libjson-glib is required if building with GStreamer 1.18 on Ubunto 20.24. The lib provides JSON serialization/deserialization services for the WebRTC Sink. **Note: the WebRTC requires GStream 1.18 or later - only available on Ubuntu 20.04**
```
sudo apt-get install libjson-glib-1.0-0
```

### Lib Soup - GStreamer 1.18
libsoup is required if building with GStreamer 1.18 on Ubunto 20.24. The lib provides services required for the WebSocket Server and WebRTC Sink API. **Note: the WebRTC requires GStream 1.18 or later - only available on Ubuntu 20.04**
```
sudo apt install libsoup-gnome2.4-dev
```

---

## Optional Documentation and Debug Dependencies

### Installing dot by graphviz
Doxygen requires **dot** to convert calling graphs to .png files and Pipeline graphs can be generated using **dot** as well
```
sudo apt-get install graphviz imagemagick
```

### Installing Doxygen
To install Doxygen on Ubuntu
```
sudo apt-get install doxygen
```

---

## Getting Started
* **Installing Dependencies**
* [Building and Importing DSL](/docs/building-dsl.md)

## API Reference
* [Overview](/docs/overview.md)
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
* [On-Screen Display](/docs/api-osd.md)
* [Demuxer and Splitter Tees](/docs/api-tee)
* [Sink](/docs/api-sink.md)
* [Pad Probe Handler](/docs/api-pph.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Accumulator](/docs/api-ode-accumulator.md)
* [ODE Action ](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [ODE Heat-Mapper](/docs/api-ode-heat-mapper.md)
* [Display Type](/docs/api-display-type.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [Mailer](/docs/api-mailer.md)
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)
