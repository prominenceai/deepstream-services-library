# Installing DSL Dependencies
The DeepStream Services Library (DSL) is built on the NVIDIA® [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) and requires all SDK components and prerequisites to be installed and verified.

Please consult the [NVIDIA DeepStream Quick Start Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html) for complete Installation Instructions.

> The following installation instructions are specific to NVIDIA® DeepStream Versions 6.4 and 7.0 on Ubuntu 22.04.

## Contents
* [Base Install - Jetson and dGPU](#base-install---jetson-and-dgpu)
* [Additional WebRTC Sink Dependencies](#additional-webrtc-sink-dependencies)
* [Enabling Extended Image Services (Optional)](#enabling-extended-image-services-optional)
* [Enabling Interpipe Services (Optional)](#enabling-interpipe-services-optional)
* [Documentation and Debug Dependencies (Optional)](#documentation-and-debug-dependencies-optional)

---

## Base Install - Jetson and dGPU
### Native Install 
Enter the following command to install the minimal requirements to build DSL
```bash
sudo apt update && sudo apt-get install \
    libgstrtspserver-1.0-dev \
    gstreamer1.0-rtsp \
    libapr1 \
    libapr1-dev \
    libaprutil1 \
    libaprutil1-dev \
    libgeos-dev \
    libcurl4-openssl-dev
```    

### Container Install 
The same as above, but without using sudo
```bash
apt update && apt-get install \
    libgstrtspserver-1.0-dev \
    gstreamer1.0-rtsp \
    libapr1 \
    libapr1-dev \
    libaprutil1 \
    libaprutil1-dev \
    libgeos-dev \
    libcurl4-openssl-dev
```    

## Additional WebRTC Sink Dependencies
### Native Install
Enter the following command to install the additional WebRTC Sink dependencies.
```bash
sudo apt-get install \
    libjson-glib-dev \
    libsoup-gnome2.4-dev \
    libgstreamer-plugins-bad1.0-dev \
    libnice-dev \
    gstreamer1.0-nice
```
### Container Install 
The same as above, but without using sudo
```bash
apt-get install \
    libjson-glib-dev \
    libsoup-gnome2.4-dev \
    libgstreamer-plugins-bad1.0-dev \
    libnice-dev \
    gstreamer1.0-nice
```

### Post Install
After installation, enter the following command to update the json-glib-1.0 package.
```bash
pkg-config --cflags json-glib-1.0
```
### Updating the Makefile
To enable the WebRTC Sink Component, search for the following section in the DSL Makefile and set the BUILD flag to true.
```makefile
# To enable the WebRTC Sink component (requires GStreamer >= 1.20)
# - set BUILD_WEBRTC:=true
BUILD_WEBRTC:=true
```

---

## Enabling Extended Image Services (Optional)
DSL provides a choice of using [FFmpeg](https://ffmpeg.org/) or [OpenCV](https://opencv.org/) to implement the Extended Images Services, which include:
* [Streaming Image Source](/docs/api-source.md#dsl_source_image_stream_new)
* [Object](/docs/api-ode-action.md#dsl_ode_action_capture_object_new) and [Frame](/docs/api-ode-action.md#dsl_ode_action_capture_frame_new) Capture [ODE Actions](/docs/api-ode-action.md).
* [Frame Capture Sink](/docs/api-sink.md#dsl_sink_frame_capture_new)

### Using FFmpeg
#### Native Install 
Building with FFmpeg on Ubuntu 22.04 requires the following development libraries to be installed.
```bash
sudo apt-get install \
    libavformat-dev \
    libswscale-dev  
```
#### Container Install
To use FFmpeg within a DeepStream Docker Container, you must clone, build, and install the latest version of the FFmpeg development libraries. 
Copy and execute each of the following commands, one at a time, to setup the required dependencies.
```bash
$ mkdir ~/ffmpeg; cd ~/ffmpeg
$ git clone https://github.com/FFmpeg/FFmpeg.git
$ cd FFmpeg
$ ./configure --enable-shared --disable-lzma
$ make
$ sudo make install
```

### Using OpenCV
#### Native Install 
Copy and execute the following command to install the OpenCV development library (as an alternative to using FFmpeg).
```bash
sudo apt-get install -y libopencv-dev
```
#### Container Install
There are insufficient dependencies in the Docker container to install and use libopencv-dev as show above. Using OpenCV in a Docker container is not recommended!

### Updating the Makefile
To enable the Extended Image Services, search for the following section in the DSL Makefile and set the appropriate BUILD_WITH flag to true.
```makefile
# To enable the extended Image Services, install either the FFmpeg or OpenCV 
# development libraries (See /docs/installing-dependencies.md), and
#  - set either BUILD_WITH_FFMPEG or BUILD_WITH_OPENCV:=true (NOT both)
BUILD_WITH_FFMPEG:=false
BUILD_WITH_OPENCV:=false
```

## Enabling Interpipe Services (Optional)
The Interpipe Sink and Source are optional/conditional DSL components.  To enable, you will need to [build and install](https://developer.ridgerun.com/wiki/index.php/GstInterpipe_-_Building_and_Installation_Guide) the RidgeRun plugins. Then update the DSL Makefile to include/build the DSL Sink and Source components. Search for the following section and set `BUILD_INTER_PIPE` to `true`,
```
# To enable the InterPipe Sink and Source components
# - set BUILD_INTER_PIPE:=true
BUILD_INTER_PIPE:=true
```

## Documentation and Debug Dependencies (Optional)

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
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [Custom Component](/docs/api-gst.md)
* [Pad Probe Handler](/docs/api-pph.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Accumulator](/docs/api-ode-accumulator.md)
* [ODE Action ](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [ODE Heat-Mapper](/docs/api-ode-heat-mapper.md)
* [Display Type](/docs/api-display-type.md)
* [Mailer](/docs/api-mailer.md)
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)
