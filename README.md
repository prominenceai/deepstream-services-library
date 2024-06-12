[![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/prominenceai/deepstream-services-library?include_prereleases)](https://github.com/canammex-tech/deepstream-services-library/releases)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/prominenceai/deepstream-services-library/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](https://github.com/prominenceai/deepstream-services-library/blob/master/docs/overview.md)
[![Ask Me Anything!](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://github.com/prominenceai/deepstream-services-library/issues/new/choose)
[![Discord](https://img.shields.io/discord/750454524849684540)](https://discord.gg/MJvY9jjpAK)

![DeepStream Services Library](/Images/dsl-github-banner.png)

## Intelligent Audio and Video Analytics (IAVA)
A library of on-demand DeepStream Pipeline services. Written in C++ 17 with an `extern "C"` API, The DeepStream Services Library (DSL) can be called from both C/C++ and Python applications.

## Examples

### C++

```C++
#include <DslApi.h>

// New V4L2 Source
uint retval = dsl_source_v4l2_new(L"my-web-cam-1", L"/dev/video0");
```

### Python3

```Python
from dsl import *

# New V4L2 Source
retval = dsl_source_v4l2_new('my-web-cam-1', '/dev/video0')
```

DSL is built on the NVIDIA® [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk), _"A complete streaming analytics toolkit for AI-based video and image understanding, as well as multi-sensor processing."_,

The DeepStream SDK and DSL use the open source [GStreamer](https://gstreamer.freedesktop.org/),  _"An extremely powerful and versatile framework for creating streaming media applications"_.

---

## Important Bulletins
The latest release `v0.30.alpha` was developed to support DeepSteam 6.4 and 7.0 on Ubuntu 22.04. 

> WARNING! There is a cricical error in the DeepStream 7.0 Installation Instructions.

Under the section [Install librdkafka](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html#install-librdkafka-to-enable-kafka-protocol-adaptor-for-message-broker), the following instructions
```bash
$ sudo mkdir -p /opt/nvidia/deepstream/deepstream/lib
$ sudo cp /usr/local/lib/librdkafka* /opt/nvidia/deepstream/deepstream/lib
```
Must be replaced with.
```bash
$ sudo mkdir -p /opt/nvidia/deepstream/deepstream-7.0/lib
$ sudo cp /usr/local/lib/librdkafka* /opt/nvidia/deepstream/deepstream-7.0/lib
```
See [Error in DeepStream 7.0 installation instructions - symlink fails to create](https://forums.developer.nvidia.com/t/error-in-deepstream-7-0-installation-instructions-symlink-fails-to-create/296026) for more information.

---

## Contributing

DSL is released under the [MIT License](LICENSE). Contributions are welcome and greatly appreciated. Please review our [code of conduct](/CODE_OF_CONDUCT.md).

***Please report issues!***

## DSL Branching Workflow

![DSL Git Branching Workflow](/Images/dsl-branching-workflow.png)

1. `Master` is always the latest release and is only updated once per release.
2. `Develop` is the next release currently in development. The branch will be named with the next release label.
3. `Feature` all work is done on feature branches, named for the feature under development. 

All pull requests should be made against the current `Develop` branch.

## Discord
Come join us on [Discord](https://discord.gg/MJvY9jjpAK), an informal place to chat, ask questions, discuss ideas, etc.

## DSL Users Guide

* [Release Notes](/Release%20Notes/dsl-releases.md)
* [Overview](/docs/overview.md)
* [Installing Dependencies](/docs/installing-dependencies.md)
* [Building and Importing DSL](/docs/building-dsl.md)
* [API Reference](/docs/api-reference-list.md)
  * [Pipeline](/docs/api-pipeline.md)
  * [Player](/docs/api-player.md)
  * [Source](/docs/api-source.md)
  * [Tap](/docs/api-tap.md)
  * [Video Dewarper](/docs/api-dewarper.md)
  * [Preprocessor](/docs/api-preproc.md)
  * [Inference Engines and Servers](/docs/api-infer.md)
  * [Multi-Object Tracker](/docs/api-tracker.md)
  * [Segmentation Visualizer](/docs/api-segvisual.md)
  * [Tiler](/docs/api-tiler.md)
  * [Demuxer and Splitter](/docs/api-tee.md)
  * [Remuxer](/docs/api-remuxer.md)
  * [On-Screen Display](/docs/api-osd.md)
  * [Sink](/docs/api-sink.md)
  * [Branch](/docs/api-branch.md)
  * [Component](/docs/api-component.md)
  * [Custom Component](/docs/api-gst.md)
  * [Pad Probe Handler](/docs/api-pph.md)
  * [ODE Trigger](/docs/api-ode-trigger.md)
  * [ODE Accumulator](/docs/api-ode-accumulator.md)
  * [ODE Acton](/docs/api-ode-action.md)
  * [ODE Area](/docs/api-ode-area.md)
  * [ODE Heat-Mapper](/docs/api-ode-heat-mapper.md)
  * [Display Type](/docs/api-display-type.md)
  * [Mailer](/docs/api-mailer.md)
  * [WebSocket Server](/docs/api-ws-server.md)
  * [Message Broker](/docs/api-msg-broker.md)
  * [Info API](/docs/api-info.md)
* [Examples](/docs/examples.md)
  * [C/C++](/docs/examples-cpp.md)
  * [Python](/docs/examples-python.md)
  * [Tkinter Reference App](/docs/examples-tkinter.md)
  * [HTML WebRTC Client](/docs/examples-webrtc-html.md)
* [Using VS Code](/docs/vscode.md)
* [Logging and Debugging](/docs/debugging-dsl.md)
