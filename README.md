[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/canammex-tech/deepstream-services-library/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](https://github.com/canammex-tech/deepstream-services-library/blob/master/docs/overview.md)

# deepstream-services-library (DSL)

![](/Images/under-construction.png)

DSL is best described as _"the NVIDIA DeepStream Reference Application — [deepstream-app](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html#page/DeepStream_Development_Guide%2Fdeepstream_app_architecture.html) — reimagined as a shared library of DeepStream pipeline services"._

Written in C++ 11 with an `extern "C"` API, DSL can be called from both C/C++ and Python applications.

**C/C++**
```C++
#include <DslApi.h>

// New CSI Live Camera Source
uint retval = dsl_source_csi_new("csi-source", 1280, 720, 30, 1);
```
**Python3**
```Python
from dsl import *

# New CSI Live Camera Source
retval = dsl_source_csi_new('csi-source', 1280, 720, 30, 1)
```

DSL is built on the NVIDA® [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk), _"A complete streaming analytics toolkit for AI-based video and image understanding, as well as multi-sensor processing."_, 

The DeepStream SDK and DSL use the open source [GStreamer](https://gstreamer.freedesktop.org/),  _"An extremely powerful and versatile framework for creating streaming media applications"_.

## Contributing
DSL is released under the [BSD 2-Clause License](LICENSE). Contributions are welcome and greatly appreciated. Contributor guidelines and code of conduct are still TBW. 

## DSL Users Guide

* [Overview](/docs/overview.md)
* [Installing Dependencies](/docs/installing-dependencies.md)
* [Building and Importing DSL](/docs/building-dsl.md)
* [API Reference](/docs/api-reference-list.md)
  * [Pipeline](/docs/api-pipeline.md)
  * [Component](/docs/api-component.md)
  * [Streaming Source](/docs/api-source.md)
  * [Video Dewarper](/docs/api-dewarper.md)
  * [Inference Engine](/docs/api-gie.md)
  * [Multi-Object Tracker](/docs/api-tracker.md)
  * [Tiled Display](/docs/api-display.md)
  * [On-Screen Display](/docs/api-osd.md)
  * [Sink](/docs/api-sink.md)
* [Examples](/docs/examples.md)
  * [C/C++](/docs/examples-cpp.md)
  * [Python](/docs/examples-python.md)
* [Logging and Debugging](/docs/debugging-dsl.md)
