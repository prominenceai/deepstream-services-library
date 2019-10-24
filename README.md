# deepstream-services-library (DSL)

![](/Images/under-construction.png)

DSL is best described as _"all the functionality of the NVIDIA DeepStream Reference Application - [deepstream-app](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html#page/DeepStream_Development_Guide%2Fdeepstream_app_architecture.html) - re-imagined as a shared library of on-demand DeepStream pipeline services"._

Written in C++ 11, the DSL API can be  called from both C/C++ and Python applications.

**C/C++**
```C++
#include <DslApi.h>

dsl_pipeline_new("myPipeline");
```
**Python**
```Python
import ctypes
libc = ctypes.CDLL("dsl-lib.so")

libc.dsl_pipeline_new("myPipeline")
```

DSL is built on the NVIDA® [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk), _"A complete streaming analytics toolkit for AI-based video and image understanding, as well as multi-sensor processing."_, 

The DeepStream SDK and DSL use the open source [GStreamer](https://gstreamer.freedesktop.org/),  _"An extremely powerful and versatile framework for creating streaming media applications"_.

## Contributing
DSL is released under the MIT license. Contributions are welcome and greatly appreciated. Contributor guidelines and code of conduct are still TBW. 

## DSL Users Guide

* [Overview](/docs/overview.md)
* [Installing Dependencies](/docs/installing-dependencies.md)
* [Building DSL](/docs/building-dsl.md)
* [API Reference](/docs/api-reference-list.md)
  * [Pipeline](/docs/api-pipeline.md)
  * [Component](/docs/api-component.md)
  * [Source](/docs/api-source.md)
  * Primary GIE
  * Secondary GIE
  * Tiled Display
  * On Screen Display
  * Sink
* [Examples](/docs/examples.md)
  * [C/C++](/docs/examples-cpp.md)
  * [Python](/docs/examples-python.md)
* [Logging and Debugging](/docs/debugging-dsl.md)
