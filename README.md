# deepstream-services-library (DSL)

![](/Images/under-construction.png)

DSL is best described as _"the NVIDIA DeepStream Reference Application — [deepstream-app](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html#page/DeepStream_Development_Guide%2Fdeepstream_app_architecture.html) — reimagined as a shared library of on-demand DeepStream pipeline services"._

Written in C++ 11, the DSL API can be  called from both C/C++ and Python applications.

**C/C++**
```C++
#include <DslApi.h>

uint retval = dsl_source_csi_new("csi-source", 1280, 720, 30, 1);
```
**Python3**
```Python
from dsl import *

# New CSI Live Camera Source
retval = dsl_source_csi_new('csi-source', 1280, 720, 30, 1)
```

DSL is built on the NVIDA® [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk), _"A complete streaming analytics toolkit for AI-based video and image understanding, as well as multi-sensor processing."_, 

The DeepStream SDK and DSL use the open source [GStreamer](https://gstreamer.freedesktop.org/),  _"An extremely powerful and versatile framework for creating streaming media applications"_.

The goals of DSL:
* To provide a high-level, multi-launguage Services API for building, and dynamically updating DeepStream pipelines.
* To abstract and encapsulate the complexity of the broad and flexible GStreamer framework with a simple DeepStream specific API. That said, DSL is designed to check-for a previously intialized instance of the GST Lib supporting integration with existing GStreamer Applications.

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
  * [Primary GIE](/docs/api-primary-gie)
  * Secondary GIE
  * [Tiled Display](/docs/api-tile-display.md)
  * [On Screen Display](/docs/api-screen-display.md)
  * [Sink](/docs/api-screen-display.md)# deepstream-services-library (DSL)

![](/Images/under-construction.png)

DSL is best described as _"the NVIDIA DeepStream Reference Application — [deepstream-app](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html#page/DeepStream_Development_Guide%2Fdeepstream_app_architecture.html) — reimagined as a shared library of on-demand DeepStream pipeline services"._

Written in C++ 11, the DSL API can be  called from both C/C++ and Python applications.

**C/C++**
```C++
#include <DslApi.h>

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

The goals of DSL:
* To provide a high-level, multi-language Services API for building, and dynamically updating DeepStream pipelines.
* To abstract and encapsulate the complexity of the broad and flexible GStreamer framework with a simple DeepStream specific API. That said, DSL is designed to check-for a previously initialized instance of the GST Lib supporting integration with existing GStreamer Applications.

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
  * [Primary GIE](/docs/api-primary-gie)
  * Secondary GIE
  * [Tiled Display](/docs/api-tile-display.md)
  * [On Screen Display](/docs/api-screen-display.md)
  * [Sink](/docs/api-screen-display.md)
  * Tracker
  * Dewarpper
* [Examples](/docs/examples.md)
  * [C/C++](/docs/examples-cpp.md)
  * [Python](/docs/examples-python.md)
* [Logging and Debugging](/docs/debugging-dsl.md)
  * Tracker
  * Dewarpper
* [Examples](/docs/examples.md)
  * [C/C++](/docs/examples-cpp.md)
  * [Python](/docs/examples-python.md)
* [Logging and Debugging](/docs/debugging-dsl.md)
