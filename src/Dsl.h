/*
The MIT License

Copyright (c) 2019-2021, Prominence AI, Inc.

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

#ifndef _DSL_H
#define _DSL_H

#include <cstdlib>

#include <gst/gst.h>
#include <gst/video/videooverlay.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <queue>
#include <iomanip>
#include <iostream> 
#include <sstream>
#include <vector>
#include <map>
#include <list> 
#include <memory> 
#include <math.h>
#include <fstream>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <typeinfo>
#include <algorithm>
#include <random>
#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>
#include <regex>

#include <deepstream_common.h>
#include <deepstream_config.h>
#include <deepstream_perf.h>
#include <nvds_version.h>
#include <gstnvdsmeta.h>
#include <nvdsmeta_schema.h>
#include <gstnvdsinfer.h>
#include <gst-nvdssr.h>
#include <cuda_runtime_api.h>
#include <geos_c.h>
#include <curl/curl.h>

#include "DslLog.h"
#include "DslMutex.h"


#endif // _DSL_H