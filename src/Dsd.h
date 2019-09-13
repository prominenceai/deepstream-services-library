/*
The MIT License

Copyright (c) 2019-Present, ROBERT HOWELL

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

#ifndef _DSD_H
#define _DSD_H

#include <cstdlib>

#include <gst/gst.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <iostream> 
#include <vector>
#include <map> 

// referenced in deepstream_tracker_bin.c
GST_DEBUG_CATEGORY_EXTERN (NVDS_APP);

#include <deepstream_app_version.h>
#include <deepstream_common.h>
#include <deepstream_config.h>
#include <deepstream_config_file_parser.h>
#include <deepstream_osd.h>
#include <deepstream_perf.h>
#include <deepstream_primary_gie.h>
#include <deepstream_sinks.h>
#include <deepstream_sources.h>
#include <deepstream_streammux.h>
#include <deepstream_tiled_display.h>
#include <deepstream_dsexample.h>
#include <deepstream_tracker.h>
#include <deepstream_secondary_gie.h>
#include <deepstream_gie.h>
#include <deepstream_dewarper.h>

#include "DsdMutex.h"
#include "DsdLog.h"



#endif // _DSD_H