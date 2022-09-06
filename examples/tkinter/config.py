################################################################################
# The MIT License
#
# Copyright (c) 2021, Prominence AI, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

#!/usr/bin/env python
import sys	
from dsl import *	

import tkinter as tk

# RTSP Source URI	
rtsp_uri = 'rtsp://user:pswd@192.168.1.64:554/Streaming/Channels/101'	

# define stream-muxer output dimensions - typically set to common source dimensions
STREAMMUX_WIDTH = DSL_STREAMMUX_DEFAULT_WIDTH
STREAMMUX_HEIGHT = DSL_STREAMMUX_DEFAULT_HEIGHT	

# Tiler and Sink Window dimensions default to the same. 
TILER_WIDTH = STREAMMUX_WIDTH
TILER_HEIGHT = STREAMMUX_HEIGHT	
WINDOW_WIDTH = STREAMMUX_WIDTH	
WINDOW_HEIGHT = STREAMMUX_HEIGHT	

# Filespecs for the Primary GIE
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt'
primary_model_engine_file = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine'

# Define the Pipeline and component names
PIPELINE = 'pipeline'
SOURCE_1 = 'source-1'
PGIE = 'primary-gie'
TRACKER = 'tracker'
TILER = 'tiler'
DISPLAY_TYPE_PPH = 'display-type-pph'
OSD = 'on-screen_display'
WINDOW_SINK = 'window-sink'

BUTTON_SIDE_LENGTH = 48
BUTTON_FRAME_SIDE_LENGTH = 52

ACTION_ROW_Y_OFFSET = 10   # from top of Window to top of frame
BOTTOM_ROW_Y_OFFSET = 64  # from bottom of Window to top of frame

COLOR_PICKER_FRAME_X_OFFSET = 64
COLOR_PICKER_FRAME_Y_OFFSET = ACTION_ROW_Y_OFFSET

COLORS_ROW_1_FRAME_X_OFFSET = COLOR_PICKER_FRAME_X_OFFSET + 296
COLORS_ROW_1_FRAME_Y_OFFSET = ACTION_ROW_Y_OFFSET + 2

COLORS_ROW_2_FRAME_X_OFFSET = COLOR_PICKER_FRAME_X_OFFSET + 296
COLORS_ROW_2_FRAME_Y_OFFSET = ACTION_ROW_Y_OFFSET + 27

BG_COLOR_FRAME_X_OFFSET = COLORS_ROW_2_FRAME_X_OFFSET + 64
BG_COLOR_FRAME_Y_OFFSET = ACTION_ROW_Y_OFFSET + 2

LINE_COLOR_FRAME_X_OFFSET = BG_COLOR_FRAME_X_OFFSET + 52
LINE_COLOR_FRAME_Y_OFFSET = ACTION_ROW_Y_OFFSET + 2

FILL_WIDTH_FRAME_WIDTH = 72
FILL_WIDTH_FRAME_X_OFFSET = LINE_COLOR_FRAME_X_OFFSET + FILL_WIDTH_FRAME_WIDTH + 18
FILL_WIDTH_FRAME_Y_OFFSET = ACTION_ROW_Y_OFFSET

ALPHA_VALUES_FRAME_WIDTH = 160
ALPHA_VALUES_FRAME_X_OFFSET = FILL_WIDTH_FRAME_X_OFFSET + ALPHA_VALUES_FRAME_WIDTH
ALPHA_VALUES_FRAME_Y_OFFSET = ACTION_ROW_Y_OFFSET

DRAW_POLYGON_FRAME_X_OFFSET = ALPHA_VALUES_FRAME_X_OFFSET + BUTTON_FRAME_SIDE_LENGTH
DRAW_POLYGON_FRAME_Y_OFFSET = ACTION_ROW_Y_OFFSET

DRAW_LINE_FRAME_X_OFFSET = DRAW_POLYGON_FRAME_X_OFFSET + BUTTON_FRAME_SIDE_LENGTH
DRAW_LINE_FRAME_Y_OFFSET = ACTION_ROW_Y_OFFSET

SELECT_TILE_FRAME_X_OFFSET = DRAW_LINE_FRAME_X_OFFSET + BUTTON_FRAME_SIDE_LENGTH
SELECT_TILE_FRAME_Y_OFFSET = ACTION_ROW_Y_OFFSET

SHOW_HIDE_FRAME_X_OFFSET = 64
SHOW_HIDE_FRAME_Y_OFFSET = BOTTOM_ROW_Y_OFFSET

VIEWER_DIMENSIONS_FRAME_WIDTH = 160
VIEWER_DIMENSIONS_FRAME_X_OFFSET = SHOW_HIDE_FRAME_X_OFFSET + VIEWER_DIMENSIONS_FRAME_WIDTH
VIEWER_DIMENSIONS_FRAME_Y_OFFSET = BOTTOM_ROW_Y_OFFSET

CURSOR_LOCATION_FRAME_WIDTH = 160
CURSOR_LOCATION_FRAME_X_OFFSET = VIEWER_DIMENSIONS_FRAME_X_OFFSET + CURSOR_LOCATION_FRAME_WIDTH
CURSOR_LOCATION_FRAME_Y_OFFSET = BOTTOM_ROW_Y_OFFSET

DISPLAY_TYPE_FRAME_WIDTH = 800
DISPLAY_TYPE_FRAME_X_OFFSET = CURSOR_LOCATION_FRAME_X_OFFSET + DISPLAY_TYPE_FRAME_WIDTH
DISPLAY_TYPE_FRAME_Y_OFFSET = BOTTOM_ROW_Y_OFFSET

PLAY_STOP_FRAME_X_OFFSET = 10
PLAY_STOP_FRAME_Y_OFFSET = BOTTOM_ROW_Y_OFFSET