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

import tkinter as tk


class Line():
    def __init__(self, x1, y1, x2, y2, line_width, line_color):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.line_width = line_width
        self.line_color = line_color.__dict__

class LineParams():
    def __init__(self, x1, y1, x2, y2, line_width, line_color):
        self.line = Line(x1, y1, x2, y2, line_width, line_color)
        self.line_color_obj = line_color

    def get_dict(self):
        return self.line.__dict__

    def copy(self, line_params):
        line_params.x1 = self.line.x1
        line_params.y1 = self.line.y1
        line_params.x2 = self.line.x2
        line_params.y2 = self.line.y2
        line_params.line_width = self.line.line_width
        line_params.line_color.red = self.line_color_obj.red
        line_params.line_color.green = self.line_color_obj.green
        line_params.line_color.blue = self.line_color_obj.blue
        line_params.line_color.alpha = self.line_color_obj.alpha
        return line_params