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

##
# RGB Color Button Class
##
class RgbColorButton(tk.Button):
    ##
    # Ctor for the RGBA Color Button Class
    # master - master TK Object, ownership 
    # name - unique name for the Button
    # red - red value for this color between 0 and 255
    # green - green value for this color between 0 and 255
    # blue - blue value for this color between 0 and 255
    # on_select - function to call on Button Select
    ##
    def __init__(self, master, name, red, green, blue, on_select):
        super().__init__(master, relief=tk.FLAT, width=0, height=1, 
            font=('normal',6), command=self.select)

        self.name = name
        self.red = red
        self.green = green
        self.blue = blue
        self.on_select = on_select

        # Pack the button now. Master will controll visibility.
        self.pack(side=tk.LEFT)

        # Define the RGB string for setting TK colors, and configure the button color.
        self.rgb_str = '#%02x%02x%02x'%(red, green, blue)
        self.config(bg=self.rgb_str, activebackground=self.rgb_str)
        
        # Setup the point values [1..0]
        if red:
            self.red_p = red / 255
        else:
            self.red_p = 0
        if green:
            self.green_p = green / 255
        else:
            self.green_p = 0
        if blue:
            self.blue_p = blue / 255
        else:
            self.blue_p = 0
        
    ##
    # Function to select this button
    ##
    def select(self):
        self.on_select(self)
        
    ##
    # Return the TK RGB String for this Color Button
    ##
    def get_rgb_str(self):
        return self.rgb_str
        
    ##
    # Return the RGB values [0..255] for this Color Button
    ##
    def get_rgb(self):
        return self.red, self.green, self.blue