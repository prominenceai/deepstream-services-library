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

from config import BUTTON_SIDE_LENGTH

class Action():
    def __init__(self, frame, image_file, action,
        width=BUTTON_SIDE_LENGTH, height=BUTTON_SIDE_LENGTH):
        self.image = tk.PhotoImage(file=image_file)
        
        self.button = tk.Button(frame, image=self.image, bg='white', relief=tk.FLAT,
            width=width, height=height, 
            command=action)
            
        self.pack()

    def pack(self):
        self.button.pack(side=tk.LEFT)
    
    def pack_forget(self):
        self.button.pack_forget()

    def disable(self):
        self.button.config(state=tk.DISABLED)

    def enable(self):
        self.button.config(state=tk.NORMAL)
        

class ToggleAction():
    def __init__(self, frame, image_file1, image_file2, action,
        width=BUTTON_SIDE_LENGTH, height=BUTTON_SIDE_LENGTH):
        self.image1 = tk.PhotoImage(file=image_file1)
        self.image2 = tk.PhotoImage(file=image_file2)
        self.action = action
        
        self.toggle_state = 1;

        self.button = tk.Button(frame, image=self.image1, bg='white', relief=tk.FLAT,
            width=width, height=height, 
            command=self.command)
            
        self.pack()

    def pack(self):
        self.button.pack(side=tk.LEFT)
    
    def pack_forget(self):
        self.button.pack_forget()

    def disable(self):
        self.button.config(state=tk.DISABLED)

    def enable(self):
        self.button.config(state=tk.NORMAL)

    def command(self):
        self.action(self.toggle_state)
        self.toggle_state = not self.toggle_state
        if self.toggle_state:
            self.button.config(image=self.image1)
        else:
            self.button.config(image=self.image2)
