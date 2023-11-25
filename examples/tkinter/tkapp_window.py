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
from tkinter import colorchooser
from PIL import Image, ImageTk
import threading

from config import *
from color import ColorParams, RgbColorButton
from action import Action, ToggleAction
from display_type import LineParams

##
# Thread to block on the DSL/GStreamer Main Loop
##
def thread_loop():
    try:
        dsl_main_loop_run()
    except:
        pass

##
# MetaData class for creating a display type. A single Object
# that will be accessed by multiple threads. Pipeline and TK APP
##
class MetaData():
    def __init__(self):
        self.mutex = threading.Lock()
        self.active_display_type = None

##
# Application Window Class
##
class AppWindow(tk.Frame):
    ##
    # Ctor for this Application Window
    ##
    def __init__(self, master, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        
        # shared meta_data object - shared between OSD Custom PPH and draw actions
        self.meta_data = MetaData()

        # Quick access properties for window width and height
        for key, value in kwargs.items():
            if key == 'width':
                self.width = value
            elif key == 'height':
                self.height = value
        try:
            self.width and self.height
        except NameError:
            raise ValueError('width and height parameters are required')

        self.sink_window_frame = tk.Frame(self, bg='black', 
            width=WINDOW_WIDTH, height=WINDOW_HEIGHT, bd=0)

        left = (self.width - WINDOW_WIDTH) / 2
        top = (self.height - WINDOW_HEIGHT) / 2
        
        self.sink_window_frame.place(x=left, y=top)

        # 
        # Setup the show/hide all toggle button
        #
        self.show_hide_frame = tk.Frame(self)
        self.show_hide = ToggleAction(self.show_hide_frame, 
            "./images/hide.png", "./images/show.png", self.show_hide_all)
        self.show_hide_frame.place(
            x = self.width - SHOW_HIDE_FRAME_X_OFFSET, 
            y = self.height - SHOW_HIDE_FRAME_Y_OFFSET)

        # 
        # Setup the play/stop pipeline toggle button
        #
        self.play_stop_frame = tk.Frame(self)
        self.play_stop = ToggleAction(self.play_stop_frame, 
            "./images/play.png", "./images/stop.png", self.play_stop_pipeline)

        # 
        # Setup the select tile action
        #
        self.select_tile_frame = tk.Frame(self)
        self.select_tile = Action(self.select_tile_frame,
            "./images/select.png", self.select_tile)

        # 
        # Setup the draw line action
        #
        self.draw_line_frame = tk.Frame(self)
        self.draw_line_action = Action(self.draw_line_frame,
            "./images/line.png", self.draw_line)
            
        # 
        # Setup the draw polygon action
        #
        self.draw_polygon_frame = tk.Frame(self)
        self.draw_polygon_action = Action(self.draw_polygon_frame,
            "./images/polygon.png", self.draw_polygon)

        # 
        # Setup the color picker action button
        #
        self.color_picker_frame = tk.Frame(self)
        self.color_picker = Action(self.color_picker_frame,
            "./images/color-picker.png", self.pick_color)

        #
        # Create the frame for the top row of color buttons and
        #
        self.colors_row_1 = tk.Frame(self)
        
        # Create all top row color buttons and add to top row frame
        self.black = RgbColorButton(self.colors_row_1, 
            name='Black', red=0, green= 0, blue=0, 
            on_select=self.on_color_select)

        self.gray_50 = RgbColorButton(self.colors_row_1, 
            name='Gray-50%', red=127, green= 127, blue=127, 
            on_select=self.on_color_select)
            
        self.dark_red = RgbColorButton(self.colors_row_1, 
            name='Dark-red', red=136, green= 0, blue=21, 
            on_select=self.on_color_select)
            
        self.red = RgbColorButton(self.colors_row_1, 
            name='Red', red=237, green= 28, blue=36, 
            on_select=self.on_color_select)
            
        self.orange = RgbColorButton(self.colors_row_1, 
            name='Orange', red=255, green= 127, blue=39, 
            on_select=self.on_color_select)

        self.yellow = RgbColorButton(self.colors_row_1, 
            name='Yellow', red=255, green=242, blue=0, 
            on_select=self.on_color_select)

        self.green = RgbColorButton(self.colors_row_1, 
            name='Green', red=34, green=177, blue=76, 
            on_select=self.on_color_select)

        self.turquoise = RgbColorButton(self.colors_row_1, 
            name='Turquoise', red=0, green=162, blue=232, 
            on_select=self.on_color_select)

        self.indigo = RgbColorButton(self.colors_row_1, 
            name='Indigo', red=63, green=72, blue=204, 
            on_select=self.on_color_select)

        self.purple = RgbColorButton(self.colors_row_1, 
            name='Purple', red=163, green=73, blue=164, 
            on_select=self.on_color_select)

        #
        # Create the frame for the bottom row of color buttons and 
        #
        self.colors_row_2 = tk.Frame(self)
        
        #
        # Create all top row color buttons and add to top row frame
        #
        self.white = RgbColorButton(self.colors_row_2, 
            name='White', red=255, green=255, blue=255, 
            on_select=self.on_color_select)

        self.gray_25 = RgbColorButton(self.colors_row_2, 
            name='Gray-25', red=195, green=195, blue=195, 
            on_select=self.on_color_select)

        self.brown = RgbColorButton(self.colors_row_2, 
            name='Brown', red=185, green=122, blue=87, 
            on_select=self.on_color_select)

        self.rose = RgbColorButton(self.colors_row_2, 
            name='Rose', red=255, green=174, blue=201, 
            on_select=self.on_color_select)

        self.gold = RgbColorButton(self.colors_row_2, 
            name='Gold', red=255, green=201, blue=14, 
            on_select=self.on_color_select)

        self.light_yellow = RgbColorButton(self.colors_row_2, 
            name='Light yellow', red=239, green=228, blue=176, 
            on_select=self.on_color_select)

        self.lime = RgbColorButton(self.colors_row_2, 
            name='Lime', red=181, green= 230, blue=29, 
            on_select=self.on_color_select)

        self.light_turquoise = RgbColorButton(self.colors_row_2, 
            name='Light turquoise', red=153, green= 217, blue=234, 
            on_select=self.on_color_select)

        self.blue_gray = RgbColorButton(self.colors_row_2, 
            name='Blue-gray', red=112, green=146, blue=190, 
            on_select=self.on_color_select)

        self.lavender = RgbColorButton(self.colors_row_2, 
            name='Lavender', red=200, green=191, blue=231, 
            on_select=self.on_color_select)

        #
        # Load the trasparency (checkered) background image
        #
        self.transparency_image = tk.PhotoImage(file='./images/transparency.png')

        #
        # New Canvas for the Line Color - with transparency background
        # Button release is bound to set_line_color_active method
        #
        self.line_color_canvas = tk.Canvas(self, bg=self.black.get_rgb_str(),
            highlightbackground='gray80', width=BUTTON_SIDE_LENGTH, height=BUTTON_SIDE_LENGTH)
        self.line_color_canvas.bind("<ButtonRelease-1>", self.set_line_color_active)
        self.line_color_canvas.create_image(1,1, image=self.transparency_image, anchor=tk.NW)
        self.line_color_canvas.color = self.black
        self.line_color_canvas.alpha = tk.IntVar(value=50)
        self.overlay_alpha_color(self.line_color_canvas, x1=1, y1=1, x2=50, y2=50)
        self.active_color_canvas = self.line_color_canvas
        
        #
        # New Canvas for the Background Color - with transparency background
        # Button release is bound to set_bg_color_active method
        #
        self.bg_color_canvas = tk.Canvas(self, bg=self.white.get_rgb_str(), 
            highlightbackground='gray80', width=BUTTON_SIDE_LENGTH, height=BUTTON_SIDE_LENGTH)
        self.bg_color_canvas.bind("<ButtonRelease-1>", self.set_bg_color_active)
        self.bg_color_canvas.create_image(1,1, image=self.transparency_image, anchor=tk.NW)
        self.bg_color_canvas.color = self.white
        self.bg_color_canvas.alpha = tk.IntVar(value=50)
        self.overlay_alpha_color(self.bg_color_canvas, x1=1, y1=1, x2=50, y2=50)

        #
        # Setup the fill enabled and line width frame and add image and entry boxes
        #
        self.fill_width_frame = tk.Frame(self, bg='white', relief=tk.FLAT)
        self.fill_width_canvas = tk.Canvas(self.fill_width_frame, bg='white', 
            width=FILL_WIDTH_FRAME_WIDTH, height=52)
        self.fill_width_canvas.pack()
        self.fill_width_image = tk.PhotoImage(file='./images/fill-size.png')
        self.fill_width_canvas.create_image(4,4, image=self.fill_width_image, anchor=tk.NW)
        self.has_bg_color = tk.IntVar(value=True)
        self.has_bg_color_cb = tk.Checkbutton(self.fill_width_canvas, bg='white', width=1,
            relief=tk.FLAT, variable=self.has_bg_color, highlightthickness=0, font=('bold',10))
        self.has_bg_color_cb.place(x=40, y=8)
        self.line_width = tk.IntVar(value=4)
        self.line_width_entry = tk.Entry(self.fill_width_canvas, bg='white', bd=0, width=2, 
            textvariable=self.line_width, relief=tk.FLAT, highlightthickness=0)        
        self.line_width_entry.place(x=50, y=28)

        #
        # Setup the display-type results frame
        #
        self.display_type_frame = tk.Frame(self, bg='white', relief=tk.FLAT)
        self.display_type_canvas = tk.Canvas(self.display_type_frame, bg='white', 
            width=DISPLAY_TYPE_FRAME_WIDTH, height=BUTTON_FRAME_SIDE_LENGTH)
        self.display_type_canvas.pack()
        self.display_type_params1 = tk.Label(
            self.display_type_canvas, bg='white', bd=0, text='', 
                width=90, font=('bold',10), anchor=tk.W)
        self.display_type_params1.place(x=4, y=12)
        self.display_type_params2 = tk.Label(
            self.display_type_canvas, bg='white', bd=0, text='', 
                width=90, font=('bold',10), anchor=tk.W)
        self.display_type_params2.place(x=4, y=30)

        #
        # Setup the curson location frame with cursor image, and label for location
        #
        self.cursor_location_frame = tk.Frame(self, bg='white', relief=tk.FLAT)
        self.cursor_location_canvas = tk.Canvas(self.cursor_location_frame, bg='white', 
            width=CURSOR_LOCATION_FRAME_WIDTH, height=BUTTON_FRAME_SIDE_LENGTH)
        self.cursor_location_canvas.pack()
        self.cursor_location_image = tk.PhotoImage(file='./images/cross-cursor.png')
        self.cursor_location_canvas.create_image(4,4, 
            image=self.cursor_location_image, anchor=tk.NW)
        self.cursor_location_label_start = tk.Label(
            self.cursor_location_canvas, bg='white', bd=0, text='', 
                width=13, font=('bold',10), anchor=tk.W)
        self.cursor_location_label_start.place(x=50, y=12)
        self.cursor_location_label_end = tk.Label(
            self.cursor_location_canvas, bg='white', bd=0, text='', 
                width=13, font=('bold',10), anchor=tk.W)
        self.cursor_location_label_end.place(x=50, y=30)

        #
        # Setup the window and selection dimensions frame 
        #
        self.view_dimensions_frame = tk.Frame(self, bg='white', relief=tk.FLAT)
        self.view_dimensions_canvas = tk.Canvas(self.view_dimensions_frame, bg='white', 
            width=VIEWER_DIMENSIONS_FRAME_WIDTH, height=BUTTON_FRAME_SIDE_LENGTH)
        self.view_dimensions_canvas.pack()
        self.view_dimensions_image = tk.PhotoImage(file='./images/source-dimensions.png')
        self.view_dimensions_canvas.create_image(4,4, image=self.view_dimensions_image, anchor=tk.NW)
        self.view_dimensions_label = tk.Label(self.view_dimensions_canvas, bg='white', bd=0, 
                text='{width} x {height} px'.format(width=self.width, height=self.height), 
                width=13, font=('bold',10), anchor=tk.W)
        self.view_dimensions_label.place(x=50, y=12)
        self.select_dimensions_label = tk.Label(
            self.view_dimensions_canvas, bg='white', bd=0, text='', 
                width=13, font=('bold',10), anchor=tk.W)
        self.select_dimensions_label.place(x=50, y=30)
        
        #
        # Setup the alpha levels frame with cursor image, and label for location
        #
        self.alpha_values_frame = tk.Frame(self, bg='white', relief=tk.FLAT)
        self.alpha_values_canvas = tk.Canvas(self.alpha_values_frame, bg='white', 
            width=ALPHA_VALUES_FRAME_WIDTH, height=52)
        self.alpha_values_canvas.pack()
        self.alpha_values_image = tk.PhotoImage(file='./images/alpha.png')
        self.alpha_values_canvas.create_image(4,4, image=self.alpha_values_image, anchor=tk.NW)

        self.color_1_alpha = tk.Scale(self.alpha_values_frame, from_=100, to=0, orient=tk.HORIZONTAL, 
            showvalue=False, sliderrelief=tk.FLAT, bg='grey90', variable=self.line_color_canvas.alpha, 
            command=self.on_line_color_alpha_update)
        self.color_1_alpha.place(x=50, y=8)
        
        self.color_2_alpha = tk.Scale(self.alpha_values_frame, from_=100, to=0, orient=tk.HORIZONTAL, 
            showvalue=False, sliderrelief=tk.FLAT, bg='grey90', variable=self.bg_color_canvas.alpha, 
            command=self.on_bg_color_alpha_update)
        self.color_2_alpha.place(x=50, y=24)

        # Thread to run and block on the DSL/GST main-loop
        self.dsl_main_loop_thread = None

        # ranging parameters for drawing display-types
        self.start_x = None
        self.start_y = None
        self.polygon_line_params = []
        
        self.placed_all = False;
        self.place_all()
        
    def get_sink_window(self):
        return self.sink_window_frame.winfo_id()

    def show_hide_all(self, toggle_state):
        if toggle_state:
            self.place_forget_all()
        else:
            self.place_all()

    def play_stop_pipeline(self, toggle_state):
        if toggle_state:
            # Thread to run and block on the DSL/GST main-loop
            self.dsl_main_loop_thread = threading.Thread(target=thread_loop, daemon=True)
            dsl_pipeline_play(PIPELINE)
            self.dsl_main_loop_thread.start()

        else:
            dsl_pipeline_stop(PIPELINE)
            # this is a hack to clear the window
            self.config(cursor="arrow")

            dsl_main_loop_quit()
            self.dsl_main_loop_thread.join()
            self.dsl_main_loop_thread = None

    def pick_color(self):
        print(colorchooser.askcolor(title ="Choose color"))

    def place_all(self):
        if not self.placed_all:
            self.color_picker_frame.place(
                x = self.width - COLOR_PICKER_FRAME_X_OFFSET,
                y = COLOR_PICKER_FRAME_Y_OFFSET)
            self.colors_row_1.place(
                x = self.width - COLORS_ROW_1_FRAME_X_OFFSET, 
                y = COLORS_ROW_1_FRAME_Y_OFFSET)
            self.colors_row_2.place(
                x = self.width - COLORS_ROW_2_FRAME_X_OFFSET, 
                y = COLORS_ROW_2_FRAME_Y_OFFSET)
            self.line_color_canvas.place(
                x = self.width - LINE_COLOR_FRAME_X_OFFSET, 
                y = LINE_COLOR_FRAME_Y_OFFSET)
            self.bg_color_canvas.place(
                x = self.width - BG_COLOR_FRAME_X_OFFSET, 
                y = BG_COLOR_FRAME_Y_OFFSET)
            self.fill_width_frame.place(
                x = self.width - FILL_WIDTH_FRAME_X_OFFSET,
                y = FILL_WIDTH_FRAME_Y_OFFSET)
            self.alpha_values_frame.place(
                x = self.width - ALPHA_VALUES_FRAME_X_OFFSET,
                y = ALPHA_VALUES_FRAME_Y_OFFSET)
            self.draw_polygon_frame.place(
                x = self.width - DRAW_POLYGON_FRAME_X_OFFSET, 
                y = DRAW_POLYGON_FRAME_Y_OFFSET)
            self.draw_line_frame.place(
                x = self.width - DRAW_LINE_FRAME_X_OFFSET, 
                y = DRAW_LINE_FRAME_Y_OFFSET)
            self.select_tile_frame.place(
                x = self.width - SELECT_TILE_FRAME_X_OFFSET, 
                y = SELECT_TILE_FRAME_Y_OFFSET)

            self.cursor_location_frame.place(
                x = self.width - CURSOR_LOCATION_FRAME_X_OFFSET, 
                y = self.height - CURSOR_LOCATION_FRAME_Y_OFFSET)
            self.view_dimensions_frame.place(
                x = self.width - VIEWER_DIMENSIONS_FRAME_X_OFFSET,
                y = self.height - VIEWER_DIMENSIONS_FRAME_Y_OFFSET)
            self.display_type_frame.place(
                x = self.width - DISPLAY_TYPE_FRAME_X_OFFSET,
                y = self.height - DISPLAY_TYPE_FRAME_Y_OFFSET)
            
            # From lower left corner
            self.play_stop_frame.place(
                x = PLAY_STOP_FRAME_X_OFFSET, 
                y = self.height - PLAY_STOP_FRAME_Y_OFFSET)
                
        
            self.placed_all = True

    def place_forget_all(self):
        if self.placed_all:
            self.color_picker_frame.place_forget()
            self.colors_row_1.place_forget()
            self.colors_row_2.place_forget()
            self.line_color_canvas.place_forget()
            self.bg_color_canvas.place_forget()
            self.draw_polygon_frame.place_forget()
            self.select_tile_frame.place_forget()
            self.draw_line_frame.place_forget()
            self.alpha_values_frame.place_forget()
            self.display_type_frame.place_forget()
            self.cursor_location_frame.place_forget()
            self.view_dimensions_frame.place_forget()
            self.fill_width_frame.place_forget()
            self.play_stop_frame.place_forget()
            
            self.placed_all = False

    ##
    # Button release event function to set the Active Color to the 
    # Line color
    ##
    def set_line_color_active(self, e):
        self.active_color_canvas = self.line_color_canvas
        
    ##
    # Button release event function to set the Active Color to the Background color
    ##
    def set_bg_color_active(self, e):
        self.active_color_canvas = self.bg_color_canvas

    ##
    # Called when the user clicks on a color botton
    ##
    def on_color_select(self, color):
        self.active_color_canvas.color = color
        self.overlay_alpha_color(self.active_color_canvas, x1=1, y1=1, x2=50, y2=505)

    ##
    # Called on Alpha slider update-event for the primary line color
    ##
    def on_line_color_alpha_update(self, e):
        self.overlay_alpha_color(self.line_color_canvas, x1=1, y1=1, x2=50, y2=50)

    ##
    # Called on Alpha slider update-event for the secondary backround color
    ##
    def on_bg_color_alpha_update(self, e):
        self.overlay_alpha_color(self.bg_color_canvas, x1=1, y1=1, x2=50, y2=50)


    ##
    # Overlays an RGBA color over a checkered transparency canvas
    ##
    def overlay_alpha_color(self, canvas, x1, y1, x2, y2):
        alpha = int(canvas.alpha.get() * 255 * 0.01)
        fill = canvas.color.get_rgb() + (alpha,)
        image = Image.new('RGBA', (x2-x1, y2-y1), fill)
        canvas.image = ImageTk.PhotoImage(image)
        canvas.create_image(x1, y1, image=canvas.image, anchor='nw')

    ##
    # Called on cursor motion when in drawing mode
    ##
    def on_cursor_motion(self, e):
        self.cursor_location_label_start.config(
            text='{}, {} px'.format(e.x,e.y))

    def select_tile(self):
        self.on_draw_action('Select')

    def draw_line(self):
        self.on_draw_action('Line')
        
    def draw_polygon(self):
        self.on_draw_action('Polygon')
        
    def on_draw_action(self, display_type):
    
        # unbind existing events
        self.sink_window_frame.unbind("<B1-Motion>")
        self.sink_window_frame.unbind("<ButtonRelease-1>")
        self.sink_window_frame.unbind("<Button-1>")
        
        # if called by the select-tile action, clear the display location,
        # unbind the currsor motion, bind the button click, and exit.
        if display_type == 'Select':
            self.cursor_location_label_start.config(text='')
            self.cursor_location_label_end.config(text='')
            self.sink_window_frame.unbind('<Motion>')
            self.sink_window_frame.config(cursor="arrow")
            self.sink_window_frame.bind("<Button-1>", self.select_tile)
            return
            
        # Else, bind the currsor motion and change to the cross-hair currsor
        self.sink_window_frame.config(cursor="cross")
        self.sink_window_frame.bind('<Motion>', self.on_cursor_motion)
        
        # Bind the correct ranging function based on display type
        if display_type == 'Line':
            self.sink_window_frame.bind("<B1-Motion>", self.line_ranging)
        elif display_type == 'Polygon':
            self.sink_window_frame.bind("<B1-Motion>", self.polygon_ranging)
        else:
            print('Unknown display_type for on_draw_action')


    def line_ranging(self, e):
        if self.start_x and self.start_y:
            if e.x > self.start_x:
                width = e.x - self.start_x
            else:
                width = self.start_x - e.x
            if e.y > self.start_y:
                height = e.y - self.start_y
            else:
                height = self.start_y - e.y
                
            x1 = self.start_x
            y1 = self.start_y
            x2 = e.x
            y2 = e.y
            
            line_color = ColorParams(
                red=self.line_color_canvas.color.red_p, 
                green=self.line_color_canvas.color.green_p, 
                blue=self.line_color_canvas.color.blue_p, 
                alpha=self.line_color_canvas.alpha.get()*0.01)
            line_params = LineParams(x1=x1, y1=y1, x2=x2, y2=y2, line_width=self.line_width.get(),
                line_color=line_color)
                
            self.meta_data.mutex.acquire()
            self.meta_data.active_display_type = {'Line': line_params}
            self.meta_data.mutex.release()

            self.select_dimensions_label.config(text='{} x {} px'.format(width,height))
            self.cursor_location_label_end.config(
                text='{}, {} px'.format(e.x, e.y))
        else:
            self.start_x = e.x
            self.start_y = e.y

        def line_make(e):
            self.line_ranging(e)
            x1 = self.start_x
            y1 = self.start_y
            x2 = e.x
            y2 = e.y
            self.display_type_params1.config(text='x1={}, y1={}, x2={}, y2={}'.format(x1,y1,x2,y2))
            line_color = ColorParams(
                red=self.line_color_canvas.color.red_p, 
                green=self.line_color_canvas.color.green_p, 
                blue=self.line_color_canvas.color.blue_p, 
                alpha=self.line_color_canvas.alpha.get()*0.01)
            self.display_type_params2.config(text='width={}, color={}'.format(
                self.line_width.get(), line_color.get_rgb_str()))
            self.sink_window_frame.bind('<Motion>', self.on_cursor_motion)
            self.select_dimensions_label.config(text='')
            self.cursor_location_label_end.config(text='')
            
            self.start_x = None
            self.start_y = None
            
        self.sink_window_frame.bind('<ButtonRelease-1>',line_make)

    def polygon_ranging(self, e):
        if self.start_x and self.start_y:
            if e.x > self.start_x:
                width = e.x - self.start_x
            else:
                width = self.start_x - e.x
            if e.y > self.start_y:
                height = e.y - self.start_y
            else:
                height = self.start_y - e.y
                
            x1 = self.start_x
            y1 = self.start_y
            x2 = e.x
            y2 = e.y
            
            line_color = ColorParams(
                red=self.line_color_canvas.color.red_p, 
                green=self.line_color_canvas.color.green_p, 
                blue=self.line_color_canvas.color.blue_p, 
                alpha=self.line_color_canvas.alpha.get()*0.01)
            self.polygon_line_params[len(self.polygon_line_params)-1] = \
                LineParams(x1=x1, y1=y1, x2=x2, y2=y2, line_width=self.line_width.get(),
                    line_color=line_color)
                
            self.meta_data.mutex.acquire()
            self.meta_data.active_display_type = {'Polygon': self.polygon_line_params}
            self.meta_data.mutex.release()

            self.select_dimensions_label.config(text='{} x {} px'.format(width,height))
            self.cursor_location_label_end.config(
                text='{}, {} px'.format(round(e.x), round(e.y)))
        else:
            self.start_x = e.x
            self.start_y = e.y
            self.polygon_line_params.clear()
            self.polygon_line_params.append(0)

        def polygon_next(e):
            self.polygon_ranging(e)
            self.polygon_line_params.append(0)
                
            self.start_x = e.x
            self.start_y = e.y

        def polygon_make(e):
            self.start_x = self.polygon_line_params[len(self.polygon_line_params)-2].line.x2
            self.start_y = self.polygon_line_params[len(self.polygon_line_params)-2].line.y2
            e.x = self.polygon_line_params[0].line.x1
            e.y = self.polygon_line_params[0].line.y1
            self.polygon_ranging(e)
            coordinates_string = ''
            self.coordinates_list = []
            for lineObj in self.polygon_line_params:
                coordinates_string += '(' + str(lineObj.line.x1) + ',' + str(lineObj.line.y1) + '),'
            
            self.display_type_params1.config(text='coordinates={}'.format(coordinates_string))
            line_color = ColorParams(
                red=self.line_color_canvas.color.red_p, 
                green=self.line_color_canvas.color.green_p, 
                blue=self.line_color_canvas.color.blue_p, 
                alpha=self.line_color_canvas.alpha.get()*0.01)
            self.display_type_params2.config(text='width={}, color={}'.format(
                self.line_width.get(), line_color.get_rgb_str()))
            self.sink_window_frame.unbind('<ButtonRelease-1>')
            self.sink_window_frame.unbind('<ButtonRelease-3>')
            
            self.sink_window_frame.bind('<Motion>', self.on_cursor_motion)
            self.select_dimensions_label.config(text='')
            self.cursor_location_label_end.config(text='')
            self.start_x = None
            self.start_y = None

        self.sink_window_frame.bind('<ButtonRelease-1>', polygon_next)
        self.sink_window_frame.bind('<ButtonRelease-3>', polygon_make)
