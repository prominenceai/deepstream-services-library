# Tkinter Reference Application
This repo includes a basic Tkinter python GUI application found [here](/examples/tkinter). The reference example is in an early stage and there are plans to expand the functionality over time. In its current state the application demonstrates:
* How to use an Application window with a [DSL Pipeline](/docs/api-pipeline.md) and [Window Sink](/docs/api-sink.md).
* How to draw [ODE Areas](/docs/api-ode-area.md) over a live stream using a [Custom Pad Probe Handler](/docs/api-pph.md) and [Display Types](/docs/api-display-type.md).

## Dependencies
The following dependencies are required to run the application:
* **python3-pip** and **python-dev**: `$ sudo apt-get install python3-pip python-dev`
* **pillow**: `$ pip3 install image`
* **DSL libdsl.so**: `$ make lib`
* [pyds](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases): [Installation](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/master/HOWTO.md)

## Configuration
All configurable parameters -- RTSP URI, component names, streammuxer and window dimensions, etc. -- are defined in the file [config.py](/examples/tkinter/config.py).

## Running the Application
When runing on Ubuntu, Tkinter applications need to execute at a slightly elevated priority. This requires the application to be launched with administration priviliges. 
```
$ sudo python3 tkapp.py
```

* To start the pipeline playing, select ![](/examples/tkinter/images/play.png)
* To stop a playing pipeline, select ![](/examples/tkinter/images/stop.png)
* To hide the controls from view, select ![](/examples/tkinter/images/hide.png)

This version is limited to polygons and lines, with plans to add rectangles, circles, and text in a future release.

**Note!** Polygons do not support fill color - requires support from NVIDIA.

### Controls

![](/Images/controls.png)

### Drawing Polygons
Select desired color, line-width, and alpha level, and then draw the polygon as follows:
1. Select the polygon button to switch into drawing mode.
2. Press the **left mouse button** at the desired location for the first coordinate.
3. Drag the mouse to the location of the next coordinate and press the **left mouse button** again.
4. Repeat step 4 until all coordinates have been drawn.
5. Select the **right mouse button** to complete/close the polygon shape.

![](/Images/partial-polygon.png)

Once complete, the polygon values will be displayed in the status bar at the bottom of the frame.
![](/Images/coordinates.png)

The coordinates can then be used when developing your own Python or C/C++ pipeline applications.
