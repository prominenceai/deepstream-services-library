# Logging and Debugging
Refer to the GStreamer [Tutorial on Debugging tools](https://gstreamer.freedesktop.org/documentation/tutorials/basic/debugging-tools.html?gi-language=c) for more information.

## Building DSL as an app with a main function
NOTE: this is the current default during initial construction

Invoke the make file with the `app` option to change the build type from static lib to 
application. `main/DslMain.cpp` will be added to the list of SRCS.
```
$ make app
```
The build will generate a console application in the current directory with name `dsl-app`

## Setting the Debug message level
DSL uses GStreamer's built-in logging with a single `GST_CATEGORY` labeled `DSL`.
The environment variable, GST_DEBUG, controls the current logging level for both GStreamer and DSL. 
The below example sets a default level of `WARNING=2` for all categories. 
```
$ export GST_DEBUG=2
```
Log level values: `NONE=0`,`ERROR=1`, `WARNING=2`, `INFO=3`, `FIXME=4`, `DEBUG=5`

Levels are "lower-level-inclusive" with `WARNING=2` including `ERROR=1` and so forth. 
DSL, all GStreamer modules, and plugins define their own unique category allowing log levels to be set for each.
```
$ export GST_DEBUG=2,DSL:3
```
The above example sets a default level of `WARNING=2`, with the `DSL` category set to level `INFO=3`

Example console output with the above setting
```
0:00:00.105501773  3721 0x556be5daa0 INFO   DSL src/DslBintr.h:58:Bintr:  : New bintr:: gie1
0:00:00.227679866  3721 0x556be5daa0 INFO   DSL src/DslBintr.h:162:MakeElement:  : Successfully linked new element primary_gie_conv for gie1
0:00:00.937156235  3721 0x556be5daa0 INFO   DSL src/DslBintr.h:162:MakeElement:  : Successfully linked new element primary_gie_classifier for gie1
0:00:00.939783162  3721 0x556be5daa0 INFO   DSL src/DslServices.cpp:384:GieNew:  : new GIE 'gie1' created successfully
```

## Creating Pipeline Graphs
DSL takes advantage of GStreamer's capability to output graph files. These are `.dot` files, readable with 
free programs like GraphViz. Pipeline Graphs describe the topology of your DSL pipeline, along with the 
caps negotiated in each link. 

DSL calls `GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS()` on transition to `GST_STATE_READY` and `GST_STATE_PLAYING`. 
The macro generates `.dot` files under `GST_DEBUG_DUMP_DOT_DIR` when the environment variable is set to a valid directory. 

```
$ export GST_DEBUG_DUMP_DOT_DIR=./.dot
```
```
$ mkdir -p $GST_DEBUG_DUMP_DOT_DIR
```
after building and running a pipeline you'll find `.dot` files under the ./.dot directory with the format of
```
    <timestamp>-pipeline-ready.dot
    <timestamp>-pipeline-playing.dot
```
Use the following command to convert a `.dot.` file to a `.png` file for viewing.
```
$ dot -Tpng 0.00.13.747142702-pipeline-ready.dot > pipeline.png
```
