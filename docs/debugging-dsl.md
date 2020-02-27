# Logging and Debugging
Refer to the GStreamer [Tutorial on Debugging tools](https://gstreamer.freedesktop.org/documentation/tutorials/basic/debugging-tools.html?gi-language=c) for more information.

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
$ export GST_DEBUG=1,DSL:3
```

## Creating Pipeline Graphs
DSL takes advantage of GStreamer's capability to output graph files. These are `.dot` files, readable with 
free programs like GraphViz. Pipeline Graphs describe the topology of your DSL pipeline, along with the 
caps negotiated in each link. 

GStreamer creates the information when a Pipeline transition into states of `GST_STATE_READY`, `GST_STATE_PAUSED` and `GST_STATE_PLAYING` when the following environment variable is set.

```
$ export GST_DEBUG_DUMP_DOT_DIR=./.dot
```
```
$ mkdir -p $GST_DEBUG_DUMP_DOT_DIR
```
The DSL Pipeline API provides two services [dsl_pipeline_dump_to_dot](/docs/api-pipeline.md#dsl_pipeline_dump_to_dot) and [dsl_pipeline_dump_to_dot_with_ts](/docs/api-pipeline.md#dsl_pipeline_dump_to_dot_with_ts) to dump the informatin to file, best called from a [dsl_state_change_listener_cb](/docs/api-pipeline.md#dsl_state_change_listener_cb)

after building and running a pipeline you'll find `.dot` files under the ./.dot directory with the format of
```
    <timestamp>-pipeline-ready.dot
    <timestamp>-pipeline-playing.dot
```
Use the following command to convert a `.dot.` file to a `.png` file for viewing.
```
$ dot -Tpng 0.00.13.747142702-pipeline-ready.dot > pipeline.png
```
