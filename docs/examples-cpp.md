# C/C++ Examples
The following C/C++ examples are available under the `/examples/cpp`. The same examples exist for Python under `/examples/python`

### [1file_ptis_ktl_osd_window.cpp](/examples/cpp/1file_ptis_ktl_osd_window.cpp)
A simple pipeline with a single `File Source`, `Primary Triton Inference Server`, `KTL Tracker`, `On-Screen Display`, and `Window Sink`. The example adds a number of Client callbacks to the Pipeline to handle `XWindow KeyRelease`, `XWindow Delete`, `Pipeline state change`, and `End-of-Stream (EOS)`.

### [ode_occurrence_4rtsp_start_record_tap_action.cpp](/examples/cpp/ode_occurrence_4rtsp_start_record_tap_action.cpp)
A more complete example with a variable number of `RTSP Sources`, each with a `Record Tap` for smart recording. The Pipeline is assembled with a `Primary GST Inference Engine`, `IOU Tracker`, `2D Tiler`, `On-Screen Display`, `Window Sink`, and most importantly a `Object Detection Event (ODE) PPH Handler`. The following `ODE Triggers` with `ODE Actions` are added to the ODE Handler for each RTSP Source:
* `Instance Trigger` - to trigger on new instances of a Person with a unique tracking id with a limit of one.
  * `Start Recording Action` - to start a new smart recording session.
* `Always Trigger` - to trigger on every frame when enabled (enabled/disabled in an `On Recording Event` callback)
  * `Display Meta Action` - to add a recording in progress indicator
    * `Display Types` - used as a `REC ON` indicator

The example adds a number of Client callbacks to the Pipeline to handle `XWindow KeyRelease`, `XWindow Delete`, `Pipeline state change`, and `End-of-Stream (EOS)`.

### [player_play_all_mp4_files_found.cpp](/examples/cpp/player_play_all_mp4_files_found.cpp)
A simple example of how to use a `Video Render Player`, a type of simplified GST Pipeline with a single `File Source` and `Window Sink`, to play all recorded files in a specified directory.

### [rtsp_player_to_test_connections.cpp](/examples/cpp/rtsp_player_to_test_connections.cpp)
A simple `Player` with an `RTSP Source` and `Window Sink` that can be very useful for testing your RTSP URI's without building a pipeline.

---

In addition to the above examples, all DSL API testing is done against the extern C API. The following test files may be useful as examples.
* [DslPipelinePlayComponentsTest.cpp](/test/api/DslPipelinePlayComponentsTest.cpp)
* [DslOdeBehaviorTest.cpp](/test/api/DslOdeBehaviorTest.cpp)
* [DslPipelinePlayTrackerTest.cpp](/test/api/DslPipelinePlayTrackerTest.cpp)
