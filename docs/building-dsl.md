# Building and Importing DSL

The DSL shared library is built using GCC and GNU Make. The MakeFile is located in the DSL root folder. There are a few simple steps involved in creating and installing the shared library, `libdsl.so`. The current default `make` -- prior to releasing DSL v1.0 -- builds DSL into a command line test-executable for running all API and Unit level test cases. The DSL source-only object files, built for the test application, are re-linked into the `libdsl.so` shared library.

## Contents / Steps
1. Clone this repository to pull down all source
2. Use the make (all) default to build the `dsl-test-app` executable
3. Use the `sudo make install` option to build the object files into `libdsl.so` and intall the lib
4. Generate caffemodel engine files (optional)
5. Import the shared lib using Python3
6. Run the `dsl-test-app` to verify DSL changes (optional)

### Clone the Repository
Clone the repository to pull all source code required to build the DSL test application - then navigate to the `deepstream-services-library` root folder.
```
git clone https://github.com/prominenceai/deepstream-services-library
cd deepstream-services-library
```

***Import Note: When building with GStreamer v1.18 on Ubuntu 20.04, you will need to set the GStreamer sub version in the Makefile to 18 for the WebRTC Sink and WebSocket source code to be included in the build.***

### Make (all)
Invoke the standard make (all) to compile all source code and test scenarios into objects files, and link them into a [Catch2](https://github.com/catchorg/Catch2) test application. On successful build, the `dsl-test-app` will be found under the same root folder.

```bash
make -j <num-cores>
```
For example:
```bash
make -j 4
```
to use 4 CPU cores for the parallel.

### Make and install the shared library and install dsl.py
Once the dsl-test-app.exe has been created, the source-only objects can be re-linked into the `libdsl.so` with the make install option. Root level privileges are required for the Makefile to copy the lib to `/usr/local/lib` once built. 

The DSL Python bindings file (dsl.py) is also copied to `.local/lib/python3.6/site-packages` with the following command.

```
sudo make install
```

### Generate caffemodel engine files (optional)
Enable DSL logging if you wish to monitor the process (optional).
```bash
export GST_DEBUG=1,DSL:4
```
execute the python script in the `deepstream-services-library` root folder.
```bash
python3 make_caffemodel_engine_files.py
```
**Note:** this script can take several minutes to run.

The following files are generated (Jetson Nano versions by default)
```
/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine
/opt/nvidia/deepstream/deepstream/samples/models/Secondary_CarColor/resnet18.caffemodel_b8_gpu0_fp16.engine
/opt/nvidia/deepstream/deepstream/samples/models/Secondary_CarMake/resnet18.caffemodel_b8_gpu0_fp16.engine
/opt/nvidia/deepstream/deepstream/samples/models/Secondary_VehicleTypesresnet18.caffemodel_b8_gpu0_fp16.engine
```
Update the Primary detector path specification in the script to generate files for other devices.


### Import the shared lib using Python3
The shared lib `libdsl.so` is mapped to Python3 using [CTypes](https://docs.python.org/3/library/ctypes.html) in the python module `dsl.py`. 

Import dsl into your python code.
```python
#!/usr/bin/env python
from dsl import *

# New CSI Live Camera Source
retval = dsl_source_csi_new('csi-source', 1280, 720, 30, 1)

if retval != DSL_RETURN_SUCCESS:
    print(retval)
    # --- handle error
```


### Optionally generate documentation.
Doxygen is used for source documentation which can be generated with the following make command
```
make dox
```

### Running the Test Application
***This step is optional unless contributing code changes.***

Current requirements and limitations.
* Many of test cases have a dependency on the Nano versions of the caffemodel engine files.
* Triton inference server installation is required.

Once the test executable has been built, it can be run with the command below.

```
./dsl-test-app.exe
```

After completion, ensure that all tests have passed before submitting a pull request.
```
===============================================================================
All tests passed (6405 assertions in 795 test cases)
```

Note: the total passed assertions and test cases are subject to change.


## Getting Started
* [Installing DSL Dependencies](/docs/installing-dependencies.md)
* **Building and Importing DSL**

## API Reference
* [Overview](/docs/overview.md)
* [List of all Services](/docs/api-reference-list.md)
* [Pipeline](/docs/api-pipeline.md)
* [Player](/docs/api-player.md)
* [Source](/docs/api-source.md)
* [Tap](/docs/api-tap.md)
* [Dewarper](/docs/api-dewarper.md)
* [Preprocessor](/docs/api-preproc.md)
* [Inference Engine and Server](/docs/api-infer.md)
* [Tracker](/docs/api-tracker.md)
* [Segmentation Visualizer](/docs/api-segvisual.md)
* [Tiler](/docs/api-tiler.md)
* [Demuxer and Splitter Tees](/docs/api-tee)
* [On-Screen Display](/docs/api-osd.md)
* [Sink](/docs/api-sink.md)
* [Pad Probe Handler](/docs/api-pph.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Accumulator](/docs/api-ode-accumulator.md)
* [ODE Action ](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [ODE Heat-Mapper](/docs/api-ode-heat-mapper.md)
* [Display Type](/docs/api-display-type.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [Mailer](/docs/api-mailer.md)
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)
