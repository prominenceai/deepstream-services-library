# Building and Importing DSL

The DSL shared library is built using GCC and GNU Make. The MakeFile is located in the DSL root folder.
There are a few simple steps to creating the shared library, `dsl-lib.so`.

1. Clone this repository to pull down all source
2. Use the make (all) default to build the `dsl-test-app` executable
3. (Optionally) Run the `dsl-test-app` to verify the build
4. Use the `make lib` command to build the object files into `dsl-lib.so`

### Cloning Repository
Clone the repository to pull all source and test scenarios to build the dsl test application
```
$ git clone https://github.com/canammex-tech/deepstream-services-library
```

### Make (all)
The current default `make`, prior to releasing v1.0, builds DSL into a command line test-executable for running all API and Unit level test cases. The DSL source-only object files, built for the test application, can then be re-linked into the `dsl.so` shared library, see below.

Invoke the standard make (all) to  compile all source code and test scenarios into objects files, and link them into a [Catch2](https://github.com/catchorg/Catch2) test application. On successful build, the `dsl-test-app` will be found under the same root folder.

```
$ make
```

### Running the Test Application
***This step is optional unless contributing changes.***

**Note: The Model Engine files, required to run the Tests, are not checked into this repository as they exceed GitHub's size restrictions.  The files will be setup for LFS tracking and added to the repo in the comming weeks.** 

Note: The test application uses Nano versions of GIE Models and Config files, which will fail to load on other platforms. The tests require an internet connection for URI and RTSP Source testing as well. 

Once the test executable has been built, it can be run with the command below.

```
$ ./dsl-test-app
```

After completion, ensure that all tests have passed before building the shared library.
```
===============================================================================
All tests passed (2243 assertions in 340 test cases)
```

Note: the total passed assertions and test cases are subject to change.

### Making the Shared Library
Once the object files have been created by calling `make` , the source-only objects are re-linked into a shared library by calling Make with the lib option

```
# make lib
```

### Importing the Shared Lib using Python3
The shared lib `dsl-lib.so` is mapped to Python3 using [CTypes](https://docs.python.org/3/library/ctypes.html) in the python module `dsl.py`, also located in the DSL root folder.

```python
import sys
# add path to dsl.py module
sys.path.insert(0, "../../")
from dsl import *

# New CSI Live Camera Source
retval = dsl_source_csi_new('csi-source', 1280, 720, 30, 1)

if retval != DSL_RETURN_SUCCESS:
    print(retval)
    # --- handle error
```
