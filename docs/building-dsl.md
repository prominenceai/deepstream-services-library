# Building and Importing DSL

The DSL shared library is built using GCC and GNU Make. The MakeFile is located in the DSL root folder.
There are a few simple steps to creating a verified shared lib, `dsl-lib.so`.

1. Clone this repository to pull down all source
2. Use the make (all) default to build the `dsl-test-app` executable
3. Run the `dsl-test-app` to verify the build
4. Use the `make lib` command to build the object files into `dsl-lib.so`

### Clone Repository
Clone the repository to pull all source and test scenarios to build the dsl test application
```
$ git clone https://github.com/canammex-tech/deepstream-services-library
```

### Make (all)
Invoke the standard make (all) to  compile all source code and test scenarios into objects files, and link them into a [Catch2](https://github.com/catchorg/Catch2) test application. On succesfull build, the `dsl-test-app` will be found under the same root folder.

```
$ make
```

### Run the Test Application
Run the generated Catch2 test application `dsl-test-app` to verify the build.
```
$ ./dsl-test-app
```

After completion, ensure that all tests have passed before building the shared library.
```
===============================================================================
All tests passed (526 assertions in 73 test cases)
```

Note: the total passed assertions and test cases are subject to change.

### Importing the Shared Lib using Python3
The shared lib `dsl-lib.so` is wrapped using native CTypes in the python module `dsl.py`, also located in the DSL root folder.

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
