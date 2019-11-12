
from ctypes import *

lib = cdll.LoadLibrary("dsl-lib.so")

print(lib.dsl_pipeline_new("pipeline1"))
print(lib.dsl_pipeline_delete("pipeline1"))

