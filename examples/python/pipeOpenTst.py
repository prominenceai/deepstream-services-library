#######################################################################
#
#   pipeOpenTst.py
#
#
#   Description:    This is a simple test that uses dsl-lib.so and trys
#                   to open a pipeline with the same name.  If it fails
#                   to open the second time, the test then deletes
#                   the pipeline and then trys to reopen the pipeline
#
#######################################################################

import ctypes

def test1 :
    libc = ctypes.CDLL("dsl-app.so")

if __name__ == "__main__":
    test1()
