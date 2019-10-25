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

def test1() :
#    libc = ctypes.cdll.LoadLibrary('/home/sscsmatrix/Projects/CanAmMex/deepstream-services-library/dsl-lib.so')
    libc =ctypes.CDLL("dsl-lib.so")
    if libc.dsl_pipeline_new("pipeline1") == 0:
        print("pipeline opened")
    else:
        print("pipeline failed to open")

    # Try to open pipeline again
    print("Try to open same pipeline it should fail to open")
    if libc.dsl_pipeline_new("pipeline1") == 0:
        print("pipeline opened")
    else:
        print("pipeline failed to open")
        print("delete pipeline")
        if libc.dsl_pipeline_delete("pipeline1") == 0:
            print("pipeline successfully deleted")
            print("try to open pipeline again")
            if libc.dsl_pipeline_new("pipeline1") == 0:
                 print("pipeline opened")
                 print("Test passed")
            else:
                 print("pipeline failed to open")
        else:
            print("test failed!!")


if __name__ == "__main__":
    test1()
