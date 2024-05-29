/*
The MIT License

Copyright (c) 2019-2024, Prominence AI, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in-
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "catch.hpp"
#include "Dsl.h"
#include "DslSinkBintr.h"
#include "DslOdeAction.h"

using namespace DSL;

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(1000)

static uint new_buffer_cb(uint data_type, 
    void* buffer, void* client_data)
{
    return DSL_FLOW_OK;
}

SCENARIO( "A new AppSinkBintr is created correctly",  "[SinkBintr]" )
{
    GIVEN( "Attributes for a new App Sink" ) 
    {
        std::string sinkName("app-sink");
        uint dataType(DSL_SINK_APP_DATA_TYPE_BUFFER);

        WHEN( "The AppSinkBintr is created" )
        {
            DSL_APP_SINK_PTR pSinkBintr = DSL_APP_SINK_NEW(sinkName.c_str(), 
                DSL_SINK_APP_DATA_TYPE_BUFFER, new_buffer_cb, NULL);
            
            THEN( "The correct attribute values are returned" )
            {
                REQUIRE( pSinkBintr->GetDataType() == dataType );
                REQUIRE( pSinkBintr->GetSyncEnabled() == true );
                REQUIRE( pSinkBintr->GetAsyncEnabled() == false );
                REQUIRE( pSinkBintr->GetMaxLateness() == -1 );
                REQUIRE( pSinkBintr->GetQosEnabled() == false );
            }
        }
    }
}

SCENARIO( "A new AppSinkBintr can LinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A new AppSinkBintr in an Unlinked state" ) 
    {
        std::string sinkName("app-sink");

        DSL_APP_SINK_PTR pSinkBintr = DSL_APP_SINK_NEW(sinkName.c_str(), 
            DSL_SINK_APP_DATA_TYPE_BUFFER, new_buffer_cb, NULL);

        REQUIRE( pSinkBintr->IsLinked() == false );

        WHEN( "A new AppSinkBintr is Linked" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );

            THEN( "The AppSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A new FrameCaptureSinkBintr is created correctly",  "[SinkBintr]" )
{
    GIVEN( "Attributes for a new App Sink" ) 
    {
        std::string actionName("ode-action");
        std::string outdir("./");

        DSL_ODE_ACTION_CAPTURE_FRAME_PTR pAction = 
            DSL_ODE_ACTION_CAPTURE_FRAME_NEW(actionName.c_str(), 
                outdir.c_str());

        std::string sinkName("frame-capture-sink");

        WHEN( "The AppSinkBintr is created" )
        {
            DSL_FRAME_CAPTURE_SINK_PTR pSinkBintr =
                DSL_FRAME_CAPTURE_SINK_NEW(sinkName.c_str(), pAction);
            
            THEN( "The correct attribute values are returned" )
            {
                REQUIRE( pSinkBintr->GetDataType() == DSL_SINK_APP_DATA_TYPE_BUFFER );
                REQUIRE( pSinkBintr->GetSyncEnabled() == true );
                REQUIRE( pSinkBintr->GetAsyncEnabled() == false );
                REQUIRE( pSinkBintr->GetMaxLateness() == -1 );
                REQUIRE( pSinkBintr->GetQosEnabled() == false );
            }
        }
    }
}

SCENARIO( "A new FakeSinkBintr is created correctly",  "[SinkBintr]" )
{
    GIVEN( "Attributes for a new Fake Sink" ) 
    {
        std::string sinkName("fake-sink");

        WHEN( "The FakeSinkBintr is created " )
        {
            DSL_FAKE_SINK_PTR pSinkBintr = 
                DSL_FAKE_SINK_NEW(sinkName.c_str());
            
            THEN( "The correct attribute values are returned" )
            {
                REQUIRE( pSinkBintr->GetSyncEnabled() == false );
                REQUIRE( pSinkBintr->GetAsyncEnabled() == false );
                REQUIRE( pSinkBintr->GetMaxLateness() == -1 );
                REQUIRE( pSinkBintr->GetQosEnabled() == false );
            }
        }
    }
}

SCENARIO( "A new FakeSinkBintr can LinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A new FakeSinkBintr in an Unlinked state" ) 
    {
        std::string sinkName("fake-sink");

        DSL_FAKE_SINK_PTR pSinkBintr = 
            DSL_FAKE_SINK_NEW(sinkName.c_str());

        REQUIRE( pSinkBintr->IsLinked() == false );

        WHEN( "A new FakeSinkBintr is Linked" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );

            THEN( "The FakeSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A new 3dSinkBintr is created correctly",  "[SinkBintr]" )
{
    GIVEN( "Attributes for a new 3D Sink" ) 
    {
        
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
        {
            std::string sinkName("3d-sink");
            uint offsetX(100);
            uint offsetY(140);
            uint sinkW(1280);
            uint sinkH(720);

            WHEN( "The 3dSinkBintr is created " )
            {
                DSL_3D_SINK_PTR pSinkBintr = 
                    DSL_3D_SINK_NEW(sinkName.c_str(), 
                        offsetX, offsetY, sinkW, sinkH);
                
                THEN( "The correct attribute values are returned" )
                {
                    REQUIRE( pSinkBintr->GetSyncEnabled() == true );
                    REQUIRE( pSinkBintr->GetAsyncEnabled() == false );
                    REQUIRE( pSinkBintr->GetMaxLateness() == -1 );
                    REQUIRE( pSinkBintr->GetQosEnabled() == false );
                }
            }
        }
    }
}

SCENARIO( "A new 3dSinkBintr can LinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A new 3dSinkBintr in an Unlinked state" ) 
    {
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
        {
            std::string sinkName("3d-sink");
            uint offsetX(100);
            uint offsetY(140);
            uint sinkW(1280);
            uint sinkH(720);

            DSL_3D_SINK_PTR pSinkBintr = 
                DSL_3D_SINK_NEW(sinkName.c_str(), 
                    offsetX, offsetY, sinkW, sinkH);

            REQUIRE( pSinkBintr->IsLinked() == false );

            WHEN( "A new 3dSinkBintr is Linked" )
            {
                REQUIRE( pSinkBintr->LinkAll() == true );

                THEN( "The 3dSinkBintr's IsLinked state is updated correctly" )
                {
                    REQUIRE( pSinkBintr->IsLinked() == true );
                }
            }
        }
    }
}

SCENARIO( "A Linked 3dSinkBintr can UnlinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A 3dSinkBintr in a linked state" ) 
    {
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
        {
            std::string sinkName("3d-sink");
            uint offsetX(100);
            uint offsetY(140);
            uint sinkW(1280);
            uint sinkH(720);

            DSL_3D_SINK_PTR pSinkBintr = 
                DSL_3D_SINK_NEW(sinkName.c_str(), 
                    offsetX, offsetY, sinkW, sinkH);

            REQUIRE( pSinkBintr->LinkAll() == true );

            WHEN( "A 3dSinkBintr is Unlinked" )
            {
                pSinkBintr->UnlinkAll();

                THEN( "The 3dSinkBintr's IsLinked state is updated correctly" )
                {
                    REQUIRE( pSinkBintr->IsLinked() == false );
                }
            }
        }
    }
}

SCENARIO( "An 3dSinkBintr's Offsets can be updated", "[SinkBintr]" )
{
    GIVEN( "A new 3dSinkBintr in memory" ) 
    {
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
        {
            std::string sinkName("3d-sink");
            uint initOffsetX(0);
            uint initOffsetY(0);
            uint sinkW(1280);
            uint sinkH(720);

            DSL_3D_SINK_PTR pSinkBintr = 
                DSL_3D_SINK_NEW(sinkName.c_str(), initOffsetX, initOffsetY, sinkW, sinkH);
                
            uint currOffsetX(0);
            uint currOffsetY(0);
        
            pSinkBintr->GetOffsets(&currOffsetX, &currOffsetY);
            REQUIRE( currOffsetX == initOffsetX );
            REQUIRE( currOffsetY == initOffsetY );

            WHEN( "The 3dSinkBintr's Offsets are Set" )
            {
                uint newOffsetX(80);
                uint newOffsetY(20);
                
                pSinkBintr->SetOffsets(newOffsetX, newOffsetY);

                THEN( "The 3dSinkBintr's new demensions are returned on Get")
                {
                    pSinkBintr->GetOffsets(&currOffsetX, &currOffsetY);
                    REQUIRE( currOffsetX == newOffsetX );
                    REQUIRE( currOffsetY == newOffsetY );
                }
            }
        }
    }
}


SCENARIO( "An 3dSinkBintr's Dimensions can be updated", "[SinkBintr]" )
{
    GIVEN( "A new 3dSinkBintr in memory" ) 
    {
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
        {
            std::string sinkName("3d-sink");
            uint offsetX(0);
            uint offsetY(0);
            uint initSinkW(300);
            uint initSinkH(200);

            DSL_3D_SINK_PTR pSinkBintr = 
                DSL_3D_SINK_NEW(sinkName.c_str(), 
                    offsetX, offsetY, initSinkW, initSinkH);
                
            uint currSinkW(0);
            uint currSinkH(0);
        
            pSinkBintr->GetDimensions(&currSinkW, &currSinkH);
            REQUIRE( currSinkW == initSinkW );
            REQUIRE( currSinkH == initSinkH );

            WHEN( "The 3dSinkBintr's dimensions are Set" )
            {
                uint newSinkW(1280);
                uint newSinkH(720);
                
                pSinkBintr->SetDimensions(newSinkW, newSinkH);

                THEN( "The 3dSinkBintr's new dimensions are returned on Get")
                {
                    pSinkBintr->GetDimensions(&currSinkW, &currSinkH);
                    REQUIRE( currSinkW == newSinkW );
                    REQUIRE( currSinkH == newSinkH );
                }
            }
        }
    }
}

SCENARIO( "A 3dSinkBintr can Get and Set its GPU ID",  "[SinkBintr]" )
{
    GIVEN( "A new 3dSinkBintr in memory" ) 
    {
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
        {
            std::string sinkName("3d-sink");
            uint offsetX(0);
            uint offsetY(0);
            uint sinkW(300);
            uint sinkH(200);

            DSL_3D_SINK_PTR p3dSinkBintr = 
                DSL_3D_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);
            
            uint GPUID0(0);
            uint GPUID1(1);

            REQUIRE( p3dSinkBintr->GetGpuId() == GPUID0 );
            
            WHEN( "The 3dSinkBintr's  GPU ID is set" )
            {
                REQUIRE( p3dSinkBintr->SetGpuId(GPUID1) == true );

                THEN( "The correct GPU ID is returned on get" )
                {
                    REQUIRE( p3dSinkBintr->GetGpuId() == GPUID1 );
                }
            }
        }
    }
}

SCENARIO( "A new EglSinkBintr is created correctly",  "[SinkBintr]" )
{
    GIVEN( "Attributes for a new EGL SInk" ) 
    {
        std::string sinkName("egl-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        WHEN( "The EglSinkBintr is created " )
        {
            DSL_EGL_SINK_PTR pSinkBintr = 
                DSL_EGL_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);
            
            THEN( "The correct attribute values are returned" )
            {
                REQUIRE( pSinkBintr->GetForceAspectRatio() == false );
                REQUIRE( pSinkBintr->GetSyncEnabled() == true );
                REQUIRE( pSinkBintr->GetAsyncEnabled() == false );
                REQUIRE( pSinkBintr->GetMaxLateness() == -1 );
                REQUIRE( pSinkBintr->GetQosEnabled() == false );
            }
        }
    }
}

SCENARIO( "A new EglSinkBintr can LinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A new EglSinkBintr in an Unlinked state" ) 
    {
        std::string sinkName("egl-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        DSL_EGL_SINK_PTR pSinkBintr = 
            DSL_EGL_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        REQUIRE( pSinkBintr->IsLinked() == false );

        WHEN( "A new EglSinkBintr is Linked" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );

            THEN( "The EglSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Linked EglSinkBintr can UnlinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A EglSinkBintr in a linked state" ) 
    {
        std::string sinkName("egl-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        DSL_EGL_SINK_PTR pSinkBintr = 
            DSL_EGL_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        REQUIRE( pSinkBintr->LinkAll() == true );

        WHEN( "A EglSinkBintr is Unlinked" )
        {
            pSinkBintr->UnlinkAll();

            THEN( "The 3dSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A EglSinkBintr can Reset, LinkAll and UnlinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A new EglSinkBintr" ) 
    {
        std::string sinkName("egl-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        DSL_EGL_SINK_PTR pSinkBintr = 
            DSL_EGL_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        WHEN( "A EglSinkBintr has been linked and unlinked" )
        {
            // A window Sink can only be reset after it has been linked/unlinked
            REQUIRE( pSinkBintr->Reset() == false );

            REQUIRE( pSinkBintr->LinkAll() == true );
            pSinkBintr->UnlinkAll();
            REQUIRE( pSinkBintr->IsLinked() == false );

            THEN( "The EglSinkBintr can be reset correctly" )
            {
                REQUIRE( pSinkBintr->Reset() == true );

            }
        }
    }
}

SCENARIO( "A EglSinkBintr can LinkAll and UnlinkAll mutlple times", "[SinkBintr]" )
{
    GIVEN( "A new EglSinkBintr" ) 
    {
        std::string sinkName("egl-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);
        std::shared_ptr<DslMutex> pSharedClientMutex = 
            std::shared_ptr<DslMutex>(new DslMutex());

        DSL_EGL_SINK_PTR pSinkBintr = 
            DSL_EGL_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        WHEN( "A EglSinkBintr is Linked and its handle prepared" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );
            REQUIRE( pSinkBintr->PrepareWindowHandle(pSharedClientMutex) == true );

            THEN( "The EglSinkBintr can UnlinkAll and LinkAll and serveral times correctly" )
            {
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                pSinkBintr->UnlinkAll();
                REQUIRE( pSinkBintr->IsLinked() == false );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( pSinkBintr->LinkAll() == true );
                REQUIRE( pSinkBintr->PrepareWindowHandle(pSharedClientMutex) == true );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                pSinkBintr->UnlinkAll();
                REQUIRE( pSinkBintr->IsLinked() == false );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( pSinkBintr->LinkAll() == true );
                REQUIRE( pSinkBintr->PrepareWindowHandle(pSharedClientMutex) == true );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                pSinkBintr->UnlinkAll();
                REQUIRE( pSinkBintr->IsLinked() == false );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( pSinkBintr->LinkAll() == true );
                REQUIRE( pSinkBintr->PrepareWindowHandle(pSharedClientMutex) == true );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                pSinkBintr->UnlinkAll();
                REQUIRE( pSinkBintr->IsLinked() == false );
            }
        }
    }
}


SCENARIO( "Multiple EGL SInks can create their XWindow correctly", 
    "[SinkBintr]" )
{
    GIVEN( "A PipelineBintr with valid XWindow dimensions" ) 
    {
        std::string sinkName1("egl-sink-1");
        std::string sinkName2("egl-sink-2");
        std::string sinkName3("egl-sink-3");
        std::string sinkName4("egl-sink-4");
        uint offsetX(0);
        uint offsetY(0);
        uint initSinkW(300);
        uint initSinkH(200);
        std::shared_ptr<DslMutex> pSharedClientMutex = 
            std::shared_ptr<DslMutex>(new DslMutex());

        DSL_EGL_SINK_PTR pSinkBintr1 = DSL_EGL_SINK_NEW(
            sinkName1.c_str(), offsetX, offsetY, initSinkW, initSinkH);
        DSL_EGL_SINK_PTR pSinkBintr2 = DSL_EGL_SINK_NEW(
            sinkName2.c_str(), offsetX, offsetY, initSinkW, initSinkH);
        DSL_EGL_SINK_PTR pSinkBintr3 = DSL_EGL_SINK_NEW(
            sinkName3.c_str(), offsetX, offsetY, initSinkW, initSinkH);
        DSL_EGL_SINK_PTR pSinkBintr4 = DSL_EGL_SINK_NEW(
            sinkName4.c_str(), offsetX, offsetY, initSinkW, initSinkH);

        WHEN( "The new PipelineBintr's XWindow is created" )
        {
            REQUIRE( pSinkBintr1->PrepareWindowHandle(pSharedClientMutex) == true );
            REQUIRE( pSinkBintr2->PrepareWindowHandle(pSharedClientMutex) == true );
            REQUIRE( pSinkBintr3->PrepareWindowHandle(pSharedClientMutex) == true );
            REQUIRE( pSinkBintr4->PrepareWindowHandle(pSharedClientMutex) == true );
                
            THEN( "The XWindow handle is available" )
            {
                REQUIRE( pSinkBintr1->GetHandle() != 0 );
                REQUIRE( pSinkBintr2->GetHandle() != 0 );
                REQUIRE( pSinkBintr3->GetHandle() != 0 );
                REQUIRE( pSinkBintr4->GetHandle() != 0 );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
            }
        }
    }
}

SCENARIO( "Multiple EGL SInks can create their XWindow correctly in full screen mode", 
    "[SinkBintr]" )
{
    GIVEN( "Four EglSinkBintr's with valid XWindow dimensions" ) 
    {
        std::string sinkName1("egl-sink-1");
        std::string sinkName2("egl-sink-2");
        std::string sinkName3("egl-sink-3");
        std::string sinkName4("egl-sink-4");
        uint offsetX(0);
        uint offsetY(0);
        uint initSinkW(300);
        uint initSinkH(200);
        std::shared_ptr<DslMutex> pSharedClientMutex = 
            std::shared_ptr<DslMutex>(new DslMutex());

        DSL_EGL_SINK_PTR pSinkBintr1 = DSL_EGL_SINK_NEW(
            sinkName1.c_str(), offsetX, offsetY, initSinkW, initSinkH);
        DSL_EGL_SINK_PTR pSinkBintr2 = DSL_EGL_SINK_NEW(
            sinkName2.c_str(), offsetX, offsetY, initSinkW, initSinkH);
        DSL_EGL_SINK_PTR pSinkBintr3 = DSL_EGL_SINK_NEW(
            sinkName3.c_str(), offsetX, offsetY, initSinkW, initSinkH);
        DSL_EGL_SINK_PTR pSinkBintr4 = DSL_EGL_SINK_NEW(
            sinkName4.c_str(), offsetX, offsetY, initSinkW, initSinkH);

        WHEN( "The all EglSinkBintr's XWindows are created" )
        {
            REQUIRE( pSinkBintr1->SetFullScreenEnabled(true) == true );
            REQUIRE( pSinkBintr2->SetFullScreenEnabled(true) == true );
            REQUIRE( pSinkBintr3->SetFullScreenEnabled(true) == true );
            REQUIRE( pSinkBintr4->SetFullScreenEnabled(true) == true );
            REQUIRE( pSinkBintr1->PrepareWindowHandle(pSharedClientMutex) == true );
            REQUIRE( pSinkBintr2->PrepareWindowHandle(pSharedClientMutex) == true );
            REQUIRE( pSinkBintr3->PrepareWindowHandle(pSharedClientMutex) == true );
            REQUIRE( pSinkBintr4->PrepareWindowHandle(pSharedClientMutex) == true );
                
            THEN( "The XWindow handle is available" )
            {
                REQUIRE( pSinkBintr1->GetHandle() != 0 );
                REQUIRE( pSinkBintr2->GetHandle() != 0 );
                REQUIRE( pSinkBintr3->GetHandle() != 0 );
                REQUIRE( pSinkBintr4->GetHandle() != 0 );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
            }
        }
    }
}

SCENARIO( "A EglSinkBintr's Offsets can be updated", "[SinkBintr]" )
{
    GIVEN( "A new EglSinkBintr in memory" ) 
    {
        std::string sinkName("egl-sink");
        uint initOffsetX(0);
        uint initOffsetY(0);
        uint sinkW(1280);
        uint sinkH(720);
        std::shared_ptr<DslMutex> pSharedClientMutex = 
            std::shared_ptr<DslMutex>(new DslMutex());

        DSL_EGL_SINK_PTR pSinkBintr = 
            DSL_EGL_SINK_NEW(sinkName.c_str(), initOffsetX, initOffsetY, sinkW, sinkH);
        REQUIRE( pSinkBintr->PrepareWindowHandle(pSharedClientMutex) == true );
            
        uint currOffsetX(0);
        uint currOffsetY(0);
    
        pSinkBintr->GetOffsets(&currOffsetX, &currOffsetY);
        REQUIRE( currOffsetX == initOffsetX );
        REQUIRE( currOffsetY == initOffsetY );

        WHEN( "The EglSinkBintr's Offsets are Set" )
        {
            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
            uint newOffsetX(80);
            uint newOffsetY(20);
            
            REQUIRE( pSinkBintr->SetOffsets(newOffsetX, newOffsetY) == true );

            THEN( "The EglSinkBintr's new offsets are returned on Get")
            {
                // must sleep to allow XWindow offset's to update
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                pSinkBintr->GetOffsets(&currOffsetX, &currOffsetY);
                
//                REQUIRE( currOffsetX == newOffsetX );
//                REQUIRE( currOffsetY == newOffsetY );
            }
        }
    }
}

SCENARIO( "An EglSinkBintr's Dimensions can be updated", "[SinkBintr]" )
{
    GIVEN( "A new EglSinkBintr in memory" ) 
    {
        std::string sinkName("window-sink");
        uint offsetX(0);
        uint offsetY(0);
        uint initSinkW(300);
        uint initSinkH(200);
        std::shared_ptr<DslMutex> pSharedClientMutex = 
            std::shared_ptr<DslMutex>(new DslMutex());

        DSL_EGL_SINK_PTR pSinkBintr = DSL_EGL_SINK_NEW(
            sinkName.c_str(), offsetX, offsetY, initSinkW, initSinkH);
        REQUIRE( pSinkBintr->PrepareWindowHandle(pSharedClientMutex) == true );
            
        uint currSinkW(0);
        uint currSinkH(0);
    
        pSinkBintr->GetDimensions(&currSinkW, &currSinkH);
        REQUIRE( currSinkW == initSinkW );
        REQUIRE( currSinkH == initSinkH );

        WHEN( "The EglSinkBintr's dimensions are Set" )
        {
            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
            uint newSinkW(1280);
            uint newSinkH(720);
            REQUIRE( pSinkBintr->SetDimensions(newSinkW, newSinkH) == true);

            THEN( "The EglSinkBintr's new dimensions are returned on Get")
            {
                // must sleep to allow XWindow dimensions's to update
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                pSinkBintr->GetDimensions(&currSinkW, &currSinkH);

                REQUIRE( currSinkW == newSinkW );
                REQUIRE( currSinkH == newSinkH );
            }
        }
    }
}

// x86_64 build only
//SCENARIO( "A EglSinkBintr can Get and Set its GPU ID",  "[SinkBintr]" )
//{
//    GIVEN( "A new EglSinkBintr in memory" ) 
//    {
//        std::string sinkName("3d-sink");
//        uint offsetX(0);
//        uint offsetY(0);
//        uint initSinkW(300);
//        uint initSinkH(200);
//
//        DSL_EGL_SINK_PTR pEglSinkBintr = 
//            DSL_EGL_SINK_NEW(sinkName.c_str(), offsetX, offsetY, initSinkW, initSinkH);
//        
//        uint GPUID0(0);
//        uint GPUID1(1);
//
//        REQUIRE( pEglSinkBintr->GetGpuId() == GPUID0 );
//        
//        WHEN( "The EglSinkBintr's  GPU ID is set" )
//        {
//            REQUIRE( pEglSinkBintr->SetGpuId(GPUID1) == true );
//
//            THEN( "The correct GPU ID is returned on get" )
//            {
//                REQUIRE( pEglSinkBintr->GetGpuId() == GPUID1 );
//            }
//        }
//    }
//}

SCENARIO( "An EglSinkBintr's force-aspect-ration setting can be updated", "[SinkBintr]" )
{
    GIVEN( "A new EglSinkBintr in memory" ) 
    {
        std::string sinkName("3d-sink");
        uint offsetX(0);
        uint offsetY(0);
        uint initSinkW(300);
        uint initSinkH(200);

        DSL_EGL_SINK_PTR pSinkBintr = 
            DSL_EGL_SINK_NEW(sinkName.c_str(), offsetX, offsetY, initSinkW, initSinkH);
            
        REQUIRE( pSinkBintr->GetForceAspectRatio() == false );

        WHEN( "The EglSinkBintr's force-aspect-ration setting is Set" )
        {
            REQUIRE( pSinkBintr->SetForceAspectRatio(true) == true );

            THEN( "The EglSinkBintr's new force-aspect-ration setting is returned on Get")
            {
                REQUIRE( pSinkBintr->GetForceAspectRatio() == true );
            }
        }
    }
}

//SCENARIO( "A new DSL_CODEC_MPEG4 FileSinkBintr is created correctly",  "[SinkBintr]" )
//{
//    GIVEN( "Attributes for a new DSL_CODEC_MPEG4 File Sink" ) 
//    {
//        std::string sinkName("file-sink");
//        std::string filePath("./output.mp4");
//        uint codec(DSL_CODEC_MPEG4);
//        uint container(DSL_CONTAINER_MP4);
//        uint bitrate(2000000);
//        uint interval(0);
//
//        WHEN( "The DSL_CODEC_MPEG4 FileSinkBintr is created " )
//        {
//            DSL_FILE_SINK_PTR pSinkBintr = 
//                DSL_FILE_SINK_NEW(sinkName.c_str(), filePath.c_str(), codec, container, bitrate, interval);
//            
//            THEN( "The correct attribute values are returned" )
//            {
//                uint retCodec(0), retContainer(0);
//                pSinkBintr->GetVideoFormats(&retCodec, &retContainer);
//                REQUIRE( retCodec == codec );
//                REQUIRE( retContainer == container);
//                bool sync(false), async(false);
//                pSinkBintr->GetSyncSettings(&sync, &async);
//                REQUIRE( sync == true );
//                REQUIRE( async == false );
//            }
//        }
//    }
//}
//
//SCENARIO( "A new DSL_CODEC_MPEG4 FileSinkBintr can LinkAll Child Elementrs", "[SinkBintr]" )
//{
//    GIVEN( "A new DSL_CODEC_MPEG4 FileSinkBintr in an Unlinked state" ) 
//    {
//        std::string sinkName("file-sink");
//        std::string filePath("./output.mp4");
//        uint codec(DSL_CODEC_MPEG4);
//        uint container(DSL_CONTAINER_MP4);
//        uint bitrate(2000000);
//        uint interval(0);
//
//        DSL_FILE_SINK_PTR pSinkBintr = 
//            DSL_FILE_SINK_NEW(sinkName.c_str(), filePath.c_str(), codec, container, bitrate, interval);
//
//        REQUIRE( pSinkBintr->IsLinked() == false );
//
//        WHEN( "A new DSL_CODEC_MPEG4 FileSinkBintr is Linked" )
//        {
//            REQUIRE( pSinkBintr->LinkAll() == true );
//
//            THEN( "The DSL_CODEC_MPEG4 FileSinkBintr's IsLinked state is updated correctly" )
//            {
//                REQUIRE( pSinkBintr->IsLinked() == true );
//            }
//        }
//    }
//}
//
//SCENARIO( "A Linked DSL_CODEC_MPEG4 FileSinkBintr can UnlinkAll Child Elementrs", "[SinkBintr]" )
//{
//    GIVEN( "A DSL_CODEC_MPEG4 FileSinkBintr in a linked state" ) 
//    {
//        std::string sinkName("file-sink");
//        std::string filePath("./output.mp4");
//        uint codec(DSL_CODEC_MPEG4);
//        uint container(DSL_CONTAINER_MP4);
//        uint bitrate(2000000);
//        uint interval(0);
//
//        DSL_FILE_SINK_PTR pSinkBintr = 
//            DSL_FILE_SINK_NEW(sinkName.c_str(), filePath.c_str(), codec, container, bitrate, interval);
//
//        REQUIRE( pSinkBintr->IsLinked() == false );
//        REQUIRE( pSinkBintr->LinkAll() == true );
//
//        WHEN( "A DSL_CODEC_MPEG4 FileSinkBintr is Unlinked" )
//        {
//            pSinkBintr->UnlinkAll();
//
//            THEN( "The DSL_CODEC_MPEG4 FileSinkBintr's IsLinked state is updated correctly" )
//            {
//                REQUIRE( pSinkBintr->IsLinked() == false );
//            }
//        }
//    }
//}

SCENARIO( "A new DSL_CODEC_H264 FileSinkBintr is created correctly",  "[SinkBintr]" )
{
    GIVEN( "Attributes for a new DSL_CODEC_H264 File Sink" ) 
    {
        std::string sinkName("file-sink");
        std::string filePath("./output.mp4");
        uint codec(DSL_CODEC_H264);
        uint container(DSL_CONTAINER_MP4);
        uint bitrate(0); // use default
        uint interval(0);

        WHEN( "The DSL_CODEC_H264 FileSinkBintr is created " )
        {
            DSL_FILE_SINK_PTR pSinkBintr = 
                DSL_FILE_SINK_NEW(sinkName.c_str(), filePath.c_str(), codec, container, bitrate, interval);
            
            THEN( "The correct attribute values are returned" )
            {
                uint retCodec(0), retBitrate(0), retInterval(0);
                pSinkBintr->GetEncoderSettings(&retCodec, &retBitrate, &retInterval);
                REQUIRE( retCodec == codec );
                REQUIRE( retBitrate == 4000000);
                REQUIRE( retInterval == interval);
                
                uint retWidth(99), retHeight(99);
                pSinkBintr->GetConverterDimensions(&retWidth, &retHeight);
                REQUIRE( retWidth == 0 );
                REQUIRE( retHeight == 0 );
                
                REQUIRE( pSinkBintr->GetSyncEnabled() == false );
                REQUIRE( pSinkBintr->GetAsyncEnabled() == false );
                REQUIRE( pSinkBintr->GetMaxLateness() == -1 );
                REQUIRE( pSinkBintr->GetQosEnabled() == false );
            }
        }
    }
}

SCENARIO( "A new DSL_CODEC_H264 FileSinkBintr can LinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A new DSL_CODEC_H264 FileSinkBintr in an Unlinked state" ) 
    {
        std::string sinkName("file-sink");
        std::string filePath("./output.mp4");
        uint codec(DSL_CODEC_H264);
        uint container(DSL_CONTAINER_MP4);
        uint bitrate(0); // use default
        uint interval(0);

        DSL_FILE_SINK_PTR pSinkBintr = 
            DSL_FILE_SINK_NEW(sinkName.c_str(), filePath.c_str(), codec, container, bitrate, interval);

        REQUIRE( pSinkBintr->IsLinked() == false );

        WHEN( "A new DSL_CODEC_H264 FileSinkBintr is Linked" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );

            THEN( "The DSL_CODEC_H264 FileSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Linked DSL_CODEC_H264 FileSinkBintr can UnlinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A DSL_CODEC_H264 FileSinkBintr in a linked state" ) 
    {
        std::string sinkName("file-sink");
        std::string filePath("./output.mp4");
        uint codec(DSL_CODEC_H264);
        uint container(DSL_CONTAINER_MP4);
        uint bitrate(0); // use default
        uint interval(0);

        DSL_FILE_SINK_PTR pSinkBintr = 
            DSL_FILE_SINK_NEW(sinkName.c_str(), filePath.c_str(), codec, container, bitrate, interval);

        REQUIRE( pSinkBintr->IsLinked() == false );
        REQUIRE( pSinkBintr->LinkAll() == true );

        WHEN( "A DSL_CODEC_H264 FileSinkBintr is Unlinked" )
        {
            pSinkBintr->UnlinkAll();

            THEN( "The DSL_CODEC_H264 FileSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A new DSL_CODEC_H265 FileSinkBintr is created correctly",  "[SinkBintr]" )
{
    GIVEN( "Attributes for a new DSL_CODEC_H265 File Sink" ) 
    {
        std::string sinkName("file-sink");
        std::string filePath("./output.mp4");
        uint codec(DSL_CODEC_H265);
        uint container(DSL_CONTAINER_MP4);
        uint bitrate(0); // use default
        uint interval(0);

        WHEN( "The DSL_CODEC_H265 FileSinkBintr is created " )
        {
            DSL_FILE_SINK_PTR pSinkBintr = 
                DSL_FILE_SINK_NEW(sinkName.c_str(), filePath.c_str(), codec, container, bitrate, interval);
            
            THEN( "The correct attribute values are returned" )
            {
                uint retCodec(0), retBitrate(0), retInterval(0);
                pSinkBintr->GetEncoderSettings(&retCodec, &retBitrate, &retInterval);
                REQUIRE( retCodec == codec );
                REQUIRE( retBitrate == 4000000 ); // default value
                REQUIRE( retInterval == interval );

                uint retWidth(99), retHeight(99);
                pSinkBintr->GetConverterDimensions(&retWidth, &retHeight);
                REQUIRE( retWidth == 0 );
                REQUIRE( retHeight == 0 );

                REQUIRE( pSinkBintr->GetSyncEnabled() == false );
                REQUIRE( pSinkBintr->GetAsyncEnabled() == false );
                REQUIRE( pSinkBintr->GetMaxLateness() == -1 );
                REQUIRE( pSinkBintr->GetQosEnabled() == false );
            }
        }
    }
}

SCENARIO( "A new DSL_CODEC_H265 FileSinkBintr can LinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A new DSL_CODEC_H265 FileSinkBintr in an Unlinked state" ) 
    {
        std::string sinkName("file-sink");
        std::string filePath("./output.mp4");
        uint codec(DSL_CODEC_H265);
        uint container(DSL_CONTAINER_MP4);
        uint bitrate(0); // use default
        uint interval(0);

        DSL_FILE_SINK_PTR pSinkBintr = 
            DSL_FILE_SINK_NEW(sinkName.c_str(), filePath.c_str(), codec, container, bitrate, interval);

        REQUIRE( pSinkBintr->IsLinked() == false );

        WHEN( "A new DSL_CODEC_H265 FileSinkBintr is Linked" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );

            THEN( "The DSL_CODEC_H265 FileSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Linked DSL_CODEC_H265 FileSinkBintr can UnlinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A DSL_CODEC_H265 FileSinkBintr in a linked state" ) 
    {
        std::string sinkName("file-sink");
        std::string filePath("./output.mp4");
        uint codec(DSL_CODEC_H265);
        uint container(DSL_CONTAINER_MP4);
        uint bitrate(0); // use default
        uint interval(0);

        DSL_FILE_SINK_PTR pSinkBintr = 
            DSL_FILE_SINK_NEW(sinkName.c_str(), filePath.c_str(), codec, container, bitrate, interval);

        REQUIRE( pSinkBintr->IsLinked() == false );
        REQUIRE( pSinkBintr->LinkAll() == true );

        WHEN( "A DSL_CODEC_H265 FileSinkBintr is Unlinked" )
        {
            pSinkBintr->UnlinkAll();

            THEN( "The DSL_CODEC_H265 FileSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A FileSinkBintr's Encoder settings can be updated", "[SinkBintr]" )
{
    GIVEN( "A new FileSinkBintr in memory" ) 
    {
        std::string sinkName("file-sink");
        std::string filePath("./output.mp4");
        uint codec(DSL_CODEC_H265);
        uint container(DSL_CONTAINER_MP4);
        uint initBitrate(0); // use default
        uint initInterval(0);

        DSL_FILE_SINK_PTR pSinkBintr = 
            DSL_FILE_SINK_NEW(sinkName.c_str(), filePath.c_str(), codec, container, initBitrate, initInterval);
            
        uint currCodec(99);
        uint currBitrate(0);
        uint currInterval(0);
    
        pSinkBintr->GetEncoderSettings(&currCodec, &currBitrate, &currInterval);
        REQUIRE( currCodec == codec );
        REQUIRE( currBitrate == 4000000 ); // default
        REQUIRE( currInterval == initInterval );

        WHEN( "The FileSinkBintr's Encoder settings are Set" )
        {
            uint newCodec(DSL_CODEC_H264);
            uint newBitrate(3000000);
            uint newInterval(5);
            
            pSinkBintr->SetEncoderSettings(newCodec, newBitrate, newInterval);

            THEN( "The FileSinkBintr's new Encoder settings are returned on Get")
            {
                pSinkBintr->GetEncoderSettings(&currCodec, &currBitrate, &currInterval);
                REQUIRE( currCodec == newCodec );
                REQUIRE( currBitrate == newBitrate );
                REQUIRE( currInterval == newInterval );
            }
        }
    }
}

SCENARIO( "A FileSinkBintr's Converter dimensions can be updated", "[SinkBintr]" )
{
    GIVEN( "A new FileSinkBintr in memory" ) 
    {
        std::string sinkName("file-sink");
        std::string filePath("./output.mp4");
        uint codec(DSL_CODEC_H265);
        uint container(DSL_CONTAINER_MP4);
        uint initBitrate(0); // use default
        uint initInterval(0);

        DSL_FILE_SINK_PTR pSinkBintr = 
            DSL_FILE_SINK_NEW(sinkName.c_str(), filePath.c_str(), 
            codec, container, initBitrate, initInterval);
            
        uint retWidth(99), retHeight(99);
    
        pSinkBintr->GetConverterDimensions(&retWidth, &retHeight);
        REQUIRE( retWidth == 0 );
        REQUIRE( retHeight == 0 );

        WHEN( "The FileSinkBintr's Converter dimensions are Set" )
        {
            uint newWidth(1280), newHeight(720);
            
            REQUIRE( pSinkBintr->SetConverterDimensions(newWidth, newHeight) == true );

            THEN( "The FileSinkBintr's new Converter dimensions are returned on Get")
            {
                pSinkBintr->GetConverterDimensions(&retWidth, &retHeight);
                REQUIRE( retWidth == newWidth );
                REQUIRE( retHeight == newHeight );
            }
        }
    }
}

SCENARIO( "A FileSinkBintr can Get and Set its GPU ID",  "[SinkBintr]" )
{
    GIVEN( "A new FileSinkBintr in memory" ) 
    {
        std::string sinkName("file-sink");
        std::string filePath("./output.mp4");
        uint codec(DSL_CODEC_H265);
        uint container(DSL_CONTAINER_MP4);
        uint initBitrate(0); // use default
        uint initInterval(0);
        
        uint GPUID0(0);
        uint GPUID1(1);

        DSL_FILE_SINK_PTR pFileSinkBintr = 
            DSL_FILE_SINK_NEW(sinkName.c_str(), filePath.c_str(), codec, container, initBitrate, initInterval);

        REQUIRE( pFileSinkBintr->GetGpuId() == GPUID0 );
        
        WHEN( "The FileSinkBintr's  GPU ID is set" )
        {
            REQUIRE( pFileSinkBintr->SetGpuId(GPUID1) == true );

            THEN( "The correct GPU ID is returned on get" )
            {
                REQUIRE( pFileSinkBintr->GetGpuId() == GPUID1 );
            }
        }
    }
}

SCENARIO( "A new DSL_CONTAINER_MP4 RecordSinkBintr is created correctly",  "[SinkBintr]" )
{
    GIVEN( "Attributes for a new DSL_CODEC_MPEG4 RecordSinkBintr" ) 
    {
        std::string sinkName("record-sink");
        std::string outdir("./");
        uint codec(DSL_CODEC_H264);
        uint bitrate(4000000);
        uint interval(0);
        uint container(DSL_CONTAINER_MP4);
        
        dsl_record_client_listener_cb clientListener;

        WHEN( "The DSL_CONTAINER_MP4 RecordSinkBintr is created" )
        {
            DSL_RECORD_SINK_PTR pSinkBintr = DSL_RECORD_SINK_NEW(sinkName.c_str(), 
                outdir.c_str(), codec, container, bitrate, interval, clientListener);
            
            THEN( "The correct attribute values are returned" )
            {
                uint width(0), height(0);
                pSinkBintr->GetDimensions(&width, &height);
                REQUIRE( width == 0 );
                REQUIRE( height == 0 );
                
                std::string retOutdir = pSinkBintr->GetOutdir();
                REQUIRE( outdir == retOutdir );
                
                REQUIRE( pSinkBintr->GetCacheSize() == DSL_DEFAULT_VIDEO_RECORD_CACHE_IN_SEC );
                
                // The following should produce warning messages as there is no record-bin context 
                // prior to linking. Unfortunately, this requires visual verification.
                REQUIRE( pSinkBintr->GotKeyFrame() == false );
                REQUIRE( pSinkBintr->IsOn() == false );
                REQUIRE( pSinkBintr->ResetDone() == false );
            }
        }
    }
}

SCENARIO( "A RecordSinkBintr's Init Parameters can be Set/Get ",  "[SinkBintr]" )
{
    GIVEN( "A new DSL_CODEC_MPEG4 RecordSinkBintr" ) 
    {
        std::string sinkName("record-sink");
        std::string outdir("./");
        uint codec(DSL_CODEC_H264);
        uint bitrate(4000000);
        uint interval(0);
        uint container(DSL_CONTAINER_MP4);
        
        dsl_record_client_listener_cb clientListener;

        DSL_RECORD_SINK_PTR pSinkBintr = DSL_RECORD_SINK_NEW(sinkName.c_str(), 
            outdir.c_str(), codec, container, bitrate, interval, clientListener);

        WHEN( "The Video Cache Size is set" )
        {
            REQUIRE( pSinkBintr->GetCacheSize() == DSL_DEFAULT_VIDEO_RECORD_CACHE_IN_SEC );
            
            uint newCacheSize(20);
            REQUIRE( pSinkBintr->SetCacheSize(newCacheSize) == true );

            THEN( "The correct cache size value is returned" )
            {
                REQUIRE( pSinkBintr->GetCacheSize() == newCacheSize );
            }
        }

        WHEN( "The Video Recording Dimensions are set" )
        {
            uint newWidth(1024), newHeight(780), retWidth(99), retHeight(99);
            pSinkBintr->GetDimensions(&retWidth, &retHeight);
            REQUIRE( retWidth == 0 );
            REQUIRE( retHeight == 0 );
            REQUIRE( pSinkBintr->SetDimensions(newWidth, newHeight) == true );

            THEN( "The correct cache size value is returned" )
            {
                pSinkBintr->GetDimensions(&retWidth, &retHeight);
                REQUIRE( retWidth == newWidth );
                REQUIRE( retHeight == retHeight );
            }
        }
    }
}

static void* record_complete_cb(dsl_recording_info* info, void* client_data)
{
    std::cout << "session_id:     " << info->session_id << "\n";
    std::wcout << L"filename:      " << info->filename << L"\n";
    std::wcout << L"dirpath:       " << info->dirpath << L"\n";
    std::cout << "container_type: " << info->container_type << "\n";
    std::cout << "width:         " << info->width << "\n";
    std::cout << "height:        " << info->height << "\n";
    
    return (void*)0x12345678;
}

SCENARIO( "A RecordSinkBintr handles a Record Complete Notification correctly",  "[SinkBintr]" )
{
    GIVEN( "A new DSL_CODEC_MPEG4 RecordSinkBintr" ) 
    {
        std::string sinkName("record-sink");
        std::string outdir("./");
        uint codec(DSL_CODEC_H264);
        uint bitrate(2000000);
        uint interval(0);
        uint container(DSL_CONTAINER_MP4);
        
        dsl_record_client_listener_cb clientListener;

        DSL_RECORD_SINK_PTR pSinkBintr = DSL_RECORD_SINK_NEW(sinkName.c_str(), 
            outdir.c_str(), codec, container, bitrate, interval, record_complete_cb);

        WHEN( "The RecordSinkBinter is called to handle a record complete" )
        {
            std::string filename("recording-file-name");
            std::string dirpath("recording-dir-path");
            NvDsSRRecordingInfo recordingInfo{0};
            recordingInfo.sessionId = 123;
            recordingInfo.filename = const_cast<gchar*>(filename.c_str());
            recordingInfo.dirpath = const_cast<gchar*>(dirpath.c_str());
            recordingInfo.containerType = NVDSSR_CONTAINER_MP4;
            recordingInfo.width = 123;
            recordingInfo.height = 456;
            
            void* retval = pSinkBintr->HandleRecordComplete(&recordingInfo);
            
            THEN( "The correct response is returned" )
            {
                REQUIRE( retval == (void*)0x12345678 );
            }
        }
    }
}

SCENARIO( "A new DSL_CONTAINER_MP4 RecordSinkBintr can LinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A new DSL_CONTAINER_MP4 RecordSinkBintr in an Unlinked state" ) 
    {
        std::string sinkName("record-sink");
        std::string outdir("./");
        uint codec(DSL_CODEC_H265);
        uint bitrate(2000000);
        uint interval(0);
        uint container(DSL_CONTAINER_MKV);
        
        dsl_record_client_listener_cb clientListener;

        DSL_RECORD_SINK_PTR pSinkBintr = DSL_RECORD_SINK_NEW(sinkName.c_str(), 
            outdir.c_str(), codec, container, bitrate, interval, clientListener);

        REQUIRE( pSinkBintr->IsLinked() == false );

        WHEN( "A new DSL_CONTAINER_MP4 RecordSinkBintr is Linked" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );

            THEN( "The DSL_CODEC_H265 RecordSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == true );

                // Once, linked will not generate warning messages.
                REQUIRE( pSinkBintr->GotKeyFrame() == false );
                REQUIRE( pSinkBintr->IsOn() == false );

                // initialized to true on context init.
                REQUIRE( pSinkBintr->ResetDone() == true );
            }
        }
    }
}

SCENARIO( "A Linked DSL_CONTAINER_MP4 RecordSinkBintr can UnlinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A DSL_CONTAINER_MP4 RecordSinkBintr in a linked state" ) 
    {
        std::string sinkName("record-sink");
        std::string outdir("./");
        uint codec(DSL_CODEC_H265);
        uint bitrate(2000000);
        uint interval(0);
        uint container(DSL_CONTAINER_MP4);
        
        dsl_record_client_listener_cb clientListener;

        DSL_RECORD_SINK_PTR pSinkBintr = DSL_RECORD_SINK_NEW(sinkName.c_str(), 
            outdir.c_str(), codec, container, bitrate, interval, clientListener);

        REQUIRE( pSinkBintr->IsLinked() == false );
        REQUIRE( pSinkBintr->LinkAll() == true );

        WHEN( "A DSL_CONTAINER_MP4 RecordSinkBintr is Unlinked" )
        {
            pSinkBintr->UnlinkAll();

            THEN( "The DSL_CONTAINER_MP4 RecordSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A Linked DSL_CONTAINER_MP4 RecordSinkBintr can Link/UnlinkAll multiple times", 
    "[SinkBintr]" )
{
    GIVEN( "A DSL_CONTAINER_MP4 RecordSinkBintr in a linked state" ) 
    {
        std::string sinkName("record-sink");
        std::string outdir("./");
        uint codec(DSL_CODEC_H265);
        uint bitrate(2000000);
        uint interval(0);
        uint container(DSL_CONTAINER_MP4);
        
        dsl_record_client_listener_cb clientListener;

        DSL_RECORD_SINK_PTR pSinkBintr = DSL_RECORD_SINK_NEW(sinkName.c_str(), 
            outdir.c_str(), codec, container, bitrate, interval, clientListener);

        REQUIRE( pSinkBintr->IsLinked() == false );

        WHEN( "A DSL_CONTAINER_MP4 RecordSinkBintr is Linked/Unlinked multiple times" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );
            pSinkBintr->UnlinkAll();
            REQUIRE( pSinkBintr->LinkAll() == true );
            pSinkBintr->UnlinkAll();
            REQUIRE( pSinkBintr->LinkAll() == true );
            pSinkBintr->UnlinkAll();
            REQUIRE( pSinkBintr->LinkAll() == true );
            pSinkBintr->UnlinkAll();
            REQUIRE( pSinkBintr->LinkAll() == true );
            pSinkBintr->UnlinkAll();

            THEN( "The DSL_CONTAINER_MP4 RecordSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A new RtmpSinkBintr is created correctly",  "[SinkBintr]" )
{
    GIVEN( "Attributes for a new Rtmp Sink" ) 
    {
        std::string sinkName("rtmp-sink");
        std::string uri("rtmp://localhost/path-to-stream");
        uint bitrate(0);
        uint interval(0);

        WHEN( "The DSL_CODEC_H264 RtspServerSinkBintr is created " )
        {
            DSL_RTMP_SINK_PTR pSinkBintr = 
                DSL_RTMP_SINK_NEW(sinkName.c_str(), uri.c_str(), 
                    bitrate, interval);
            
            THEN( "The correct attribute values are returned" )
            {
                std::string retUri(pSinkBintr->GetUri());
                REQUIRE( retUri == uri );
                
                REQUIRE( pSinkBintr->GetSyncEnabled() == true );
                REQUIRE( pSinkBintr->GetAsyncEnabled() == false );
                REQUIRE( pSinkBintr->GetMaxLateness() == -1 );
                REQUIRE( pSinkBintr->GetQosEnabled() == false );
            }
        }
    }
}

SCENARIO( "A new RtmpSinkBintr can LinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A new RtmpSinkBintr in an Unlinked state" ) 
    {
        std::string sinkName("rtmp-sink");
        std::string uri("rtmp://localhost/path-to-stream");
        uint bitrate(0);
        uint interval(0);

        DSL_RTMP_SINK_PTR pSinkBintr = 
            DSL_RTMP_SINK_NEW(sinkName.c_str(), uri.c_str(), 
                bitrate, interval);

        REQUIRE( pSinkBintr->IsLinked() == false );

        WHEN( "A new DSL_CODEC_H264 RtspServerSinkBintr is Linked" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );

            THEN( "The DSL_CODEC_H264 RtspServerSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Linked RtmpSinkBintr can UnlinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A RtmpSinkBintr in a linked state" ) 
    {
        std::string sinkName("rtmp-sink");
        std::string uri("rtmp://localhost/path-to-stream");
        uint bitrate(0);
        uint interval(0);

        DSL_RTMP_SINK_PTR pSinkBintr = 
            DSL_RTMP_SINK_NEW(sinkName.c_str(), uri.c_str(), 
                bitrate, interval);

        REQUIRE( pSinkBintr->IsLinked() == false );
        REQUIRE( pSinkBintr->LinkAll() == true );

        WHEN( "A DSL_CODEC_H264 RtspServerSinkBintr is Unlinked" )
        {
            pSinkBintr->UnlinkAll();

            THEN( "The DSL_CODEC_H264 RtspServerSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A new DSL_CODEC_H264 RtspClientSinkBintr is created correctly",
    "[SinkBintr]" )
{
    GIVEN( "Attributes for a new DSL_CODEC_H264 RTSP Client Sink" ) 
    {
        std::string sinkName("rtsp-client-sink");
        std::string uri("rtsp://server_endpoint/stream");
        uint codec(DSL_CODEC_H264);
        uint bitrate(0); // use default
        uint interval(0);

        WHEN( "The DSL_CODEC_H264 RtspClientSinkBintr is created " )
        {
            DSL_RTSP_CLIENT_SINK_PTR pSinkBintr = 
                DSL_RTSP_CLIENT_SINK_NEW(sinkName.c_str(), 
                    uri.c_str(), codec, bitrate, interval);
            
            THEN( "The correct attribute values are returned" )
            {
                REQUIRE( pSinkBintr->GetLatency() == 2000 );
                REQUIRE( pSinkBintr->GetProfiles() == DSL_RTSP_PROFILE_AVP );
                REQUIRE( pSinkBintr->GetProtocols() == (DSL_RTSP_LOWER_TRANS_TCP |
                    DSL_RTSP_LOWER_TRANS_UDP_MCAST | DSL_RTSP_LOWER_TRANS_UDP) );
                REQUIRE( pSinkBintr->GetTlsValidationFlags() == DSL_TLS_CERTIFICATE_VALIDATE_ALL );

                uint retCodec(0), retBitrate(0), retInterval(0);
                pSinkBintr->GetEncoderSettings(&retCodec, &retBitrate, &retInterval);
                REQUIRE( retCodec == codec );
                REQUIRE( retBitrate == 4000000);
                REQUIRE( retInterval == interval);
                
                uint retWidth(99), retHeight(99);
                pSinkBintr->GetConverterDimensions(&retWidth, &retHeight);
                REQUIRE( retWidth == 0 );
                REQUIRE( retHeight == 0 );
                
            }
        }
    }
}

SCENARIO( "A new RtspClientSinkBintr can LinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A new DSL_CODEC_H265 RtspClientSinkBintr in an Unlinked state" ) 
    {
        std::string sinkName("rtsp-client-sink");
        std::string uri("rtsp://server_endpoint/stream");
        uint codec(DSL_CODEC_H264);
        uint bitrate(0); // use default
        uint interval(0);

        DSL_RTSP_CLIENT_SINK_PTR pSinkBintr = 
            DSL_RTSP_CLIENT_SINK_NEW(sinkName.c_str(), 
                uri.c_str(), codec, bitrate, interval);

        REQUIRE( pSinkBintr->IsLinked() == false );

        WHEN( "A new RtspClientSinkBintr is Linked" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );

            THEN( "The RtspClientSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Linked RtspClientSinkBintr can UnlinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A MultiImageSinkBintr in a linked state" ) 
    {
        std::string sinkName("rtsp-client-sink");
        std::string uri("rtsp://server_endpoint/stream");
        uint codec(DSL_CODEC_H264);
        uint bitrate(0); // use default
        uint interval(0);

        DSL_RTSP_CLIENT_SINK_PTR pSinkBintr = 
            DSL_RTSP_CLIENT_SINK_NEW(sinkName.c_str(), 
                uri.c_str(), codec, bitrate, interval);

        REQUIRE( pSinkBintr->IsLinked() == false );
        REQUIRE( pSinkBintr->LinkAll() == true );

        WHEN( "A RtspClientSinkBintr is Unlinked" )
        {
            pSinkBintr->UnlinkAll();

            THEN( "The RtspClientSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A new DSL_CODEC_H264 RtspServerSinkBintr is created correctly",  "[SinkBintr]" )
{
    GIVEN( "Attributes for a new DSL_CODEC_H264 File Sink" ) 
    {
        std::string sinkName("rtsp-sink");
        std::string host("224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint codec(DSL_CODEC_H264);
        uint bitrate(4000000);
        uint interval(0);

        WHEN( "The DSL_CODEC_H264 RtspServerSinkBintr is created " )
        {
            DSL_RTSP_SERVER_SINK_PTR pSinkBintr = 
                DSL_RTSP_SERVER_SINK_NEW(sinkName.c_str(), host.c_str(), 
                    udpPort, rtspPort, codec, bitrate, interval);
            
            THEN( "The correct attribute values are returned" )
            {
                uint retUdpPort(0), retRtspPort(0), retCodec(0);
                pSinkBintr->GetServerSettings(&retUdpPort, &retRtspPort);
                REQUIRE( retUdpPort == udpPort );
                REQUIRE( retRtspPort == rtspPort );
                REQUIRE( retCodec == codec );
                REQUIRE( pSinkBintr->GetSyncEnabled() == true );
                REQUIRE( pSinkBintr->GetAsyncEnabled() == false );
                REQUIRE( pSinkBintr->GetMaxLateness() == -1 );
                REQUIRE( pSinkBintr->GetQosEnabled() == false );
            }
        }
    }
}

SCENARIO( "A new DSL_CODEC_H264 RtspServerSinkBintr can LinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A new DSL_CODEC_H264 RtspServerSinkBintr in an Unlinked state" ) 
    {
        std::string sinkName("rtsp-sink");
        std::string host("224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint codec(DSL_CODEC_H264);
        uint bitrate(4000000);
        uint interval(0);

        DSL_RTSP_SERVER_SINK_PTR pSinkBintr = 
            DSL_RTSP_SERVER_SINK_NEW(sinkName.c_str(), host.c_str(), 
                udpPort, rtspPort, codec, bitrate, interval);

        REQUIRE( pSinkBintr->IsLinked() == false );

        WHEN( "A new DSL_CODEC_H264 RtspServerSinkBintr is Linked" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );

            THEN( "The DSL_CODEC_H264 RtspServerSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Linked DSL_CODEC_H264 RtspServerSinkBintr can UnlinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A DSL_CODEC_H264 RtspServerSinkBintr in a linked state" ) 
    {
        std::string sinkName("rtsp-sink");
        std::string host("224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint codec(DSL_CODEC_H264);
        uint bitrate(4000000);
        uint interval(0);

        DSL_RTSP_SERVER_SINK_PTR pSinkBintr = 
            DSL_RTSP_SERVER_SINK_NEW(sinkName.c_str(), host.c_str(), 
                udpPort, rtspPort, codec, bitrate, interval);

        REQUIRE( pSinkBintr->IsLinked() == false );
        REQUIRE( pSinkBintr->LinkAll() == true );

        WHEN( "A DSL_CODEC_H264 RtspServerSinkBintr is Unlinked" )
        {
            pSinkBintr->UnlinkAll();

            THEN( "The DSL_CODEC_H264 RtspServerSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A new DSL_CODEC_H265 RtspServerSinkBintr is created correctly",  "[SinkBintr]" )
{
    GIVEN( "Attributes for a new DSL_CODEC_H265 File Sink" ) 
    {
        std::string sinkName("rtsp-sink");
        std::string host("224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint codec(DSL_CODEC_H265);
        uint bitrate(4000000);
        uint interval(0);

        WHEN( "The DSL_CODEC_H265 RtspServerSinkBintr is created " )
        {
            DSL_RTSP_SERVER_SINK_PTR pSinkBintr = 
                DSL_RTSP_SERVER_SINK_NEW(sinkName.c_str(), host.c_str(), 
                    udpPort, rtspPort, codec, bitrate, interval);
            
            THEN( "The correct attribute values are returned" )
            {
                uint retUdpPort(0), retRtspPort(0);
                pSinkBintr->GetServerSettings(&retUdpPort, &retRtspPort);
                REQUIRE( retUdpPort == udpPort);
                REQUIRE( retRtspPort == rtspPort);
                REQUIRE( pSinkBintr->GetSyncEnabled() == true );
            }
        }
    }
}

SCENARIO( "A new DSL_CODEC_H265 RtspServerSinkBintr can LinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A new DSL_CODEC_H265 RtspServerSinkBintr in an Unlinked state" ) 
    {
        std::string sinkName("rtsp-sink");
        std::string host("224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint codec(DSL_CODEC_H265);
        uint bitrate(4000000);
        uint interval(0);

        DSL_RTSP_SERVER_SINK_PTR pSinkBintr = 
            DSL_RTSP_SERVER_SINK_NEW(sinkName.c_str(), host.c_str(), 
                udpPort, rtspPort, codec, bitrate, interval);

        REQUIRE( pSinkBintr->IsLinked() == false );

        WHEN( "A new DSL_CODEC_H265 RtspServerSinkBintr is Linked" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );

            THEN( "The DSL_CODEC_H265 RtspServerSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Linked DSL_CODEC_H265 RtspServerSinkBintr can UnlinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A DSL_CODEC_H265 RtspServerSinkBintr in a linked state" ) 
    {
        std::string sinkName("rtsp-sink");
        std::string host("224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint codec(DSL_CODEC_H265);
        uint bitrate(4000000);
        uint interval(0);

        DSL_RTSP_SERVER_SINK_PTR pSinkBintr = 
            DSL_RTSP_SERVER_SINK_NEW(sinkName.c_str(), host.c_str(), 
                udpPort, rtspPort, codec, bitrate, interval);

        REQUIRE( pSinkBintr->IsLinked() == false );
        REQUIRE( pSinkBintr->LinkAll() == true );

        WHEN( "A DSL_CODEC_H265 RtspServerSinkBintr is Unlinked" )
        {
            pSinkBintr->UnlinkAll();

            THEN( "The DSL_CODEC_H265 RtspServerSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A RtspServerSinkBintr can Get and Set its GPU ID",  "[SinkBintr]" )
{
    GIVEN( "A new RtspServerSinkBintr in memory" ) 
    {
        std::string sinkName("rtsp-sink");
        std::string host("224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint codec(DSL_CODEC_H265);
        uint bitrate(4000000);
        uint interval(0);
        
        uint GPUID0(0);
        uint GPUID1(1);

        DSL_RTSP_SERVER_SINK_PTR pRtspServerSinkBintr = 
            DSL_RTSP_SERVER_SINK_NEW(sinkName.c_str(), host.c_str(), 
                udpPort, rtspPort, codec, bitrate, interval);

        REQUIRE( pRtspServerSinkBintr->GetGpuId() == GPUID0 );
        
        WHEN( "The RtspServerSinkBintr's  GPU ID is set" )
        {
            REQUIRE( pRtspServerSinkBintr->SetGpuId(GPUID1) == true );

            THEN( "The correct GPU ID is returned on get" )
            {
                REQUIRE( pRtspServerSinkBintr->GetGpuId() == GPUID1 );
            }
        }
    }
}

SCENARIO( "A new MultImageSinkBintr is created correctly",  "[SinkBintr]" )
{
    GIVEN( "Attributes for a new MultiImageSinkBintr Sink" ) 
    {
        std::string sinkName("multi-image-sink");
        
        uint width(1920), height(1080);
        uint fpsN(1), fpsD(2);
        std::string filePath("./frame-%05d.jpg");

        
        WHEN( "The MultiImageSinkBintr is created " )
        {
            DSL_MULTI_IMAGE_SINK_PTR pSinkBintr =
                DSL_MULTI_IMAGE_SINK_NEW(sinkName.c_str(), filePath.c_str(), 
                    width, height, fpsN, fpsD);
            
            THEN( "The correct attribute values are returned" )
            {
                std::string retFilePath = pSinkBintr->GetFilePath();
                REQUIRE( retFilePath == filePath );
                
                uint retWidth(0), retHeight(0);
                pSinkBintr->GetDimensions(&retWidth, &retHeight);
                REQUIRE( retWidth == width );
                REQUIRE( retHeight == height );

                uint retFpsN(0), retFpsD(0);
                pSinkBintr->GetFrameRate(&retFpsN, &retFpsD);
                REQUIRE( retFpsN == fpsN );
                REQUIRE( retFpsD == fpsD );

                REQUIRE( pSinkBintr->GetMaxFiles() == 0 );
                
                REQUIRE( pSinkBintr->GetSyncEnabled() == false );
                REQUIRE( pSinkBintr->GetAsyncEnabled() == false );
                REQUIRE( pSinkBintr->GetMaxLateness() == -1 );
                REQUIRE( pSinkBintr->GetQosEnabled() == false );
            }
        }
    }
}

SCENARIO( "A new MultiImageSinkBintr can LinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A new DSL_CODEC_H265 MultiImageSinkBintr in an Unlinked state" ) 
    {
        std::string sinkName("multi-image-sink");
        
        uint width(1920), height(1080);
        uint fpsN(1), fpsD(2);
        std::string filePath("./frame-%05d.jpg");

        DSL_MULTI_IMAGE_SINK_PTR pSinkBintr =
            DSL_MULTI_IMAGE_SINK_NEW(sinkName.c_str(), filePath.c_str(), 
                width, height, fpsN, fpsD);

        REQUIRE( pSinkBintr->IsLinked() == false );

        WHEN( "A new MultiImageSinkBintr is Linked" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );

            THEN( "The MultiImageSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Linked MultiImageSinkBintr can UnlinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A MultiImageSinkBintr in a linked state" ) 
    {
        std::string sinkName("multi-image-sink");
        
        uint width(1920), height(1080);
        uint fpsN(1), fpsD(2);
        std::string filePath("./frame-%05d.jpg");

        DSL_MULTI_IMAGE_SINK_PTR pSinkBintr =
            DSL_MULTI_IMAGE_SINK_NEW(sinkName.c_str(), filePath.c_str(), 
                width, height, fpsN, fpsD);

        REQUIRE( pSinkBintr->IsLinked() == false );
        REQUIRE( pSinkBintr->LinkAll() == true );

        WHEN( "A MultiImageSinkBintr is Unlinked" )
        {
            pSinkBintr->UnlinkAll();

            THEN( "The MultiImageSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A new V4l2SinkBintr is created correctly",  "[SinkBintr]" )
{
    GIVEN( "Attributes for a new V4L2 Sink" ) 
    {
        std::string sinkName("v4l2-sink");
        std::string deviceLocation("/dev/video0");

        WHEN( "The V4l2SinkBintr is created" )
        {
            DSL_V4L2_SINK_PTR pSinkBintr = DSL_V4L2_SINK_NEW(sinkName.c_str(), 
                deviceLocation.c_str());
            
            THEN( "The correct attribute values are returned" )
            {
                std::string retDeviceLocation = pSinkBintr->GetDeviceLocation();
                REQUIRE( retDeviceLocation == deviceLocation);

                std::string retDeviceName = pSinkBintr->GetDeviceName();
                REQUIRE( retDeviceName == "" );

                REQUIRE( pSinkBintr->GetDeviceFd() ==  -1 );
                REQUIRE( pSinkBintr->GetDeviceFlags() == DSL_V4L2_DEVICE_TYPE_NONE );
                
                int retBrightness(-99), retContrast(-99), retSaturation(-99);
                pSinkBintr->GetPictureSettings(&retBrightness,
                    &retContrast, &retSaturation);
                REQUIRE( retBrightness == 0 );
                REQUIRE( retContrast == 0 );
                REQUIRE( retSaturation == 0 );
                
                REQUIRE( pSinkBintr->GetSyncEnabled() == true );
                REQUIRE( pSinkBintr->GetAsyncEnabled() == false );
                REQUIRE( pSinkBintr->GetMaxLateness() == -1 );
                REQUIRE( pSinkBintr->GetQosEnabled() == false );
            }
        }
    }
}

SCENARIO( "A new V4l2SinkBintr can LinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A new V4l2SinkBintr in an Unlinked state" ) 
    {
        std::string sinkName("v4l2-sink");
        std::string deviceName("/dev/video0");

        DSL_V4L2_SINK_PTR pSinkBintr = DSL_V4L2_SINK_NEW(sinkName.c_str(), 
            deviceName.c_str());

        REQUIRE( pSinkBintr->IsLinked() == false );

        WHEN( "A new V4l2SinkBintr is Linked" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );

            THEN( "The V4l2SinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A new V4l2SinkBintr can UnlinkAll Child Elementrs", "[SinkBintr]" )
{
    GIVEN( "A new V4l2SinkBintr in an Linked state" ) 
    {
        std::string sinkName("v4l2-sink");
        std::string deviceName("/dev/video0");

        DSL_V4L2_SINK_PTR pSinkBintr = DSL_V4L2_SINK_NEW(sinkName.c_str(), 
            deviceName.c_str());

        REQUIRE( pSinkBintr->LinkAll() == true );
        REQUIRE( pSinkBintr->IsLinked() == true );

        // second call should fail
        REQUIRE( pSinkBintr->LinkAll() == false );

        WHEN( "A the V4l2SinkBintr is Unlinked" )
        {
            pSinkBintr->UnlinkAll();
            
            THEN( "The V4l2SinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == false );
            }
        }
    }
}
