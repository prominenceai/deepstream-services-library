/*
The MIT License

Copyright (c) 2019-2021, Prominence AI, Inc.

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
#include "DslTapBintr.h"

using namespace DSL;


SCENARIO( "A RecordTapBintr is created correctly",  "[RecordTapBintr]" )
{
    GIVEN( "Attributes for a new RecordTapBintr" ) 
    {
        std::string recordTapName("record-tap");
        std::string outDir("./");
        uint container(DSL_CONTAINER_MKV);

        dsl_record_client_listener_cb clientListener;

        WHEN( "The Dewarper is created" )
        {
            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                DSL_RECORD_TAP_NEW(recordTapName.c_str(), outDir.c_str(), container, clientListener);

            THEN( "The Dewarper's config file is found, loaded, and returned correctly")
            {
                std::string retOutDir(pRecordTapBintr->GetOutdir());
                
                REQUIRE( retOutDir == outDir );
                REQUIRE( pRecordTapBintr->GetCacheSize() == DSL_DEFAULT_VIDEO_RECORD_CACHE_IN_SEC );
                
                uint width(99), height(99);
                pRecordTapBintr->GetDimensions(&width, &height);
                REQUIRE( width == 0 );
                REQUIRE( height == 0 );
            }
        }
    }
}

SCENARIO( "A RecordTapBintr's Init Parameters can be Set/Get ",  "[RecordTapBintr]" )
{
    GIVEN( "A new DSL_CONTAINER_MKV RecordTapBintr" ) 
    {
        std::string recordTapName("record-tap");
        std::string outDir("./");
        uint container(DSL_CONTAINER_MKV);
        
        dsl_record_client_listener_cb clientListener;

        DSL_RECORD_TAP_PTR pRecordTapBintr = 
            DSL_RECORD_TAP_NEW(recordTapName.c_str(), outDir.c_str(), container, clientListener);

        WHEN( "The Video Cache Size is set" )
        {
            uint newCacheSize(20);
            REQUIRE( pRecordTapBintr->SetCacheSize(newCacheSize) == true );

            THEN( "The correct cache size value is returned" )
            {
                REQUIRE( pRecordTapBintr->GetCacheSize() == newCacheSize );
            }
        }

        WHEN( "The Video Recording Dimensions are set" )
        {
            uint newWidth(1024), newHeight(780), retWidth(99), retHeight(99);
            pRecordTapBintr->GetDimensions(&retWidth, &retHeight);
            REQUIRE( retWidth == 0 );
            REQUIRE( retHeight == 0 );
            REQUIRE( pRecordTapBintr->SetDimensions(newWidth, newHeight) == true );

            THEN( "The correct cache size value is returned" )
            {
                pRecordTapBintr->GetDimensions(&retWidth, &retHeight);
                REQUIRE( retWidth == newWidth );
                REQUIRE( retHeight == retHeight );
            }
        }
    }
}

SCENARIO( "A new DSL_CONTAINER_MKV RecordTapBintr can LinkAll Child Elementrs", "[RecordTapBintr]" )
{
    GIVEN( "A new DSL_CONTAINER_MKV RecordTapBintr in an Unlinked state" ) 
    {
        std::string recordTapName("record-tap");
        std::string outDir("./");
        uint container(DSL_CONTAINER_MKV);
        
        dsl_record_client_listener_cb clientListener;

        DSL_RECORD_TAP_PTR pRecordTapBintr = 
            DSL_RECORD_TAP_NEW(recordTapName.c_str(), outDir.c_str(), container, clientListener);
        
        REQUIRE( pRecordTapBintr->IsLinked() == false );

        WHEN( "A new DSL_CONTAINER_MP4 RecordTapBintr is Linked" )
        {
            REQUIRE( pRecordTapBintr->LinkAll() == true );

            THEN( "The DSL_CODEC_H265 RecordTapBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pRecordTapBintr->IsLinked() == true );

                // Once, linked will not generate warning messages.
                REQUIRE( pRecordTapBintr->GotKeyFrame() == false );
                REQUIRE( pRecordTapBintr->IsOn() == false );

                // initialized to true on context init.
                REQUIRE( pRecordTapBintr->ResetDone() == true );
            }
        }
    }
}

SCENARIO( "A Linked DSL_CONTAINER_MKV RecordTapBintr can UnlinkAll Child Elementrs", "[RecordTapBintr]" )
{
    GIVEN( "A DSL_CONTAINER_MKV RecordTapBintr in a linked state" ) 
    {
        std::string recordTapName("record-tap");
        std::string outDir("./");
        uint container(DSL_CONTAINER_MKV);
        
        dsl_record_client_listener_cb clientListener;

        DSL_RECORD_TAP_PTR pRecordTapBintr = 
            DSL_RECORD_TAP_NEW(recordTapName.c_str(), outDir.c_str(), container, clientListener);
        
        REQUIRE( pRecordTapBintr->IsLinked() == false );
        REQUIRE( pRecordTapBintr->LinkAll() == true );

        WHEN( "A DSL_CONTAINER_MKV RecordTapBintr is Unlinked" )
        {
            pRecordTapBintr->UnlinkAll();

            THEN( "The DSL_CONTAINER_MKV RecordTapBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pRecordTapBintr->IsLinked() == false );
            }
        }
    }
}
