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
#include "Dsl.h"
#include "DslNodetr.h"

using namespace DSL; 

SCENARIO( "A new Nodetr is created correctly", "[Nodetr]" )
{
    GIVEN( "A name for a new Nodetr" ) 
    {
        std::string nodetrName  = "parent";

        WHEN( "The Nodetr is created" )
        {
            DSL_NODETR_PTR pNodetr = DSL_NODETR_NEW(nodetrName.c_str());
                
            THEN( "All memeber variables are initialized correctly" )
            {
                REQUIRE( pNodetr->GetName() == nodetrName );
                REQUIRE( pNodetr->GetNumChildren() == 0 );
                
                REQUIRE( pNodetr->GetParentGstObject() == NULL );
                REQUIRE( pNodetr->GetSink() == nullptr );
                REQUIRE( pNodetr->GetSource() == nullptr );
                
                // Ensure that GST has been intialized after first Nodetr
                REQUIRE( gst_is_initialized() == TRUE );
            }
        }
    }
}

SCENARIO( "Parent-child Binter releationship is setup on Add", "[Nodetr]" )
{
    GIVEN( "A new Parent and Child Nodetr in memory" ) 
    {
        std::string parentNodetrName  = "parent";
        std::string childNodetrName = "child";

        DSL_NODETR_PTR pParentNodetr = DSL_NODETR_NEW(parentNodetrName.c_str());
        DSL_NODETR_PTR pChildNodetr = DSL_NODETR_NEW(childNodetrName.c_str());
        
        REQUIRE( pParentNodetr->GetName() == parentNodetrName );
        REQUIRE( pParentNodetr->GetNumChildren() == 0 );
        
        REQUIRE( pChildNodetr->GetName() == childNodetrName );
        REQUIRE( pChildNodetr->GetParentGstObject() == NULL );

        WHEN( "The Parent Nodetr is called to add to Child" )
        {
            // Need to fake GstObj creation when testing the base Nodetr
            pParentNodetr->SetGstObject((GstObject*)0x12345678);
            REQUIRE( pParentNodetr->AddChild(pChildNodetr) == true );
        
            THEN( "The Parent-Child relationship is created" )
            {
                REQUIRE( pParentNodetr->GetNumChildren() == 1 );
                REQUIRE( pParentNodetr->IsChild(pChildNodetr) );
//                REQUIRE( pChildNodetr->GetParentGstObject() == pParentNodetr->GetGstObject() );            
            }
        }
    }
}    
    
SCENARIO( "Parent-child Nodetr releationship is cleared on Remove", "[Nodetr]" )
{
    GIVEN( "A new Parent Nodetr with a Child Nodetr in memory" ) 
    {
        std::string parentNodetrName  = "parent";
        std::string childNodetrName = "child";

        DSL_NODETR_PTR pParentNodetr = DSL_NODETR_NEW(parentNodetrName.c_str());
        DSL_NODETR_PTR pChildNodetr = DSL_NODETR_NEW(childNodetrName.c_str());

        // Need to fake GstObj creation when testing the base Nodetr
        pParentNodetr->SetGstObject((GstObject*)0x12345678);
        REQUIRE( pParentNodetr->AddChild(pChildNodetr) == true );

        REQUIRE( pParentNodetr->GetNumChildren() == 1 );
        REQUIRE( pParentNodetr->IsChild(pChildNodetr) == true );

        // Can't check as it requires casting of the object that is fake.
//        REQUIRE( pChildNodetr->GetParentGstObject() == pParentNodetr->GetGstObject() );

        WHEN( "When the Parent is called to  RemoveChild" )
        {
            REQUIRE( pParentNodetr->RemoveChild(pChildNodetr) == true );
            
            THEN( " The Parent-Child relationship is deleted ")
            {
                REQUIRE( pParentNodetr->GetNumChildren() == 0 );
                REQUIRE( pChildNodetr->GetParentGstObject() == NULL );            
            }
        }
    }
}    

SCENARIO( "Source-Sink Nodetr releationship is setup on LinkToSink", "[Nodetr]" )
{
    GIVEN( "A new Source Nodetr and Sink Nodetr in memory" ) 
    {
        std::string sourceNodetrName  = "source";
        std::string sinkNodetrName = "sink";

        DSL_NODETR_PTR pSourceNodetr = DSL_NODETR_NEW(sourceNodetrName.c_str());
        DSL_NODETR_PTR pSinkNodetr = DSL_NODETR_NEW(sinkNodetrName.c_str());

        WHEN( "When the Source Nodetr is linked to the Sink Nodetr" )
        {
            REQUIRE( pSourceNodetr->LinkToSink(pSinkNodetr) == true );
            
            THEN( "The Source-Sync relationship is created" )
            {
                REQUIRE( pSourceNodetr->GetSink() == pSinkNodetr );            
            }
        }
    }
}

SCENARIO( "Source-Sink Nodetr releationship is removed on UnlinFromSink", "[Nodetr]" )
{
    GIVEN( "A Source Nodetr linked to a Sink Nodetr" ) 
    {
        std::string sourceNodetrName  = "source";
        std::string sinkNodetrName = "sink";

        DSL_NODETR_PTR pSourceNodetr = DSL_NODETR_NEW(sourceNodetrName.c_str());
        DSL_NODETR_PTR pSinkNodetr = DSL_NODETR_NEW(sinkNodetrName.c_str());

        REQUIRE( pSourceNodetr->LinkToSink(pSinkNodetr) == true );
        
        WHEN( "When the SourceNodetr Unlinks the SinkNodetr" )
        {
            pSourceNodetr->UnlinkFromSink();
            
            THEN( "The Source-Sync relationship is removed" )
            {
                REQUIRE( pSourceNodetr->GetSink() == nullptr );            
            }
        }
    }
}

SCENARIO( "Source-Sink Nodetr releationship is setup on LinkToSource", "[Nodetr]" )
{
    GIVEN( "A new Source Nodetr and Sink Nodetr in memory" ) 
    {
        std::string sourceNodetrName  = "source";
        std::string sinkNodetrName = "sink";

        DSL_NODETR_PTR pSourceNodetr = DSL_NODETR_NEW(sourceNodetrName.c_str());
        DSL_NODETR_PTR pSinkNodetr = DSL_NODETR_NEW(sinkNodetrName.c_str());

        WHEN( "When the Source Nodetr is linked to the Sink Nodetr" )
        {
            REQUIRE( pSinkNodetr->LinkToSource(pSourceNodetr) == true );
            
            THEN( "The Source-Sync relationship is created" )
            {
                REQUIRE( pSinkNodetr->GetSource() == pSourceNodetr );            
            }
        }
    }
}

SCENARIO( "Source-Sink Nodetr releationship is removed on UnlinkFromSource", "[Nodetr]" )
{
    GIVEN( "A Source Nodetr linked to a Sink Nodetr" ) 
    {
        std::string sourceNodetrName  = "source";
        std::string sinkNodetrName = "sink";

        DSL_NODETR_PTR pSourceNodetr = DSL_NODETR_NEW(sourceNodetrName.c_str());
        DSL_NODETR_PTR pSinkNodetr = DSL_NODETR_NEW(sinkNodetrName.c_str());

        REQUIRE( pSinkNodetr->LinkToSource(pSourceNodetr) == true );

        WHEN( "When the SourceNodetr Unlinks the SinkNodetr" )
        {
            pSinkNodetr->UnlinkFromSource();
            
            THEN( "The Source-Sync relationship is removed" )
            {
                REQUIRE( pSinkNodetr->GetSource() == nullptr );            
            }
        }
    }
}
