/*
The MIT License

Copyright (c) 2019-Present, ROBERT HOWELL

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
                REQUIRE( pNodetr->m_name == nodetrName );
                REQUIRE( pNodetr->m_pChildren.size() == 0 );
                
                REQUIRE( pNodetr->m_pParentGstObj == NULL );
                REQUIRE( pNodetr->m_pSink == nullptr );
                REQUIRE( pNodetr->m_pSource == nullptr );
                
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
        
        REQUIRE( pParentNodetr->m_name == parentNodetrName );
        REQUIRE( pParentNodetr->m_pChildren.size() == 0 );
        
        REQUIRE( pChildNodetr->m_name == childNodetrName );
        REQUIRE( pChildNodetr->m_pParentGstObj == NULL );

        WHEN( "The Parent Nodetr is called to add to Child" )
        {
            // Need to fake GstObj creation when testing the base Nodetr
            pParentNodetr->m_pGstObj = (GstObject*)0x12345678;
            pParentNodetr->AddChild(pChildNodetr);
        
            THEN( "The Parent-Child relationship is created" )
            {
                REQUIRE( pParentNodetr->m_pChildren.size() == 1 );
                REQUIRE( pParentNodetr->m_pChildren[pChildNodetr->m_name] == pChildNodetr );
                REQUIRE( pChildNodetr->m_pParentGstObj == pParentNodetr->m_pGstObj );            
                REQUIRE( pChildNodetr->IsInUse() == true );
            }
            pParentNodetr->m_pGstObj = NULL;
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
        pParentNodetr->m_pGstObj = (GstObject*)0x12345678;
        pParentNodetr->AddChild(pChildNodetr);

        REQUIRE( pParentNodetr->m_pChildren.size() == 1 );
        REQUIRE( pParentNodetr->m_pChildren[pChildNodetr->m_name] == pChildNodetr );
        REQUIRE( pChildNodetr->m_pParentGstObj == pParentNodetr->m_pGstObj );
        REQUIRE( pChildNodetr->IsInUse() == true );

        WHEN( "When the Parent is called to  RemoveChild" )
        {
            pParentNodetr->RemoveChild(pChildNodetr);
            
            THEN( " The Parent-Child relationship is deleted ")
            {
                REQUIRE( pParentNodetr->m_pChildren.size() == 0 );
                REQUIRE( pChildNodetr->m_pParentGstObj == NULL );            
                REQUIRE( pChildNodetr->IsInUse() == false );            
            }
        }
        pParentNodetr->m_pGstObj = NULL;
    }
}    

SCENARIO( "Source-Sink Nodetr releationship is setup on Link", "[Nodetr]" )
{
    GIVEN( "A new Source Nodetr and Sink Nodetr in memory" ) 
    {
        std::string sourceNodetrName  = "source";
        std::string sinkNodetrName = "sink";

        DSL_NODETR_PTR pSourceNodetr = DSL_NODETR_NEW(sourceNodetrName.c_str());
        DSL_NODETR_PTR pSinkNodetr = DSL_NODETR_NEW(sinkNodetrName.c_str());

        WHEN( "When the Source Nodetr is linked to the Sink Nodetr" )
        {
            pSourceNodetr->LinkTo(pSinkNodetr);
            
            THEN( "The Source-Sync relationship is created" )
            {
                REQUIRE( pSourceNodetr->m_pSink == pSinkNodetr );            
                REQUIRE( pSinkNodetr->m_pSource == pSourceNodetr );            
                REQUIRE( pSourceNodetr->IsInUse() == true );            
                REQUIRE( pSinkNodetr->IsInUse() == true );            
                
                // Ensure no Parent-Child Nodetr relationships exist
                REQUIRE( pSourceNodetr->m_pChildren.size() == 0 );
                REQUIRE( pSourceNodetr->m_pParentGstObj == NULL );
                REQUIRE( pSinkNodetr->m_pChildren.size() == 0 );
                REQUIRE( pSinkNodetr->m_pParentGstObj == NULL );
            }
        }
    }
}

SCENARIO( "Source-Sink Nodetr releationship is removed on Unink", "[Nodetr]" )
{
    GIVEN( "A Source Nodetr linked to a Sink Nodetr" ) 
    {
        std::string sourceNodetrName  = "source";
        std::string sinkNodetrName = "sink";

        DSL_NODETR_PTR pSourceNodetr = DSL_NODETR_NEW(sourceNodetrName.c_str());
        DSL_NODETR_PTR pSinkNodetr = DSL_NODETR_NEW(sinkNodetrName.c_str());

        pSourceNodetr->LinkTo(pSinkNodetr);

        // Ensure no Parent-Child Nodetr relationships exist
        REQUIRE( pSourceNodetr->m_pChildren.size() == 0 );
        REQUIRE( pSourceNodetr->m_pParentGstObj == NULL );
        REQUIRE( pSinkNodetr->m_pChildren.size() == 0 );
        REQUIRE( pSinkNodetr->m_pParentGstObj == NULL );

        WHEN( "When the Source Nodetr is unlinked from the Sink Nodetr" )
        {
            pSourceNodetr->Unlink();
            
            THEN( "The Source-Sync relationship is removed" )
            {
                REQUIRE( pSourceNodetr->m_pSink == nullptr );            
                REQUIRE( pSinkNodetr->m_pSource == nullptr );            
                REQUIRE( pSourceNodetr->IsInUse() == false );            
                REQUIRE( pSinkNodetr->IsInUse() == false );            
            }
        }
    }
}
