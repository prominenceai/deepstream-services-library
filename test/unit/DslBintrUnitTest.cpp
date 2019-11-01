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
#include "DslBintr.h"

using namespace DSL; 

SCENARIO( "A new Binter is created correctly", "[Bintr]" )
{
    GIVEN( "A name for a new Bintr" ) 
    {
        std::string bintrName  = "parent";

        WHEN( "The Bintr is created" )
        {
            DSL_BINTR_PTR pBintr = DSL_BINTR_NEW(bintrName.c_str());
                
            THEN( "All memeber variables are initialized correctly" )
            {
                REQUIRE( pBintr->m_name == bintrName );
                REQUIRE( pBintr->m_pChildBintrs.size() == 0 );
                
                REQUIRE( pBintr->m_pParentBintr == nullptr );
                REQUIRE( pBintr->m_pSinkBintr == nullptr );
                REQUIRE( pBintr->m_pSourceBintr == nullptr );
            }
        }
    }
}

SCENARIO( "Parent-child Binter releationship is setup on Add", "[Bintr]" )
{
    GIVEN( "A new Parent and Child Bintr in memory" ) 
    {
        std::string parentBintrName  = "parent";
        std::string childBintrName = "child";

        DSL_BINTR_PTR pParentBintr = DSL_BINTR_NEW(parentBintrName.c_str());
        DSL_BINTR_PTR pChildBintr = DSL_BINTR_NEW(childBintrName.c_str());
        
        REQUIRE( pParentBintr->m_name == parentBintrName );
        REQUIRE( pParentBintr->m_pChildBintrs.size() == 0 );
        
        REQUIRE( pChildBintr->m_name == childBintrName );
        REQUIRE( pChildBintr->m_pParentBintr == nullptr );            

        WHEN( "The Parent Bintr is called to add to Child" )
        {
            pParentBintr->AddChild(pChildBintr);
        
            THEN( "The Parent-Child relationship is created" )
            {
                REQUIRE( pParentBintr->m_pChildBintrs.size() == 1 );
                REQUIRE( pParentBintr->m_pChildBintrs[pChildBintr->m_name] == pChildBintr );
                REQUIRE( pChildBintr->m_pParentBintr == pParentBintr );            
                REQUIRE( pChildBintr->IsInUse() == true );
            }
        }
        WHEN( "When the Child Bintr is called to add to Parent" )
        {
            pChildBintr->AddToParent(pParentBintr);
            
            THEN( " The Parent-Child relationship is created ")
            {
                REQUIRE( pParentBintr->m_pChildBintrs.size() == 1 );
                REQUIRE( pParentBintr->m_pChildBintrs[pChildBintr->m_name] == pChildBintr );
                REQUIRE( pChildBintr->m_pParentBintr == pParentBintr );            
                REQUIRE( pChildBintr->IsInUse() == true );            
            }
        }
    }
}    
    
SCENARIO( "Parent-child Bintr releationship is cleared on Remove", "[pipeline]" )
{
    GIVEN( "A new Parent Bintr with a Child Bintr in memory" ) 
    {
        std::string parentBintrName  = "parent";
        std::string childBintrName = "child";

        DSL_BINTR_PTR pParentBintr = DSL_BINTR_NEW(parentBintrName.c_str());
        DSL_BINTR_PTR pChildBintr = DSL_BINTR_NEW(childBintrName.c_str());

        pParentBintr->AddChild(pChildBintr);

        REQUIRE( pParentBintr->m_pChildBintrs.size() == 1 );
        REQUIRE( pParentBintr->m_pChildBintrs[pChildBintr->m_name] == pChildBintr );
        REQUIRE( pChildBintr->m_pParentBintr == pParentBintr );            
        REQUIRE( pChildBintr->IsInUse() == true );

        WHEN( "When the Parent is called to  RemoveChild" )
        {
            pParentBintr->RemoveChild(pChildBintr);
            
            THEN( " The Parent-Child relationship is deleted ")
            {
                REQUIRE( pParentBintr->m_pChildBintrs.size() == 0 );
                REQUIRE( pChildBintr->m_pParentBintr == nullptr );            
                REQUIRE( pChildBintr->IsInUse() == false );            
            }
        }
        WHEN( "When the Child is called to RemoveFromParent" )
        {
            pChildBintr->RemoveFromParent(pParentBintr);
            
            THEN( " The Parent-Child relationship is deleted ")
            {
                REQUIRE( pParentBintr->m_pChildBintrs.size() == 0 );
                REQUIRE( pChildBintr->m_pParentBintr == nullptr );            
                REQUIRE( pChildBintr->IsInUse() == false );            
            }
        }
    }
}    

// TODO Add elements with sink and source pad to complete test cases..

//SCENARIO( "Source-Sink Bintr releationship is setup on Link", "[pipeline]" )
//{
//    GIVEN( "A new Source Bintr and Sink Bintr in memory" ) 
//    {
//        std::string sourceBintrName  = "source";
//        std::string sinkBintrName = "sink";
//
//        DSL_BINTR_PTR pSourceBintr = DSL_BINTR_NEW(sourceBintrName.c_str());
//        DSL_BINTR_PTR pSinkBintr = DSL_BINTR_NEW(sinkBintrName.c_str());
//
//        WHEN( "When the Source Bintr is linked to the Sink Bintr" )
//        {
//            pSourceBintr->LinkTo(pSinkBintr);
//            
//            THEN( "The Source-Sync relationship is created" )
//            {
//                REQUIRE( pSourceBintr->m_pSinkBintr == pSinkBintr );            
//                REQUIRE( pSinkBintr->m_pSourceBintr == pSourceBintr );            
//                REQUIRE( pSourceBintr->IsInUse() == true );            
//                REQUIRE( pSinkBintr->IsInUse() == true );            
//            }
//        }
//    }
//}
//
//SCENARIO( "Source-Sink Bintr releationship is removed on Unink", "[pipeline]" )
//{
//    GIVEN( "A Source Bintr linked to a Sink Bintr" ) 
//    {
//        std::string sourceBintrName  = "source";
//        std::string sinkBintrName = "sink";
//
//        DSL_BINTR_PTR pSourceBintr = DSL_BINTR_NEW(sourceBintrName.c_str());
//        DSL_BINTR_PTR pSinkBintr = DSL_BINTR_NEW(sinkBintrName.c_str());
//
//        pSourceBintr->LinkTo(pSinkBintr);
//
//        WHEN( "When the Source Bintr is unlinked from the Sink Bintr" )
//        {
//            pSourceBintr->Unlink();
//            
//            THEN( "The Source-Sync relationship is removed" )
//            {
//                REQUIRE( pSourceBintr->m_pSinkBintr == nullptr );            
//                REQUIRE( pSinkBintr->m_pSourceBintr == nullptr );            
//                REQUIRE( pSourceBintr->IsInUse() == false );            
//                REQUIRE( pSinkBintr->IsInUse() == false );            
//            }
//        }
//    }
//}
