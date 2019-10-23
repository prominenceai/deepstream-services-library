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
#include "DslBintr.h"

SCENARIO( "Parent-child releationship is setup on Add", "[pipeline]" )
{
    GIVEN( "A new Parent and Child Bintr in memory" ) 
    {
        std::string parentBintrName  = "parent";
        std::string childBintrName = "child";

        std::shared_ptr<DSL::Bintr> pParentBintr = 
            std::shared_ptr<DSL::Bintr>(new DSL::Bintr(parentBintrName.c_str()));
        std::shared_ptr<DSL::Bintr> pChildBintr = 
            std::shared_ptr<DSL::Bintr>(new DSL::Bintr(childBintrName.c_str()));
        REQUIRE( pParentBintr->m_name == parentBintrName );
        REQUIRE( pParentBintr->m_pChildBintrs.size() == 0 );
        
        REQUIRE( pChildBintr->m_name == childBintrName );
        REQUIRE( pChildBintr->m_pParentBintr == nullptr );            

        WHEN( "The Parent is called to Add and Remove Child" )
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
        WHEN( "When the Child is called to AddToParent" )
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
    
SCENARIO( "Parent-child releationship is cleared on Remove", "[pipeline]" )
{
    GIVEN( "A new Parent Bintr with a Child Bintr in memory" ) 
    {
        std::string parentBintrName  = "parent";
        std::string childBintrName = "child";

        std::shared_ptr<DSL::Bintr> pParentBintr = 
            std::shared_ptr<DSL::Bintr>(new DSL::Bintr(parentBintrName.c_str()));
        std::shared_ptr<DSL::Bintr> pChildBintr = 
            std::shared_ptr<DSL::Bintr>(new DSL::Bintr(childBintrName.c_str()));

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