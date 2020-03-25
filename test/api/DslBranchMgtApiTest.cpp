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
#include "DslApi.h"

SCENARIO( "A single Branch is created and deleted correctly", "[branch-mgt-api]" )
{
    GIVEN( "An empty list of Branchs" ) 
    {
        std::wstring actualName  = L"test-branch";
        
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Branch is created" ) 
        {

            REQUIRE( dsl_branch_new(actualName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete(actualName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 0 );
    }
}

SCENARIO( "Multiple Branches are created and deleted correctly", "[branch-mgt-api]" )
{
    GIVEN( "A map of multiple Branches" ) 
    {
        int sampleSize = 6;
        
        for(int i = 0; i < sampleSize; i++)
        {
            REQUIRE( dsl_branch_new(std::to_wstring(i).c_str()) == DSL_RESULT_SUCCESS );
        }
        REQUIRE( dsl_component_list_size() == sampleSize );

        WHEN( "Multiple Branches are deleted" ) 
        {
            const wchar_t* branchList[] = {L"1",L"3", NULL};
            
            REQUIRE( dsl_component_delete_many(branchList) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == sampleSize - 2 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 0 );
    }
}

SCENARIO( "Many Branches are created correctly", "[branch-mgt-api]" )
{
    const wchar_t* branchNames[] = {L"BranchA", L"BranchB", L"BranchC", NULL};
    
    GIVEN( "An empty container of Branches" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "Many Branches are created at once" ) 
        {

            REQUIRE( dsl_branch_new_many(branchNames) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 3 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 0 );
    }
}

SCENARIO( "Many Branches are deleted correctly", "[branch-mgt-api]" )
{
    const wchar_t* branchNames[] = {L"BranchA", L"BranchB", L"BranchC", NULL};
    
    GIVEN( "Many Branches created with an array of names" ) 
    {
        REQUIRE( dsl_branch_new_many(branchNames) == DSL_RESULT_SUCCESS );

        WHEN( "Many Branches are deleted with the same array of names" ) 
        {
            REQUIRE( dsl_component_delete_many(branchNames) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A non-unique Branch name fails when creating Many Branches", "[branch-mgt-api]" )
{
    const wchar_t* branchNames[] = {L"BranchA", L"BranchB", L"BranchA", NULL};
    
    GIVEN( "An empty container of Branches" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A non-unique Branch Name is used when creating many Branches" ) 
        {

            REQUIRE( dsl_branch_new_many(branchNames) == 
                DSL_RESULT_BRANCH_NAME_NOT_UNIQUE );

            THEN( "The list size and contents are updated correctly" ) 
            {
                // Only the first two were added?
                // TODO - check for uniqueness first
                REQUIRE( dsl_component_list_size() == 2 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 0 );
    }
}
