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
#include "DslApi.h"

SCENARIO( "The Components container is updated correctly on new Demuxer",
    "[tee-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring demuxerName(L"demuxer");

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Demuxer is created" ) 
        {
            REQUIRE( dsl_tee_demuxer_new(demuxerName.c_str(), 
                2) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Components container is updated correctly on Demuxer delete",
    "[tee-api]" )
{
    GIVEN( "A new Demuxer in memory" ) 
    {
        std::wstring demuxerName(L"demuxer");

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_tee_demuxer_new(demuxerName.c_str(),
            2) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "The new Demuxer is deleted" ) 
        {
            REQUIRE( dsl_component_delete(demuxerName.c_str()) 
                == DSL_RESULT_SUCCESS );
            
            THEN( "The list size is updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Demuxer can add and remove a Branch", "[tee-api]" )
{
    GIVEN( "A Demuxer and Branch" ) 
    {
        std::wstring demuxerName(L"demuxer");
        std::wstring branchName(L"branch");

        REQUIRE( dsl_tee_demuxer_new(demuxerName.c_str(), 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branchName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 2 );

        uint count(0);
        
        REQUIRE( dsl_tee_branch_count_get(demuxerName.c_str(), 
            &count) == DSL_RESULT_SUCCESS );
        REQUIRE( count == 0 );

        WHEN( "A the Branch is added to the Demuxer" ) 
        {
            REQUIRE( dsl_tee_branch_add(demuxerName.c_str(), 
                branchName.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_tee_branch_count_get(demuxerName.c_str(), 
                &count) == DSL_RESULT_SUCCESS );
            REQUIRE( count == 1 );

            THEN( "The same branch can be removed" ) 
            {
                REQUIRE( dsl_tee_branch_remove(demuxerName.c_str(), 
                    branchName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_tee_branch_count_get(demuxerName.c_str(), 
                    &count) == DSL_RESULT_SUCCESS );
                REQUIRE( count == 0 );
                REQUIRE( dsl_component_delete(demuxerName.c_str()) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Demuxer can add and remove Branches at a specific stream-id", "[tee-api]" )
{
    GIVEN( "A Demuxer and three Branches" ) 
    {
        std::wstring demuxerName(L"demuxer");
        std::wstring branchName0(L"branch-0");
        std::wstring branchName1(L"branch-1");
        std::wstring branchName2(L"branch-2");

        REQUIRE( dsl_tee_demuxer_new(demuxerName.c_str(), 
            3) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_branch_new(branchName0.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branchName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branchName2.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 4 );

        uint count(0);
        
        REQUIRE( dsl_tee_branch_count_get(demuxerName.c_str(), 
            &count) == DSL_RESULT_SUCCESS );
        REQUIRE( count == 0 );

        WHEN( "We attempt to add a Branch to the Demuxer beyond max_branches" ) 
        {
            REQUIRE( dsl_tee_demuxer_branch_add_to(demuxerName.c_str(), 
                branchName0.c_str(), 3) == DSL_RESULT_TEE_BRANCH_ADD_FAILED );

            THEN( "The branch should fail to add" ) 
            {
                REQUIRE( dsl_tee_branch_count_get(demuxerName.c_str(), 
                    &count) == DSL_RESULT_SUCCESS );
                REQUIRE( count == 0 );
                REQUIRE( dsl_component_delete(demuxerName.c_str()) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "We add a Branch to the Demuxer witin max_branches" ) 
        {
            REQUIRE( dsl_tee_demuxer_branch_add_to(demuxerName.c_str(), 
                branchName0.c_str(), 2) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_tee_branch_count_get(demuxerName.c_str(), 
                &count) == DSL_RESULT_SUCCESS );
            REQUIRE( count == 1 );

            // adding the same branch twice must fail
            REQUIRE( dsl_tee_demuxer_branch_add_to(demuxerName.c_str(), 
                branchName0.c_str(), 2) == DSL_RESULT_COMPONENT_IN_USE );
            REQUIRE( dsl_tee_branch_count_get(demuxerName.c_str(), 
                &count) == DSL_RESULT_SUCCESS );
            REQUIRE( count == 1 );

            THEN( "The same branch can be removed" ) 
            {
                REQUIRE( dsl_tee_branch_remove(demuxerName.c_str(), 
                    branchName0.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_tee_branch_count_get(demuxerName.c_str(), 
                    &count) == DSL_RESULT_SUCCESS );
                REQUIRE( count == 0 );

                // second call to remove the same branch must fail
                REQUIRE( dsl_tee_branch_remove(demuxerName.c_str(), 
                    branchName0.c_str()) == DSL_RESULT_TEE_BRANCH_IS_NOT_CHILD );
                
                REQUIRE( dsl_component_delete(demuxerName.c_str()) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Demuxer can move Branches correctly", "[tee-api]" )
{
    GIVEN( "A Demuxer and three Branches" ) 
    {
        std::wstring demuxerName(L"demuxer");
        std::wstring branchName0(L"branch-0");
        std::wstring branchName1(L"branch-1");
        std::wstring branchName2(L"branch-2");

        REQUIRE( dsl_tee_demuxer_new(demuxerName.c_str(), 
            3) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_branch_new(branchName0.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branchName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branchName2.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 4 );

        uint count(0);
        
        REQUIRE( dsl_tee_branch_count_get(demuxerName.c_str(), 
            &count) == DSL_RESULT_SUCCESS );
        REQUIRE( count == 0 );

        WHEN( "We add a Branch to the Demuxer witin max_branches" ) 
        {
            REQUIRE( dsl_tee_demuxer_branch_add_to(demuxerName.c_str(), 
                branchName0.c_str(), 2) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_tee_branch_count_get(demuxerName.c_str(), 
                &count) == DSL_RESULT_SUCCESS );
            REQUIRE( count == 1 );

            THEN( "The same branch can be moved an remove" ) 
            {
                REQUIRE( dsl_tee_demuxer_branch_move_to(demuxerName.c_str(), 
                    branchName0.c_str(), 0) == DSL_RESULT_SUCCESS );
                
                REQUIRE( dsl_tee_branch_remove(demuxerName.c_str(), 
                    branchName0.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_tee_branch_count_get(demuxerName.c_str(), 
                    &count) == DSL_RESULT_SUCCESS );
                REQUIRE( count == 0 );

                REQUIRE( dsl_component_delete(demuxerName.c_str()) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
            THEN( "Moving a branch to an occupied stream fails" ) 
            {
                REQUIRE( dsl_tee_demuxer_branch_add_to(demuxerName.c_str(), 
                    branchName1.c_str(), 0) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_tee_demuxer_branch_move_to(demuxerName.c_str(), 
                    branchName0.c_str(), 0) == DSL_RESULT_TEE_BRANCH_MOVE_FAILED );
                
                // remove should now fail
                REQUIRE( dsl_tee_branch_remove(demuxerName.c_str(), 
                    branchName0.c_str()) == DSL_RESULT_TEE_BRANCH_IS_NOT_CHILD );

                REQUIRE( dsl_component_delete(demuxerName.c_str()) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Demuxer can manages its max-branches correctly", "[tee-api]" )
{
    GIVEN( "A Demuxer and three Branchs" ) 
    {
        std::wstring demuxerName(L"demuxer");
        std::wstring branchName1(L"branch1");
        std::wstring branchName2(L"branch2");
        std::wstring branchName3(L"branch3");

        REQUIRE( dsl_tee_demuxer_new(demuxerName.c_str(),
            2) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branchName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branchName2.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branchName3.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "When max-branches number of Braches have been added" ) 
        {
            REQUIRE( dsl_tee_branch_add(demuxerName.c_str(), 
                branchName1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_tee_branch_add(demuxerName.c_str(), 
                branchName2.c_str()) == DSL_RESULT_SUCCESS );

            uint count(99);
            REQUIRE( dsl_tee_branch_count_get(demuxerName.c_str(), 
                &count) == DSL_RESULT_SUCCESS );
            REQUIRE( count == 2 );

            THEN( "Adding one more Branch must fail " ) 
            {
                REQUIRE( dsl_tee_branch_add(demuxerName.c_str(), 
                    branchName3.c_str()) == DSL_RESULT_TEE_BRANCH_ADD_FAILED );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Demuxer can update its max-branches correctly", "[tee-api]" )
{
    GIVEN( "A Demuxer and three Branchs" ) 
    {
        std::wstring demuxerName(L"demuxer");
        std::wstring branchName1(L"branch1");
        std::wstring branchName2(L"branch2");
        std::wstring branchName3(L"branch3");
        
        uint org_max_branches(2);

        REQUIRE( dsl_tee_demuxer_new(demuxerName.c_str(),
            org_max_branches) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branchName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branchName2.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branchName3.c_str()) == DSL_RESULT_SUCCESS );

        uint ret_max_branches(99);
        REQUIRE( dsl_tee_demuxer_max_branches_get(demuxerName.c_str(),
            &ret_max_branches) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_max_branches == org_max_branches );

        WHEN( "When the max-branches setting is updated prior to adding branches" ) 
        {
            uint new_max_branches(5);
            REQUIRE( dsl_tee_demuxer_max_branches_set(demuxerName.c_str(),
                new_max_branches) == DSL_RESULT_SUCCESS );
            
            THEN( "Correct value is returned on get" )
            {
                REQUIRE( dsl_tee_demuxer_max_branches_get(demuxerName.c_str(),
                    &ret_max_branches) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_branches == new_max_branches );
                
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "When max-branches branches have been added" ) 
        {
            REQUIRE( dsl_tee_branch_add(demuxerName.c_str(), 
                branchName1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_tee_branch_add(demuxerName.c_str(), 
                branchName2.c_str()) == DSL_RESULT_SUCCESS );

            uint count(99);
            REQUIRE( dsl_tee_branch_count_get(demuxerName.c_str(), 
                &count) == DSL_RESULT_SUCCESS );
            REQUIRE( count == 2 );

            THEN( "Reducing the max-branches setting must fail " ) 
            {
                REQUIRE( dsl_tee_demuxer_max_branches_set(demuxerName.c_str(),
                    org_max_branches-1) == DSL_RESULT_TEE_SET_FAILED );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Tee can update its blocking-timeout setting correctly", "[tee-api]" )
{
    GIVEN( "A Demuxer and three Branchs" ) 
    {
        std::wstring demuxerName(L"demuxer");
        
        REQUIRE( dsl_tee_demuxer_new(demuxerName.c_str(),
            10) == DSL_RESULT_SUCCESS );

        uint ret_blocking_timeout(99);
        REQUIRE( dsl_tee_blocking_timeout_get(demuxerName.c_str(),
            &ret_blocking_timeout) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_blocking_timeout == DSL_TEE_DEFAULT_BLOCKING_TIMEOUT_IN_SEC );

        WHEN( "When the blocking-timeout setting is updated" ) 
        {
            uint new_blocking_timeout(5);
            REQUIRE( dsl_tee_blocking_timeout_set(demuxerName.c_str(),
                new_blocking_timeout) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" )
            {
                REQUIRE( dsl_tee_blocking_timeout_get(demuxerName.c_str(),
                    &ret_blocking_timeout) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_blocking_timeout == new_blocking_timeout );
                
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

static boolean pad_probe_handler_cb1(void* buffer, void* user_data)
{
    return true;
}
static boolean pad_probe_handler_cb2(void* buffer, void* user_data)
{
    return true;
}

SCENARIO( "A Sink Pad Batch Meta Handler can be added and removed from a Demuxer", "[tee-api]" )
{
    GIVEN( "A new Demuxer and Custom PPH" ) 
    {
        std::wstring demuxerName(L"demuxer");
        std::wstring customPpmName(L"custom-ppm");

        REQUIRE( dsl_tee_demuxer_new(demuxerName.c_str(),
            1) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pph_custom_new(customPpmName.c_str(), pad_probe_handler_cb1, NULL) == DSL_RESULT_SUCCESS );

        WHEN( "A Sink Pad Probe Handler is added to the OSD" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE( dsl_tee_pph_remove(demuxerName.c_str(), customPpmName.c_str()) == DSL_RESULT_TEE_HANDLER_REMOVE_FAILED );

            REQUIRE( dsl_tee_pph_add(demuxerName.c_str(), customPpmName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Sink Pad Probe Handler can then be removed" ) 
            {
                REQUIRE( dsl_tee_pph_remove(demuxerName.c_str(), customPpmName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "A Sink Pad Probe Handler is added to the Primary GIE" ) 
        {
            REQUIRE( dsl_tee_pph_add(demuxerName.c_str(), customPpmName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "Attempting to add the same Sink Pad Probe Handler twice failes" ) 
            {
                REQUIRE( dsl_tee_pph_add(demuxerName.c_str(), customPpmName.c_str()) == DSL_RESULT_TEE_HANDLER_ADD_FAILED );
                REQUIRE( dsl_tee_pph_remove(demuxerName.c_str(), customPpmName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Tee API checks for NULL input parameters", "[tee-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring teeName  = L"test-tee";
        uint count(0);
        
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_tee_demuxer_new(NULL, 
                    1) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_demuxer_new_branch_add_many(teeName.c_str(), 
                    1, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_demuxer_max_branches_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_demuxer_max_branches_get(teeName.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_demuxer_max_branches_set(NULL, 
                    1) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_tee_demuxer_branch_add_to(NULL, 
                    NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_demuxer_branch_add_to(teeName.c_str(), 
                    NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_demuxer_branch_move_to(NULL, 
                    NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_demuxer_branch_move_to(teeName.c_str(), 
                    NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_tee_splitter_new(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_splitter_new_branch_add_many(teeName.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_tee_branch_add(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_branch_add(teeName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_branch_add_many(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_branch_add_many(teeName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_branch_remove(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_branch_remove(teeName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_branch_remove_many(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_branch_remove_many(teeName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_branch_count_get(NULL, &count) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_pph_add(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_pph_add(teeName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_pph_remove(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_pph_remove(teeName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_tee_blocking_timeout_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_blocking_timeout_get(teeName.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_blocking_timeout_set(NULL, 
                    1) == DSL_RESULT_INVALID_INPUT_PARAM );
                
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

