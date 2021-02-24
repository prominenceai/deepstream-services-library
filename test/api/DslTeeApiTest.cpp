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

SCENARIO( "The Components container is updated correctly on new Demuxer", "[tee-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring demuxerName(L"demuxer");

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Demuxer is created" ) 
        {
            REQUIRE( dsl_tee_demuxer_new(demuxerName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Components container is updated correctly on Demuxer delete", "[tee-api]" )
{
    GIVEN( "A new Demuxer in memory" ) 
    {
        std::wstring demuxerName(L"demuxer");

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_tee_demuxer_new(demuxerName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "The new Demuxer is deleted" ) 
        {
            REQUIRE( dsl_component_delete(demuxerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size is updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

static boolean pad_probe_handler_cb1(void* buffer, void* user_data)
{
}
static boolean pad_probe_handler_cb2(void* buffer, void* user_data)
{
}

SCENARIO( "A Sink Pad Batch Meta Handler can be added and removed from a Demuxer", "[tee-api]" )
{
    GIVEN( "A new Demuxer and Custom PPH" ) 
    {
        std::wstring demuxerName(L"demuxer");
        std::wstring customPpmName(L"custom-ppm");

        REQUIRE( dsl_tee_demuxer_new(demuxerName.c_str()) == DSL_RESULT_SUCCESS );

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

SCENARIO( "A Demuxer can add and remove a Branch", "[tee-api]" )
{
    GIVEN( "A Demuxer and Branch" ) 
    {
        std::wstring demuxerName(L"demuxer");
        std::wstring branchName(L"branch");

        REQUIRE( dsl_tee_demuxer_new(demuxerName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branchName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 2 );

        uint count(0);
        
        REQUIRE( dsl_tee_branch_count_get(demuxerName.c_str(), &count) == DSL_RESULT_SUCCESS );
        REQUIRE( count == 0 );

        WHEN( "A the Branch is added to the Demuxer" ) 
        {
            REQUIRE( dsl_tee_branch_add(demuxerName.c_str(), branchName.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_tee_branch_count_get(demuxerName.c_str(), &count) == DSL_RESULT_SUCCESS );
            REQUIRE( count == 1 );

            THEN( "The same branch can be removed" ) 
            {
                REQUIRE( dsl_tee_branch_remove(demuxerName.c_str(), branchName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_tee_branch_count_get(demuxerName.c_str(), &count) == DSL_RESULT_SUCCESS );
                REQUIRE( count == 0 );
                REQUIRE( dsl_component_delete(demuxerName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Demuxer can add and remove many Branches", "[tee-api]" )
{
    GIVEN( "A Demuxer and Branch" ) 
    {
        std::wstring demuxerName(L"demuxer");
        std::wstring branchName1(L"branch1");
        std::wstring branchName2(L"branch2");
        std::wstring branchName3(L"branch3");

        REQUIRE( dsl_tee_demuxer_new(demuxerName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branchName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branchName2.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branchName3.c_str()) == DSL_RESULT_SUCCESS );

        uint count(0);
        
        REQUIRE( dsl_tee_branch_count_get(demuxerName.c_str(), &count) == DSL_RESULT_SUCCESS );
        REQUIRE( count == 0 );

        WHEN( "A the Branch is added to the Demuxer" ) 
        {
            const wchar_t* branches[] = {L"branch1", L"branch2", L"branch3", NULL};
            
            REQUIRE( dsl_tee_branch_add_many(demuxerName.c_str(), branches) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_tee_branch_count_get(demuxerName.c_str(), &count) == DSL_RESULT_SUCCESS );
            REQUIRE( count == 3 );

            THEN( "The same branch can be removed" ) 
            {
                REQUIRE( dsl_tee_branch_remove_many(demuxerName.c_str(), branches) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_tee_branch_count_get(demuxerName.c_str(), &count) == DSL_RESULT_SUCCESS );
                REQUIRE( count == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
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
                REQUIRE( dsl_tee_demuxer_new(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_splitter_new(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_demuxer_new_branch_add_many(teeName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_splitter_new_branch_add_many(teeName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
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
                
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

