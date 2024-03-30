/*
The MIT License

Copyright (c) 2024, Prominence AI, Inc.

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
#include "DslApi.h"


SCENARIO( "The Components container is updated correctly on new GST Pin", 
    "[gst-bin-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring bin_name(L"test-bin");

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new GST Thin is created" ) 
        {

            REQUIRE( dsl_gst_bin_new(bin_name.c_str()) == DSL_RESULT_SUCCESS );
            
            // second call must fail
            REQUIRE( dsl_gst_bin_new(bin_name.c_str()) ==
                DSL_RESULT_GST_BIN_NAME_NOT_UNIQUE );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}

SCENARIO( "The Components container is updated correctly on GST Bin delete", 
    "[gst-bin-api]" )
{
    GIVEN( "A new GST Bin in memory" ) 
    {
        std::wstring bin_name(L"test-bin");

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_gst_bin_new(bin_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "The new GST Bin is deleted" ) 
        {
            REQUIRE( dsl_component_delete(bin_name.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size is updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A GST Bin inin use can't be deleted", "[gst-bin-api]" )
{
    GIVEN( "A new GST Display and new Pipeline" ) 
    {
        std::wstring pipeline_name(L"test-pipeline");
        std::wstring bin_name(L"test-bin");

        REQUIRE( dsl_gst_bin_new(bin_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        WHEN( "The GST Bin is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
                bin_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The GST Bin can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(bin_name.c_str()) == 
                DSL_RESULT_COMPONENT_IN_USE );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A GST Bin, once removed from a Pipeline, can be deleted", 
    "[gst-bin-api]" )
{
    GIVEN( "A new Pipeline with a child GST Bin" ) 
    {
        std::wstring pipeline_name(L"test-pipeline");
        std::wstring bin_name(L"test-bin");

        REQUIRE( dsl_gst_bin_new(bin_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
            bin_name.c_str()) == DSL_RESULT_SUCCESS );
            
        WHEN( "The GST Bin is from the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipeline_name.c_str(), 
                bin_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The GST Bin can be deleted" ) 
            {
                REQUIRE( dsl_component_delete(bin_name.c_str()) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An GST Bin in use can't be added to a second Pipeline", 
    "[gst-bin-api]" )
{
    GIVEN( "A new GST Bin and two new pPipelines" ) 
    {
        std::wstring pipeline_name1(L"test-pipeline-1");
        std::wstring pipeline_name2(L"test-pipeline-2");
        
        std::wstring bin_name(L"test-bin");

        REQUIRE( dsl_gst_bin_new(bin_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipeline_name1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipeline_name2.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The GST Bin is added to the first Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipeline_name1.c_str(), 
                bin_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The GST Bin can't be added to the second Pipeline" ) 
            {
                REQUIRE( dsl_pipeline_component_add(pipeline_name2.c_str(), 
                    bin_name.c_str()) == DSL_RESULT_COMPONENT_IN_USE );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                
                REQUIRE( dsl_component_delete(bin_name.c_str()) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new GST Bin can Add and Remove a GST Element", "[gst-bin-api]" )
{
    GIVEN( "A new GST Bin and new GST Element" ) 
    {
        std::wstring bin_name(L"test-bin");

        std::wstring element_name(L"test-element");
        std::wstring factory_name(L"queue");

        REQUIRE( dsl_gst_bin_new(bin_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_gst_element_new(element_name.c_str(),
            factory_name.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The GST Element is added to the GST Bin" ) 
        {
            REQUIRE( dsl_gst_bin_element_add(bin_name.c_str(), 
                element_name.c_str()) == DSL_RESULT_SUCCESS );
            
            // Adding the same Element twice must fail
            REQUIRE( dsl_gst_bin_element_add(bin_name.c_str(), 
                element_name.c_str()) == DSL_RESULT_GST_ELEMENT_IN_USE );
            
            THEN( "The same GST Element can be removed correctly" ) 
            {
                REQUIRE( dsl_gst_bin_element_remove(bin_name.c_str(), 
                    element_name.c_str()) == DSL_RESULT_SUCCESS );

                // Removing the same Element twice must fail
                REQUIRE( dsl_gst_bin_element_remove(bin_name.c_str(), 
                    element_name.c_str()) == DSL_RESULT_GST_BIN_ELEMENT_NOT_IN_USE );
                
                REQUIRE( dsl_component_delete(bin_name.c_str()) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( dsl_gst_element_delete(element_name.c_str()) == 
                DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A new GST Bin can Add and Remove a Multiple GST Elements", 
    "[gst-bin-api]" )
{
    GIVEN( "A new GST Bin and new GST Element" ) 
    {
        std::wstring bin_name(L"test-bin");

        std::wstring element_name1(L"test-element1");
        std::wstring element_name2(L"test-element2");
        std::wstring element_name3(L"test-element3");
        std::wstring factory_name(L"queue");

        REQUIRE( dsl_gst_bin_new(bin_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_gst_element_new(element_name1.c_str(),
            factory_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_gst_element_new(element_name2.c_str(),
            factory_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_gst_element_new(element_name3.c_str(),
            factory_name.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The GST Elements are added to the GST Bin" ) 
        {
            REQUIRE( dsl_gst_bin_element_add(bin_name.c_str(), 
                element_name1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_gst_bin_element_add(bin_name.c_str(), 
                element_name2.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_gst_bin_element_add(bin_name.c_str(), 
                element_name3.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The same GST Elements can be removed correctly" ) 
            {
                REQUIRE( dsl_gst_bin_element_remove(bin_name.c_str(), 
                    element_name1.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_gst_bin_element_remove(bin_name.c_str(), 
                    element_name2.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_gst_bin_element_remove(bin_name.c_str(), 
                    element_name3.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_gst_element_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The GST Bin API checks for NULL input parameters", "[gst-bin-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring bin_name(L"test-bin");
        
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_gst_bin_new(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                
                REQUIRE( dsl_gst_bin_new_element_add_many(NULL, NULL)
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_bin_new_element_add_many(bin_name.c_str(), NULL)
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                
                REQUIRE( dsl_gst_bin_element_add(NULL, NULL)
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_bin_element_add(bin_name.c_str(), NULL)
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_gst_bin_element_add_many(NULL, NULL)
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_bin_element_add_many(bin_name.c_str(), NULL)
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_gst_bin_element_remove(NULL, NULL)
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_bin_element_remove(bin_name.c_str(), NULL)
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_gst_bin_element_remove_many(NULL, NULL)
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_bin_element_remove_many(bin_name.c_str(), NULL)
                    == DSL_RESULT_INVALID_INPUT_PARAM );
            }
        }
    }
}