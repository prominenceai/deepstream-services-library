/*
The MIT License

Copyright (c) 2022, Prominence AI, Inc.

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

static const std::wstring nmp_pph_name(L"nmp-pph");

static std::wstring label_file1(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/labels.txt");

static std::wstring label_file2(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Secondary_CarColor/labels.txt");

SCENARIO( "A new Non Maximum Processor (NMP) Pad Probe Handler", "[pph-nmp-api]" )
{
    GIVEN( "Attributes for a NMP PPH" ) 
    {
        REQUIRE( dsl_pph_list_size() == 0 );

        WHEN( "A new ODE PPH is created with Supress and IOU settings" ) 
        {
            uint process_method(DSL_NMP_PROCESS_METHOD_SUPRESS);
            uint match_method(DSL_NMP_MATCH_METHOD_IOU);
            float match_threshold(0.5);
            
            REQUIRE( dsl_pph_nmp_new(nmp_pph_name.c_str(), label_file1.c_str(),
                process_method, match_method, match_threshold) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_pph_list_size() == 1 );
                
                const wchar_t* c_ret_label_file;
                REQUIRE( dsl_pph_nmp_label_file_get(nmp_pph_name.c_str(),
                    &c_ret_label_file) == DSL_RESULT_SUCCESS );
                std::wstring ret_label_file(c_ret_label_file);
                REQUIRE( ret_label_file == label_file1);
                
                uint ret_process_method(99);
                REQUIRE( dsl_pph_nmp_process_method_get(nmp_pph_name.c_str(),
                    &ret_process_method) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_process_method == process_method );
                
                uint ret_match_method(99);
                float ret_match_threshold(0);
                REQUIRE( dsl_pph_nmp_match_settings_get(nmp_pph_name.c_str(),
                    &ret_match_method, &ret_match_threshold) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_match_method == match_method );
                REQUIRE( ret_match_threshold == match_threshold );
                
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_list_size() == 0 );
            }
        }
        WHEN( "A new ODE PPH is created with Merge and IOS settings" ) 
        {
            uint process_method(DSL_NMP_PROCESS_METHOD_MERGE);
            uint match_method(DSL_NMP_MATCH_METHOD_IOS);
            float match_threshold(0.5);
            
            REQUIRE( dsl_pph_nmp_new(nmp_pph_name.c_str(), label_file1.c_str(),
                process_method, match_method, match_threshold) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_pph_list_size() == 1 );
                
                const wchar_t* c_ret_label_file;
                REQUIRE( dsl_pph_nmp_label_file_get(nmp_pph_name.c_str(),
                    &c_ret_label_file) == DSL_RESULT_SUCCESS );
                std::wstring ret_label_file(c_ret_label_file);
                REQUIRE( ret_label_file == label_file1);
                
                uint ret_process_method(99);
                REQUIRE( dsl_pph_nmp_process_method_get(nmp_pph_name.c_str(),
                    &ret_process_method) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_process_method == process_method );
                
                uint ret_match_method(99);
                float ret_match_threshold(0);
                REQUIRE( dsl_pph_nmp_match_settings_get(nmp_pph_name.c_str(),
                    &ret_match_method, &ret_match_threshold) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_match_method == match_method );
                REQUIRE( ret_match_threshold == match_threshold );
                
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Non Maximum Processor (NMP) Pad Probe Handler can set/get its label file correctly", 
    "[pph-nmp-api]" )
{
    GIVEN( "Attributes for a NMP PPH" ) 
    {
        REQUIRE( dsl_pph_list_size() == 0 );

        uint process_method(DSL_NMP_PROCESS_METHOD_SUPRESS);
        uint match_method(DSL_NMP_MATCH_METHOD_IOU);
        float match_threshold(0.5);
        
        REQUIRE( dsl_pph_nmp_new(nmp_pph_name.c_str(), label_file1.c_str(),
            process_method, match_method, match_threshold) == DSL_RESULT_SUCCESS );

        const wchar_t* c_ret_label_file;
        REQUIRE( dsl_pph_nmp_label_file_get(nmp_pph_name.c_str(),
            &c_ret_label_file) == DSL_RESULT_SUCCESS );
        std::wstring ret_label_file(c_ret_label_file);
        REQUIRE( ret_label_file == label_file1);

        WHEN( "The label file is set to a new file path" ) 
        {
            REQUIRE( dsl_pph_nmp_label_file_set(nmp_pph_name.c_str(),
                label_file2.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct file path is returned on get call" ) 
            {
                REQUIRE( dsl_pph_nmp_label_file_get(nmp_pph_name.c_str(),
                    &c_ret_label_file) == DSL_RESULT_SUCCESS );
                ret_label_file = c_ret_label_file;
                REQUIRE( ret_label_file == label_file2);
                
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_list_size() == 0 );
            }
        }
        WHEN( "The label file is set to NULL for agnostic non-maximum processing" ) 
        {
            REQUIRE( dsl_pph_nmp_label_file_set(nmp_pph_name.c_str(),
                NULL) == DSL_RESULT_SUCCESS );

            THEN( "NULL is correctly returned on get" ) 
            {
                REQUIRE( dsl_pph_nmp_label_file_get(nmp_pph_name.c_str(),
                    &c_ret_label_file) == DSL_RESULT_SUCCESS );
                REQUIRE( c_ret_label_file == NULL);
                
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_list_size() == 0 );
            }
        }
        WHEN( "An invalid label file path is used " ) 
        {
            std::wstring invalid_path(L"./bad/path");

            THEN( "The set call fails with the correct result code" ) 
            {
                REQUIRE( dsl_pph_nmp_label_file_set(nmp_pph_name.c_str(),
                    invalid_path.c_str()) == DSL_RESULT_PPH_SET_FAILED );

                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Non Maximum Processor (NMP) Pad Probe Handler can set/get its process mode correctly", 
    "[pph-nmp-api]" )
{
    GIVEN( "Attributes for a NMP PPH" ) 
    {
        REQUIRE( dsl_pph_list_size() == 0 );

        uint process_method(DSL_NMP_PROCESS_METHOD_SUPRESS);
        uint match_method(DSL_NMP_MATCH_METHOD_IOU);
        float match_threshold(0.5);
        
        REQUIRE( dsl_pph_nmp_new(nmp_pph_name.c_str(), label_file1.c_str(),
            process_method, match_method, match_threshold) == DSL_RESULT_SUCCESS );

        uint ret_process_method(99);
        REQUIRE( dsl_pph_nmp_process_method_get(nmp_pph_name.c_str(),
            &ret_process_method) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_process_method == process_method );
        
        WHEN( "The process method is set to Merge" ) 
        {
            REQUIRE( dsl_pph_nmp_process_method_set(nmp_pph_name.c_str(),
                DSL_NMP_PROCESS_METHOD_MERGE) == DSL_RESULT_SUCCESS );

            THEN( "The correct setting is returned on get call" ) 
            {
                REQUIRE( dsl_pph_nmp_process_method_get(nmp_pph_name.c_str(),
                    &ret_process_method) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_process_method == DSL_NMP_PROCESS_METHOD_MERGE );
                
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_list_size() == 0 );
            }
        }
        WHEN( "An invalid process method is used on set" ) 
        {
            uint new_process_mode = DSL_NMP_PROCESS_METHOD_MERGE+1;

            THEN( "The set call fails with the correct result code" ) 
            {
                REQUIRE( dsl_pph_nmp_process_method_set(nmp_pph_name.c_str(),
                    new_process_mode) == DSL_RESULT_PPH_SET_FAILED );
                
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Non Maximum Processor (NMP) Pad Probe Handler can set/get its match settings correctly", 
    "[pph-nmp-api]" )
{
    GIVEN( "Attributes for a NMP PPH" ) 
    {
        REQUIRE( dsl_pph_list_size() == 0 );

        uint process_method(DSL_NMP_PROCESS_METHOD_SUPRESS);
        uint match_method(DSL_NMP_MATCH_METHOD_IOU);
        float match_threshold(0.5);
        
        REQUIRE( dsl_pph_nmp_new(nmp_pph_name.c_str(), label_file1.c_str(),
            process_method, match_method, match_threshold) == DSL_RESULT_SUCCESS );

        uint ret_match_method(99);
        float ret_match_threshold(0);
        REQUIRE( dsl_pph_nmp_match_settings_get(nmp_pph_name.c_str(),
            &ret_match_method, &ret_match_threshold) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_match_method == match_method );
        REQUIRE( ret_match_threshold == match_threshold );
        
        WHEN( "The match settings are updated" ) 
        {
            uint new_match_method(DSL_NMP_MATCH_METHOD_IOS);
            float new_match_threshold(0.75);
            
            REQUIRE( dsl_pph_nmp_match_settings_set(nmp_pph_name.c_str(),
                new_match_method, new_match_threshold) == DSL_RESULT_SUCCESS );

            THEN( "The correct settings are returned on get call" ) 
            {
                REQUIRE( dsl_pph_nmp_match_settings_get(nmp_pph_name.c_str(),
                    &ret_match_method, &ret_match_threshold) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_match_method == new_match_method );
                REQUIRE( ret_match_threshold == new_match_threshold );
                
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_list_size() == 0 );
            }
        }
        WHEN( "An invalid match method is used on set" ) 
        {
            uint new_match_method(DSL_NMP_MATCH_METHOD_IOS+1);
            float new_match_threshold(0.75);

            THEN( "The set call fails with the correct result code" ) 
            {
                REQUIRE( dsl_pph_nmp_match_settings_set(nmp_pph_name.c_str(),
                    new_match_method, new_match_threshold) == DSL_RESULT_PPH_SET_FAILED );
                
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_list_size() == 0 );
            }
        }
        WHEN( "An invalid match threshold is used on set" ) 
        {
            uint new_match_method(DSL_NMP_MATCH_METHOD_IOS);
            float new_match_threshold(1.01);

            THEN( "The set call fails with the correct result code" ) 
            {
                REQUIRE( dsl_pph_nmp_match_settings_set(nmp_pph_name.c_str(),
                    new_match_method, new_match_threshold) == DSL_RESULT_PPH_SET_FAILED );
                
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A invalid Non Maximum Processor (NMP) Pad Probe Handler fails on all set/get calls", 
    "[pph-nmp-api]" )
{
    GIVEN( "Attributes for an invalid NMP PPH - using ODE" ) 
    {
        std::wstring invalid_pph_name(L"invalid-pph");

        uint process_method(DSL_NMP_PROCESS_METHOD_SUPRESS);
        uint match_method(DSL_NMP_MATCH_METHOD_IOU);
        float match_threshold(0.5);
        
        WHEN( "The invalid NMP PPH is created" ) 
        {
            REQUIRE( dsl_pph_ode_new(invalid_pph_name.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The all set/get calls fails with the correct result code" ) 
            {
                const wchar_t* ret_lable_file;
                REQUIRE( dsl_pph_nmp_label_file_get(invalid_pph_name.c_str(),
                    &ret_lable_file) == 
                    DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE );
                REQUIRE( dsl_pph_nmp_label_file_set(invalid_pph_name.c_str(),
                    label_file1.c_str()) == 
                    DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE );
                REQUIRE( dsl_pph_nmp_process_method_get(invalid_pph_name.c_str(),
                    &process_method) == 
                    DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE );
                REQUIRE( dsl_pph_nmp_process_method_set(invalid_pph_name.c_str(),
                    process_method) == 
                    DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE );
                REQUIRE( dsl_pph_nmp_match_settings_get(invalid_pph_name.c_str(),
                    &match_method, &match_threshold) == 
                    DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE );
                REQUIRE( dsl_pph_nmp_match_settings_set(invalid_pph_name.c_str(),
                    match_method, match_threshold) == 
                    DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE );
                
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_list_size() == 0 );
            }
        }
    }
}            

SCENARIO( "The Non Maximum Processor (NMP) Pad Probe Handler API checks for NULL pointers correctly", 
    "[pph-nmp-api]" )
{
    GIVEN( "Attributes for an invalid NMP PPH - using ODE" ) 
    {
        std::wstring pph_name(L"nmp-pph");

        uint process_method(DSL_NMP_PROCESS_METHOD_SUPRESS);
        uint match_method(DSL_NMP_MATCH_METHOD_IOU);
        float match_threshold(0.5);
        
        WHEN( "The invalid NMP PPH is created" ) 
        {
            
            THEN( "The all set/get calls fails with the correct result code" ) 
            {
                const wchar_t* ret_lable_file;
                
                REQUIRE( dsl_pph_nmp_new(NULL, label_file1.c_str(),
                    process_method, match_method, match_threshold) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                
                REQUIRE( dsl_pph_nmp_label_file_get(NULL, &ret_lable_file) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pph_nmp_label_file_get(pph_name.c_str(), NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pph_nmp_label_file_set(NULL, label_file1.c_str()) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_pph_nmp_process_method_get(NULL, &process_method) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pph_nmp_process_method_get(pph_name.c_str(), NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pph_nmp_process_method_set(NULL,
                    process_method) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_pph_nmp_match_settings_get(NULL,
                    &match_method, &match_threshold) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pph_nmp_match_settings_get(pph_name.c_str(),
                    NULL, &match_threshold) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pph_nmp_match_settings_get(pph_name.c_str(),
                    &match_method, NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pph_nmp_match_settings_set(NULL,
                    match_method, match_threshold) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_list_size() == 0 );
            }
        }
    }
}            

        
        
        