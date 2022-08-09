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
