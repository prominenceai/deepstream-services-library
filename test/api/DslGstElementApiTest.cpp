
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

SCENARIO( "An invalid factory name causes an exception", "[gst-element-api]" )
{
    GIVEN( "An empty list of Events" ) 
    {
        std::wstring element_name(L"element");
        
        std::wstring factory_name(L"non-element");
        
        REQUIRE( dsl_gst_element_list_size() == 0 );

        WHEN( "Several new Elements are created" ) 
        {
            REQUIRE( dsl_gst_element_new(element_name.c_str(),
                factory_name.c_str()) == DSL_RESULT_GST_ELEMENT_THREW_EXCEPTION );

            THEN( "The list size and events are not updated" ) 
            {
                REQUIRE( dsl_gst_element_list_size() == 0 );
            }
        }
    }
}
                
SCENARIO( "The GST Elements container is updated correctly on multiple new Elements", "[gst-element-api]" )
{
    GIVEN( "An empty list of Events" ) 
    {
        std::wstring element_name1(L"element-1");
        std::wstring element_name2(L"element-2");
        std::wstring element_name3(L"element-3");
        
        std::wstring factory_name(L"queue");
        
        REQUIRE( dsl_gst_element_list_size() == 0 );

        WHEN( "Several new Elements are created" ) 
        {
            REQUIRE( dsl_gst_element_new(element_name1.c_str(),
                factory_name.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_gst_element_new(element_name2.c_str(),
                factory_name.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_gst_element_new(element_name3.c_str(),
                factory_name.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size and events are updated correctly" ) 
            {
                REQUIRE( dsl_gst_element_list_size() == 3 );

                REQUIRE( dsl_gst_element_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_gst_element_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The GST Elements container is updated correctly on Delete GST Element", "[gst-element-api]" )
{
    GIVEN( "A list of several GST Elements" ) 
    {
        std::wstring element_name1(L"element-1");
        std::wstring element_name2(L"element-2");
        std::wstring element_name3(L"element-3");
        
        std::wstring factory_name(L"queue");
        
        REQUIRE( dsl_gst_element_list_size() == 0 );

        REQUIRE( dsl_gst_element_new(element_name1.c_str(),
            factory_name.c_str()) == DSL_RESULT_SUCCESS );

        // second creation of the same name must fail
        REQUIRE( dsl_gst_element_new(element_name1.c_str(),
            factory_name.c_str()) == DSL_RESULT_GST_ELEMENT_NAME_NOT_UNIQUE );
            
        REQUIRE( dsl_gst_element_new(element_name2.c_str(),
            factory_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_gst_element_new(element_name3.c_str(),
            factory_name.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A single Element is deleted" ) 
        {
            REQUIRE( dsl_gst_element_delete(element_name1.c_str()) == 
                DSL_RESULT_SUCCESS );
           
            // Deleting twice must faile
            REQUIRE( dsl_gst_element_delete(element_name1.c_str()) == 
                DSL_RESULT_GST_ELEMENT_NAME_NOT_FOUND );

            THEN( "The list size and events are updated correctly" ) 
            {
                REQUIRE( dsl_gst_element_list_size() == 2 );

                REQUIRE( dsl_gst_element_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_gst_element_list_size() == 0 );
            }
        }
        WHEN( "Multiple Elements are deleted" ) 
        {
            const wchar_t* elements[] = {L"element-2", L"element-3", NULL};
            
            REQUIRE( dsl_gst_element_delete_many(elements) == DSL_RESULT_SUCCESS );
            THEN( "The list size and events are updated correctly" ) 
            {
                REQUIRE( dsl_gst_element_list_size() == 1 );

                REQUIRE( dsl_gst_element_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_gst_element_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A GST Element can get and set properperties correctly",  "[gst-element-api]" )
{
    GIVEN( "An empty list of Events" ) 
    {
        std::wstring element_name1(L"element-1");
        std::wstring element_name2(L"element-2");
        std::wstring element_name3(L"element-3");
        
        std::wstring factory_name1(L"queue");
        std::wstring factory_name2(L"identity");
        std::wstring factory_name3(L"v4l2sink");
        
        std::wstring property_boolean(L"flush-on-eos");
        std::wstring property_float(L"drop-probability");
        std::wstring property_int(L"datarate");
        std::wstring property_uint(L"max-size-buffers");
        std::wstring property_int64(L"ts-offset");
        std::wstring property_uint64(L"max-size-time");
        std::wstring property_string(L"device");
        
        REQUIRE( dsl_gst_element_new(element_name1.c_str(),
            factory_name1.c_str()) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_gst_element_new(element_name2.c_str(),
            factory_name2.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_gst_element_new(element_name3.c_str(),
            factory_name3.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A boolean value is updated" ) 
        {
            // Test default value first
            boolean defValue(false), curValue(true), newValue(true);
            REQUIRE( dsl_gst_element_property_boolean_get(element_name1.c_str(),
                property_boolean.c_str(), &curValue) == DSL_RESULT_SUCCESS );
            REQUIRE( curValue == defValue );
            REQUIRE( dsl_gst_element_property_boolean_set(element_name1.c_str(),
                property_boolean.c_str(), newValue) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_gst_element_property_boolean_get(element_name1.c_str(),
                    property_boolean.c_str(), &curValue) == DSL_RESULT_SUCCESS );
                REQUIRE( curValue == newValue );
            
                REQUIRE( dsl_gst_element_delete_all() == 0 );
            }
        }
        WHEN( "A float value is updated" ) 
        {
            // Test default value first
            float defValue(0), curValue(99), newValue(0.23);
            REQUIRE( dsl_gst_element_property_float_get(element_name2.c_str(),
                property_float.c_str(), &curValue) == DSL_RESULT_SUCCESS );
            REQUIRE( curValue == defValue );
            REQUIRE( dsl_gst_element_property_float_set(element_name2.c_str(),
                property_float.c_str(), newValue) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_gst_element_property_float_get(element_name2.c_str(),
                    property_float.c_str(), &curValue) == DSL_RESULT_SUCCESS );
                REQUIRE( curValue == newValue );
            
                REQUIRE( dsl_gst_element_delete_all() == 0 );
            }
        }
        WHEN( "An int value is updated" ) 
        {
            // Test default value first
            int defValue(0), curValue(99), newValue(123);
            REQUIRE( dsl_gst_element_property_int_get(element_name2.c_str(),
                property_int.c_str(), &curValue) == DSL_RESULT_SUCCESS );
            REQUIRE( curValue == defValue );
            REQUIRE( dsl_gst_element_property_int_set(element_name2.c_str(),
                property_int.c_str(), newValue) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_gst_element_property_int_get(element_name2.c_str(),
                    property_int.c_str(), &curValue) == DSL_RESULT_SUCCESS );
                REQUIRE( curValue == newValue );
            
                REQUIRE( dsl_gst_element_delete_all() == 0 );
            }
        }
        WHEN( "An uint value is updated" ) 
        {
            // Test default value first
            uint defValue(200), curValue(99), newValue(123);
            REQUIRE( dsl_gst_element_property_uint_get(element_name1.c_str(),
                property_uint.c_str(), &curValue) == DSL_RESULT_SUCCESS );
            REQUIRE( curValue == defValue );
            REQUIRE( dsl_gst_element_property_uint_set(element_name1.c_str(),
                property_uint.c_str(), newValue) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_gst_element_property_uint_get(element_name1.c_str(),
                    property_uint.c_str(), &curValue) == DSL_RESULT_SUCCESS );
                REQUIRE( curValue == newValue );
            
                REQUIRE( dsl_gst_element_delete_all() == 0 );
            }
        }
        WHEN( "An int64 value is updated" ) 
        {
            // Test default value first
            int64_t defValue(0), curValue(99), newValue(123);
            REQUIRE( dsl_gst_element_property_int64_get(element_name2.c_str(),
                property_int64.c_str(), &curValue) == DSL_RESULT_SUCCESS );
            REQUIRE( curValue == defValue );
            REQUIRE( dsl_gst_element_property_int64_set(element_name2.c_str(),
                property_int64.c_str(), newValue) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_gst_element_property_int64_get(element_name2.c_str(),
                    property_int64.c_str(), &curValue) == DSL_RESULT_SUCCESS );
                REQUIRE( curValue == newValue );
            
                REQUIRE( dsl_gst_element_delete_all() == 0 );
            }
        }
        WHEN( "An uint64 value is updated" ) 
        {
            // Test default value first
            uint64_t defValue(1000000000), curValue(99), newValue(123);
            REQUIRE( dsl_gst_element_property_uint64_get(element_name1.c_str(),
                property_uint64.c_str(), &curValue) == DSL_RESULT_SUCCESS );
            REQUIRE( curValue == defValue );
            REQUIRE( dsl_gst_element_property_uint64_set(element_name1.c_str(),
                property_uint64.c_str(), newValue) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_gst_element_property_uint64_get(element_name1.c_str(),
                    property_uint64.c_str(), &curValue) == DSL_RESULT_SUCCESS );
                REQUIRE( curValue == newValue );
            
                REQUIRE( dsl_gst_element_delete_all() == 0 );
            }
        }
        WHEN( "A string value is updated" ) 
        {
            // Test default value first
            const wchar_t* curCValue;
            std::wstring defValue(L"/dev/video1");
            std::wstring newValue(L"/dev/video2");
            REQUIRE( dsl_gst_element_property_string_get(element_name3.c_str(),
                property_string.c_str(), &curCValue) == DSL_RESULT_SUCCESS );
            std::wstring curValue(curCValue);
            REQUIRE( curValue == defValue );
            REQUIRE( dsl_gst_element_property_string_set(element_name3.c_str(),
                property_string.c_str(), newValue.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_gst_element_property_string_get(element_name3.c_str(),
                    property_string.c_str(), &curCValue) == DSL_RESULT_SUCCESS );
                std::wstring curValue = curCValue;
                REQUIRE( curValue == newValue );
            
                REQUIRE( dsl_gst_element_delete_all() == 0 );
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
SCENARIO( "A Sink Pad Probe Handler can be added and removed from a GST Element", 
    "[gst-element-api]" )
{
    GIVEN( "A new GST Element and Custom PPH" ) 
    {
        std::wstring element_name(L"element");
        std::wstring factory_name(L"queue");

        std::wstring customPpmName(L"custom-ppm");

        REQUIRE( dsl_gst_element_new(element_name.c_str(),
            factory_name.c_str()) == DSL_RESULT_SUCCESS );
 
        REQUIRE( dsl_pph_custom_new(customPpmName.c_str(), 
            pad_probe_handler_cb1, NULL) == DSL_RESULT_SUCCESS );

        WHEN( "A Sink Pad Probe Handler is added to the GST Element" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE( dsl_gst_element_pph_remove(element_name.c_str(), customPpmName.c_str(), 
                DSL_PAD_SINK) == DSL_RESULT_GST_ELEMENT_HANDLER_REMOVE_FAILED );

            REQUIRE(  dsl_gst_element_pph_add(element_name.c_str(), customPpmName.c_str(), 
                DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
            
            THEN( "The Padd Probe Handler can then be removed" ) 
            {
                REQUIRE( dsl_gst_element_pph_remove(element_name.c_str(), 
                    customPpmName.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_gst_element_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "A Sink Pad Probe Handler is added to the GST Element" ) 
        {
            REQUIRE(  dsl_gst_element_pph_add(element_name.c_str(), customPpmName.c_str(), 
                DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
            
            THEN( "Attempting to add the same Sink Pad Probe Handler twice failes" ) 
            {
                REQUIRE(  dsl_gst_element_pph_add(element_name.c_str(), customPpmName.c_str(), 
                    DSL_PAD_SINK) == DSL_RESULT_GST_ELEMENT_HANDLER_ADD_FAILED );
                REQUIRE( dsl_gst_element_pph_remove(element_name.c_str(), customPpmName.c_str(), 
                    DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_gst_element_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Source Pad Probe Handler can be added and removed from a GST Element", 
    "[gst-element-api]" )
{
    GIVEN( "A new GST Element and Custom PPH" ) 
    {
        std::wstring element_name(L"element");
        std::wstring factory_name(L"queue");

        std::wstring customPpmName(L"custom-ppm");

        REQUIRE( dsl_gst_element_new(element_name.c_str(),
            factory_name.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pph_custom_new(customPpmName.c_str(), pad_probe_handler_cb1, 
            NULL) == DSL_RESULT_SUCCESS );

        WHEN( "A Source Pad Probe Handler is added to the GST Element" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE( dsl_gst_element_pph_remove(element_name.c_str(), customPpmName.c_str(), 
                DSL_PAD_SRC) == DSL_RESULT_GST_ELEMENT_HANDLER_REMOVE_FAILED );

            REQUIRE(  dsl_gst_element_pph_add(element_name.c_str(), customPpmName.c_str(), 
                DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
            
            THEN( "The Padd Probe Handler can then be removed" ) 
            {
                REQUIRE( dsl_gst_element_pph_remove(element_name.c_str(), 
                    customPpmName.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_gst_element_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "A Source Pad Probe Handler is added to the GST Element" ) 
        {
            REQUIRE(  dsl_gst_element_pph_add(element_name.c_str(), customPpmName.c_str(), 
                DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
            
            THEN( "Attempting to add the same Source Pad Probe Handler twice failes" ) 
            {
                REQUIRE(  dsl_gst_element_pph_add(element_name.c_str(), customPpmName.c_str(), 
                    DSL_PAD_SRC) == DSL_RESULT_GST_ELEMENT_HANDLER_ADD_FAILED );
                REQUIRE( dsl_gst_element_pph_remove(element_name.c_str(), 
                    customPpmName.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_gst_element_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Element API checks for NULL input parameters", "[gst-element-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring element_name(L"element");
        std::wstring property(L"property");
        
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_gst_element_new(NULL,
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_new(element_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_gst_element_property_boolean_get(NULL,
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_boolean_get(element_name.c_str(),
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_boolean_get(element_name.c_str(),
                    property.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_boolean_set(NULL,
                    NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_boolean_set(element_name.c_str(),
                    NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_gst_element_property_float_get(NULL,
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_float_get(element_name.c_str(),
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_float_get(element_name.c_str(),
                    property.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_float_set(NULL,
                    NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_float_set(element_name.c_str(),
                    NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_gst_element_property_int_get(NULL,
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_int_get(element_name.c_str(),
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_int_get(element_name.c_str(),
                    property.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_int_set(NULL,
                    NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_int_set(element_name.c_str(),
                    NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_gst_element_property_uint_get(NULL,
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_uint_get(element_name.c_str(),
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_uint_get(element_name.c_str(),
                    property.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_uint_set(NULL,
                    NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_uint_set(element_name.c_str(),
                    NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_gst_element_property_int64_get(NULL,
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_int64_get(element_name.c_str(),
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_int64_get(element_name.c_str(),
                    property.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_int64_set(NULL,
                    NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_int64_set(element_name.c_str(),
                    NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_gst_element_property_uint64_get(NULL,
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_uint64_get(element_name.c_str(),
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_uint64_get(element_name.c_str(),
                    property.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_uint64_set(NULL,
                    NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_uint64_set(element_name.c_str(),
                    NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_gst_element_property_string_get(NULL,
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_string_get(element_name.c_str(),
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_string_get(element_name.c_str(),
                    property.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_string_set(NULL,
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_string_set(element_name.c_str(),
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_gst_element_property_string_set(element_name.c_str(),
                    property.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
            }
        }
    }
}
