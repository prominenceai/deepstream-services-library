/*
The MIT License

Copyright (c) 2021, Prominence AI, Inc.

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

SCENARIO( "All SMTP Properties can be set and returned back correctly", "[mailer-api]" )
{
    GIVEN( "A set of SMTP credentials" ) 
    {
        std::wstring mailer_name(L"mailer");
        std::wstring username(L"joe.blow");
        std::wstring password(L"3littlepigs");
        std::wstring mail_server(L"smtp://mail.example.com");
        std::wstring from_name(L"Joe Blow");
        std::wstring from_address(L"joe.blow@example.com");
        std::wstring to_name1(L"Joe Blow");
        std::wstring to_address1(L"joe.blow@example.org");
        std::wstring to_name2(L"Jack Black");
        std::wstring to_address2(L"jack.black@example.org");
        std::wstring cc_name1(L"Jane Doe");
        std::wstring cc_address1(L"jane.doe@example.org");
        std::wstring cc_name2(L"Bill Williams");
        std::wstring cc_address2(L"bill.williams@example.org");
        
        REQUIRE( dsl_mailer_new(mailer_name.c_str()) == DSL_RESULT_SUCCESS );
        
        WHEN( "All Properties are set" ) 
        {
            REQUIRE( dsl_mailer_credentials_set(mailer_name.c_str(), username.c_str(), 
                password.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_mailer_server_url_set(mailer_name.c_str(), 
                mail_server.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_mailer_address_from_set(mailer_name.c_str(), 
                from_name.c_str(), from_address.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_mailer_ssl_enabled_set(mailer_name.c_str(), false) == DSL_RESULT_SUCCESS );
            
            REQUIRE( dsl_mailer_address_to_add(mailer_name.c_str(),
                to_name1.c_str(), to_address1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_mailer_address_to_add(mailer_name.c_str(),
                to_name2.c_str(), to_address2.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_mailer_address_to_add(mailer_name.c_str(),
                cc_name1.c_str(), cc_address1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_mailer_address_to_add(mailer_name.c_str(),
                cc_name2.c_str(), cc_address2.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                const wchar_t* ret_mail_server_str;
                const wchar_t* ret_from_name_str;
                const wchar_t* ret_from_address_str;

                REQUIRE( dsl_mailer_server_url_get(mailer_name.c_str(),
                    &ret_mail_server_str) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_mailer_address_from_get(mailer_name.c_str(),
                    &ret_from_name_str, &ret_from_address_str) == DSL_RESULT_SUCCESS );

                std::wstring ret_mail_server(ret_mail_server_str);
                std::wstring ret_from_name(ret_from_name_str);
                std::wstring ret_from_address(ret_from_address_str);
                boolean ret_ssl_enabled(true);
                
                REQUIRE( ret_mail_server == mail_server );
                REQUIRE( ret_from_name == from_name );
                REQUIRE( ret_from_address == from_address );
                REQUIRE( dsl_mailer_ssl_enabled_get(mailer_name.c_str(),
                    &ret_ssl_enabled) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_ssl_enabled == false );
                
                dsl_mailer_address_to_remove_all(mailer_name.c_str());
                dsl_mailer_address_cc_remove_all(mailer_name.c_str());
                
                REQUIRE( dsl_mailer_delete(mailer_name.c_str()) == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "A SMTP Test Message can be Queued", "[mailer-api]" )
{
    GIVEN( "A set of SMTP credentials and setup parameters " ) 
    {
        std::wstring mailer_name(L"mailer");
        std::wstring username(L"joe.blow");
        std::wstring password(L"3littlepigs");
        std::wstring mail_server(L"smtp://mail.gmail.com");
        std::wstring from_name(L"Joe Blow");
        std::wstring from_address(L"joe.blow@gmail.com");
        std::wstring to_name1(L"Joe Blow");
        std::wstring to_address1(L"joe.blow@gmail.com");

        REQUIRE( dsl_mailer_new(mailer_name.c_str()) == DSL_RESULT_SUCCESS );
        
        WHEN( "All Properties are set" ) 
        {
            REQUIRE( dsl_mailer_credentials_set(mailer_name.c_str(),
                username.c_str(), password.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_mailer_server_url_set(mailer_name.c_str(),
                mail_server.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_mailer_address_from_set(mailer_name.c_str(),
                from_name.c_str(), from_address.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_mailer_ssl_enabled_set(mailer_name.c_str(),
                false) == DSL_RESULT_SUCCESS );
            
            REQUIRE( dsl_mailer_address_to_add(mailer_name.c_str(),
            to_name1.c_str(), to_address1.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "A Test Message can be queued" ) 
            {
                REQUIRE( dsl_mailer_test_message_send(mailer_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_mailer_delete(mailer_name.c_str()) == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "The SMTP API checks for NULL input parameters", "[mailer-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring mailer_name(L"mailer");
        std::wstring username(L"joe.blow");
        std::wstring password(L"3littlepigs");

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                
                REQUIRE( dsl_mailer_credentials_set(NULL, NULL,
                    password.c_str()) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_mailer_credentials_set(mailer_name.c_str(),NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_mailer_credentials_set(mailer_name.c_str(),username.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
            }
        }
    }
}
