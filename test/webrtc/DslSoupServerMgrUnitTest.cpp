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
#include "Dsl.h"
#include <glib/gstdio.h>
#include <gtypes.h>
#include "DslSoupServerMgr.h"

using namespace DSL;

SCENARIO( "A Signaling Transceiver is created correctly", "[SoupServerMgr]" )
{
    GIVEN( "Attribute for a new Client Reciever" )
    {
        WHEN( "The Signaling Transceiver is created" )
        {
            std::unique_ptr<SignalingTransceiver> pSignalingTransceiver= 
                std::unique_ptr<SignalingTransceiver>(new SignalingTransceiver());

            THEN( "All attributes are setup correctly" )
            {
                REQUIRE( pSignalingTransceiver->GetConnection() == NULL);
            }
        }
    }
}

SCENARIO( "A Soup Server Manager can listen on and disconnect from a specific port number", "[SoupServerMgr]" )
{
    GIVEN( "A the Soup Server manager" )
    {
        uint initialPortNumber(99);

        REQUIRE( SoupServerMgr::GetMgr()->GetListeningState(&initialPortNumber) == false );
        REQUIRE( initialPortNumber == 0 );

        WHEN( "The Soup Server Manager starts listening on a specified port" )
        {
            uint newPortNumber(DSL_WEBSOCKET_SERVER_DEFAULT_WEBSOCKET_PORT);
            uint retPortNumber(0);
            REQUIRE( SoupServerMgr::GetMgr()->StartListening(newPortNumber) == true );
            REQUIRE( SoupServerMgr::GetMgr()->GetListeningState(&retPortNumber) == true );
            REQUIRE( retPortNumber == newPortNumber );

            // Second call to start when already listening must fail
            REQUIRE( SoupServerMgr::GetMgr()->StartListening(newPortNumber) == false );

            THEN( "The Soup Server Manager can stop listening successfully." )
            {
                REQUIRE( SoupServerMgr::GetMgr()->StopListening() == true );

                // Second call with the same Signaling Transceiver must fail
                REQUIRE( SoupServerMgr::GetMgr()->StopListening() == false );
                REQUIRE( SoupServerMgr::GetMgr()->GetListeningState(&initialPortNumber) == false );
                REQUIRE( initialPortNumber == 0 );
            }
        }
    }
}

SCENARIO( "A Soup Server Manager can add and listen for a new URI", "[SoupServerMgr]" )
{
    GIVEN( "A the Soup Server manager" )
    {
        uint initialPortNumber(99);

        REQUIRE( SoupServerMgr::GetMgr()->GetListeningState(&initialPortNumber) == false );
        REQUIRE( initialPortNumber == 0 );

        WHEN( "A new URI is added to the Soup Server Manager" )
        {
            std::string newPath("/ws/new_path");
            REQUIRE( SoupServerMgr::GetMgr()->AddPath(newPath.c_str()) == true );

            THEN( "The Soup Server Manager starts and stops listening correctly." )
            {
                uint newPortNumber(DSL_WEBSOCKET_SERVER_DEFAULT_WEBSOCKET_PORT);
                uint retPortNumber(0);
                REQUIRE( SoupServerMgr::GetMgr()->StartListening(newPortNumber) == true );
                REQUIRE( SoupServerMgr::GetMgr()->GetListeningState(&retPortNumber) == true );
                REQUIRE( retPortNumber == newPortNumber );

                REQUIRE( SoupServerMgr::GetMgr()->StopListening() == true );
                REQUIRE( SoupServerMgr::GetMgr()->GetListeningState(&initialPortNumber) == false );
                REQUIRE( initialPortNumber == 0 );
            }
        }
    }
}

SCENARIO( "A Signaling Transceiver can be added and removed ", "[SoupServerMgr]" )
{
    GIVEN( "A new Client Reciever" )
    {
        std::unique_ptr<SignalingTransceiver> pSignalingTransceiver= 
            std::unique_ptr<SignalingTransceiver>(new SignalingTransceiver());
        WHEN( "The Signaling Transceiver is added the Soup Server Manager" )
        {
            // First call initializes the singlton, if not called already
            REQUIRE( SoupServerMgr::GetMgr()->AddSignalingTransceiver(pSignalingTransceiver.get()) == true);

            // Second call with the same Signaling Transceiver must fail
            REQUIRE( SoupServerMgr::GetMgr()->AddSignalingTransceiver(pSignalingTransceiver.get()) == false);

            THEN( "The same Signaling Transceiver can be removed" )
            {
                REQUIRE( SoupServerMgr::GetMgr()->RemoveSignalingTransceiver(pSignalingTransceiver.get()) == true);

                // Second call with the same Signaling Transceiver must fail
                REQUIRE( SoupServerMgr::GetMgr()->RemoveSignalingTransceiver(pSignalingTransceiver.get()) == false);
            }
        }
    }
}

SCENARIO( "A Connection event after removing the only client is handled correctly", "[SoupServerMgr]" )
{
    GIVEN( "A new Client Reciever" )
    {
        std::unique_ptr<SignalingTransceiver> pSignalingTransceiver= 
            std::unique_ptr<SignalingTransceiver>(new SignalingTransceiver());

        std::string pathString("ws");

        WHEN( "The Signaling Transceiver is added the Soup Server Manager" )
        {
            // First call initializes the singlton
            REQUIRE( SoupServerMgr::GetMgr()->AddSignalingTransceiver(pSignalingTransceiver.get()) == true);

            // Remove the client reciever leaving the Soup Server without a client. 
            REQUIRE( SoupServerMgr::GetMgr()->RemoveSignalingTransceiver(pSignalingTransceiver.get()) == true);

            THEN( "When a new connection is simulated" )
            {
                SoupWebsocketConnection connection;
                SoupServerMgr::GetMgr()->HandleOpen(&connection, pathString.c_str());
            }
        }
    }
}

static void websocket_server_client_listener_1(const wchar_t* path, 
    void* client_data)
{
    std::cout << "Websocket client listener 1\n";
}

static void websocket_server_client_listener_2(const wchar_t* path, 
    void* client_data)
{
    std::cout << "Websocket client listener 2\n";
}

SCENARIO( "A Websocket client listener can be added and removed from the Soup Server Manger", "[SoupServerMgr]" )
{
    GIVEN( "The Soup Server Manger" )
    {
        WHEN( "The client listener is added to the Soup Server Manager" )
        {
            // First call initializes the singlton, if not called previously by other test cases.
            REQUIRE( SoupServerMgr::GetMgr()->AddClientListener(
                websocket_server_client_listener_1, NULL) == true );

            // Second call with the same Signaling Transceiver must fail
            REQUIRE( SoupServerMgr::GetMgr()->AddClientListener(
                websocket_server_client_listener_1, NULL) == false );

            THEN( "The same Signaling Transceiver can be removed" )
            {
                REQUIRE( SoupServerMgr::GetMgr()->RemoveClientListener(
                    websocket_server_client_listener_1) == true );

                // Second call with the same Signaling Transceiver must fail
                REQUIRE( SoupServerMgr::GetMgr()->RemoveClientListener(
                    websocket_server_client_listener_1) == false );
            }
        }
    }
}

SCENARIO( "A Connection event results in the Soup Server client listener to be called", "[SoupServerMgr]" )
{
    GIVEN( "A new Client Reciever" )
    {
        std::string pathString("ws");

        // First call initializes the singlton, if not called previously by other test cases.
        REQUIRE( SoupServerMgr::GetMgr()->AddClientListener(
            websocket_server_client_listener_1, NULL) == true );

        REQUIRE( SoupServerMgr::GetMgr()->AddClientListener(
            websocket_server_client_listener_2, NULL) == true );

        WHEN( "When a new socket open event occurrs" )
        {
            SoupWebsocketConnection connection;
            SoupServerMgr::GetMgr()->HandleOpen(&connection, pathString.c_str());

            THEN( "Both client listners are called " )
            {
                // Note: requires visual confirmation of console statements
            }
        }
    }
}
