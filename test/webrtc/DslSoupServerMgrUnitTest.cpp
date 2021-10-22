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

SCENARIO( "A Signaling Transceiver can be added, creating the Soup Server Manger on first call", "[SoupServerMgr]" )
{
    GIVEN( "A new Client Reciever" )
    {
        std::unique_ptr<SignalingTransceiver> pSignalingTransceiver= 
            std::unique_ptr<SignalingTransceiver>(new SignalingTransceiver());
        WHEN( "The Signaling Transceiver is added the Soup Server Manager" )
        {
            // First call initializes the singlton
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
        WHEN( "The Signaling Transceiver is added the Soup Server Manager" )
        {
            // First call initializes the singlton
            REQUIRE( SoupServerMgr::GetMgr()->AddSignalingTransceiver(pSignalingTransceiver.get()) == true);

            // Remove the client reciever leaving the Soup Server without a client. 
            REQUIRE( SoupServerMgr::GetMgr()->RemoveSignalingTransceiver(pSignalingTransceiver.get()) == true);

            THEN( "When a new connection is simulated" )
            {
                SoupWebsocketConnection connection;
                SoupServerMgr::GetMgr()->HandleOpen(&connection);
            }
        }
    }
}

