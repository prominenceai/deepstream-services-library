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
#include "DslServices.h"
#include "DslComms.h"

using namespace DSL;

SCENARIO( "A new Email Address is created correctly", "[Comms]" )
{
    GIVEN( "Attributes for a new Email Address" )
    {
        std::string name{"Joe Blow"};
        std::string inAddress{"joe.blow@example.org"};
        std::string outAddress{"<joe.blow@example.org>"};

        WHEN( "When the email address is created" )
        {
            EmailAddress emailAddress(name.c_str(), inAddress.c_str());

            THEN( "All members are setup correctly" )
            {
                std::string retAddress((const char*)emailAddress);
                REQUIRE( retAddress == outAddress );
                
                std::cout << emailAddress << "\r\n";
            }
        }
        
    }
}

SCENARIO( "A new SMTP message is created correctly", "[Comms]" )
{
    GIVEN( "Attributes for a new SMTP message" )
    {
        std::string senderName("John Henry");
        std::string senderAddress("john.henry@example.org");
        EmailAddress fromAddress(senderName.c_str(), senderAddress.c_str());
        
        std::string content("this is our specific email content");
        std::string mailServer("smtp://mail.example.com");

        std::string toName1("Joe Blow");
        std::string toAddress1("joe.blow@example.org");
        EmailAddress toEmail1(toName1.c_str(), toAddress1.c_str());

        std::string toName2("Jack Black");
        std::string toAddres2("jack.black@example.org");
        EmailAddress toEmail2(toName2.c_str(), toAddres2.c_str());

        std::string ccName1("Jane Doe");
        std::string ccAddress1("jane.doe@example.org");
        EmailAddress ccEmail1(ccName1.c_str(), ccAddress1.c_str());

        std::string ccName2("Bill Williams");
        std::string ccAddress2("bill.williams@example.org");
        EmailAddress ccEmail2(ccName2.c_str(), ccAddress2.c_str());
        
        std::string subject("this is the subject of the message");

        // body text gets \r\n from action
        std::string bodyLine1("this is unique content for line 1 \r\n");
        std::string bodyLine2("this is unique content for line 2 \r\n");
        std::string bodyLine3("this is unique content for line 3 \r\n");

        EmailAddresses toAddresses{toEmail1, toEmail2};
        EmailAddresses ccAddresses{ccEmail1, ccEmail2};
        std::vector<std::string> body{bodyLine1, bodyLine2, bodyLine3};
        
        WHEN( "When two new SMTP Messages are created" )
        {
            std::shared_ptr<SmtpMessage> pMessage1 = 
                std::shared_ptr<SmtpMessage>(new SmtpMessage(toAddresses, 
                    fromAddress, ccAddresses, subject, body));
                    
            std::shared_ptr<SmtpMessage> pMessage2 = 
                std::shared_ptr<SmtpMessage>(new SmtpMessage(toAddresses, 
                    fromAddress, ccAddresses, subject, body));

            THEN( "All members are setup correctly" )
            {
                REQUIRE( pMessage1->GetId() == 1 );
                REQUIRE( pMessage1->IsInProgress() == false );
                REQUIRE( pMessage1->IsComplete() == false );
                REQUIRE( pMessage2->GetId() == 2 );
                REQUIRE( pMessage2->IsInProgress() == false );
                REQUIRE( pMessage2->IsComplete() == false );
            }
        }
    }
}

SCENARIO( "A new SMTP message can update its state correctly", "[Comms]" )
{
    GIVEN( "Attributes for a new SMTP message" ) 
    {
        std::string senderName("John Henry");
        std::string senderAddress("john.henry@example.org");
        EmailAddress fromAddress(senderName.c_str(), senderAddress.c_str());
        
        std::string content("this is our specific email content");
        std::string mailServer("smtp://mail.example.com");

        std::string toName1("Joe Blow");
        std::string toAddress1("joe.blow@example.org");
        EmailAddress toEmail1(toName1.c_str(), toAddress1.c_str());

        std::string toName2("Jack Black");
        std::string toAddres2("jack.black@example.org");
        EmailAddress toEmail2(toName2.c_str(), toAddres2.c_str());

        std::string ccName1("Jane Doe");
        std::string ccAddress1("jane.doe@example.org");
        EmailAddress ccEmail1(ccName1.c_str(), ccAddress1.c_str());

        std::string ccName2("Bill Williams");
        std::string ccAddress2("bill.williams@example.org");
        EmailAddress ccEmail2(ccName2.c_str(), ccAddress2.c_str());
        
        std::string subject("this is the subject of the message");

        // body text gets \r\n from action
        std::string bodyLine1("this is unique content for line 1 \r\n");
        std::string bodyLine2("this is unique content for line 2 \r\n");
        std::string bodyLine3("this is unique content for line 3 \r\n");

        std::vector<EmailAddress> toAddresses{toEmail1, toEmail2};
        std::vector<EmailAddress> ccAddresses{ccEmail1, ccEmail2};
        std::vector<std::string> body{bodyLine1, bodyLine2, bodyLine3};

        WHEN( "When the message is set to In-Progress" )
        {
            std::shared_ptr<SmtpMessage> pMessage = 
                std::shared_ptr<SmtpMessage>(new SmtpMessage(toAddresses, 
                    fromAddress, ccAddresses, subject, body));

            REQUIRE( pMessage->IsInProgress() == false );
            pMessage->NowInProgress();
            
            THEN( "The state change is reported correctly" )
            {
                REQUIRE( pMessage->IsComplete() == false );
                REQUIRE( pMessage->IsInProgress() == true );
                REQUIRE( pMessage->IsFailure() == false );
            }
        }        
        WHEN( "When the message is set to Failure" )
        {
            std::shared_ptr<SmtpMessage> pMessage = 
                std::shared_ptr<SmtpMessage>(new SmtpMessage(toAddresses, 
                    fromAddress, ccAddresses, subject, body));

            REQUIRE( pMessage->IsFailure() == false );
            pMessage->NowFailure();
            
            THEN( "The state change is reported correctly" )
            {
                REQUIRE( pMessage->IsComplete() == false );
                REQUIRE( pMessage->IsInProgress() == false );
                REQUIRE( pMessage->IsFailure() == true );
            }
        }        
        WHEN( "When the message is set to Complete" )
        {
            std::shared_ptr<SmtpMessage> pMessage = 
                std::shared_ptr<SmtpMessage>(new SmtpMessage(toAddresses, 
                    fromAddress, ccAddresses, subject, body));

            REQUIRE( pMessage->IsComplete() == false );
            pMessage->NowComplete();
            
            THEN( "The state change is reported correctly" )
            {
                REQUIRE( pMessage->IsComplete() == true );
                REQUIRE( pMessage->IsInProgress() == false );
                REQUIRE( pMessage->IsFailure() == false );
            }
        }        
    }
}

SCENARIO( "A new SMTP message can be Read", "[Comms]" )
{
    GIVEN( "Attributes for a new SMTP message" )
    {
        std::string senderName("John Henry");
        std::string senderAddress("john.henry@example.org");
        EmailAddress fromAddress(senderName.c_str(), senderAddress.c_str());
        
        std::string content("this is our specific email content");
        std::string mailServer("smtp://mail.example.com");

        std::string toName1("Joe Blow");
        std::string toAddress1("joe.blow@example.org");
        EmailAddress toEmail1(toName1.c_str(), toAddress1.c_str());

        std::string toName2("Jack Black");
        std::string toAddres2("jack.black@example.org");
        EmailAddress toEmail2(toName2.c_str(), toAddres2.c_str());

        std::string ccName1("Jane Doe");
        std::string ccAddress1("jane.doe@example.org");
        EmailAddress ccEmail1(ccName1.c_str(), ccAddress1.c_str());

        std::string ccName2("Bill Williams");
        std::string ccAddress2("bill.williams@example.org");
        EmailAddress ccEmail2(ccName2.c_str(), ccAddress2.c_str());
        
        std::string subject("this is the subject of the message");

        // body text gets \r\n from action
        std::string bodyLine1("this is unique content for line 1 \r\n");
        std::string bodyLine2("this is unique content for line 2 \r\n");
        std::string bodyLine3("this is unique content for line 3 \r\n");

        std::vector<EmailAddress> toAddresses{toEmail1, toEmail2};
        std::vector<EmailAddress> ccAddresses{ccEmail1, ccEmail2};
        std::vector<std::string> body{bodyLine1, bodyLine2, bodyLine3};

        std::shared_ptr<SmtpMessage> pMessage = 
            std::shared_ptr<SmtpMessage>(new SmtpMessage(toAddresses, 
                fromAddress, ccAddresses, subject, body));
        
        WHEN( "Sufficient memory is provided" )
        {
            size_t len(0);
            char line[128] = {0};
            
            THEN( "All lines can be read correctly" )
            {
                do
                {
                    len = pMessage->ReadLine(line, sizeof(line), 10);
                    if (len)
                    {
                        std::cout << line;
                    }
                    
                }while (len);
                    
                REQUIRE( pMessage->IsComplete() == true );
            }
        }
        WHEN( "Insufficient memory is provided" )
        {
            size_t len(0);
            char line[10] = {0};
            
            THEN( "All lines can be read correctly" )
            {
                REQUIRE( pMessage->ReadLine(line, sizeof(line), 10) == 0 );
                REQUIRE( pMessage->IsFailure() == true );
            }
        }
    }
}

SCENARIO( "A new SMTP Message Queue is created correctly", "[Comms]" )
{
    GIVEN( "Attributes for a new SMTP Message Queue" ) 
    {
        // No members ...
        
        WHEN( "When two new SMTP Message Queue is created" )
        {
            SmtpMessageQueue queue;

            THEN( "All members are setup correctly" )
            {
                REQUIRE( queue.IsEmpty() == true );
                REQUIRE( queue.Front() == nullptr );
                REQUIRE( queue.Size() == 0 );
            }
        }        
    }
}

SCENARIO( "A SMTP Message Queue Purges messages correctly", "[Comms]" )
{
    GIVEN( "A new empty SMTP Message Queue" ) 
    {
        std::string senderName("John Henry");
        std::string senderAddress("john.henry@example.org");
        EmailAddress fromAddress(senderName.c_str(), senderAddress.c_str());
        
        std::string content("this is our specific email content");
        std::string mailServer("smtp://mail.example.com");

        std::string toName1("Joe Blow");
        std::string toAddress1("joe.blow@example.org");
        EmailAddress toEmail1(toName1.c_str(), toAddress1.c_str());

        std::string toName2("Jack Black");
        std::string toAddres2("jack.black@example.org");
        EmailAddress toEmail2(toName2.c_str(), toAddres2.c_str());

        std::string ccName1("Jane Doe");
        std::string ccAddress1("jane.doe@example.org");
        EmailAddress ccEmail1(ccName1.c_str(), ccAddress1.c_str());

        std::string ccName2("Bill Williams");
        std::string ccAddress2("bill.williams@example.org");
        EmailAddress ccEmail2(ccName2.c_str(), ccAddress2.c_str());
        
        std::string subject("this is the subject of the message");

        // body text gets \r\n from action
        std::string bodyLine1("this is unique content for line 1 \r\n");
        std::string bodyLine2("this is unique content for line 2 \r\n");
        std::string bodyLine3("this is unique content for line 3 \r\n");

        std::vector<EmailAddress> toAddresses{toEmail1, toEmail2};
        std::vector<EmailAddress> ccAddresses{ccEmail1, ccEmail2};
        std::vector<std::string> body{bodyLine1, bodyLine2, bodyLine3};

        std::shared_ptr<SmtpMessage> pMessage = 
            std::shared_ptr<SmtpMessage>(new SmtpMessage(toAddresses, 
                fromAddress, ccAddresses, subject, body));
            
        WHEN( "When a single new message is added" )
        {
            SmtpMessageQueue queue;
            queue.Push(pMessage);

            THEN( "The Message Queue is updated correctly" )
            {
                REQUIRE( queue.IsEmpty() == false );
                REQUIRE( queue.Front() == pMessage );
                REQUIRE( queue.Size() == 1 );
            }
        }        
        WHEN( "When a single new Message is PENDING (default)" )
        {
            SmtpMessageQueue queue;
            queue.Push(pMessage);

            THEN( "Purge returns true without updating the queue" )
            {
                // should return true - restart timer
                REQUIRE( queue.Purge() == true );
                REQUIRE( queue.Front() == pMessage );
                REQUIRE( queue.Size() == 1 );
            }
        }        
        WHEN( "When a single Message is INPROGRESS" )
        {
            SmtpMessageQueue queue;
            queue.Push(pMessage);
            
            pMessage->NowInProgress();

            THEN( "Purge returns true without updating the queue" )
            {
                // should return true - restart timer
                REQUIRE( queue.Purge() == true );
                REQUIRE( queue.Front() == pMessage );
                REQUIRE( queue.Size() == 1 );
            }
        }        
        WHEN( "When a single Message is COMPLETE" )
        {
            SmtpMessageQueue queue;
            queue.Push(pMessage);
            
            pMessage->NowComplete();

            THEN( "Purge returns false after purging the queue" )
            {
                // should return false - complete, don't restart timer
                REQUIRE( queue.Purge() == false );
                REQUIRE( queue.Front() == nullptr );
                REQUIRE( queue.Size() == 0 );
            }
        }        
        WHEN( "When multiple Messages transition through their states" )
        {
            std::shared_ptr<SmtpMessage> pMessage1 = 
                std::shared_ptr<SmtpMessage>(new SmtpMessage(toAddresses, 
                    fromAddress, ccAddresses, subject, body));
            std::shared_ptr<SmtpMessage> pMessage2 = 
                std::shared_ptr<SmtpMessage>(new SmtpMessage(toAddresses, 
                    fromAddress, ccAddresses, subject, body));
            std::shared_ptr<SmtpMessage> pMessage3 = 
                std::shared_ptr<SmtpMessage>(new SmtpMessage(toAddresses, 
                    fromAddress, ccAddresses, subject, body));
            std::shared_ptr<SmtpMessage> pMessage4 = 
                std::shared_ptr<SmtpMessage>(new SmtpMessage(toAddresses, 
                    fromAddress, ccAddresses, subject, body));
            std::shared_ptr<SmtpMessage> pMessage5 = 
                std::shared_ptr<SmtpMessage>(new SmtpMessage(toAddresses, 
                    fromAddress, ccAddresses, subject, body));

            SmtpMessageQueue queue;
            
            pMessage1->NowComplete();
            queue.Push(pMessage1);
            pMessage2->NowInProgress();
            queue.Push(pMessage2);
            // leave the rest pending
            queue.Push(pMessage3);
            queue.Push(pMessage4);
            queue.Push(pMessage5);

            REQUIRE( queue.Front() == pMessage1 );
            REQUIRE( queue.Size() == 5 );

            THEN( "They are each purged in turn correctly" )
            {
                REQUIRE( queue.Purge() == true );
                REQUIRE( queue.Front() == pMessage2 );
                REQUIRE( queue.Size() == 4 );

                pMessage2->NowComplete();
                pMessage3->NowComplete();

                REQUIRE( queue.Purge() == true );
                REQUIRE( queue.Front() == pMessage4 );
                REQUIRE( queue.Size() == 2 );

                // should never occur out of order, however....
                pMessage5->NowComplete();

                REQUIRE( queue.Purge() == true );
                REQUIRE( queue.Front() == pMessage4 );
                REQUIRE( queue.Size() == 2 );

                pMessage4->NowComplete();

                // should return false - complete, don't restart timer
                REQUIRE( queue.Purge() == false );
                REQUIRE( queue.Front() == nullptr );
                REQUIRE( queue.Size() == 0 );
            }
        }        
    }
}

SCENARIO( "A Comms Object can set and get all SMTP properties", "[Comms]" )
{
    GIVEN( "A new Comms Object" ) 
    {
        std::string senderName("John Henry");
        std::string senderAddress("john.henry@example.org");
        EmailAddress fromAddress(senderName.c_str(), senderAddress.c_str());
        
        std::string content("this is our specific email content");
        std::string mailServer("smtp://mail.example.com");

        std::string toName1("Joe Blow");
        std::string toAddress1("joe.blow@example.org");
        EmailAddress toEmail1(toName1.c_str(), toAddress1.c_str());

        std::string toName2("Jack Black");
        std::string toAddress2("jack.black@example.org");
        EmailAddress toEmail2(toName2.c_str(), toAddress2.c_str());

        std::string ccName1("Jane Doe");
        std::string ccAddress1("jane.doe@example.org");
        EmailAddress ccEmail1(ccName1.c_str(), ccAddress1.c_str());

        std::string ccName2("Bill Williams");
        std::string ccAddress2("bill.williams@example.org");
        EmailAddress ccEmail2(ccName2.c_str(), ccAddress2.c_str());
        
        std::string subject("this is the subject of the message");

        const std::shared_ptr<Comms> pComms = DSL::Services::GetServices()->GetComms();
        
        WHEN( "When the SMTP Parameters are set" )
        {
            pComms->SetSmtpServerUrl(mailServer.c_str()); 
            pComms->SetSmtpFromAddress(senderName.c_str(), senderAddress.c_str());
            
            pComms->SetSmtpSslEnabled(false);
            
            pComms->AddSmtpToAddress(toName1.c_str(), toAddress1.c_str());
            pComms->AddSmtpToAddress(toName2.c_str(), toAddress2.c_str());
            pComms->AddSmtpCcAddress(ccName1.c_str(), ccAddress1.c_str());
            pComms->AddSmtpCcAddress(ccName1.c_str(), ccAddress2.c_str());

            THEN( "The same Parameters are returned on get" )
            {
                const char* pRetMailServer;
                const char* pRetFromName;
                const char* pRetFromAddress;
                pComms->GetSmtpServerUrl(&pRetMailServer);
                pComms->GetSmtpFromAddress(&pRetFromName, &pRetFromAddress);
                
                pComms->RemoveAllSmtpToAddresses();
                pComms->RemoveAllSmtpCcAddresses();
                
                std::string retMailServer(pRetMailServer);
                std::string retFromName(pRetFromName);
                std::string retFromAddress(pRetFromAddress);
                
                REQUIRE( retMailServer == mailServer );
                REQUIRE( retFromName == senderName );
                REQUIRE( retFromAddress == senderAddress );
            }
        }
    }
}

SCENARIO( "A Comms Object can Queue an SMTP Email with specific content", "[Comms]" )
{
    GIVEN( "A new Comms Object" ) 
    {
        std::string senderName("John Henry");
        std::string senderAddress("john.henry@example.org");
        EmailAddress fromAddress(senderName.c_str(), senderAddress.c_str());
        
        std::string content("this is our specific email content");
        std::string mailServer("smtp://mail.example.com");

        std::string toName1("Joe Blow");
        std::string toAddress1("joe.blow@example.org");
        EmailAddress toEmail1(toName1.c_str(), toAddress1.c_str());

        std::string toName2("Jack Black");
        std::string toAddress2("jack.black@example.org");
        EmailAddress toEmail2(toName2.c_str(), toAddress2.c_str());

        std::string ccName1("Jane Doe");
        std::string ccAddress1("jane.doe@example.org");
        EmailAddress ccEmail1(ccName1.c_str(), ccAddress1.c_str());

        std::string ccName2("Bill Williams");
        std::string ccAddress2("bill.williams@example.org");
        EmailAddress ccEmail2(ccName2.c_str(), ccAddress2.c_str());
        
        std::string subject("this is the subject of the message");

        // body text gets \r\n from action
        std::string bodyLine1("this is unique content for line 1 \r\n");
        std::string bodyLine2("this is unique content for line 2 \r\n");
        std::string bodyLine3("this is unique content for line 3 \r\n");
        std::vector<std::string> body{bodyLine1, bodyLine2, bodyLine3};
        
        const std::shared_ptr<Comms> pComms = DSL::Services::GetServices()->GetComms();
        
        WHEN( "The Comms object is intialized correctly" )
        {
            pComms->SetSmtpServerUrl(mailServer.c_str()); 
            pComms->SetSmtpFromAddress(senderName.c_str(), senderAddress.c_str());
            
            pComms->AddSmtpToAddress(toName1.c_str(), toAddress1.c_str());
            pComms->AddSmtpToAddress(toName2.c_str(), toAddress2.c_str());
            pComms->AddSmtpCcAddress(ccName1.c_str(), ccAddress1.c_str());
            pComms->AddSmtpCcAddress(ccName1.c_str(), ccAddress2.c_str());
            
            THEN( "The Comms object can queue a new email" )
            {
                REQUIRE( pComms->QueueSmtpMessage(subject, body) == true );
            }
        }
    }
}
 
SCENARIO( "A Comms Object handles a failed SMTP Email because of invalid options", "[Comms]" )
{
    GIVEN( "A new Comms Object" ) 
    {
        std::string senderName("John Henry");
        std::string senderAddress("john.henry@example.org");
        EmailAddress fromAddress(senderName.c_str(), senderAddress.c_str());
        
        std::string content("this is our specific email content");
        std::string mailServer("smtp://mail.example.com");

        std::string toName1("Joe Blow");
        std::string toAddress1("joe.blow@example.org");
        EmailAddress toEmail1(toName1.c_str(), toAddress1.c_str());

        std::string toName2("Jack Black");
        std::string toAddress2("jack.black@example.org");
        EmailAddress toEmail2(toName2.c_str(), toAddress2.c_str());

        std::string ccName1("Jane Doe");
        std::string ccAddress1("jane.doe@example.org");
        EmailAddress ccEmail1(ccName1.c_str(), ccAddress1.c_str());

        std::string ccName2("Bill Williams");
        std::string ccAddress2("bill.williams@example.org");
        EmailAddress ccEmail2(ccName2.c_str(), ccAddress2.c_str());
        
        std::string subject("this is the subject of the message");

        // body text gets \r\n from action
        std::string bodyLine1("this is unique content for line 1 \r\n");
        std::string bodyLine2("this is unique content for line 2 \r\n");
        std::string bodyLine3("this is unique content for line 3 \r\n");
        std::vector<std::string> body{bodyLine1, bodyLine2, bodyLine3};
        
        const std::shared_ptr<Comms> pComms = DSL::Services::GetServices()->GetComms();

        pComms->SetSmtpServerUrl(mailServer.c_str()); 
        pComms->SetSmtpFromAddress(senderName.c_str(), senderAddress.c_str());
        pComms->AddSmtpToAddress(toName1.c_str(), toAddress1.c_str());
        pComms->AddSmtpToAddress(toName2.c_str(), toAddress2.c_str());
        pComms->AddSmtpCcAddress(ccName1.c_str(), ccAddress1.c_str());
        pComms->AddSmtpCcAddress(ccName1.c_str(), ccAddress2.c_str());
        
        WHEN( "A message is Queued with invalid options" )
        {
            REQUIRE( pComms->QueueSmtpMessage(subject, body) == true );
            
            THEN( "The Comms object handles the failure correctly" )
            {
                REQUIRE( pComms->SendSmtpMessage() == false );
            }
        }
    }
}           