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
#include "DslMailer.h"

using namespace DSL;

SCENARIO( "A new Email Address is created correctly", "[Mailer]" )
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

SCENARIO( "A new SMTP message is created correctly", "[Mailer]" )
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
        
        std::string attachment("");

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
                    fromAddress, ccAddresses, subject, body, attachment));
                    
            std::shared_ptr<SmtpMessage> pMessage2 = 
                std::shared_ptr<SmtpMessage>(new SmtpMessage(toAddresses, 
                    fromAddress, ccAddresses, subject, body, attachment));

            THEN( "All members are setup correctly" )
            {
                REQUIRE( pMessage1->GetId() > 0 );
                REQUIRE( pMessage2->GetId() == (pMessage1->GetId() +1) );
            }
        }
    }
}


SCENARIO( "A new SMTP Message Queue is created correctly", "[Mailer]" )
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
                REQUIRE( queue.PopFront() == nullptr );
                REQUIRE( queue.Size() == 0 );
            }
        }        
    }
}


SCENARIO( "A Mailer Object can set and get all SMTP properties", "[Mailer]" )
{
    GIVEN( "A new Mailer Object" ) 
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
        
        std::string mailerName("mailer");

        DSL_MAILER_PTR pMailer = DSL_MAILER_NEW(mailerName.c_str());
        
        WHEN( "When the SMTP Parameters are set" )
        {
            pMailer->SetServerUrl(mailServer.c_str()); 
            pMailer->SetFromAddress(senderName.c_str(), senderAddress.c_str());
            
            pMailer->SetSslEnabled(false);
            
            pMailer->AddToAddress(toName1.c_str(), toAddress1.c_str());
            pMailer->AddToAddress(toName2.c_str(), toAddress2.c_str());
            pMailer->AddCcAddress(ccName1.c_str(), ccAddress1.c_str());
            pMailer->AddCcAddress(ccName1.c_str(), ccAddress2.c_str());

            THEN( "The same Parameters are returned on get" )
            {
                const char* pRetMailServer;
                const char* pRetFromName;
                const char* pRetFromAddress;
                pMailer->GetServerUrl(&pRetMailServer);
                pMailer->GetFromAddress(&pRetFromName, &pRetFromAddress);
                
                pMailer->RemoveAllToAddresses();
                pMailer->RemoveAllCcAddresses();
                
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

SCENARIO( "A Mailer Object can Queue an SMTP Email with specific content", "[Mailer]" )
{
    GIVEN( "A new Mailer Object" ) 
    {
        std::string userName("john.henry");
        std::string password("3littlepigs");
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

        std::string filePath("./test/streams/first-person-occurrence-438.jpeg");
        char absolutePath[PATH_MAX+1];
        std::string fullFilePath = realpath(filePath.c_str(), absolutePath);
        
        std::string mailerName("mailer");

        DSL_MAILER_PTR pMailer = DSL_MAILER_NEW(mailerName.c_str());
        
        WHEN( "The Mailer object can queue a message without an attachment" )
        {
            pMailer->SetCredentials(userName.c_str(), password.c_str());
            pMailer->SetServerUrl(mailServer.c_str()); 
            pMailer->SetFromAddress(senderName.c_str(), senderAddress.c_str());
            
            pMailer->AddToAddress(toName1.c_str(), toAddress1.c_str());
            pMailer->AddToAddress(toName2.c_str(), toAddress2.c_str());
            pMailer->AddCcAddress(ccName1.c_str(), ccAddress1.c_str());
            pMailer->AddCcAddress(ccName1.c_str(), ccAddress2.c_str());
            
            THEN( "The Mailer object can queue a new email" )
            {
                REQUIRE( pMailer->QueueMessage(subject, body) == true );
            }
        }
        WHEN( "The Mailer object can queue a message with an attachment" )
        {
            
            pMailer->SetCredentials(userName.c_str(), password.c_str());
            pMailer->SetServerUrl(mailServer.c_str()); 
            pMailer->SetFromAddress(senderName.c_str(), senderAddress.c_str());
            
            pMailer->AddToAddress(toName1.c_str(), toAddress1.c_str());
            pMailer->AddToAddress(toName2.c_str(), toAddress2.c_str());
            pMailer->AddCcAddress(ccName1.c_str(), ccAddress1.c_str());
            pMailer->AddCcAddress(ccName1.c_str(), ccAddress2.c_str());
            
            THEN( "The Mailer object can queue a new email" )
            {
                REQUIRE( pMailer->QueueMessage(subject, body, fullFilePath) == true );
            }
        }
    }
}
 
SCENARIO( "A Mailer Object handles a failed SMTP Send because of an invalid server url", "[Mailer]" )
{
    GIVEN( "A new Mailer Object" ) 
    {
        std::string userName("john.henry");
        std::string password("3littlepigs");
        std::string senderName("John Henry");
        std::string senderAddress("john.henry@example.org");
        EmailAddress fromAddress(senderName.c_str(), senderAddress.c_str());
        
        std::string content("this is our specific email content");
        std::string mailServer("smtps://smtp.gmail.com:465");

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
        
        std::string mailerName("mailer");

        DSL_MAILER_PTR pMailer = DSL_MAILER_NEW(mailerName.c_str());

        pMailer->SetCredentials(userName.c_str(), password.c_str());
        pMailer->SetServerUrl(mailServer.c_str()); 
        pMailer->SetFromAddress(senderName.c_str(), senderAddress.c_str());
        pMailer->AddToAddress(toName1.c_str(), toAddress1.c_str());
        pMailer->AddToAddress(toName2.c_str(), toAddress2.c_str());
        pMailer->AddCcAddress(ccName1.c_str(), ccAddress1.c_str());
        pMailer->AddCcAddress(ccName1.c_str(), ccAddress2.c_str());
        
        WHEN( "A message is Queued with invalid options" )
        {
            REQUIRE( pMailer->QueueMessage(subject, body) == true );
            
            THEN( "The Mailer object handles the failure correctly" )
            {
                REQUIRE( pMailer->SendMessage() == false );
            }
        }
    }
}           