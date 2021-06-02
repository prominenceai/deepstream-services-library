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

#include "DslMailer.h"
#include "DslApi.h"

#define DATE_BUFF_LENGTH 37

namespace DSL
{
    // Initialize static Message Id counter
    uint SmtpMessage::s_nextMessageId = 1;

    std::vector<std::string> SmtpMessage::m_htmlBegin 
    {
        {"<html>\r\n"},
        {"<head>\r\n"},
        {"<style>\r\n"},
        {".p1 {font-family: 'Lucida Console', 'Courier New', monospace; }\r\n"},
        {"</style>\r\n"},
        {"</head>\r\n"},
        {"<body>\r\n"},
        {"<p class='p1'>\r\n"},
        {"<pre>\r\n"}
    };

    std::vector<std::string> SmtpMessage::m_htmlEnd 
    {
        {"</p>\r\n"},
        {"<pre>\r\n"},
        {"</body\r\n>"},
        {"</html\r\n>"}
    };


    SmtpMessage::SmtpMessage(const EmailAddresses& toList,
        const EmailAddress& from, const EmailAddresses& ccList,
        const std::string& subject, const std::vector<std::string>& body,
        const std::string& attachment)
        : m_attachment(attachment.c_str())
    {
        LOG_FUNC();

        m_messageId = s_nextMessageId++;
        
        m_header.push_back(DateTimeLine());
        m_header.push_back(AddressLine(TO, toList));
        m_header.push_back(FromLine(from));
        m_header.push_back(AddressLine(CC, ccList));
        m_header.push_back(MessageIdLine(from));
        m_header.push_back(SubjectLine(subject));

        // concatenate HTML begin formating, body, and HTML end fromating
        m_content.reserve(m_htmlBegin.size() + body.size() + m_htmlEnd.size());
        m_content.insert(m_content.end(), m_htmlBegin.begin(), m_htmlBegin.end() );
        m_content.insert(m_content.end(), body.begin(), body.end() );
        m_content.insert(m_content.end(), m_htmlEnd.begin(), m_htmlEnd.end() );

        g_mutex_init(&m_messageMutex);
    };

    SmtpMessage::~SmtpMessage()
    {
        LOG_FUNC();

        g_mutex_clear(&m_messageMutex);
    }
    
    std::string SmtpMessage::DateTimeLine()
    {
        char dateTime[DATE_BUFF_LENGTH] = {0};
        time_t seconds = time(NULL);
        struct tm currentTm;
        localtime_r(&seconds, &currentTm);

        strftime(dateTime, DATE_BUFF_LENGTH, "%a, %d %b %Y %H:%M:%S %z", &currentTm);
        std::string dateTimeStr(dateTime);
        std::string dateTimeline("Date: " + dateTimeStr);
        
        return dateTimeline;
    }    

    std::string SmtpMessage::AddressLine(addressType type,
        const EmailAddresses& addresses)
    {
        std::ostringstream oss;
        oss << ((type == TO) ? "To: " : "Cc: ");
        oss << addresses;
        return oss.str();
    }
    
    std::string SmtpMessage::FromLine(const EmailAddress& from)
    {
        std::string fromLine("From: " + std::string(from));
        return fromLine;
    }
    
    std::string SmtpMessage::MessageIdLine(EmailAddress from)
    {
        char dateTime[DATE_BUFF_LENGTH] = {0};
        time_t seconds = time(NULL);
        struct tm currentTm;
        time(&seconds);
        gmtime_r(&seconds, &currentTm);

        size_t dateLen = std::strftime(dateTime, DATE_BUFF_LENGTH, "%Y%m%d%H%M%S", &currentTm);

        static const std::string alphaNum {
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz" };

        std::mt19937 gen;
        std::uniform_int_distribution<> distr(0, alphaNum.length() - 1);
        std::string dateTimeStr(dateTime);
        dateTimeStr.resize(DATE_BUFF_LENGTH);
        std::generate_n(dateTimeStr.begin() + dateLen,
                        DATE_BUFF_LENGTH - dateLen,
                        [&]() { return alphaNum[distr(gen)]; });

        std::string messageIdLine{"Message-ID: <" + dateTimeStr + "@" + from.domain()};
        return messageIdLine; 
    }
    
    std::string SmtpMessage::SubjectLine(const std::string& subject)
    {
        std::string subjectLine("Subject: " + subject);
        return subjectLine;
    }

    // ------------------------------------------------------------------------------

    SmtpMessageQueue::SmtpMessageQueue()
        : m_enabled(true)
        , m_purgeTimerId(0)
    {
        LOG_FUNC();
        
        g_mutex_init(&m_queueMutex);
    };
    
    SmtpMessageQueue::~SmtpMessageQueue()
    {
        LOG_FUNC();

        if (m_purgeTimerId)
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_queueMutex);
            g_source_remove(m_purgeTimerId);
        }
        g_mutex_clear(&m_queueMutex);
    }
    
    bool SmtpMessageQueue::GetEnabled()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_queueMutex);

        return m_enabled;
    }
    
    void SmtpMessageQueue::SetEnabled(bool enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_queueMutex);

        m_enabled = enabled;
    }

    bool SmtpMessageQueue::Push(std::shared_ptr<SmtpMessage> pMessage)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_queueMutex);
        
        if (!m_enabled)
        {
            LOG_ERROR("SMTP Message Queue is currently disabled, unable to push new message");
            return false;
        }
        
        LOG_INFO("Pushing: SMTP Message with Id = " << pMessage->GetId());

        m_queue.push(pMessage);
        return true;
    }

    std::shared_ptr<SmtpMessage> SmtpMessageQueue::PopFront()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_queueMutex);
        
        if (m_queue.empty())
        {
            return nullptr;
        }

        std::shared_ptr<SmtpMessage> pFront = m_queue.front();
        m_queue.pop();
        return pFront;
    }
    
    // ------------------------------------------------------------------------------
    
    Mailer::Mailer(const char* name)
        : Base(name)
        , m_sendMessageThreadId(0)
        , m_sslEnabled(true)
    {
        LOG_FUNC();
        
        g_mutex_init(&m_commsMutex);
    }
    
    Mailer::~Mailer()
    {
        LOG_FUNC();
        
        if (m_sendMessageThreadId)
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
            g_source_remove(m_sendMessageThreadId);
        }
        g_mutex_clear(&m_commsMutex);
    }
    
    bool Mailer::GetEnabled()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
        
        return m_pMessageQueue.GetEnabled();
    }

    void Mailer::SetEnabled(bool enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
        
        m_pMessageQueue.SetEnabled(enabled);
    }

    void Mailer::SetCredentials(const char* username, const char* password)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);

        m_username.assign(username);
        m_password.assign(password);
    }

    void Mailer::GetServerUrl(const char** serverUrl)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
        
        *serverUrl = m_mailServerUrl.c_str();
    }
    
    void Mailer::SetServerUrl(const char* serverUrl)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);

        m_mailServerUrl.assign(serverUrl);
    }
    
    void Mailer::GetFromAddress(const char** name, const char** address)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
        
        m_fromAddress.Get(name, address);
    }
    
    void Mailer::SetFromAddress(const char* name, const char* address)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);

        m_fromAddress.Set(name, address);
    }
    
    bool Mailer::GetSslEnabled()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
        
        return m_sslEnabled;
    }

    void Mailer::SetSslEnabled(bool enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
        
        m_sslEnabled = enabled;
    }

    void Mailer::AddToAddress(const char* name, const char* address)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
        
        EmailAddress newToAddress(name, address);
        m_toAddresses.push_back(newToAddress);
    }

    void Mailer::RemoveAllToAddresses()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
        
        m_toAddresses.clear();
    }

    void Mailer::AddCcAddress(const char* name, const char* address)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
        
        EmailAddress newCcAddress(name, address);
        m_ccAddresses.push_back(newCcAddress);
    }

    void Mailer::RemoveAllCcAddresses()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
        
        m_ccAddresses.clear();
    }
    
    bool Mailer::IsSetup()
    {
        bool result(true);
        
        if (m_mailServerUrl.empty())
        {
            LOG_INFO("SMTP Mail Server has not be set");
            result = false;
        }
        if (m_username.empty() or m_password.empty())
        {
            LOG_INFO("SMTP Mail Server Username and Password have not be set");
            result = false;
        }
        if (!m_fromAddress.IsSet())
        {
            LOG_INFO("SMTP From Address has not be set");
            result = false;
        }
        if (m_toAddresses.empty())
        {
            LOG_INFO("SMTP one or more To Addressess have not be set");
            result = false;
        }
        return result;
    }

    bool Mailer::QueueMessage(const std::string& subject, 
        const std::vector<std::string>& body, const std::string& attachment)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
        
        if (!IsSetup())
        {
            LOG_ERROR("Unable to queue Message - SMTP Mail settings are incomplete.");
            return false;
        }

        // Create a new message with the caller's unique content
        std::shared_ptr<SmtpMessage> pMessage = 
            std::shared_ptr<SmtpMessage>(new SmtpMessage(m_toAddresses, 
                m_fromAddress, m_ccAddresses, subject, body, attachment));
        
        // queue the new Message, return on falure
        if (!m_pMessageQueue.Push(pMessage))
        {
            return false;
        }
        
        // if the SendMessage thread is not currently running
        if (!m_sendMessageThreadId)
        {
            m_sendMessageThreadId = g_idle_add(MailerSendMessage, this);
        }
        return (bool)m_sendMessageThreadId;
        
    }
    
    bool Mailer::SendMessage()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
        
        // Make sure the queue has a message to process
        if (m_pMessageQueue.IsEmpty())
        {
            m_sendMessageThreadId = 0;
            return false;
        }
        
        std::shared_ptr<SmtpMessage> message = m_pMessageQueue.PopFront();

        CURL* pCurl = curl_easy_init();
        if(!pCurl)
        {
            LOG_ERROR("curl_easy_init() failed");
        }
        
        // Set the options for this curl sesion
        if (m_sslEnabled)
        {
            curl_easy_setopt(pCurl, CURLOPT_USE_SSL, CURLUSESSL_ALL);
            curl_easy_setopt(pCurl, CURLOPT_USERNAME, m_username.c_str());
            curl_easy_setopt(pCurl, CURLOPT_PASSWORD, m_password.c_str());
        }
        curl_easy_setopt(pCurl, CURLOPT_URL, m_mailServerUrl.c_str());
        curl_easy_setopt(pCurl, CURLOPT_MAIL_FROM, (const char*)m_fromAddress);
        
        
        // build a recipient list of all TO and CC addresses
        curl_slist* recipients(NULL);
        
        for (auto &ivec: m_toAddresses)
        {
            recipients = curl_slist_append(recipients, (const char*)ivec);
        }
        for (auto &ivec: m_ccAddresses)
        {
            recipients = curl_slist_append(recipients, (const char*)ivec);
        }
        curl_easy_setopt(pCurl, CURLOPT_MAIL_RCPT, recipients);
        
        // Build and set the message header list.
        curl_slist* headers(NULL);
        for (auto &ivec: message->m_header)
        {
            headers = curl_slist_append(headers, ivec.c_str());
        }
        curl_easy_setopt(pCurl, CURLOPT_HTTPHEADER, headers);
 
        // Build the mime message. The inline part is an alternative proposing 
        // the html and the text versions of the e-mail.
        curl_mime* mime = curl_mime_init(pCurl);
        curl_mime* alt = curl_mime_init(pCurl);

        std::ostringstream inlineHtml;
        for (auto &ivec: message->m_content)
        {
            inlineHtml << ivec;
        }
     
        // HTML message.
        curl_mimepart* part = curl_mime_addpart(alt);
        curl_mime_data(part, inlineHtml.str().c_str(), CURL_ZERO_TERMINATED);
        curl_mime_type(part, "text/html");

        // Create the inline part.
        part = curl_mime_addpart(mime);
        curl_mime_subparts(part, alt);
        curl_mime_type(part, "multipart/alternative");
        curl_slist* slist = curl_slist_append(NULL, "Content-Disposition: inline");
        curl_mime_headers(part, slist, 1);

        // Add optional file attachement
        if (message->m_attachment.size())
        {
            part = curl_mime_addpart(mime);
            curl_mime_filedata(part, message->m_attachment.c_str());
            curl_mime_encoder(part, "base64");
        }

        curl_easy_setopt(pCurl, CURLOPT_MIMEPOST, mime);
        
        // perform the actual send, ReadLine function is called in this context
        CURLcode result = curl_easy_perform(pCurl);
        if (result == CURLE_OK)
        {
            LOG_INFO("Email Message with id " << message->GetId() << " sent successfully");
        }
        else
        {
            LOG_ERROR("libcurl returned " << result << ": '"
                << curl_easy_strerror(result) << "' sending message");
        }

        // free up all recipients/headers
        curl_slist_free_all(recipients);       
        curl_slist_free_all(headers);       

        // clean up resources
        curl_easy_cleanup(pCurl);
 
        // Free multipart message
        curl_mime_free(mime);        
        
        // if there is more than one message to send, return true to reschedule
        if (!m_pMessageQueue.Size() > 1)
        {
            return true;
        }
        
        // otherwise, end the thread
        m_sendMessageThreadId = 0;
        return false;
    }
    
    static gboolean MailerSendMessage(gpointer pMailer)
    {
        return static_cast<Mailer*>(pMailer)->SendMessage();
    }
}

