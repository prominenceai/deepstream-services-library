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

#include "DslComms.h"
#include "DslApi.h"

#define DATE_BUFF_LENGTH 37

namespace DSL
{
    // Initialize static Message Id counter
    uint SmtpMessage::s_nextMessageId = 1;

    SmtpMessage::SmtpMessage(const EmailAddresses& toList,
        const EmailAddress& from, const EmailAddresses& ccList,
        const std::string& subject, const std::vector<std::string>& body)
        : m_linesRead(0)
        , m_messageId(0)
        , m_messageState(PENDING)
    {
        LOG_FUNC();

        m_messageId = s_nextMessageId++;
        
        std::vector<std::string> header;
        header.push_back(DateTimeLine());
        header.push_back(AddressLine(TO, toList));
        header.push_back(FromLine(from));
        header.push_back(AddressLine(CC, ccList));
        header.push_back(MessageIdLine(from));
        header.push_back(SubjectLine(subject));

        // empty line to seperate header from body, see RFC5322
        std::string seperator("\r\n");
        header.push_back(seperator);

        // concatenate header and body
        m_content.reserve(header.size() + body.size() );
        m_content.insert(m_content.end(), header.begin(), header.end() );
        m_content.insert(m_content.end(), body.begin(), body.end() );

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
        std::string dateTimeline("Date: " + dateTimeStr + "\r\n");
        
        return dateTimeline;
    }    

    std::string SmtpMessage::AddressLine(addressType type,
        const EmailAddresses& addresses)
    {
        std::ostringstream oss;
        oss << ((type == TO) ? "To: " : "Cc: ");
        oss << addresses << "\r\n";
        return oss.str();
    }
    
    std::string SmtpMessage::FromLine(const EmailAddress& from)
    {
        std::string fromLine("From: " + std::string(from) + "\r\n");
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
        std::cout << dateTimeStr << "\r\n";
        std::generate_n(dateTimeStr.begin() + dateLen,
                        DATE_BUFF_LENGTH - dateLen,
                        [&]() { return alphaNum[distr(gen)]; });

        std::string messageIdLine{"Message-ID: <" + dateTimeStr + "@" + from.domain() + ">\r\n"};
        return messageIdLine; 
    }
    
    std::string SmtpMessage::SubjectLine(const std::string& subject)
    {
        std::string subjectLine("Subject: " + subject + "\r\n");
        return subjectLine;
    }

    
    size_t SmtpMessage::ReadLine(char *ptr, size_t size, size_t nmemb)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_messageMutex);
        
        LOG_INFO("size = " << size << " nmemb = " << nmemb);
        LOG_INFO("m_linesRead = " << m_linesRead << " size = " << m_content.size());
        
        // if there are no more lines to read
        if (m_linesRead >= m_content.size())
        {
            m_messageState = COMPLETE;
            return 0;
        }
        size_t currentLineSize(m_content[m_linesRead].size());

        // make sure there is sufficient memory to copy the current line
        if((size == 0) or (nmemb == 0) or ( currentLineSize > (size*nmemb)))
        {
            m_messageState = FAILURE;
            LOG_ERROR("Insufficient memory to copy message " << m_messageId);
            return 0;
        }
    
        // copy the current line, add the null character, and increment the lines read
        m_content[m_linesRead].copy(ptr, currentLineSize);
        //ptr[currentLineSize] = 0;
        m_linesRead++;
        
        LOG_DEBUG("Returning " << currentLineSize);
        
        return currentLineSize;
    }
    
    void SmtpMessage::NowInProgress()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_messageMutex);

        m_messageState = INPROGRESS;
    }
    
    bool SmtpMessage::IsInProgress()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_messageMutex);

        return m_messageState == INPROGRESS;
    }
    
    void SmtpMessage::NowComplete()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_messageMutex);

        m_messageState = COMPLETE;
    }

    bool SmtpMessage::IsComplete()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_messageMutex);

        return bool(m_messageState == COMPLETE);
    }
    
    void SmtpMessage::NowFailure()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_messageMutex);

        m_messageState = FAILURE;
    }

    bool SmtpMessage::IsFailure()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_messageMutex);

        return bool(m_messageState == FAILURE);
    }
    
    uint SmtpMessage::GetState()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_messageMutex);

        LOG_DEBUG("Message state = " << m_messageState 
            << " for message with Id = " << m_messageId);
        return m_messageState;
    }
    
    static size_t SmtpMessageReadLine(char *ptr, 
        size_t size, size_t nmemb, void* pMessage)
    {
        return static_cast<SmtpMessage*>(pMessage)->
            ReadLine(ptr, size, nmemb);
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
        
        // Start the Purge timer if not already running
        if (!m_purgeTimerId)
        {
            m_purgeTimerId = g_timeout_add(1, SmtpMessageQueuePurge, this);
        }
        return bool(m_purgeTimerId);
    }

    std::shared_ptr<SmtpMessage> SmtpMessageQueue::Front()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_queueMutex);
        
        return (m_queue.empty()) ? nullptr : m_queue.front();
    }
    
    bool SmtpMessageQueue::Purge()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_queueMutex);
        
        if (m_queue.empty())
        {
            LOG_ERROR("Queue Purge timer running without messages");
        }
        while (m_queue.size() and 
            (m_queue.front()->IsComplete() or m_queue.front()->IsFailure()))
        {
            if (m_enabled and m_queue.front()->IsFailure())
            {
                LOG_ERROR("SMTP Message with Id " << m_queue.front()->GetId() 
                    << " failed to send. ");
            }
            LOG_INFO("Purging: SMTP Message with Id = " << m_queue.front()->GetId());
            m_queue.pop();
        }
        
        // If the queue is backing up, then disable new messages from being added
        m_enabled = (m_queue.size() > DSL_SMTP_MAX_PENDING_MESSAGES) ? false : true;
        
        // If there are remaining messages in the queue waiting for completion 
        // - either pending or in progress - return true to restart the timer.
        return m_queue.size();
    }
    
    static int SmtpMessageQueuePurge(gpointer pMessageQueue)
    {
        return static_cast<SmtpMessageQueue*>(pMessageQueue)->Purge();
    }
    
    // ------------------------------------------------------------------------------
    
    Comms::Comms()
        : m_initResult(curl_global_init(CURL_GLOBAL_NOTHING))
        , m_smtpSendMessageThreadId(0)
        , m_sslEnabled(true)
    {
        LOG_FUNC();

        // One-time init of Curl with no addition features
        if (m_initResult != CURLE_OK)
        {
            LOG_ERROR("curl_global_init failed: " << curl_easy_strerror(m_initResult));
            throw;
        }
        curl_version_info_data* info = curl_version_info(CURLVERSION_NOW);
        
        LOG_INFO("Libcurl Initialized Successfully");
        LOG_INFO("Version: " << info->version);
        LOG_INFO("Host: " << info->host);
        LOG_INFO("Features: " << info->features);
        LOG_INFO("SSL Version: " << info->ssl_version);
        LOG_INFO("Libz Version: " << info->libz_version);
        LOG_INFO("Protocols: " << info->protocols);
        
        g_mutex_init(&m_commsMutex);
    }
    
    Comms::~Comms()
    {
        LOG_FUNC();
        
        if (m_initResult == CURLE_OK)
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
            curl_global_cleanup();
        }
        g_mutex_clear(&m_commsMutex);
    }
    
    bool Comms::GetSmtpMailEnabled()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
        
        return m_pMessageQueue.GetEnabled();
    }

    void Comms::SetSmtpMailEnabled(bool enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
        
        m_pMessageQueue.SetEnabled(enabled);
    }

    void Comms::SetSmtpCredentials(const char* username, const char* password)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);

        m_username.assign(username);
        m_password.assign(password);
    }

    void Comms::GetSmtpServerUrl(const char** serverUrl)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
        
        *serverUrl = m_mailServerUrl.c_str();
    }
    
    void Comms::SetSmtpServerUrl(const char* serverUrl)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);

        m_mailServerUrl.assign(serverUrl);
    }
    
    void Comms::GetSmtpFromAddress(const char** name, const char** address)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
        
        m_fromAddress.Get(name, address);
    }
    
    void Comms::SetSmtpFromAddress(const char* name, const char* address)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);

        m_fromAddress.Set(name, address);
    }
    
    bool Comms::GetSmtpSslEnabled()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
        
        return m_sslEnabled;
    }

    void Comms::SetSmtpSslEnabled(bool enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
        
        m_sslEnabled = enabled;
    }

    void Comms::AddSmtpToAddress(const char* name, const char* address)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
        
        EmailAddress newToAddress(name, address);
        m_toAddresses.push_back(newToAddress);
    }

    void Comms::RemoveAllSmtpToAddresses()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
        
        m_toAddresses.clear();
    }

    void Comms::AddSmtpCcAddress(const char* name, const char* address)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
        
        EmailAddress newCcAddress(name, address);
        m_ccAddresses.push_back(newCcAddress);
    }

    void Comms::RemoveAllSmtpCcAddresses()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
        
        m_ccAddresses.clear();
    }
    
    bool Comms::IsSetup()
    {
        bool result(true);
        
        if (m_initResult != CURLE_OK)
        {
            LOG_INFO("CURL initialization failed");
            result = false;
        }
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

    bool Comms::QueueSmtpMessage(const std::string& subject, 
            const std::vector<std::string>& body)
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
                m_fromAddress, m_ccAddresses, subject, body));
        
        // queue the new Message, return on falure
        if (!m_pMessageQueue.Push(pMessage))
        {
            return false;
        }
        
        // if the SendMessage thread is not currently running
        if (!m_smtpSendMessageThreadId)
        {
            m_smtpSendMessageThreadId = g_idle_add(CommsSendSmtpMessage, this);
        }
        return (bool)m_smtpSendMessageThreadId;
        
    }
    
    bool Comms::SendSmtpMessage()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_commsMutex);
        
        // Make sure the queue has a message to process
        if (m_pMessageQueue.IsEmpty())
        {
            m_smtpSendMessageThreadId = 0;
            return false;
        }
        
        std::shared_ptr<SmtpMessage> message = m_pMessageQueue.Front();
        
        if (message->IsInProgress())
        {
            return true;
        }

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
        
        curl_easy_setopt(pCurl, CURLOPT_READFUNCTION, SmtpMessageReadLine);
        curl_easy_setopt(pCurl, CURLOPT_READDATA, &(*message));
        curl_easy_setopt(pCurl, CURLOPT_UPLOAD, 1L);

        message->NowInProgress();            
        
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
            message->NowFailure();
        }

        // free up all recipients;
        curl_slist_free_all(recipients);       

        // clean up resources
        curl_easy_cleanup(pCurl); 
        
        // if there is more than one messages to send, return true to reschedule
        if (!m_pMessageQueue.Size() > 1)
        {
            return true;
        }
        
        // otherwise, end the thread
        m_smtpSendMessageThreadId = 0;
        return false;
    }
    
    static gboolean CommsSendSmtpMessage(gpointer pComms)
    {
        return static_cast<Comms*>(pComms)->SendSmtpMessage();
    }
}

