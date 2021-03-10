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

#ifndef _DSL_COMMS_H
#define _DSL_COMMS_H

#include "Dsl.h"

namespace DSL {

    /**
     * @class EmailAddress
     * @brief Implements an Email Address class with display name and 
     */
    class EmailAddress 
    {
    public:
    
        /**
         * @brief ctor for an undefined EmailAdress class
         */
        EmailAddress()
            : m_isSet(false)
        {
        }
        
        /**
         * @brief ctor for the EmailAdress class
         * @param name display name for the Email Address
         * @param address actual email address
         */
        EmailAddress(const char* name, const char* address)
            : m_name(name)
            , m_address(address)
            , m_nameStr{"\"" + std::string(name) + "\""}
            , m_addressStr{address ? "<" + std::string(address) + ">" : ""}
            , m_isSet(true)
        {
        }
        
        /**
         * @brief Updates and existing Email address
         * @param name new display name to assign to this email address
         * @param address new email address to assign to the email address
         */
        void Set(const char* name, const char* address)
        {
            m_name.assign(name);
            m_address.assign(address);
            m_nameStr.assign("\"" + std::string(name) + "\"");
            m_addressStr.assign(address ? "<" + std::string(address) + ">" : "");
            m_isSet = true;
        }
        
        /**
         * @brief Gets the name and address components used to create/set this object
         * @param name display name assigned to this email address
         * @param address email address assigned to this email address object
         */
        void Get(const char** name, const char** address)
        {
            *name = m_name.c_str();
            *address = m_address.c_str();
        }
        
        /**
         * @brief returns the state of the email address, set or not
         * @return true if the email is set, false otherwise 
         */
        bool IsSet()
        {
            return m_isSet;
        }

        /**
         * @brief returns the domain from the email address
         * @return domain name for the email address
         */
        std::string domain() const
        {
            return m_address.substr(m_address.find('@') + 1);
        }

        /**
         * @brief operator to return the char* string for the address
         */
        explicit operator const char *() const
        {
            return m_addressStr.c_str();
        }
        
        /**
         * @brief operator to return the string object for Address
         */
        explicit operator const std::string () const
        {
            return m_addressStr;
        }
        
        friend std::ostream &operator<<(std::ostream &out, const EmailAddress &emailAddress)
        {
            return out << emailAddress.m_nameStr << " " << emailAddress.m_addressStr;
        }

    private:
        bool m_isSet;
        std::string m_name;
        std::string m_address;
        std::string m_nameStr;
        std::string m_addressStr;
    };
    
    /**
     * @brief type defining a list of email addresses, either To or Cc
     */
    typedef std::vector<EmailAddress> EmailAddresses;
    
    /**
     * @brief stream operator to assemble a comma seperated email
     * list with display names and email addresses
     * @param out stream to update and return
     * @param emailAddresses vector of EmailAddress objects
     */
    static std::ostream &operator<<(std::ostream &out, const EmailAddresses &emailAddresses)
    {
        if (!emailAddresses.empty()) 
        {
            auto const &iter = emailAddresses.begin();
            out << *iter;

            for (auto iter = emailAddresses.begin()+1; iter != emailAddresses.end(); ++iter)
            {
                out << "," << *iter;
            }        
        }
        return out;
    }    
    
    /**
     * @brief SMTP Message states: PENDING on create, INPROGRESS, on 
     * first Read() call from libcurl, COMPLETE on message fully read.
     */
    enum MessageState{ PENDING, INPROGRESS, COMPLETE, FAILURE };
    
    /**
     * @brief Address lines types used to specify which Email Address line to build
     */
    enum addressType{ TO, CC };

    /**
     * @class SmtpMessage
     * @brief Implements a SMPT Message, ready for reading by libcurl on message upload.
     */
    class SmtpMessage
    {
    public:
    
        /**
         * @brief ctor for the SmtpMessageData class
         * @param[in] toList recipient TO list of emails
         * @param[in] from sender's email
         * @param[in] ccList recipient CC list of eamils.
         * @param[in] subject subject of this message.
         * @param[in] body unique body content for this message
         */
        SmtpMessage(const EmailAddresses& toList,
            const EmailAddress& from, const EmailAddresses& ccList,
            const std::string& subject, const std::vector<std::string>& body);
        
        /**
         * @brief dtor for the SmtpMessageData class
         */
        ~SmtpMessage();
        
        /**
         * @brief Callback function to read successive lines of email
         * content - called by libcurl in response to an UPLOAD command. 
         * @param[out] ptr memory to copy the next line into
         * @param[in] size of remaining memory to copy into
         * @param[in] nmemb number of memory blocks??
         * @return the length of the line read.
         */
        size_t ReadLine(char *ptr, size_t size, size_t nmemb);
        
        /**
         * @brief Sets the SMTP Message into an in-progress state
         */
        void NowInProgress();

        /**
         * @brief Queries the In-Progress state of the Message
         * @return true if the Message reead is in progerss, false otherwise
         */
        bool IsInProgress();
        
        /**
         * @brief Sets the SMTP Message into a COMPLETE state
         */
        void NowComplete();

        /**
         * @brief Queries the Completion state of the Message
         * @return true if the Message has been Read completely, false otherwise
         */
        bool IsComplete();
        
        /**
         * @brief Sets the SMTP Message into a FAILED state
         */
        void NowFailure();
        
        /**
         * @brief Queries the Failure state of the Message
         * @return true if the Message failed to send, false otherwise
         */
        bool IsFailure();
        
        /**
         * @brief returns the unique Id for this message
         * @return unique message Id
         */
        uint GetId(){return m_messageId;};
    
    private:
    
        /**
         * @brief Returns an RFC5322 formated Date-Time line to start the message
         * @return new, current Date-Time string object for the "Date:" line
         */
        std::string DateTimeLine();
        
        /**
         * @brief Returns an RFC5322 formated Address line, either To or Cc
         * @param[in] type specifes which line type to build: To or Cc
         * @param[in] addresses a list of EmailAddress object to create the Address line from
         * @return formated "To:" or "Cc: string object for one of the address lines
         */
        std::string AddressLine(addressType type,
            const EmailAddresses& addresses);

        /**
         * @brief Returns an RFC5322 formated From line with sender's email addres
         * @param[in] from email address of the sender
         * @return formated "From:" string object with sender's email address
         */
        std::string FromLine(const EmailAddress& from);

        /**
         * @brief Returns a randum/unique MessageIdLine using the senders domain 
         * @param from sender's email address to extract email domain
         * @return formated "Message ID:" string object
         */
        std::string MessageIdLine(EmailAddress from);
        
        /**
         * @brief Returns an RFC5322 formated Subject line
         * @param[in] subject content for the subject line
         * @return formated "Subject:" string object 
         */
        std::string SubjectLine(const std::string& subject);

        /**
         * @brief mutex to protect mutual access to queue data
         */
        GMutex m_messageMutex;
        
        /**
         * @brief static message counter for all messages, 
         * incremented on message creation
         */
        static uint s_nextMessageId;

        /**
         * @brief static message counter for all messages, 
         * incremented on message creation
         */
        uint m_messageId;
        
        /**
         * @brief unique message content added to the body of
         * the message, all other content is template based
         */
        std::string m_uniqueContent;
        
        /**
         * @brief complete message content provided to libcurl
         * with the ReadLine callback. One vector element per line
         */
        std::vector<std::string> m_content;
        
        /**
         * @brief maintains the number of lines read by libcurl over
         * successive calls to the ReadLine callback
         */
        uint m_linesRead;

        /**
         * @brief message send states
         */
        MessageState m_messageState;
    };

    static size_t SmtpMessageReadLine(char *ptr, 
        size_t size, size_t nmemb, void* pMessage);

    /**
     * @class SmtpMessageQueue
     * @brief Implements a self-purging outgoing SMPT message queue
     */
    class SmtpMessageQueue
    {
    public:
    
        /**
         * @brief ctor for the SmtpMessageQueue class
         */
        SmtpMessageQueue();
        
        /**
         * @brief dtor for the SmtpMessageQueue class
         */
        ~SmtpMessageQueue();
        
        /**
         * @brief inserts a new SMTP Message
         * @param message new message to queue
         * @return true if the message could be queue successfully, false otherwise.
         */
        bool Push(std::shared_ptr<SmtpMessage> message);
        
        /**
         * @brief queries the queue to see if the Queue is empty
         * @return true if the queue is emptry, false other
         */
        bool IsEmpty(){return m_queue.empty();};

        /**
         * @brief queries the queue for its current size - number of entries
         * @return the current size of the queue
         */
        uint Size(){return m_queue.size();};
        
        /**
         * @brief returns a pointer to the element at the front of the Queue
         * @return shared pointer to the message at the front.
         */
        std::shared_ptr<SmtpMessage> Front();
        
        /**
         * @brief purges all messages in the queue that have been fully read by libcurl
         */
        bool Purge();
        
        /**
         * @brief gets the current enabled setting for the queue
         * @return current enabled setting
         */
        bool GetEnabled();
        
        /**
         * @brief sets the enabled setting for the queue
         */
        void SetEnabled(bool enabled);
        
    private:
    
        /**
         * specifies the current queue state, true if able to queue message.
         */
        bool m_enabled;
        
        /**
         * @brief mutex to protect mutual access to queue data
         */
        GMutex m_queueMutex;

        /**
         * @brief Queue of SMTP Messages in one of three states.
         */
        std::queue<std::shared_ptr<SmtpMessage>> m_queue;
        
        /**
         * @brief gnome timer id for the self purging 
         */
        uint m_purgeTimerId;
    };

    static int SmtpMessageQueuePurge(gpointer pMessageQueue);
    
    
    /**
     * @class Comms
     * @brief Implements a Comms abstraction class for libcurl
     */
    class Comms
    {
    public:
    
        /**
         * @brief ctor for the Comms class
         */
        Comms();
        
        /**
         * @brief dtor for the Comms class
         */
        ~Comms();

        /**
         * @brief Get the current enabled state for the SMTP Email Queue
         * @return true if enabled, false otherwise.
         */
        bool GetSmtpMailEnabled();
        
        /**
         * @brief Sets the enabled state for the SMTP Email Queue
         * @param enabled set to true to enable. false to disable.
         */
        void SetSmtpMailEnabled(bool enabled);
        
        /**
         * @brief Sets the current SMTP Credentials for outgoing mail
         * @param[in] userId mail account user Id
         * @param[in] password mail account password
         */
        void SetSmtpCredentials(const char* username, const char* password);
        
        /**
         * @brief Gets the current SMTP Sender and Mail server string values
         * @param[out] serverUrl current Sender's SMTP mail server URL
         */
        void GetSmtpServerUrl(const char** serverUrl);
        
        /**
         * @brief Sets the current SMTP Sender and Mail server string values
         * @param[in] serverUrl new Sender's SMTP mail server URL to use
         */
        void SetSmtpServerUrl(const char* serverUrl);
        
        /**
         * @brief Gets the current "From:" email adress 
         * @param[out] name the display name used for the "From:" address
         * @param[out] address the action email address
         */
        void GetSmtpFromAddress(const char** name, const char** address);

        /**
         * @brief Sets the current "From:" email adress to use for all 
         * subsequent emails
         * @param[in] name the display name to use for the "From:" address
         * @param[in] address the action email address to use
         */
        void SetSmtpFromAddress(const char* name, const char* address);
        
        /**
         * @brief Get the current SSL enabled setting, default = true.
         * @return true if SSL is enabled, false otherwise
         */
        bool GetSmtpSslEnabled();
        
        /**
         * @brief Set the SSL enabled setting for subsequent emails
         * @param enabled set to true to enable SSL, false otherwise
         */
        void SetSmtpSslEnabled(bool enabled);
        
        /**
         * @brief Add a new TO address for subsequent emails
         * @param[in] toAddress new email address to add to the TO recepients
         */
        void AddSmtpToAddress(const char* name, const char* address);
        
        /**
         * @brief Deletes all TO address from the recepients list
         */
        void RemoveAllSmtpToAddresses();

        /**
         * @brief Add a new CC address for supsequent emails
         * @param[in] toAddress new email address to add to the TO recepients
         */
        void AddSmtpCcAddress(const char* name, const char* address);
        
        /**
         * @brief Deletes all CC address from the recepients list
         */
        void RemoveAllSmtpCcAddresses();
        
        /**
         * @brief Returs the initialization state of SMTP services
         * @return true if setup false otherwise
         */
        bool IsSetup();
        
        /**
         * @brief Queues a SMTP Message to be sent to all current recepients
         * @param[in] subject subject line for the email /r/n terminated
         * @param[in] body message body to add, each line /r/n terminated
         * @return true if successfully queued, false otherwise
         */
        bool QueueSmtpMessage(const std::string& subject, 
            const std::vector<std::string>& body);

        /**
         * @brief background thread function to send a pending SMTP message
         * @return true if the message was sent successfully, false otherwise
         */
        bool SendSmtpMessage();
        
    private:

        /**
         * @breif result of the curl_global_init() call, used by
         * the dtor to know whether to call curl_global_cleanup()
         */
        CURLcode m_initResult; 
        
        /**
         * @brief mutex to protect mutual access to comms data
         */
        GMutex m_commsMutex;
        
        /**
         * @brief Sender's mail account User Name
         */
        std::string m_username;

        /**
         * @brief Sender's mail account Password
         */
        std::string m_password;

        /**
         * @brief Sender's Mail Server URL
         */
        std::string m_mailServerUrl;

        /**
         * @brief Sender's email address 
         */
        EmailAddress m_fromAddress;
        
        /**
         * @brief Boolean flag to use SSL on send, default = true;
         */
        bool m_sslEnabled;
        
        /** 
         * @brief List of TO email address
         */
        EmailAddresses m_toAddresses;

        /** 
         * @brief List of CC email address
         */
        EmailAddresses m_ccAddresses;

        /**
         * @brief gnome thread id for the background send message thread
         */
        uint m_smtpSendMessageThreadId;
        
        /**
         * @brief queue of pending, in-progress, and complete (in a 
         * state of waiting to be purged) messages.
         */
        SmtpMessageQueue m_pMessageQueue;
    };

    static int CommsSendSmtpMessage(gpointer pComms);
}


#endif // _DSL_DRIVER_H    
