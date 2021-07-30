/*
The MIT License

Copyright (c)   2021, Prominence AI, Inc.

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

#include "Dsl.h"
#include "DslApi.h"
#include "DslServices.h"
#include "DslServicesValidate.h"
#include "DslMailer.h"

namespace DSL
{
    DslReturnType Services::MailerNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            if (m_mailers.find(name) != m_mailers.end())
            {   
                LOG_ERROR("Mailer name '" << name << "' is not unique");
                return DSL_RESULT_MAILER_NAME_NOT_UNIQUE;
            }
            m_mailers[name] = std::shared_ptr<Mailer>(new Mailer(name));
            
            LOG_INFO("New Mailer '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Mailer '" << name << "' threw exception on create");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::MailerEnabledGet(const char* name, 
        boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);
            
            *enabled = m_mailers[name]->GetEnabled();
            
            LOG_INFO("Returning Mailer Enabled = " << *enabled);
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception on get enabled setting");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MailerEnabledSet(const char* name, 
        boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);

            m_mailers[name]->SetEnabled(enabled);
            
            LOG_INFO("Setting SMTP Mail Enabled = " << enabled);
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception on set enabled setting");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
     
    DslReturnType Services::MailerCredentialsSet(const char* name,
        const char* username, const char* password)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);
            
            m_mailers[name]->SetCredentials(username, password);

            LOG_INFO("New SMTP Username and Password set");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception setting the enabled setting");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MailerServerUrlGet(const char* name,
        const char** serverUrl)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);

            m_mailers[name]->GetServerUrl(serverUrl);

            LOG_INFO("Returning SMTP Server URL = '" << *serverUrl << "'");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception getting the Server URL");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MailerServerUrlSet(const char* name,
        const char* serverUrl)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);

            m_mailers[name]->SetServerUrl(serverUrl);

            LOG_INFO("New SMTP Server URL = '" << serverUrl << "' set");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception setting the Server URL");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MailerFromAddressGet(const char* name,
        const char** displayName, const char** address)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);

            m_mailers[name]->GetFromAddress(displayName, address);

            LOG_INFO("Returning SMTP From Address with Name = '" << *name 
                << "', and Address = '" << *address << "'" );
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception getting the From Address");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MailerFromAddressSet(const char* name,
        const char* displayName, const char* address)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);

            m_mailers[name]->SetFromAddress(displayName, address);

            LOG_INFO("New SMTP From Address with Name = '" << name 
                << "', and Address = '" << address << "' set");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception setting the From Address");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MailerSslEnabledGet(const char* name,
        boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);

            *enabled = m_mailers[name]->GetSslEnabled();
            
            LOG_INFO("Returning SSL Enabled = '" << *enabled  << "'" );
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception getting the SSL Enabled Setting");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MailerSslEnabledSet(const char* name,
        boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);

            m_mailers[name]->SetSslEnabled(enabled);
            LOG_INFO("Set SSL Enabled = '" << enabled  << "'" );
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception getting the SSL Enabled Setting");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MailerToAddressAdd(const char* name,
        const char* displayName, const char* address)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);

            m_mailers[name]->AddToAddress(displayName, address);

            LOG_INFO("New To Address with Name = '" << name 
                << "', and Address = '" << address << "' added");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception adding a To Address");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MailerToAddressesRemoveAll(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);

            m_mailers[name]->RemoveAllToAddresses();

            LOG_INFO("All To Addresses removed");
        
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception removing SSL Enabled Setting");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MailerCcAddressAdd(const char* name,
        const char* displayName, const char* address)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);

            m_mailers[name]->AddCcAddress(displayName, address);

            LOG_INFO("New Cc Address with Name = '" << name 
                << "', and Address = '" << address << "' set");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception adding a Cc Address");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::MailerCcAddressesRemoveAll(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);

            m_mailers[name]->RemoveAllCcAddresses();

            LOG_INFO("All Cc Addresses removed");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception removing all Cc Addresses");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MailerSendTestMessage(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);

            std::string subject("Test message");
            std::string bline1("Test message.\r\n");
            
            std::vector<std::string> body{bline1};

            if (!m_mailers[name]->QueueMessage(subject, body))
            {
                LOG_ERROR("Failed to queue SMTP Test Message");
                return DSL_RESULT_FAILURE;
            }
            LOG_INFO("Test message Queued successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception queuing a Test Message");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
 
    boolean Services::MailerExists(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            return (boolean)(m_mailers.find(name) != m_mailers.end());
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw an exception on check for Exists");
            return false;
        }
    }

    DslReturnType Services::MailerDelete(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);
            
            if (m_mailers[name]->IsInUse())
            {
                LOG_ERROR("Cannot delete Mailer '" << name 
                    << "' as it is currently in use");
                return DSL_RESULT_MAILER_IN_USE;
            }

            m_mailers.erase(name);

            LOG_INFO("Mailer '" << name << "' deleted successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw an exception on Delete");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }

    }

    DslReturnType Services::MailerDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            if (m_mailers.empty())
            {
                return DSL_RESULT_SUCCESS;
            }
            for (auto &imap: m_mailers)
            {
                // In the case of Delete all
                if (imap.second.use_count() > 1)
                {
                    LOG_ERROR("Can't delete Player '" << imap.second->GetName() 
                        << "' as it is currently in use");
                    return DSL_RESULT_MAILER_IN_USE;
                }
            }
            m_mailers.clear();

            LOG_INFO("All Mailers deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception on MailerDeleteAll");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }

    uint Services::MailerListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_mailers.size();
    }

}