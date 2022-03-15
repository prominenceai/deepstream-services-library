/*
The MIT License

Copyright (c)   2022, Prominence AI, Inc.

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

namespace DSL
{

    std::string Services::GST_DEBUG = "GST_DEBUG";
    std::string Services::GST_DEBUG_FILE = "GST_DEBUG_FILE";
    
    DslReturnType Services::InfoInitDebugSettings()
    {
        LOG_FUNC();

        try
        {
            // Get the current GST_DEBUG log level if set.
            const char* logLevel = getenv(GST_DEBUG.c_str());
            
            if (logLevel)
            {
                m_gstDebugLogLevel.assign(logLevel);
            }
            
            // Get the current GST_DEBUG_FILE if set.
            const char* logFile = getenv(GST_DEBUG_FILE.c_str());
            
            if (logFile)
            {
                m_debugLogFilePath.assign(logFile);
            }
            
            LOG_INFO("Pre-start log level  = " << m_gstDebugLogLevel);
            LOG_INFO("Pre-start Log file  = " << m_debugLogFilePath);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception redirecting stdout");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::InfoDeinitDebugSettings()
    {
        LOG_FUNC();

        try
        {
            if (m_debugLogFileHandle)
            {
                fclose(m_debugLogFileHandle);
                LOG_INFO("DSL closed the current log file = '" 
                    << m_debugLogFilePath.c_str() << "'");
                m_debugLogFileHandle = NULL;
            }
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception redirecting stdout");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::InfoStdOutRedirect(const char* filepath)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            if (m_stdOutRedirectFile.is_open())
            {
                LOG_ERROR("stdout is currently/already in a redirected state");
                return DSL_RESULT_FAILURE;
            }
            
            // backup the default 
            m_stdOutRdBufBackup = std::cout.rdbuf();
            
            // open the redirect file and the rdbuf
            m_stdOutRedirectFile.open(filepath, std::ios::out);
            std::streambuf* redirectFileRdBuf = m_stdOutRedirectFile.rdbuf();
            
            // assign the file's rdbuf to the stdout's
            std::cout.rdbuf(redirectFileRdBuf);
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception redirecting stdout");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }
    
    void Services::InfoStdOutRestore()
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            if (!m_stdOutRedirectFile.is_open())
            {
                LOG_ERROR("stdout is not currently in a redirected state");
                return;
            }

            // restore the stdout to the initial backupt
            std::cout.rdbuf(m_stdOutRdBufBackup);

            // close the redirct file
            m_stdOutRedirectFile.close();
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception close stdout redirect file");
        }
    }
    
    DslReturnType Services::InfoLogLevelGet(const char** level)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            *level = m_gstDebugLogLevel.c_str();
            
            LOG_INFO("The GST debug log level = " << *level);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception getting log-level");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::InfoLogLevelSet(const char*  level)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            gst_debug_set_threshold_from_string(level, true);
            m_gstDebugLogLevel.assign(level);
            
            LOG_INFO("DSL set the GST debug log level = " << level);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception setting log-level");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::InfoLogFileGet(const char** filePath)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            *filePath = m_debugLogFilePath.c_str();
            LOG_INFO("The GST debug log file = " << *filePath);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception getting log-file");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::InfoLogFileSet(const char* filePath)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            if (m_debugLogFileHandle)
            {
                fclose(m_debugLogFileHandle);
                LOG_INFO("DSL closed the current log file = '" 
                    << m_debugLogFilePath.c_str() << "'");
                m_debugLogFileHandle = NULL;
            }    
            m_debugLogFilePath.assign(filePath);
            m_debugLogFilePath.append(".log");
            m_debugLogFileHandle = fopen(m_debugLogFilePath.c_str(),"w");
            
            if (!m_debugLogFileHandle)
            {
                LOG_ERROR("DSL failed to create log-file = '" << filePath << "'");
                return DSL_RESULT_FAILURE;
            }
            
            gst_debug_remove_log_function(gst_debug_log_default);
            gst_debug_add_log_function(gst_debug_log_override, this, NULL);
            LOG_INFO("DSL set the debug log file = " << filePath);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception on setting log-file");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::InfoLogFileSetWithTs(const char* filePath)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            std::string stringFilePath = filePath;
            auto t = std::time(nullptr);
            auto tm = *std::localtime(&t);

            std::ostringstream filePathStream;
            filePathStream << stringFilePath << std::put_time(&tm, "-%Y%m%d-%H%M%S.log");
            std::string fullFilePath = filePathStream.str();
            
            if (m_debugLogFileHandle)
            {
                fclose(m_debugLogFileHandle);
                LOG_INFO("DSL closed the current log file = '" 
                    << m_debugLogFilePath.c_str() << "'");
                m_debugLogFileHandle = NULL;
            }    
            m_debugLogFileHandle = fopen(fullFilePath.c_str(),"w");
            
            if (!m_debugLogFileHandle)
            {
                LOG_ERROR("DSL failed to create log-file = '" << fullFilePath << "'");
                return DSL_RESULT_FAILURE;
            }
            m_debugLogFilePath.assign(fullFilePath);
            
            gst_debug_remove_log_function(gst_debug_log_default);
            gst_debug_add_log_function(gst_debug_log_override, this, NULL);
            LOG_INFO("DSL set the debug log file = " << filePath);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception on setting log-file with timestamp");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }
    
    FILE* Services::InfoLogFileHandleGet()
    {
        return m_debugLogFileHandle;
    }
    
    DslReturnType Services::InfoLogFunctionRestore()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            if (m_debugLogFileHandle)
            {
                fclose(m_debugLogFileHandle);
                LOG_INFO("DSL closed the current log file = '" 
                    << m_debugLogFilePath.c_str() << "'");
                m_debugLogFileHandle = NULL;
                
                gst_debug_remove_log_function(gst_debug_log_override);
                gst_debug_add_log_function(gst_debug_log_default, NULL, NULL);
                LOG_INFO("DSL Restored default log function");
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception on setting log-file with timestamp");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }

    static void gst_debug_log_override(GstDebugCategory * category, GstDebugLevel level,
        const gchar * file, const gchar * function, gint line,
        GObject * object, GstDebugMessage * message, gpointer unused)
    {
        gst_debug_log_default(category, level, file, function, line, 
            object, message, Services::GetServices()->InfoLogFileHandleGet());
    }
}
