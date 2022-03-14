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
    
    DslReturnType Services::DebugLogLevelGet(const char** level)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            *level = getenv(GST_DEBUG.c_str());
            
            if (!*level)
            {
                LOG_INFO("The GST_DEBUG environment variable is not set");
            }
            else
            {
                LOG_INFO("The GST_DEBUG environment variable = " << *level);
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception getting log-level");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::DebugLogLevelSet(const char*  level)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            setenv(GST_DEBUG.c_str(), level, true);
            LOG_INFO("DSL set the GST_DEBUG environment variable = " << level);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception setting log-level");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::DebugLogFileGet(const char** filePath)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            *filePath = getenv(GST_DEBUG_FILE.c_str());
            if (!*filePath)
            {
                LOG_INFO("The GST_DEBUG_FILE environment variable is not set");
            }
            else
            {
                LOG_INFO("The GST_DEBUG_FILE environment variable = " << *filePath);
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception getting log-file");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DebugLogFileSet(const char* filePath)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            setenv(GST_DEBUG_FILE.c_str(), filePath, true);
            LOG_INFO("DSL set the GST_DEBUG_FILE environment variable = " << filePath);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception on setting log-file");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DebugLogFileSetWithTs(const char* filePath)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            setenv(GST_DEBUG_FILE.c_str(), filePath, true);
            LOG_INFO("DSL set the GST_DEBUG_FILE environment variable = " << filePath);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception on setting log-file with timestamp");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }
}
