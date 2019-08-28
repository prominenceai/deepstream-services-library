/*
The MIT License

Copyright (c) 2019-Present, ROBERT HOWELL

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

#ifndef _DSS_LOG4CXX_H
#define _DSS_LOG4CXX_H

#include <log4cxx/logger.h>


/**
returns a pointer to a log4cxx logger, specific to the calling function.
 * Each function will create a new logger on first use.
 * The logger instance will be reused on each subsequent call to the same function.
 * Note: used by the logging macros below, not to be called directly.

@param message the message string to log.
*/
#define LOG4CXX_LOGGER LogMgr::Ptr()->Log4cxxLogger(__builtin_FUNCTION())

/**
Logs the Entry and Exit of a Function with the DEBUG level.
 * Add macro as the first statement to each function of interest.
 * Consider the intrussion/penalty of this call when adding.

@param message the message string to log.
*/
#define LOG_FUNC() LogFunc lf(__builtin_FUNCTION())

/**
Logs a message with the DEBUG level.

*/
#define LOG_DEBUG(message) LOG4CXX_DEBUG(LOG4CXX_LOGGER, message)

/**
Logs a message with the INFO level.

@param message the message string to log.
*/
#define LOG_INFO(message) LOG4CXX_INFO(LOG4CXX_LOGGER, message)

/**
Logs a message with the WARN level.

@param message the message string to log.
*/
#define LOG_WARN(message) LOG4CXX_WARN(LOG4CXX_LOGGER, message)

/**
Logs a message with the ERROR level.

@param message the message string to log.
*/
#define LOG_ERROR(message) LOG4CXX_ERROR(LOG4CXX_LOGGER, message)

/**
Logs a message with the FATAL level.

@param message the message string to log.
*/
#define LOG_FATAL(message) LOG4CXX_FATAL(LOG4CXX_LOGGER, message)

namespace DSS
{

    class LogMgr
    {
    public:
        static LogMgr* Ptr();
        log4cxx::LoggerPtr Log4cxxLogger(const std::string& name);
    
    private:
        LogMgr(){};
        static LogMgr* m_pInstatnce;
    };  
    
    class LogFunc
    {
    public:
        LogFunc(const std::string& name);
        ~LogFunc();
    private:
        const std::string& m_name; 
    };

} // namespace 

#endif // _DSS_LOG4CXX_H