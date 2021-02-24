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

#ifndef _DSL_LOGGST_H
#define _DSL_LOGGST_H

GST_DEBUG_CATEGORY_EXTERN(GST_CAT_DSL);

#include "Dsl.h"

namespace DSL
{

/**
 * Logs the Entry and Exit of a Function with the DEBUG level.
 * Add macro as the first statement to each function of interest.
 * Consider the intrussion/penalty of this call when adding.
 */
#define LOG_FUNC() LogFunc lf(__METHOD_NAME__)

#define LOG(message, level) \
    do \
    { \
        std::stringstream logMessage; \
        logMessage  << " : " << message; \
        GST_CAT_LEVEL_LOG(GST_CAT_DSL, level, NULL, logMessage.str().c_str(), NULL); \
    } while (0)

#define LOG_DEBUG(message) LOG(message, GST_LEVEL_DEBUG)

#define LOG_INFO(message) LOG(message, GST_LEVEL_INFO)

#define LOG_WARN(message) LOG(message, GST_LEVEL_WARNING)

#define LOG_ERROR(message) LOG(message, GST_LEVEL_ERROR)
 
    /**
     * @class LogFunc
     * @brief Used to log entry and exit of a function.
     */
    class LogFunc
    {
    public:
        LogFunc(const std::string& method) 
        {
            m_logMessage  << method;
            GST_CAT_LEVEL_LOG(GST_CAT_DSL, GST_LEVEL_DEBUG, NULL, 
                m_logMessage.str().c_str(), "");
        };
        
        ~LogFunc()
        {
            GST_CAT_LEVEL_LOG(GST_CAT_DSL, GST_LEVEL_DEBUG, NULL, 
                m_logMessage.str().c_str(), "");
        };
        
    private:
        std::stringstream m_logMessage; 
    };

} // namespace 


#endif // _DSL_LOGGST_H
