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

#include <log4cxx/basicconfigurator.h>

#include "dss_log4cxx.h"

namespace ChildAlone
{
     
    LogMgr* LogMgr::m_pInstatnce = NULL;

    LogMgr* LogMgr::Ptr()
    {
        if (!m_pInstatnce)
        {
            // Set up a simple configuration that logs on the console.
            log4cxx::BasicConfigurator::configure();
            
            m_pInstatnce = new LogMgr;
        }
        return m_pInstatnce;
    }

    log4cxx::LoggerPtr LogMgr::Log4cxxLogger(const std::string& name)
    {
        return log4cxx::Logger::getLogger(name);
    }
    
    LogFunc::LogFunc(const std::string& name)
        : m_name(name)
    {
        LOG4CXX_DEBUG(LogMgr::Ptr()->Log4cxxLogger(m_name), "--- ENTER ---");        
    }

    LogFunc::~LogFunc()
    {
        LOG4CXX_DEBUG(LogMgr::Ptr()->Log4cxxLogger(m_name), "--- EXIT ---");        
    }

} // namespace 