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

#ifndef _DSL_EVENT_H
#define _DSL_EVENT_H

#include "Dsl.h"
#include "DslApi.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_EVENT_PTR std::shared_ptr<EventBase>
        
    class EventBase
    {
    public: 
    
        EventBase(const char* name)
            : m_name(name)
        {
            LOG_FUNC();

            g_mutex_init(&m_eventMutex);
        }

        virtual ~EventBase()
        {
            LOG_FUNC();

            g_mutex_clear(&m_eventMutex);
        }
        
        /**
         * @brief Allows a client to determined derived type from base pointer
         * @param[in] typeInfo to compare against
         * @return true if this Element is of typeInfo, false otherwise
         */
        bool IsType(const std::type_info& typeInfo)
        {
            LOG_FUNC();
            
            return (typeInfo.hash_code() == typeid(*this).hash_code());
        }

        /**
         * @brief returns the name given to this Event on creation
         * @return const std::string name given to this Event
         */
        const std::string& GetName()
        {
            LOG_FUNC();
            
            return m_name;
        }
        
        /**
         * @brief return the name given to this Event on creation
         * @return const c_str name given to this Event
         */
        const char* GetCStrName()
        {
            LOG_FUNC();
            
            return m_name.c_str();
        }

        /**
         * @brief called to determine if an Element is currently in use - i.e. has a Parent
         * @return true if the Bintr has a Parent, false otherwise
         */
        bool IsInUse()
        {
            LOG_FUNC();
            
            return (bool)m_parentName.size();
        }
        
        void AssignParentName(const std::string& name)
        {
            LOG_FUNC();

            m_parentName.assign(name);
        }
        void ClearParentName()
        {
            LOG_FUNC();
            
            m_parentName.clear();
        }

    protected:

        /**
         * @brief Unique name for this Event
         */
        std::string m_name;

        /**
         * @brief Unique name of the Parent reporter if in use.
         * Empty string if not-in-use
         */
        std::string m_parentName;

        /**
         * @brief mutex to gaurd system/client propery access
        */
        GMutex m_eventMutex;
    };
}


#endif // _DSL_EVENT_H
