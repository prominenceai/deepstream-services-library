

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

#ifndef _DSL_MUTEX_H
#define _DSL_MUTEX_H

#include "Dsl.h"


 
namespace DSL
{
    #define LOCK_MUTEX_FOR_CURRENT_SCOPE(mutex) LockMutexForCurrentScope lock(mutex)

    /**
     * @class LockMutexForCurrentScope
     * @brief Locks a GMutex for the current scope {}.
     */
    class LockMutexForCurrentScope
    {
    public:
        LockMutexForCurrentScope(GMutex* mutex) : m_pMutex(mutex) 
        {
            g_mutex_lock(m_pMutex);
        };
        
        ~LockMutexForCurrentScope()
        {
            g_mutex_unlock(m_pMutex);
        };
        
    private:
        GMutex* m_pMutex; 
    };

    #define UNREF_MESSAGE_ON_RETURN(message) UnrefMessageOnReturn ref(message)

    /**
     * @class UnrefMessageOnReturn
     * @brief Unreferences a GstMessage on return from current function scope.
     * i.e. the point at which an object of this class is destroyed
     */
    class UnrefMessageOnReturn
    {
    public:
        UnrefMessageOnReturn(GstMessage* message) : m_pMessage(message){};
        
        ~UnrefMessageOnReturn()
        {
            gst_message_unref(m_pMessage);
        };
        
    private:
        GstMessage* m_pMessage; 
    };

} // namespace 

#endif // _DSL_MUTEX_H