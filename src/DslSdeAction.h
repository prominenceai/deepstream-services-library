/*
The MIT License

Copyright (c) 2019-2024, Prominence AI, Inc.

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

#ifndef _DSL_SDE_ACTION_H
#define _DSL_SDE_ACTION_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslDeBase.h"

namespace DSL
{
    
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_SDE_ACTION_PTR std::shared_ptr<SdeAction>

    #define DSL_SDE_ACTION_PRINT_PTR std::shared_ptr<PrintSdeAction>
    #define DSL_SDE_ACTION_PRINT_NEW(name, forceFlush) \
        std::shared_ptr<PrintSdeAction>(new PrintSdeAction(name, forceFlush))

// ********************************************************************

    class SdeAction : public DeBase
    {
    public: 
    
        /**
         * @brief ctor for the SDE Action virtual base class
         * @param[in] name unique name for the SDE Action
         */
        SdeAction(const char* name);

        ~SdeAction();

        /**
         * @brief Virtual function to handle the occurrence of an SDE by taking.
         * a specific Action as implemented by the derived class.
         * @param[in] pSdeTrigger shared pointer to SDE Trigger that triggered the event.
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event.
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event.
         */
        virtual void HandleOccurrence(DSL_BASE_PTR pSdeTrigger, 
            GstBuffer* pBuffer, NvDsAudioFrameMeta* pFrameMeta) = 0;
        
    protected:

        std::string Ntp2Str(uint64_t ntp);

    };

    // ********************************************************************

    /**
     * @class PrintSdeAction
     * @brief Print SDE Action class
     */
    class PrintSdeAction : public SdeAction
    {
    public:
    
        /**
         * @brief ctor for the SDE Print Action class
         * @param[in] name unique name for the SDE Action
         */
        PrintSdeAction(const char* name, bool forceFlush);
        
        /**
         * @brief dtor for the Print SDE Action class
         */
        ~PrintSdeAction();
        
        /**
         * @brief Handles the SDE occurrence by printing the  
         * the occurrence data to the console
         * @param[in] pSdeTrigger shared pointer to SDE Trigger that triggered the event
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         */
        void HandleOccurrence(DSL_BASE_PTR pSdeTrigger, 
            GstBuffer* pBuffer, NvDsAudioFrameMeta* pFrameMeta);

        /**
         * @brief Flushes the stdout buffer. ** To be called by the idle thread only **.
         * @return false to unschedule always - single flush operation.
         */
        bool Flush();

    private:

        /**
         * @brief flag to enable/disable forced stream buffer flushing
         */
        bool m_forceFlush;
    
        /**
         * @brief gnome thread id for the background thread to flush
         */
        uint m_flushThreadFunctionId;

        /**
         * @brief mutex to protect mutual access to m_flushThreadFunctionId
         */
        DslMutex m_ostreamMutex;
    };

}

#endif // _DSL_SDE_ACTION_H
