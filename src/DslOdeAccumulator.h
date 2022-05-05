/*
The MIT License

Copyright (c) 2022, Prominence AI, Inc.

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

#ifndef _DSL_ODE_ACCUMULATOR_H
#define _DSL_ODE_ACCUMULATOR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslOdeBase.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_ODE_ACCUMULATOR_PTR std::shared_ptr<OdeAccumulator>
    #define DSL_ODE_ACCUMULATOR_NEW(name) \
        std::shared_ptr<OdeAccumulator>(new OdeAccumulator(name))

    // *****************************************************************************

    /**
     * @class OdeTrigger
     * @brief Implements a super/abstract class for all ODE Triggers
     */
    class OdeAccumulator : public OdeBase
    {
    public: 
    
        OdeAccumulator(const char* name);

        ~OdeAccumulator();

        /**
         * @brief Handles the ODE occurrence by calling the client handler
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrences(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta);
        
        /**
         * @brief Adds an ODE Action as a child to this OdeAccumulator
         * @param[in] pChild pointer to ODE Action to add
         * @return true if successful, false otherwise
         */
        bool AddAction(DSL_BASE_PTR pChild);
        
        /**
         * @brief Removes a child ODE Action from this OdeAccumulator
         * @param[in] pChild pointer to ODE Action to remove
         * @return true if successful, false otherwise
         */
        bool RemoveAction(DSL_BASE_PTR pChild);
        
        /**
         * @brief Removes all child ODE Actions from this OdeAccumulator
         */
        void RemoveAllActions(); 

    private:
    
        /**
         * @brief running accumulation of ODE Occurrences.
         */
        uint64_t m_accumulation;
    
        /**
         * @brief Index variable to incremment/assign on ODE Action add.
         */
        uint m_nextActionIndex;

        /**
         * @brief Map of child ODE Actions owned by this OdeAccumulator
         */
        std::map <std::string, DSL_BASE_PTR> m_pOdeActions;
        
        /**
         * @brief Map of child ODE Actions indexed by their add-order for execution
         */
        std::map <uint, DSL_BASE_PTR> m_pOdeActionsIndexed;

    };

}

#endif // _DSL_ODE_ACCUMULATOR_H

