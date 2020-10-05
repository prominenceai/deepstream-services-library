
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

#ifndef _DSL_STATE_CHANGE_H
#define _DSL_STATE_CHANGE_H

#include "Dsl.h"

namespace DSL
{

    /**
     * @struct DslStateChange
     * @brief defines a change of state as "previous and new" states
     */
    struct DslStateChange
    {
        /**
         * @brief ctor for the stateChange struct
         * @param[in] prevsious GstState
         * @param[in] new GstState
         */
        DslStateChange(GstState previousState, GstState newState)
            : m_previousState(previousState)
            , m_newState(newState)
        {
            LOG_FUNC();
        };

        /**
         * @brief dtor for the DslStateChange struct
         */
        ~DslStateChange()
        {
            LOG_FUNC();
        };
            
        /**
         * @brief defines the previous state value
         */
        GstState m_previousState;
        
        /**
         * @brief defines the new state value
         */
        GstState m_newState;
    };
}
#endif // _DSL_STATE_CHANGE_H
