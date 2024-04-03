
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

#ifndef _DSL_OFV_BINTR_H
#define _DSL_OFV_BINTR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslElementr.h"
#include "DslBintr.h"

namespace DSL
{
    
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_OFV_PTR std::shared_ptr<OfvBintr>
    #define DSL_OFV_NEW(name) \
        std::shared_ptr<OfvBintr>(new OfvBintr(name))

    /**
     * @class OfvBintr
     * @brief Implements an Optical Flow bin container
     */
    class OfvBintr : public Bintr
    {
    public: 
    
        /**
         * @brief ctor for the OfvBintr class
         * @param[in] name name to give the new OfvBintr
         */
        OfvBintr(const char* name);

        /**
         * @brief dtor for the OfvBintr class
         */
        ~OfvBintr();

        /**
         * @brief Adds this OfvBintr to a Parent Pipline Bintr
         * @param[in] pParentBintr
         */
        bool AddToParent(DSL_BASE_PTR pParentBintr);
        
        /**
         * @brief Removes this OfvBintr from a Parent Pipline Bintr
         * @param[in] pParentBintr
         */
        bool RemoveFromParent(DSL_BASE_PTR pParentBintr);
        
        /**
         * @brief Links all child elements of this OfvBintr
         * @return true if all elements were succesfully linked, false otherwise.
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all child elements of the OfvBintr
         */
        void UnlinkAll();

    private:
        
        /**
         @brief
         */
        
        DSL_ELEMENT_PTR m_pOptFlowQueue;
        DSL_ELEMENT_PTR m_pOptFlow;
        DSL_ELEMENT_PTR m_pOptFlowVisualQueue;
        DSL_ELEMENT_PTR m_pOptFlowVisual;
    };
}

#endif // _DSL_OFV_BINTR_H