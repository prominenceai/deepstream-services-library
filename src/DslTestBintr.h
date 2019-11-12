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

#ifndef _DSL_TEST_BINTR_H
#define _DSL_TEST_BINTR_H

#include "Dsl.h"
#include "DslBintr.h"

namespace DSL
{
    #define DSL_TEST_BINTR_PTR std::shared_ptr<TestBintr>
    #define DSL_TEST_BINTR_NEW(name) \
        std::shared_ptr<TestBintr>(new TestBintr(name))
        
    /**
     * @class TestBintr
     * @brief Implements a derived Bintr class for the purpose of testing
     * the Bintr abstract class, and with Elementrs and Padtrs as well
     */
    class TestBintr : public Bintr
    {
    public: 
    
        TestBintr(const char* name)
            : Bintr(name)
        {
            LOG_FUNC();
        };

        ~TestBintr()
        {
            LOG_FUNC();
        }

        bool AddToParent(DSL_NODETR_PTR pParent)
        {
            LOG_FUNC();

            return pParent->AddChild(shared_from_this());
        };

        bool LinkAll()
        {
            LOG_FUNC();

            return true;
        };
        
        void UnlinkAll()
        {
            LOG_FUNC();
        };
        
    };
    
}

#endif // _DSL_TEST_BINTR_H
