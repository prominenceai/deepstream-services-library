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

#ifndef _DSL_DEWARPER_BINTR_H
#define _DSL_DEWARPER_BINTR_H

#include "Dsl.h"
#include "DslElementr.h"
#include "DslBintr.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_DEWARPER_PTR std::shared_ptr<DewarperBintr>
    #define DSL_DEWARPER_NEW(name, configFile) \
        std::shared_ptr<DewarperBintr>(new DewarperBintr(name, configFile))
        
    class DewarperBintr : public Bintr
    {
    public: 
    
        DewarperBintr(const char* name, const char* configFile);

        ~DewarperBintr();

        /**
         * @brief Adds the DewarperBintr to a Parent Pipeline Bintr
         * @param[in] pParentBintr Parent Pipeline to add this Bintr to
         */
        bool AddToParent(DSL_NODETR_PTR pParentBintr);

        /**
         * @brief Links all Child Elementrs owned by this Bintr
         * @return true if all links were succesful, false otherwise
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elementrs owned by this Bintr
         * Calling UnlinkAll when in an unlinked state has no effect.
         */
        void UnlinkAll();

        /**
         * @brief gets the name of the Dewarper Config File in use by this Bintr
         * @return fully qualified patspec used to create this Bintr
         */
        const char* GetConfigFile();
        
    protected:

        /**
         * @brief pathspec to the config file used by this DewarperBintr
         */
        std::string m_configFile;
        /**
         * @brief Sink Queue for input Stream
         */
        DSL_ELEMENT_PTR  m_pSinkQueue;

        /**
         * @brief Video Converter for this DewarperBintr
         */
        DSL_ELEMENT_PTR  m_pVidConv;

        /**
         * @brief Video Capabilities Filter for this DewarperBintr
         */
        DSL_ELEMENT_PTR  m_pVidCaps;

        /**
         * @brief Dewarper Element for the DewarperBintr
         */
        DSL_ELEMENT_PTR  m_pDewarper;

        /**
         * @brief Dewarper Capabilities Filter for this DewarperBintr
         */
        DSL_ELEMENT_PTR  m_pDewarperCaps;

        /**
         * @brief Source Queue for ouput stream
         */
        DSL_ELEMENT_PTR  m_pSrcQueue;
    };
    
} // DSL

#endif // _DSL_DEWARPER_BINTR_H
