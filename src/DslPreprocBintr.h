
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

#ifndef _DSL_PREPROC_BINTR_H
#define _DSL_PREPROC_BINTR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslElementr.h"
#include "DslBintr.h"

namespace DSL
{
    
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_PREPROC_PTR std::shared_ptr<PreprocBintr>
    #define DSL_PREPROC_NEW(name, configFile) \
        std::shared_ptr<PreprocBintr>(new PreprocBintr(name, \
            configFile))

    /**
     * @class PreprocBintr
     * @brief Implements an On-Screen-Display bin container
     */
    class PreprocBintr : public Bintr
    {
    public: 
    
        /**
         * @brief ctor for the PreprocBintr class
         * @param[in] name name to give the new PreprocBintr
         * @param[in] configFile absolute or relative path to the Pre-Process 
         * config text file.
         */
        PreprocBintr(const char* name, const char* configFile);

        /**
         * @brief dtor for the PreprocBintr class
         */
        ~PreprocBintr();

        /**
         * @brief Adds this PreprocBintr to a Parent Branch Bintr
         * @param[in] pParentBintr parent Pipeline to add to
         * @return true on successful add, false otherwise
         */
        bool AddToParent(DSL_BASE_PTR pParentBintr);
        
        /**
         * @brief Removes this PreprocBintr to a Parent Branch Bintr
         * @param[in] pParentBintr parent Pipeline to remove from
         * @return true on successful add, false otherwise
         */
        bool RemoveFromParent(DSL_BASE_PTR pParentBintr);
        
        /**
         * @brief Links all child elements of this PreprocBintr
         * @return true if all elements were succesfully linked, false otherwise.
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all child elements of the PreprocBintr
         */
        void UnlinkAll();

        /**
         * @brief Sets the GPU ID for all Elementrs.
         * @return true if successfully set, false otherwise.
         */
        bool SetGpuId(uint gpuId);
        
        /**
         * @brief gets the name of the Pre-Process plugin config file in use 
         * by this PreprocBintr.
         * @return absolute or relative patspec used to create this PreprocBintr.
         */
        const char* GetConfigFile();
        
        /**
         * @brief sets the name of the Pre-Process plugin config file to use 
         * for this PreprocBintr.
         * @param[in] absolute or relative path to the config file to use.
         * @return true on successful update, false otherwise.
         */
        bool SetConfigFile(const char* configFile);
        
        /**
         * @brief Gets the current enabled setting for the PreprocBintr.
         * @return true if enabled, false otherwise.
         */
        bool GetEnabled();
        
        /**
         * @brief Sets the enabled setting for the PreprocBintr.
         * @param[in] enabled set to true to enabled, false otherwise.
         * @return true on successful update, false otherwise.
         */
        bool SetEnabled(bool enabled);
        
        /**
         * @brief Gets the assigned unique Id for the PreprocBintr.
         * @return unique Id.
         */
        uint GetUniqueId();

        /**
         * @brief static list of unique Pre-Process plugin IDs to be used/recycled by all
         * PreprocBintr cto/dtor
         */
        static std::list<uint> s_uniqueIds;

    private:
    
        /**
         * @brief assigned unique id for the PreprocBintr
         */
        uint m_uniqueId;
        
        /**
         * @brief specifies whether the Pre-Process plugin is enabled
         * or set in passthrough mode.
         */
        boolean m_enabled;
        
        /**
         * @brief Absolute or relative path to the Pre-Process plugin config file.
         */
        std::string m_configFile;
        
        
        /**
         * @brief Queue element for the PreprocBintr.
         */
        DSL_ELEMENT_PTR m_pQueue;
        
        /**
         * @brief nvdspreprocess element for the PreprocBintr.
         */
        DSL_ELEMENT_PTR m_pPreproc;
    };
    
}

#endif // _DSL_PREPROC_BINTR_H