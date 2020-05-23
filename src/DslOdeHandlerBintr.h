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

#ifndef _DSL_ODE_HANDLER_BINTR_H
#define _DSL_ODE_HANDLER_BINTR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslElementr.h"
#include "DslBintr.h"
#include "DslOdeType.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_ODE_HANDLER_PTR std::shared_ptr<OdeHandlerBintr>
    #define DSL_ODE_HANDLER_NEW(name) \
        std::shared_ptr<OdeHandlerBintr>(new OdeHandlerBintr(name))
        
    class OdeHandlerBintr : public Bintr
    {
    public: 
    
        OdeHandlerBintr(const char* name);

        ~OdeHandlerBintr();

        /**
         * @brief Adds the OdeHandlerBintr to a Parent Pipeline Bintr
         * @param[in] pParentBintr Parent Pipeline to add this Bintr to
         */
        bool AddToParent(DSL_BASE_PTR pParentBintr);

        /**
         * @brief Links all Child Elementrs owned by this Bintr
         * @return true if all links were succesful, false otherwise
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elemntrs owned by this Bintr
         * Calling UnlinkAll when in an unlinked state has no effect.
         */
        void UnlinkAll();
        
        /**
         * @brief Adds a uniquely named ODE Type to this OdeHandlerBintr
         * @param[in] pChild shared pointer to detection event to add
         * @return true if successful add, false otherwise
         */
        bool AddChild(DSL_BASE_PTR pChild);
        
        /**
         * @brief Removes a uniquely named Event from this RepoterBintr
         * @param[in] name unique name of the Event to remove
         * @return true if successful remove, false otherwise
         */
        bool RemoveChild(DSL_BASE_PTR pChild);

        /**
         * @brief Gets the current state of the Handler enabled flag
         * @return true if Handler is current enabled, false otherwise
         */
        bool GetEnabled();

        /**
         * @brief Sets the current state of the Handler enabled flag. 
         * The default state on creation is True
         * @param[in] enabled set to true if Repororting is to be enabled, false otherwise
         */
        bool SetEnabled(bool enabled);
        
        /**
         * @brief Handles a Pad buffer, by iterating through each child ODE Type
         * checking for an occurrence of such an event
         * @param pBuffer Pad buffer
         * @return true to continue handling, false to stop and self remove callback
         */
        bool HandlePadBuffer(GstBuffer* pBuffer);

    private:
    
        /**
         * @brief Handler enabled setting, default = true (enabled), 
         */ 
        bool m_isEnabled;

        /**
         * @brief Queue Elementr as both Sink and Source for this OdeHandlerBintr
         */
        DSL_ELEMENT_PTR m_pQueue;

    };
    
    static boolean PadBufferHandler(void* pBuffer, void* user_data);    
}

#endif // _DSL_ODE_HANDLER_BINTR_H
