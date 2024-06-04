/*
The MIT License

Copyright (c) 2024, Prominence AI, Inc.

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

#ifndef _DSL_QBINTR_H
#define _DSL_QBINTR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslBintr.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_QBINTR_PTR std::shared_ptr<QBintr>
    #define DSL_QBINTR_NEW(name) \
        std::shared_ptr<Bintr>(new QBintr(name))    

    /**
     * @class QBintr
     * @brief Implements a base container class for a GST Bin with a 
     * read/write Queue.
     */
    class QBintr : public Bintr
    {
    public:

        /**
         * @brief Ctor for the QBintr class
         * @brief name unique name for the QBintr class 
         */
        QBintr(const char* name);


        /**
         * @brief Dtor for the QBintr class
         */
        ~QBintr();

        /**
         * @brief Gets the current level for the Queue element by unit.
         * @param[in] unit one of the DSL_COMPONENT_QUEUE_UNIT_OF constants.
         * @return current level for the specified unit.
         */
        uint64_t GetQueueCurrentLevel(uint unit);

        /**
         * @brief Gets the leaky property for the Queue element.
         * @return One of the DSL_COMPONENT_QUEUE_LEAKY constant values.
         */
        uint GetQueueLeaky();

        /**
         * @brief Sets the leaky property for the Queue element.
         * @param[in] leaky one of the DSL_COMPONENT_QUEUE_LEAKY constant values.
         * @return true if successfully set, false otherwise.
         */
        bool SetQueueLeaky(uint leaky);

        /**
         * @brief Gets the maximum size for the Queue element by unit.
         * @param[in] unit one of the DSL_COMPONENT_QUEUE_UNIT_OF constants.
         * @return current maximum size for the specified unit.
         */
        uint64_t GetQueueMaxSize(uint unit);

        /**
         * @brief Sets the maximum size for the Queue element by unit.
         * @param[in] unit one of the DSL_COMPONENT_QUEUE_UNIT_OF constants.
         * @param[in] maxSize new maximum size for the queue for the specified unit.
         * @return true if successfully set, false otherwise.
         */
        bool SetQueueMaxSize(uint unit, uint64_t maxSize);

        /**
         * @brief Gets the minimum threshold for the Queue element by unit.
         * @param[in] unit one of the DSL_COMPONENT_QUEUE_UNIT_OF constants.
         * @return current minimum threshold for the specified unit.
         */
        uint64_t GetQueueMinThreshold(uint unit);

        /**
         * @brief Sets the minimum threshold for the Queue element by unit.
         * @brief unit[in] one of the DSL_COMPONENT_QUEUE_UNIT_OF constants.
         * @brief minThreshold[in] new minimum threshold for the queue for the 
         * specified unit.
         * @return true if successfully set, false otherwise.
         */
        bool SetQueueMinThreshold(uint unit, uint64_t minThreshold);

        /**
         * @brief Adds a callback to be notified on queue overrun
         * @param[in] listener pointer to the client's function to call on overrun
         * @param[in] clientData opaque pointer to client data passed into the listner function.
         * @return true on successful listener add, false otherwise.
         */
        bool AddQueueOverrunListener(
                dsl_component_queue_overrun_listener_cb listener, void* clientData);
  
        /**
         * @brief removes a previously added queue overrun callback
         * @param[in] listener pointer to the client's function to remove
         * @return true on successful listener remove, false otherwise.
         */
        bool RemoveQueueOverrunListener(
                dsl_component_queue_overrun_listener_cb listener);
            
        /**
         * @brief Adds a callback to be notified on queue underrun
         * @param[in] listener pointer to the client's function to call on underrun
         * @param[in] clientData opaque pointer to client data passed into the listner function.
         * @return true on successful listener add, false otherwise.
         */
        bool AddQueueUnderrunListener(
                dsl_component_queue_underrun_listener_cb listener, void* clientData);
  
        /**
         * @brief removes a previously added queue underrun callback
         * @param[in] listener pointer to the client's function to remove
         * @return true on successful listener remove, false otherwise.
         */
        bool RemoveQueueUnderrunListener(
                dsl_component_queue_underrun_listener_cb listener);
            
    protected:

        /**
         * @brief Current leaky value for the Primary Queue.
         */
        uint m_leaky;

        /**
         * @brief Maximimum number of buffers in the queue (0=disable).
         */
        uint m_maxSizeBuffers;

        /**
         * @brief Maximum amount of data in the queue (bytes, 0=disable).
         */
        uint m_maxSizeBytes;

        /**
         * @brief Maximum amount of data in the queue (in ns, 0=disable).
         */
        uint64_t m_maxSizeTime;

        /**
         * @brief Minimum number of buffers in the queue to allow reading (0=disable).
         */
        uint m_minThresholdBuffers;

        /**
         * @brief Minimum amount of data in the queue to allow reading 
         * (in bytes, 0=disable).
         */
        uint m_minThresholdBytes;
 
        /**
         * @brief Minimum amount of data in the queue to allow reading 
         * (in ns, 0=disable).
         */
        uint m_minThresholdTime;

        /**
         * @brief map of all currently registered queue-overrun-listener
         * callback functions mapped with the user provided data
         */
        std::map<dsl_component_queue_overrun_listener_cb, void*> 
            m_queueOverrunListeners;

        /**
         * @brief map of all currently registered queue-underrun-listener
         * callback functions mapped with the user provided data
         */
        std::map<dsl_component_queue_underrun_listener_cb, void*> 
            m_queueUnderrunListeners;

        /**
         * @brief Primary Queue Elementr for this QBintr
         */
        DSL_ELEMENT_PTR  m_pQueue;

    };

} // DSL

#endif // _DSL_QBINTR_H
