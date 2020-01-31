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

#ifndef _DSL_PAD_PROBETR_H
#define _DSL_PAD_PROBETR_H

#include "Dsl.h"
#include "DslElementr.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_PAD_PROBE_PTR std::shared_ptr<PadProbetr>
    #define DSL_PAD_PROBE_NEW(name, factoryName, parentElement) \
        std::shared_ptr<PadProbetr>(new PadProbetr(name, factoryName, parentElement))    

    /**
     * @class PadProbetr
     * @brief Implements a container class for GST Pad Probe
     */
    class PadProbetr : public std::enable_shared_from_this<PadProbetr>
    {
    public:
        
        /**
         * @brief ctor for the PadProbetr class
         * @param[in] name name for the new PadProbetr
         * @param[in] factoryNme "sink" or "src" Pad Probe type
         */
        PadProbetr(const char* name, const char* factoryName, DSL_ELEMENT_PTR parentElement);
        
        /**
         * @brief dtor for the PadProbetr base class
         */
        ~PadProbetr();

        /**
         * @brief Adds a Batch Meta Handler callback function to the PadProbetr
         * @param pClientBatchMetaHandler callback function pointer to add
         * @param pClientUserData user data to return on callback
         * @return false if the PadProbetr has an existing Batch Meta Handler
         */
        bool AddBatchMetaHandler(dsl_batch_meta_handler_cb pClientBatchMetaHandler, 
            void* pClientUserData);
            
        /**
         * @brief Removes the current Batch Meta Handler callback function from the PadProbetr
         * @param pClientBatchMetaHandler callback function pointer to remove
         * @return false if the PadProbetr does not have a Meta Batch Handler to remove.
         */
        bool RemoveBatchMetaHandler(dsl_batch_meta_handler_cb pClientBatchMetaHandler);
        
        /**
         * @brief Queries the PadProbetr for a child Batch Meta Handler relationship
         * @param pClientBatchMetaHandler
         * @return true if the Batch Meta Handler is currently a child of PadProbetr
         */
        bool IsChild(dsl_batch_meta_handler_cb pClientBatchMetaHandler);
        
        /**
         * @brief 
         * @param pPad
         * @param pInfo
         * @return 
         */
        GstPadProbeReturn HandlePadProbe(
            GstPad* pPad, GstPadProbeInfo* pInfo);

        /**
         * @brief Enables/disables bounding-box date output in Kitti format
         * @param enabled true if date should be written to file, false to disable
         * @param path relative or absolute dir path specification
         * @return true if success, false otherwise.
         */
        bool SetKittiOutputEnabled(bool enabled, const char* path);

    private:
    
        /**
         * @brief unique name for this PadProbetr
         */
        std::string m_name;

        /**
         * @brief mutex fo the Pad Probe handler
         */
        GMutex m_padProbeMutex;
        
        /**
         * @brief sink/src pad probe handle
         */
        uint m_padProbeId;

        /**
         * @brief map of Client Pad Batch Meta handlers
         */
        std::map<dsl_batch_meta_handler_cb, void*> m_pClientBatchMetaHandlers;

        /**
         * true if kitti file output is currently enabled.
         */
        bool m_kittiOutputEnabled;
        
        /**
         * @brief absolute or relative pathspec to the Kitti output dir used by this PadProbetr
         */
        std::string m_kittiOutputPath;
    };
    
    static GstPadProbeReturn PadProbeCB(GstPad* pPad, 
        GstPadProbeInfo* pInfo, gpointer pPadProbetr);
    
} // DSL namespace    

#endif // _DSL_PAD_PROBETR_H    
