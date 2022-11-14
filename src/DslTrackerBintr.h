/*
The MIT License

Copyright (c) 2019-2021, Prominence AI, Inc.

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

#ifndef _DSL_TRACKER_BINTR_H
#define _DSL_TRACKER_BINTR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslElementr.h"
#include "DslBintr.h"
#include "DslPadProbeHandler.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_TRACKER_PTR std::shared_ptr<TrackerBintr>
    #define DSL_TRACKER_NEW(name, configFile, width, height) \
        std::shared_ptr<TrackerBintr>(new TrackerBintr(name, configFile, width, height))

    class TrackerBintr : public Bintr
    {
    public: 
    
        TrackerBintr(const char* name,
            const char* configFile, guint width, guint height);

        ~TrackerBintr();

        /**
         * @brief gets the name of the Tracker Config File in use by this GieBintr
         * @return fully qualified patspec used to create this Bintr
         */
        const char* GetLlConfigFile();
        
        /**
         * @brief gets the name of the Tracker Lib File in use by this PrimaryGieBintr
         * @return fully qualified patspec used to create this Bintr
         */
        const char* GetLlLibFile();

        /**
         * @brief Adds the TrackerBintr to a Parent Branch Bintr
         * @param[in] pParentBintr Parent Pipeline to add this Bintr to
         */
        bool AddToParent(DSL_BASE_PTR pParentBintr);

        /**
         * @brief Removes this TrackerBintr from its Parent Branch Bintr
         * @param[in] pParentBintr parent Pipeline to remove from
         * @return true on successful add, false otherwise
         */
        bool RemoveFromParent(DSL_BASE_PTR pParentBintr);

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
         * @brief Gets the path to the low-level library in use by this TrackerBintr.
         * @return absolute path to the library file in use.
         */
        const char* GetLibFile();

        /**
         * @brief Sets the low-level library file to use for this TrackerBintr
         * @param[in] libFile fully absolute or relative path to the Lib file to use
         * @return true on successful update, false otherwise
         */
        bool SetLibFile(const char* libFile);

        /**
         * @brief Gets the path to the the config file in use by this TrackerBintr.
         * @return absolute path to the config file in use
         */
        const char* GetConfigFile();
        
        /**
         * @brief sets the config file to use for this TrackerBintr
         * @param[in] configFile fully absolute or relative path to the Lib file to use
         * @return true on successful update, false otherwise
         */
        bool SetConfigFile(const char* configFile);
        
        /**
         * @brief Gets the current width and height settings for this Tracker
         * @param[out] width the current width setting in pixels
         * @param[out] height the current height setting in pixels
         */ 
        void GetDimensions(uint* width, uint* height);
        
        /**
         * @brief Sets the current width and height settings for this Tracker
         * The caller is required to provide valid width and height values
         * @param[in] width the width value to set in pixels
         * @param[in] height the height value to set in pixels
         * @return false if the Tracker is currently in Use. True otherwise
         */ 
        bool SetDimensions(uint width, uint hieght);

        /**
         * @brief Gets the current batch-processing-enabled setting for this Tracker
         * @return True if enabled, false otherwise
         */
        bool GetBatchProcessingEnabled();
        
        /**
         * @brief Sets the enable-batch-processing setting for this Tracker
         * @return Set to true to enable, false otherwise. 
         * Note: This call is only effective if the low-level library supports 
         * both batch and per-stream processing.
         */
        bool SetBatchProcessingEnabled(bool enabled);
        
        /**
         * @brief Gets the enable-past-frame setting for this Tracker
         * @return True if enabled, false otherwise
         */
        bool GetPastFrameReportingEnabled();

        /**
         * @brief Sets the enable-past-frame setting for this Tracker
         * @return Set to true if enable, false otherwise
         * Note: This call is only effective if the low-level library supports 
         * past frame reporting.
         */
        bool SetPastFrameReportingEnabled(bool enabled);
        
        /**
         * @brief This Bintr uses the common SetBatchSize bintr method to check
         * if batch-processing is disabled and batchSize for the Pipeline > 1
         * The function logs a WARN message if this case is found to be true.
         * @param the pipeline batchSize to check
         * @return true always.
         */
        bool SetBatchSize(uint batchSize);

        /**
         * @brief Sets the GPU ID for all Elementrs
         * @return true if successfully set, false otherwise.
         */
        bool SetGpuId(uint gpuId);
        
    private:

        /**
         * @brief pathspec to the tracker config file used by this TrackerBintr
         */
        std::string m_llConfigFile;
        
        /**
         * @brief pathspec to the tracker lib file used by this TrackerBintr
         */
        std::string m_llLibFile;
    
        /**
         * @brief max frame width of the input buffer in pixels
         */
        uint m_width; 
        
        /**
         * @brief max frame height of the input buffer in pixels
         */
        uint m_height;
        
        /**
         * @brief Tracker Elementr for this TrackerBintr
         */
        DSL_ELEMENT_PTR  m_pTracker;

        /**
         * @brief true if the enable-batch-processing setting is set, false otherwise.
         */
        bool m_batchProcessingEnabled;
        
        /**
         * @brief true if the enable-past-frame setting is set, false otherwise.
         */
        bool m_pastFrameReporting;
        

    };

} // DSL

#endif // _DSL_TRACKER_BINTR_H
