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

#ifndef _DSL_TILER_BINTR_H
#define _DSL_TILER_BINTR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslElementr.h"
#include "DslBintr.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_TILER_PTR std::shared_ptr<TilerBintr>
    #define DSL_TILER_NEW(name, width, height) \
        std::shared_ptr<TilerBintr>(new TilerBintr(name, width, height))
        
    class TilerBintr : public Bintr
    {
    public: 
    
        TilerBintr(const char* name, uint width, uint height);

        ~TilerBintr();

        /**
         * @brief Adds the TilerBintr to a Parent Pipeline Bintr
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
         * @brief Gets the current number of rows and columns for the TilerBintr
         * @param[out] rows the current number of rows
         * @param[out] columns the current number of columns
         */
        void GetTiles(uint* columns, uint* rows);
        
        /**
         * @brief Sets the number of rows and columns for the TilerBintr
         * The caller is required to provide valid row and column values
         * @param[in] rows the number of rows to set
         * @param[in] columns the number of columns to set
         * @return false if the TilerBintr is currently in Use. True otherwise
         */
        bool SetTiles(uint columns, uint rows);

        /**
         * @brief Gets the current width and height settings for this TilerBintr
         * @param[out] width the current width setting in pixels
         * @param[out] height the current height setting in pixels
         */ 
        void GetDimensions(uint* width, uint* height);
        
        /**
         * @brief Sets the current width and height settings for this TilerBintr
         * The caller is required to provide valid width and height values
         * @param[in] width the width value to set in pixels
         * @param[in] height the height value to set in pixels
         * @return false if the TilerBintr is currently in Use. True otherwise
         */ 
        bool SetDimensions(uint width, uint hieght);
        
        /**
         * @brief Get the current enabled state for Frame-Number Adder.
         * @return current enabled state.
         */
        bool GetFrameNumberingEnabled();

        /**
         * @brief Set the enabled state for the Frame-Number Adder.
         * @param enabled set to true to enabled, false to disable.
         * @return true if the state was updated correctly.
         */
        bool SetFrameNumberingEnabled(bool enabled);
        
        /**
         * @brief Gets the current show-source setting for the TilerBintr
         * @return the current show-source setting, -1 equals all sources/tiles shown
         */
        void GetShowSource(int* sourceId, uint* timeout);
        
        /**
         * @brief Sets the current show-source setting for the TilerBintr to a single source
         * @param[in] the new show-source setting to use
         * Note: sourceId must be less than current batch-size, which is 0 until the Pipeline is linked/played
         * @param[in] timeout the time in seconds to show the current source
         * @param[in] hasPrecedence if true will take precedence over a currently showing single source
         * @return true if set value is successful, false otherwise.
         */
        bool SetShowSource(int sourceId, uint timeout, bool hasPrecedence);
        
        /**
         * @brief Handler routine for show-source timer experation
         */
        int HandleShowSourceTimer();
        
        /**
         * @brief Sets the show source setting to -1 showing all sources
         * The show-source timeout, if running, will be canceled on call 
         */
        void ShowAllSources();
        
        /**
         * @brief Cycles through all sources showing each until timeout.
         * @param[in] timeout the time in seconds to show the current source
         */
        bool CycleAllSources(uint timeout);
        
        /**
         * @brief Sets the GPU ID for all Elementrs
         * @return true if successfully set, false otherwise.
         */
        bool SetGpuId(uint gpuId);

        /**
         * @brief Sets the NVIDIA buffer memory type.
         * @brief nvbufMemType new memory type to use, one of the 
         * DSL_NVBUF_MEM_TYPE constant values.
         * @return true if successfully set, false otherwise.
         */
        bool SetNvbufMemType(uint nvbufMemType);
        
    private:
    
        /**
         * @brief number of rows for the TilerBintr
         */
        uint m_rows; 
        
        /**
         * @brief number of rows for the TilerBintr
         */
        uint m_columns;
        
        /**
         * @brief width of the TilerBintr in pixels
         */
        uint m_width; 
        
        /**
         * @brief height of the TilerBintr in pixels
         */
        uint m_height;
        
        /**
         * @brief true if the Frame-Number Adder has been enabled
         * to add frame_numbers to each frame crossing the Source
         * pad of the TilerBintr
         */
        bool m_frameNumberingEnabled;
        
        /**
         * @brief Frame-Number Adder Pad Probe Handler
         * to add frame_numbers to each frame crossing the Source
         * pad of the TilerBintr - when endabled
         */
        DSL_PPEH_FRAME_NUMBER_ADDER_PTR m_pFrameNumberAdder;
        
        /**
         * @brief Queue Elementr as Sink for this TilerBintr
         */
        DSL_ELEMENT_PTR m_pQueue;
 
        /**
         * @brief Tiler Elementr as Source for this TilerBintr
         */
        DSL_ELEMENT_PTR  m_pTiler;
        
        /**
         * @brief mutex to prevent callback reentry
         */
        GMutex m_showSourceMutex;
        
        /**
         * @brief current show-source id, -1 == show-a;-sources
         */
        int m_showSourceId;
        
        /**
         * @brief client provided timeout in seconds
         */
        uint m_showSourceTimeout;
        
        /**
         * @brief show-source count down counter value. 0 == time expired
         */
        uint m_showSourceCounter;

        /**
         * @brief show-source timer-id, non-zero == currently running
         */
        uint m_showSourceTimerId;
        
        /**
         * @brief true if source cycling is enabled, false otherwise
         */
        bool m_showSourceCycle;
    };

    //----------------------------------------------------------------------------------------------

    static int ShowSourceTimerHandler(void* user_data);
}

#endif // _DSL_TILER_BINTR_H
