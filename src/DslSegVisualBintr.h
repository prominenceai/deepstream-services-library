/*
The MIT License

Copyright (c) 2021, Prominence AI, Inc.

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

#ifndef _DSL_SEG_VISUAL_ELEMENTR_H
#define _DSL_SEG_VISUAL_ELEMENTR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslElementr.h"
#include "DslBintr.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
        
    #define DSL_SEG_VISUAL_ELEMENT_PTR std::shared_ptr<SegVisualElementr>
    #define DSL_SEG_VISUAL_ELEMENT_NEW(name, width, height) \
        std::shared_ptr<SegVisualElementr>(new SegVisualElementr(name, width, height))
        
    class SegVisualElementr : public Elementr
    {
    public: 
    
        SegVisualElementr(const char* name, uint width, uint height);

        ~SegVisualElementr();
        
        /**
         * @brief Gets the current output width and height settings for this SegVisualElementr
         * @param[out] width the current width setting in pixels
         * @param[out] height the current height setting in pixels
         */ 
        void GetDimensions(uint* width, uint* height);
        
        /**
         * @brief Sets the current output width and height settings for this SegVisualElementr
         * The caller is required to provide valid width and height values
         * @param[in] width the width value to set in pixels
         * @param[in] height the height value to set in pixels
         * @return True on successful update, false otherwise
         */ 
        bool SetDimensions(uint width, uint hieght);

        /**
         * @brief gets the current batchSize in use by this SigVisualElementr
         * @return the current batchSize
         */
        uint GetBatchSize();

        /**
         * @brief sets the batch size for this SegVisualElementr
         * @param the new batchSize to use
         */
        bool SetBatchSize(uint batchSize);

        /**
         * @brief Gets the current GPU ID used by this SegVisualElementr
         * @return the ID for the current GPU in use.
         */
        uint GetGpuId();

        /**
         * @brief Sets the GPU ID for this SegVisualElementr
         * @param[in] gpuId new GPU ID setting.
         * @return true if successfully set, false otherwise.
         */
        bool SetGpuId(uint gpuId);

    protected:

        /**
         * @brief output frame width of the input buffer in pixels
         */
        uint m_width; 
        
        /**
         * @brief output frame height of the input buffer in pixels
         */
        uint m_height;
        
        /**
         * @brief current batch size for this Bintr
         */
        uint m_batchSize;
        
        
        /**
         * @brief current GPU Id used by this SegVisualElementr
         */
        guint m_gpuId;
        
    };

} // DSL

#endif // _DSL_SEG_VISUAL_ELEMENTR_H
        