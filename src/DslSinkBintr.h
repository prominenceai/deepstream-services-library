
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

#ifndef _DSL_SINK_BINTR_H
#define _DSL_SINK_BINTR_H

#include "Dsl.h"
#include "DslBintr.h"
#include "DslElementr.h"

namespace DSL
{
    #define DSL_SINK_PTR std::shared_ptr<SinkBintr>

    #define DSL_OVERLAY_SINK_PTR std::shared_ptr<OverlaySinkBintr>
    #define DSL_OVERLAY_SINK_NEW(name, offsetX, offsetY, width, height) \
        std::shared_ptr<OverlaySinkBintr>( \
        new OverlaySinkBintr(name, offsetX, offsetY, width, height))

    #define DSL_WINDOW_SINK_PTR std::shared_ptr<WindowSinkBintr>
    #define DSL_WINDOW_SINK_NEW(name, offsetX, offsetY, width, height) \
        std::shared_ptr<WindowSinkBintr>( \
        new WindowSinkBintr(name, offsetX, offsetY, width, height))

    class SinkBintr : public Bintr
    {
    public: 
    
        SinkBintr(const char* sink);

        ~SinkBintr();
  
        bool AddToParent(DSL_NODETR_PTR pParentBintr);

        bool IsParent(DSL_NODETR_PTR pParentBintr);
        
        bool RemoveFromParent(DSL_NODETR_PTR pParentBintr);
        
        bool IsWindowCapable();
        
        bool LinkToSource(DSL_NODETR_PTR pTee);

        bool UnlinkFromSource();

        /**
         * @brief returns the current, sink Id as managed by the Parent pipeline
         * @return -1 when source Id is not assigned, i.e. source is not currently in use
         */
        int GetSinkId();
        
        /**
         * @brief Sets the unique id for this Sink bintr
         * @param id value to assign [0...MAX]
         */
        void SetSinkId(int id);
        
    protected:

        /**
         * @brief Queue element as sink for all Sink Bintrs.
         */
        DSL_ELEMENT_PTR m_pQueue;

        /**
         * @brief true if the Sink is capable of Windowed Video rendering, false otherwise
         */
        bool m_isWindowCapable;
        
        /**
         * @brief unique stream source identifier managed by the 
         * parent pipeline from Source add until removed
         */
        int m_sinkId;
        
        
    };

    class OverlaySinkBintr : public SinkBintr
    {
    public: 
    
        OverlaySinkBintr(const char* sink, guint offsetX, guint offsetY, guint width, guint height);

        ~OverlaySinkBintr();
  
        bool LinkAll();
        
        void UnlinkAll();

        int GetDisplayId();

        void SetDisplayId(int id);

        /**
         * @brief Gets the current X and Y offset settings for this OverlaySinkBintr
         * @param[out] offsetX the current offset in the X direction in pixels
         * @param[out] offsetY the current offset in the Y direction setting in pixels
         */ 
        void GetOffsets(uint* offsetX, uint* offsetY);

        /**
         * @brief Sets the current X and Y offset settings for this OverlaySinkBintr
         * The caller is required to provide valid width and height values
         * @param[in] offsetX the offset in the X direct to set in pixels
         * @param[in] offsetY the offset in the Y direct to set in pixels
         * @return false if the OverlaySink is currently in Use. True otherwise
         */ 
        bool SetOffsets(uint offsetX, uint offsetY);
        
        /**
         * @brief Gets the current width and height settings for this OverlaySinkBintr
         * @param[out] width the current width setting in pixels
         * @param[out] height the current height setting in pixels
         */ 
        void GetDimensions(uint* width, uint* height);
        
        /**
         * @brief Sets the current width and height settings for this OverlaySinkBintr
         * The caller is required to provide valid width and height values
         * @param[in] width the width value to set in pixels
         * @param[in] height the height value to set in pixels
         * @return false if the OverlaySink is currently in Use. True otherwise
         */ 
        bool SetDimensions(uint width, uint hieght);
        
    private:

        boolean m_sync;
        boolean m_async;
        boolean m_qos;
        uint m_overlayId;
        uint m_displayId;
        uint m_uniqueId;
        uint m_offsetX;
        uint m_offsetY;
        uint m_width;
        uint m_height;
        uint m_depth;

        DSL_ELEMENT_PTR m_pOverlay;
        
    };

    class WindowSinkBintr : public SinkBintr
    {
    public: 
    
        WindowSinkBintr(const char* sink, guint offsetX, guint offsetY, guint width, guint height);

        ~WindowSinkBintr();
  
        bool LinkAll();
        
        void UnlinkAll();

        /**
         * @brief Gets the current X and Y offset settings for this WindowSinkBintr
         * @param[out] offsetX the current offset in the X direction in pixels
         * @param[out] offsetY the current offset in the Y direction setting in pixels
         */ 
        void GetOffsets(uint* offsetX, uint* offsetY);

        /**
         * @brief Sets the current X and Y offset settings for this WindowSinkBintr
         * The caller is required to provide valid width and height values
         * @param[in] offsetX the offset in the X direct to set in pixels
         * @param[in] offsetY the offset in the Y direct to set in pixels
         * @return false if the OverlaySink is currently in Use. True otherwise
         */ 
        bool SetOffsets(uint offsetX, uint offsetY);
        
        /**
         * @brief Gets the current width and height settings for this WindowSinkBintr
         * @param[out] width the current width setting in pixels
         * @param[out] height the current height setting in pixels
         */ 
        void GetDimensions(uint* width, uint* height);
        
        /**
         * @brief Sets the current width and height settings for this WindowSinkBintr
         * The caller is required to provide valid width and height values
         * @param[in] width the width value to set in pixels
         * @param[in] height the height value to set in pixels
         * @return false if the OverlaySink is currently in Use. True otherwise
         */ 
        bool SetDimensions(uint width, uint hieght);

        Window GetXWindowHandle();
        /**
         * @brief Sets the XWindow handle for this WindowSinkBintr
         * @param[in] pXWindow handle to set
         */
        void SetXWindowHandle(Window pXWindow);
        
    private:

        boolean m_sync;
        boolean m_async;
        boolean m_qos;
        uint m_offsetX;
        uint m_offsetY;
        uint m_width;
        uint m_height;
        Window m_pXWindow;

        DSL_ELEMENT_PTR m_pTransform;
        DSL_ELEMENT_PTR m_pEglGles;
    };
}

#endif // _DSL_SINK_BINTR_H
    