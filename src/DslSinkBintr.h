
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
    class SinksBintr : public Bintr
    {
    public: 
    
        SinksBintr(const char* sink);

        ~SinksBintr();
        
        void AddChild(std::shared_ptr<Bintr> pChildBintr);

//        std::shared_ptr<StaticPadtr> m_pStaticSinkPadtr;        

    private:

        std::shared_ptr<Elementr> m_pQueue;
        std::shared_ptr<Elementr> m_pTee;
    };

    class SinkBintr : public Bintr
    {
    public: 
    
        SinkBintr(const char* sink)
            : Bintr(sink) 
        {
            LOG_FUNC();
        };

        ~SinkBintr()
        {
            LOG_FUNC();
        };
  
        std::shared_ptr<StaticPadtr> m_pStaticSinkPadtr;        
    };

    class OverlaySinkBintr : public SinkBintr
    {
    public: 
    
        OverlaySinkBintr(const char* sink, guint offsetX, guint offsetY, guint width, guint height);

        ~OverlaySinkBintr();
  
        std::shared_ptr<StaticPadtr> m_pStaticSinkPadtr;        
      
        void LinkAll();
        
        void UnlinkAll();

        void AddToParent(std::shared_ptr<Bintr> pParentBintr);

        int GetDisplayId()
        {
            LOG_FUNC();
            
            return m_displayId;
        }

        int GetOverlayId()
        {
            LOG_FUNC();
            
            return m_overlayId;
        }

        void SetDisplayId(int id)
        {
            LOG_FUNC();
            
            m_displayId = id;
        }

        void SetOverlayId(int id)
        {
            LOG_FUNC();
            
            m_overlayId = id;
        }

        
    private:

        gboolean m_sync;
        gboolean m_async;
        gboolean m_qos;
        guint m_displayId;
        guint m_overlayId;
        guint m_offsetX;
        guint m_offsetY;
        guint m_width;
        guint m_height;

        std::shared_ptr<Elementr> m_pQueue;
        std::shared_ptr<Elementr> m_pTransform;
        std::shared_ptr<Elementr> m_pOverlay;
    };

}

#endif // _DSL_SINK_BINTR_H
    