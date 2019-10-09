
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

#ifndef _DSL_SOURCE_BINTR_H
#define _DSL_SOURCE_BINTR_H

#include "Dsl.h"
#include "DslBintr.h"

namespace DSL
{
    class SourcesBintr : public Bintr
    {
    public: 
    
        SourcesBintr(const char* name);

        ~SourcesBintr();
        
        void AddChild(std::shared_ptr<Bintr> pChildBintr);
        
        void AddSourceGhostPad();
        
        void SetStreamMuxProperties(gboolean areSourcesLive, guint batchSize, guint batchTimeout, 
            guint width, guint height);

    private:

        GstElement* m_pStreamMux;
        
        /**
         @brief
         */
        gboolean m_areSourcesLive;

        /**
         @brief
         */
        gint m_batchSize;

        /**
         @brief
         */
        gint m_batchTimeout;
        /**
         @brief
         */
        gint m_streamMuxWidth;

        /**
         @brief
         */
        gint m_streamMuxHeight;

        /**
         @brief
         */
        gboolean m_enablePadding;
    };

    /**
     * @class SourceBintr
     * @brief Implements a Source Bintr for all derived Source types.
     * CSI, V4L2, URI, and RTSP
     */
    class SourceBintr : public Bintr
    {
    public: 
    
        SourceBintr(const char* source);

        SourceBintr(const char* source, guint width, guint height, 
            guint fps_n, guint fps_d);

        ~SourceBintr();
        
        /**
         * @brief unique stream source identifier managed by the 
         * parent pipeline from Source add until removed
         */
        guint m_sourceId;

    public:
            
        /**
         * @brief
         */
        gboolean m_isLive;

        /**
         * @brief
         */
        guint m_width;

        /**
         * @brief
         */
        guint m_height;

        /**
         * @brief
         */
        guint m_fps_n;

        /**
         * @brief
         */
        guint m_fps_d;

        /**
         * @brief
         */
        guint m_latency;

        /**
         * @brief
         */
        guint m_numDecodeSurfaces;

        /**
         * @brief
         */
        guint m_numExtraSurfaces;

        /**
         * @brief
         */
        GstElement * m_pSourceElement;
    };

    /**
     * @class CsiSourceBintr
     * @brief 
     */
    class CsiSourceBintr : public SourceBintr
    {
    public: 
    
        CsiSourceBintr(const char* source, guint width, guint height, 
            guint fps_n, guint fps_d);

        ~CsiSourceBintr();
        
        void AddToParent(std::shared_ptr<Bintr> pParentBintr);

    private:
        /**
         * @brief
         */
        GstElement * m_pCapsFilter;
    };


    /**
     * @class UriSourceBintr
     * @brief 
     */
    class UriSourceBintr : public SourceBintr
    {
    public: 
    
        UriSourceBintr(const char* source, const char* uri, 
            guint cudadecMemType, guint intraDecode);

        ~UriSourceBintr();
        
        /**
         * @brief 
         * @param pParentBintr
         */
        void AddToParent(std::shared_ptr<Bintr> pParentBintr);
        
        /**
         * @brief 
         * @param pBin
         * @param pPad
         */
        void HandleOnPadAdded(GstElement* pBin, GstPad* pPad);
        
        /**
         * @brief 
         * @param pChildProxy
         * @param pObject
         * @param name
         */
        void HandleOnChildAdded(GstChildProxy* pChildProxy, 
            GObject* pObject, gchar* name);
        
        /**
         * @brief 
         * @param pObject
         * @param arg0
         */
        void HandleOnSourceSetup(GstElement* pObject, GstElement* arg0);

        /**
         * @brief 
         * @param pPad
         * @param pInfo
         * @return 
         */
        GstPadProbeReturn HandleStreamBufferRestart(GstPad* pPad, GstPadProbeInfo* pInfo);
        
        /**
         * @brief 
         * @return 
         */
        gboolean HandleStreamBufferSeek();
        
    private:

        /**
         * @brief
         */
        std::string m_uriString; 
        
        /**
         * @brief
         */
        guint m_cudadecMemtype;
        
        /**
         * @brief
         */
        guint m_intraDecode;
        
        /**
         * @brief
         */
        guint m_dropFrameInterval;
        
        /**
         * @brief
         */
        guint m_accumulatedBase;

        /**
         * @brief
         */
        guint m_prevAccumulatedBase;
        
        /**
         * @brief
         */
        guint m_bufferProbeId;

        /**
         * @brief
         */
        GstElement* m_pTee;

        /**
         @brief
         */
        GstElement* m_pSourceQueue;

        /**
         * @brief
         */
        GstElement* m_pFakeSink;

        /**
         * @brief
         */
        GstElement* m_pFakeSinkQueue;
    };

    /**
     * @brief 
     * @param[in] pBin
     * @param[in] pPad
     * @param[in] pSource (callback user data) pointer to the unique URI source opject
     */
    static void OnPadAddedCB(GstElement* pBin, GstPad* pPad, gpointer pSource);

    /**
     * @brief 
     * @param[in] pChildProxy
     * @param[in] pObject
     * @param[in] name
     * @param[in] pSource (callback user data) pointer to the unique URI source opject
     */
    static void OnChildAddedCB(GstChildProxy* pChildProxy, GObject* pObject,
        gchar* name, gpointer pSource);

    /**
     * @brief 
     * @param[in] pObject
     * @param[in] arg0
     * @param[in] pSource
     */
    static void OnSourceSetupCB(GstElement* pObject, GstElement* arg0, gpointer pSource);

    /**
     * Probe function to drop certain events to support custom
     * logic of looping of each source stream.
     */

    /**
     * @brief Probe function to drop certain events to support
     * custom logic of looping of each URI source (file) stream.
     * @param pPad
     * @param pInfo
     * @param pSource
     * @return 
     */
    static GstPadProbeReturn StreamBufferRestartProbCB(GstPad* pPad, 
        GstPadProbeInfo* pInfo, gpointer pSource);

    /**
     * @brief 
     * @param pSource
     * @return 
     */
    static gboolean StreamBufferSeekCB(gpointer pSource);
}

#endif // _DSL_SOURCE_BINTR_H
