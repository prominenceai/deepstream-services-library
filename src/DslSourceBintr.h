
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
#include "DslElementr.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_SOURCE_PTR std::shared_ptr<SourceBintr>
    #define DSL_SOURCE_NEW(name) \
        std::shared_ptr<SourceBintr>(new SourceBintr(name))

    #define DSL_CSI_SOURCE_PTR std::shared_ptr<CsiSourceBintr>
    #define DSL_CSI_SOURCE_NEW(name, width, height, fps_n, fps_d) \
        std::shared_ptr<CsiSourceBintr>(new CsiSourceBintr(name, width, height, fps_n, fps_d))
        
    #define DSL_URI_SOURCE_PTR std::shared_ptr<UriSourceBintr>
    #define DSL_URI_SOURCE_NEW(name, uri, isLive, cudadecMemType, intraDecode, dropFrameInterval) \
        std::shared_ptr<UriSourceBintr>(new UriSourceBintr(name, uri, isLive, cudadecMemType, intraDecode, dropFrameInterval))
        
    #define DSL_RTSP_SOURCE_PTR std::shared_ptr<RtspSourceBintr>
    #define DSL_RTSP_SOURCE_NEW(name, uri, protocol, cudadecMemType, intraDecode, dropFrameInterval) \
        std::shared_ptr<RtspSourceBintr>(new RtspSourceBintr(name, uri, protocol, cudadecMemType, intraDecode, dropFrameInterval))

    /**
     * @class SourceBintr
     * @brief Implements a base Source Bintr for all derived Source types.
     * CSI, V4L2, URI, and RTSP
     */
    class SourceBintr : public Bintr
    {
    public: 
    
        SourceBintr(const char* name);

        ~SourceBintr();

        bool AddToParent(DSL_NODETR_PTR pParentBintr);

        bool IsParent(DSL_NODETR_PTR pParentBintr);
                        
        bool RemoveFromParent(DSL_NODETR_PTR pParentBintr);
        
        /**
         * @brief returns the current, sensor Id as managed by the Parent pipeline
         * @return -1 when source Id is not assigned, i.e. source is not currently in use
         * Unique source Id [0...MAX] when the source is in use.
         */
        int GetSourceId();
        
        /**
         * @brief Sets the unique sensor id for this Source bintr
         * @param id value to assign [0...MAX]
         */
        void SetSourceId(int id);
        
        /**
         * @brief returns the Live state of this Streaming Source
         * @return true if the Source is Live, false otherwise.
         */
        bool IsLive()
        {
            LOG_FUNC();
            
            return m_isLive;
        }
        
        /**
         * @brief Gets the current width and height settings for this SourceBintr
         * @param[out] width the current width setting in pixels
         * @param[out] height the current height setting in pixels
         */ 
        void GetDimensions(uint* width, uint* height);
        
        /**
         * @brief Gets the current FPS numerator and denominator settings for this SourceBintr
         * @param[out] fps_n the FPS numerator
         * @param[out] fps_d the FPS denominator
         */ 
        void GetFrameRate(uint* fps_n, uint* fps_d);
        
        /**
         * @brief Links the Streaming Source to a Stream Muxer
         * @param[in] pStreamMux
         */
        bool LinkToSink(DSL_NODETR_PTR pStreamMux);
        
        /**
         * @brief Unlinks this Streaming Source from a previously linked to Stream Muxer
         */
        bool UnlinkFromSink();

    public:
            
        /**
         * @brief unique stream source identifier managed by the 
         * parent pipeline from Source add until removed
         */
        int m_sourceId;
        
        /**
         * @brief True if the source is live and cannot be paused without losing data, False otherwise.
         */
        bool m_isLive;

        /**
         * @brief current width of the streaming source in Pixels.
         */
        uint m_width;

        /**
         * @brief current height of the streaming source in Pixels.
         */
        uint m_height;

        /**
         * @brief current frames-per-second numerator value for the Streaming Source
         */
        uint m_fps_n;

        /**
         * @brief current frames-per-second denominator value for the Streaming Source
         */
        uint m_fps_d;

        /**
         * @brief
         */
        guint m_latency;

        /**
         * @brief
         */
        uint m_numDecodeSurfaces;

        /**
         * @brief
         */
        uint m_numExtraSurfaces;

        /**
         * @brief Soure Element for this SourceBintr
         */
        DSL_ELEMENT_PTR m_pSourceElement;

    };

    //*********************************************************************************
    /**
     * @class CsiSourceBintr
     * @brief 
     */
    class CsiSourceBintr : public SourceBintr
    {
    public: 
    
        CsiSourceBintr(const char* name, uint width, uint height, 
            uint fps_n, uint fps_d);

        ~CsiSourceBintr();

        /**
         * @brief Links all Child Elementrs owned by this Source Bintr
         * @return True success, false otherwise
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elementrs owned by this Source Bintr
         */
        void UnlinkAll();
        
    private:
    
        uint m_sensorId;
        
        /**
         * @brief
         */
        DSL_ELEMENT_PTR m_pCapsFilter;
    };    

    //*********************************************************************************

    /**
     * @class DecodeSourceBintr
     * @brief 
     */
    class DecodeSourceBintr : public SourceBintr
    {
    public: 
    
        DecodeSourceBintr(const char* name, const char* factoryName, const char* uri, 
            bool isLive, uint cudadecMemType, uint intraDecode, uint dropFrameInterval);

        /**
         * @brief returns the current URI source for this DecodeSourceBintr
         * @return const string for either live or file source
         */
        const char* GetUri()
        {
            LOG_FUNC();
            
            return m_uri.c_str();
        }
        
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
        
    protected:

        /**
         * @brief
         */
        std::string m_uri; 
        
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
         * @brief A dynamic collection of requested Source Pads for the Tee 
         */
        std::map<std::string, GstPad*> m_pGstRequestedSourcePads;

        /**
         @brief
         */
        DSL_ELEMENT_PTR m_pSourceQueue;

        /**
         * @brief
         */
        DSL_ELEMENT_PTR m_pTee;

        /**
         * @brief
         */
        DSL_ELEMENT_PTR m_pFakeSink;

        /**
         * @brief
         */
        DSL_ELEMENT_PTR m_pFakeSinkQueue;

    };
    
    //*********************************************************************************

    /**
     * @class UriSourceBintr
     * @brief 
     */
    class UriSourceBintr : public DecodeSourceBintr
    {
    public: 
    
        UriSourceBintr(const char* name, const char* uri, bool isLive,
            uint cudadecMemType, uint intraDecode, uint dropFrameInterval);

        ~UriSourceBintr();

        /**
         * @brief Links all Child Elementrs owned by this Source Bintr
         * @return True success, false otherwise
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elementrs owned by this Source Bintr
         */
        void UnlinkAll();

        void HandleSourceElementOnPadAdded(GstElement* pBin, GstPad* pPad);

    private:


    };

    /**
     * @class RtspSourceBintr
     * @brief 
     */
    class RtspSourceBintr : public DecodeSourceBintr
    {
    public: 
    
        RtspSourceBintr(const char* name, const char* uri, uint protocol,
            uint cudadecMemType, uint intraDecode, uint dropFrameInterval);

        ~RtspSourceBintr();

        /**
         * @brief Links all Child Elementrs owned by this Source Bintr
         * @return True success, false otherwise
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elementrs owned by this Source Bintr
         */
        void UnlinkAll();

        void HandleSourceElementOnPadAdded(GstElement* pBin, GstPad* pPad);

        void HandleDecodeElementOnPadAdded(GstElement* pBin, GstPad* pPad);

    private:

        /**
         @brief 
         */
        uint m_rtpProtocols;
        
        /**
         @brief
         */
        DSL_ELEMENT_PTR m_pDepayload;
        
        /**
         @brief
         */
        DSL_ELEMENT_PTR m_pDecodeBin;
        
        /**
         @brief
         */
        DSL_ELEMENT_PTR m_pDecodeQueue;
    };

    /**
     * @brief 
     * @param[in] pBin
     * @param[in] pPad
     * @param[in] pSource (callback user data) pointer to the unique source opject
     */
    static void UriSourceElementOnPadAddedCB(GstElement* pBin, GstPad* pPad, gpointer pSource);

    /**
     * @brief 
     * @param pBin
     * @param pPad
     * @param pSource
     */
    static void RtspSourceElementOnPadAddedCB(GstElement* pBin, GstPad* pPad, gpointer pSource);
    
    /**
     * @brief 
     * @param pChildProxy
     * @param pObject
     * @param name
     * @param pSource
     */
    static void RtspDecodeElementOnPadAddedCB(GstElement* pBin, GstPad* pPad, gpointer pSource);

    /**
     * @brief 
     * @param[in] pChildProxy
     * @param[in] pObject
     * @param[in] name
     * @param[in] pSource (callback user data) pointer to the unique source opject
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
     * @brief Probe function to drop certain events to support
     * custom logic of looping of each decode source (file) stream.
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

} // DSL
#endif // _DSL_SOURCE_BINTR_H
