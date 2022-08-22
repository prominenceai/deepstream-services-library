
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

#ifndef _DSL_SOURCE_BINTR_H
#define _DSL_SOURCE_BINTR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslBintr.h"
#include "DslElementr.h"
#include "DslDewarperBintr.h"
#include "DslTapBintr.h"
#include "DslStateChange.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_SOURCE_PTR std::shared_ptr<SourceBintr>
    #define DSL_SOURCE_NEW(name) \
        std::shared_ptr<SourceBintr>(new SourceBintr(name))

    #define DSL_CSI_SOURCE_PTR std::shared_ptr<CsiSourceBintr>
    #define DSL_CSI_SOURCE_NEW(name, width, height, fpsN, fpsD) \
        std::shared_ptr<CsiSourceBintr>(new CsiSourceBintr(name, width, height, fpsN, fpsD))
        
    #define DSL_USB_SOURCE_PTR std::shared_ptr<UsbSourceBintr>
    #define DSL_USB_SOURCE_NEW(name, width, height, fpsN, fpsD) \
        std::shared_ptr<UsbSourceBintr>(new UsbSourceBintr(name, width, height, fpsN, fpsD))

    #define DSL_RESOURCE_SOURCE_PTR std::shared_ptr<ResourceSourceBintr>
        
    #define DSL_DECODE_SOURCE_PTR std::shared_ptr<DecodeSourceBintr>
        
    #define DSL_URI_SOURCE_PTR std::shared_ptr<UriSourceBintr>
    #define DSL_URI_SOURCE_NEW(name, uri, isLive, intraDecode, dropFrameInterval) \
        std::shared_ptr<UriSourceBintr>(new UriSourceBintr(name, \
            uri, isLive, intraDecode, dropFrameInterval))
        
    #define DSL_FILE_SOURCE_PTR std::shared_ptr<FileSourceBintr>
    #define DSL_FILE_SOURCE_NEW(name, uri, repeatEnabled) \
        std::shared_ptr<FileSourceBintr>(new FileSourceBintr(name, uri, repeatEnabled))

    #define DSL_IMAGE_SOURCE_PTR std::shared_ptr<ImageSourceBintr>

    #define DSL_SINGLE_IMAGE_SOURCE_PTR std::shared_ptr<SingleImageSourceBintr>
    #define DSL_SINGLE_IMAGE_SOURCE_NEW(name, uri) \
        std::shared_ptr<SingleImageSourceBintr>(new SingleImageSourceBintr(name, uri))

    #define DSL_MULTI_IMAGE_SOURCE_PTR std::shared_ptr<MultiImageSourceBintr>
    #define DSL_MULTI_IMAGE_SOURCE_NEW(name, uri, fpsN, fpsD) \
        std::shared_ptr<MultiImageSourceBintr>(new MultiImageSourceBintr(name, uri, fpsN, fpsD))

    #define DSL_IMAGE_STREAM_SOURCE_PTR std::shared_ptr<ImageStreamSourceBintr>
    #define DSL_IMAGE_STREAM_SOURCE_NEW(name, filePath, isLive, fpsN, fpsD, timeout) \
        std::shared_ptr<ImageStreamSourceBintr>(new ImageStreamSourceBintr(name, \
            filePath, isLive, fpsN, fpsD, timeout))

    #define DSL_INTERPIPE_SOURCE_PTR std::shared_ptr<InterpipeSourceBintr>
    #define DSL_INTERPIPE_SOURCE_NEW(name, listenTo, isLive, acceptEos, acceptEvents) \
        std::shared_ptr<InterpipeSourceBintr>(new InterpipeSourceBintr(name, \
            listenTo, isLive, acceptEos, acceptEvents))

    #define DSL_RTSP_SOURCE_PTR std::shared_ptr<RtspSourceBintr>
    #define DSL_RTSP_SOURCE_NEW(name, uri, protocol, \
        intraDecode, dropFrameInterval, latency, timeout) \
        std::shared_ptr<RtspSourceBintr>(new RtspSourceBintr(name, uri, protocol, \
            intraDecode, dropFrameInterval, latency, timeout))

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
        
        void UnlinkAll(){};

        bool AddToParent(DSL_BASE_PTR pParentBintr);

        bool IsParent(DSL_BASE_PTR pParentBintr);
                        
        bool RemoveFromParent(DSL_BASE_PTR pParentBintr);
        
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
         * @brief For sources that manage EOS Consumers, this service must
         * called before sending the source an EOS Event to stop playing.
         */
        virtual void DisableEosConsumer(){};
        
        /**
         * @brief Gets the current width and height settings for this SourceBintr
         * @param[out] width the current width setting in pixels
         * @param[out] height the current height setting in pixels
         */ 
        void GetDimensions(uint* width, uint* height);
        
        /**
         * @brief Gets the current FPS numerator and denominator settings for this SourceBintr
         * @param[out] fpsN the FPS numerator
         * @param[out] fpsD the FPS denominator
         */ 
        void GetFrameRate(uint* fpsN, uint* fpsD);
        
        /**
         * @brief Returns the current linkable state of the Source. Camera Sources
         * HTTP, and RTSP sources are linkable for the life of the Source.
         * File and Image Sources can be created without a file path and are unlinkable
         * until they are updated with a valid path.
         * @return true if the Source if linkable (able to link), false otherwise
         */
        virtual bool IsLinkable(){return true;};

    public:

            /**
         * @brief Device Properties, used for aarch64/x86_64 conditional logic
         */
        cudaDeviceProp m_cudaDeviceProp;

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
        uint m_fpsN;

        /**
         * @brief current frames-per-second denominator value for the Streaming Source
         */
        uint m_fpsD;

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

        /**
         * @brief Single, optional dewarper for the DecodeSourceBintr
         */ 
        DSL_DEWARPER_PTR m_pDewarperBintr;
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
            uint fpsN, uint fpsD);

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
     * @class UsbSourceBintr
     * @brief 
     */
    class UsbSourceBintr : public SourceBintr
    {
    public: 
    
        UsbSourceBintr(const char* name, uint width, uint height, 
            uint fpsN, uint fpsD);

        ~UsbSourceBintr();

        /**
         * @brief Links all Child Elementrs owned by this Source Bintr
         * @return True success, false otherwise
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elementrs owned by this Source Bintr
         */
        void UnlinkAll();
        
        /**
         * @brief Sets the GPU ID for all Elementrs
         * @return true if successfully set, false otherwise.
         */
        bool SetGpuId(uint gpuId);

    private:

        /**
         * @brief Unique sensor ID for this USB Source
         */
        uint m_sensorId;
        
        /**
         * @brief
         */
        DSL_ELEMENT_PTR m_pCapsFilter;
        
        /**
         * @brief
         */
        DSL_ELEMENT_PTR m_pVidConv1;

        /**
         * @brief
         */
        DSL_ELEMENT_PTR m_pVidConv2;
    }; 

    //*********************************************************************************
    
    class ResourceSourceBintr: public SourceBintr
    {
    public:
    
        ResourceSourceBintr(const char* name, const char* uri)
            : SourceBintr(name)
            , m_uri(uri)
        {
            LOG_FUNC();
        };
            
        ResourceSourceBintr(const char* name, const char** uris)
            : SourceBintr(name)
        {
            LOG_FUNC();
        };
            
        ~ResourceSourceBintr()
        {
            LOG_FUNC();
        }
        
        /**
         * @brief returns the current URI source for this ResourceSourceBintr
         * @return const string for either live or file source
         */
        const char* GetUri()
        {
            LOG_FUNC();
            
            return m_uri.c_str();
        }
        
        /**
         * @brief Virtual method to be implented by eached derived Resource Source
         * @param uri Source specific use of URI varries.
         * @return true on successful update, false otherwise
         */
        virtual bool SetUri(const char* uri) = 0;
        
       /**
         * @brief Returns the current linkable state of the Source. File and 
         * Image Sources can be created without a file path and are unlinkable
         * until they are updated with a valid path.
         * @return true if the Source is linkable (able to link), false otherwise
         */
        bool IsLinkable(){return bool(m_uri.size());};        
    
    protected:
    
        /**
         * @brief Universal Resource Identifier (URI) for this ResourceSourceBintr
         */
        std::string m_uri;
    };

    //*********************************************************************************

    /**
     * @class DecodeSourceBintr
     * @brief 
     */
    class DecodeSourceBintr : public ResourceSourceBintr
    {
    public: 
    
        DecodeSourceBintr(const char* name, const char* factoryName, const char* uri, 
            bool isLive, uint intraDecode, uint dropFrameInterval);
            
        ~DecodeSourceBintr();

        /**
         * @brief Sets the URL for file decode sources only
         * @param uri relative or absolute path to the file decode source
         */
        bool SetFileUri(const char* uri);
        
        /**
         * @brief Sets the unique source id for this Source bintr
         * @param id value to assign [0...MAX]
         */
        void SetSourceId(int id);
        
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

        /**
         * @brief adds a single Dewarper Bintr to this DecodeSourceBintr 
         * @param[in] pDewarperBintr shared pointer to Dewarper to add
         * @returns true if the Dewarper could be added, false otherwise
         */
        bool AddDewarperBintr(DSL_BASE_PTR pDewarperBintr);

        /**
         * @brief remove a previously added Dewarper Bintr from this DecodeSourceBintr 
         * @returns true if the Dewarper could be removed, false otherwise
         */
        bool RemoveDewarperBintr();
        
        /**
         * @brief call to query the Decode Source if it has a Dewarper
         * @return true if the Source has a Child
         */
        bool HasDewarperBintr();

        /**
         * @brief Disables Auto Repeat without updating the RepeatEnabled flag 
         * which will take affect on next Play Pipeline command. This function
         * should be called on non-live sources before sending the source an EOS
         */
        void DisableEosConsumer();
        
    protected:

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
         * nvv4l2decoder sink pad to add the Buffer Probe to
         */  
        GstPad* m_pDecoderStaticSinkpad;
        
        /**
         * @brief probe id for nvv412decoder Buffer Probe used to handle EOS
         * events and initiate the Restart process for a non-live source
         */
        guint m_bufferProbeId;
        
        /**
         * @brief A dynamic collection of requested Source Pads for the Tee 
         */
        std::map<std::string, GstPad*> m_pGstRequestedSourcePads;

        /**
         * @brief mutual exclusion of the repeat enabled setting.
         */
        GMutex m_repeatEnabledMutex;
        
        /**
         * @brief is set to true, non-live source will restart on EOS
         */
        bool m_repeatEnabled;

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
            uint intraDecode, uint dropFrameInterval);

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

        bool SetUri(const char* uri);

        void HandleSourceElementOnPadAdded(GstElement* pBin, GstPad* pPad);
        
    private:

    };

    //*********************************************************************************

    /**
     * @class FileSourceBintr
     * @brief 
     */
    class FileSourceBintr : public UriSourceBintr
    {
    public: 
    
        FileSourceBintr(const char* name, const char* uri, bool repeatEnabled);
        
        ~FileSourceBintr();

        /**
         * @brief Sets the URL for FileSourceBintr 
         * @param uri relative or absolute path to the file decode source
         */
        bool SetUri(const char* uri);

        /**
         * @brief Gets the current repeat enabled setting, non-live URI sources only
         * @return true if enabled, false otherwise.
         */
        bool GetRepeatEnabled();
        
        /**
         * @brief Sets the repeat enabled setting, non-live URI source only.
         * @param enabled set true to enable, false to disable
         * @return true on succcess, false otherwise
         */
        bool SetRepeatEnabled(bool enabled);
        
    private:

    };

    //*********************************************************************************

    /**
     * @class ImageSourceBintr
     * @brief Implements a Image Decode Source  - Super class
     */
    class ImageSourceBintr : public ResourceSourceBintr
    {
    public: 
    
        /**
         * @brief ctor for the MultiImageSourceBintr
         * @param[in] name unique name for the Image Source
         * @param[in] uri relative or absolute path to the input file source.
         * @param[in] type on of the DSL_IMAGE_TYPE_* constants
         */
        ImageSourceBintr(const char* name, const char* uri, uint type);
        
        /**
         * @brief dtor for the ImageSourceBintr
         */
        ~ImageSourceBintr();

        /**
         * @brief Links all Child Elementrs owned by this Source Bintr
         * @return True success, false otherwise
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elementrs owned by this Source Bintr
         */
        void UnlinkAll();

    protected:
        /**
         * @brief one of the DSL_IMAGE_EXTENTION_* constants
         */
        std::string m_ext;


    private:
    
        /**
         * @brief one of the DSL_IMAGE_FORMAT_* constants
         */
        uint m_format;
        
        /**
         * @brief JPEG or PNG Parser for this ImageSourceBintr
         */
        DSL_ELEMENT_PTR m_pParser;

        /**
         * @brief V4L2 Decoder for this ImageSourceBintr
         */
        DSL_ELEMENT_PTR m_pDecoder;

    };

    //*********************************************************************************

    /**
     * @class SingleImageSourceBintr
     * @brief Implements a Image Decode Source  
     */
    class SingleImageSourceBintr : public ImageSourceBintr
    {
    public: 
    
        /**
         * @brief ctor for the MultiImageSourceBintr
         * @param[in] name unique name for the Image Source
         * @param[in] uri relative or absolute path to the input file source.
         */
        SingleImageSourceBintr(const char* name, const char* uri);
        
        /**
         * @brief dtor for the ImageSourceBintr
         */
        ~SingleImageSourceBintr();

        /**
         * @brief Sets the URIs for ImageFrameSourceBintr 
         * @param uri relative or absolute path to the input file source.
         */
        bool SetUri(const char* uri);
        
    private:
    
    };

    //*********************************************************************************

    /**
     * @class MultiImageSourceBintr
     * @brief Implements a Multi Image Decode Source  
     */
    class MultiImageSourceBintr : public ImageSourceBintr
    {
    public: 
    
        /**
         * @brief ctor for the MultiImageSourceBintr
         * @param[in] name unique name for the Image Source
         * @param[in] uri relative or absolute path to the input file source.
         * @param[in] fpsN the FPS numerator
         * @param[in] fpsD the FPS denominator
         */
        MultiImageSourceBintr(const char* name, 
            const char* uri, uint fpsN, uint fpsD);
        
        /**
         * @brief dtor for the ImageSourceBintr
         */
        ~MultiImageSourceBintr();

        /**
         * @brief Sets the URIs for MultiImageFrameSourceBintr 
         * @param uri relative or absolute path to the input file source.
         */
        bool SetUri(const char* uri);
        
    private:

    };

    //*********************************************************************************

    /**
     * @class ImageStreamSourceBintr
     * @brief 
     */
    class ImageStreamSourceBintr : public ResourceSourceBintr
    {
    public: 
    
        /**
         * @brief Ctor for the ImageStreamSourceBintr class
         */
        ImageStreamSourceBintr(const char* name, 
            const char* uri, bool isLive, uint fpsN, uint fpsD, uint timeout);
        
        /**
         * @brief Dtor for the ImageStreamSourceBintr class
         */
        ~ImageStreamSourceBintr();

        /**
         * @brief Links all Child Elementrs owned by this Source Bintr
         * @return True success, false otherwise
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elementrs owned by this Source Bintr
         */
        void UnlinkAll();

        /**
         * @brief Sets the URI (filepath) to use by this ImageSoureceBintr
         * @param filePath absolute or relative path to the image file
         * @return true if set successfully, false otherwise
         */
        bool SetUri(const char* uri);

        /**
         * @brief Gets the current display timeout setting
         * @return current timeout setting.
         */
        uint GetTimeout();
        
        /**
         * @brief Sets the display timeout setting to send EOS on timeout
         * @param timeout timeout value in seconds, 0 to disable
         * @return true on succcess, false otherwise
         */
        bool SetTimeout(uint timeout);
        
        /**
         * @brief Handles the Image display timeout by sending and EOS event.
         * @return 0 always to clear the timer resource
         */
        int HandleDisplayTimeout();
        
    private:
        
        /**
         * @brief display timeout in units of seconds
         */
        uint m_timeout;

        /**
         * @brief gnome timer Id for the display timeout
         */
        uint m_timeoutTimerId;
        
        /**
         * @brief mutux to guard the display timeout callback.
         */
        GMutex m_timeoutTimerMutex;

        /**
         * @brief Caps Filter for the File Source Element
         */
        DSL_ELEMENT_PTR m_pSourceCapsFilter;

        /**
         * @brief Image Overlay element for the FileSourceBintr
         */
        DSL_ELEMENT_PTR m_pImageOverlay;

        DSL_ELEMENT_PTR m_pCapsFilter;
        DSL_ELEMENT_PTR m_pVidConv;

    };

    //*********************************************************************************

    /**
     * @class InterpipeSourceBintr
     * @brief 
     */
    class InterpipeSourceBintr : public SourceBintr
    {
    public: 
    
        /**
         * @brief Ctor for the ImageStreamSourceBintr class
         * @param[in] name unique name to assign to the Source Bintr
         * @param listenTo unique name of the InterpipeSinkBintr to listen to.
         * @param acceptEos if true, accepts the EOS event received from the 
         * Inter-Pipe Sink.
         * @param acceptEvents if true, accepts the downstream events (except 
         * for EOS) from the Inter-Pipe Sink.
         */
        InterpipeSourceBintr(const char* name, 
            const char* listenTo, bool isLive, bool acceptEos, bool acceptEvents);
        
        /**
         * @brief Dtor for the ImageStreamSourceBintr class
         */
        ~InterpipeSourceBintr();

        /**
         * @brief Links all Child Elementrs owned by this Source Bintr
         * @return True success, false otherwise
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elementrs owned by this Source Bintr
         */
        void UnlinkAll();

        /**
         * @brief Gets the name of the InterpipeSinkBintr the Source Bintr 
         * is listening to
         * @return name of the InterpipeSinkBintr this Bintr is listening to.
         */
        const char* GetListenTo();

        /**
         * @brief Sets the name of the InterpipeSinkBintr to listen to.
         * @param listenTo unique name of the InterpipeSinkBintr to listen to.
         */
        void SetListenTo(const char* listenTo);
        
        /**
         * @brief Gets the current Accept settings in use by the Source Bintr.
         * @param[out] acceptEos if true, the Source accepts EOS events from 
         * the Inter-Pipe Sink.
         * @param[out] acceptEvent if true, the Source accepts events (except EOS event) from 
         * the Inter-Pipe Sink.
         */
        void GetAcceptSettings(bool* acceptEos, bool* acceptEvents);

        /**
         * @brief Sets the Accept settings for the Source Bintr to use
         * @param[in] acceptEos set to true to accept EOS events from the Inter-Pipe Sink,
         * false otherwise.
         * @param[in] acceptEvent set to true to accept events (except EOS event) from 
         * the Inter-Pipe Sink, false otherwise.
         * @return DSL_RESULT_SUCCESS on success, DSL_RESULT_SOURCE_RESULT otherwise.
         */
        bool SetAcceptSettings(bool acceptEos, bool acceptEvents);
        
    private:
    
        /**
         * @brief uniqune name of the InterpipeSinkBintr to listen to.
         */
        std::string m_listenTo;

        /**
         * @brief uniqune name of the InterpipeSinkBintr's pluginto listen to.
         */
        std::string m_listenToFullName;

        /**
         * @brief if true, accepts the EOS event received from the Inter-Pipe Sink
         */
        bool m_acceptEos;

        /**
         * @brief if true, accepts the downstream events (except for EOS) from the 
         * Inter-Pipe Sink.
         */
        bool m_acceptEvents;

    };

    //*********************************************************************************

    /**
     * @class RtspSourceBintr
     * @brief 
     */
    class RtspSourceBintr : public DecodeSourceBintr
    {
    public: 
    
        RtspSourceBintr(const char* name, const char* uri, uint protocol, 
            uint intraDecode, uint dropFrameInterval, 
            uint latency, uint timeout);

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

        bool SetUri(const char* uri);
       
        /**
         * @brief Gets the current buffer timeout value controlling reconnection attemtps
         * @return buffer timeout in seconds, with 0 indicating that Stream Reconnection Management is disbled.
         */
        uint GetBufferTimeout();
        
        /**
         * @brief Sets the current buffer timeout to control reconnection attempts
         * @param[in] max time between successive buffers in units of seconds, set to 0 to diable
         */
        void SetBufferTimeout(uint timeout);

        /**
         * @brief Gets the current reconnection params in use by the named RTSP Source. The parameters are set
         * to DSL_RTSP_RECONNECT_SLEEP_MS and DSL_RTSP_RECONNECT_TIMEOUT_MS on source creation.
         * @param[out] sleep time to sleep between successively checking the status of the asynchrounus reconnection
         * @param[out] timeout current time to wait before terminating the current reconnection try, and
         * restarting the reconnection cycle again.
         */
        void GetReconnectionParams(uint* sleep, uint* timeout);
        
        /**
         * @brief Sets the current reconnection params in use by the named RTSP Source. The parameters are set
         * to DSL_RTSP_RECONNECT_SLEEP_MS and DSL_RTSP_RECONNECT_TIMEOUT_MS on source creation. Both must be > 10ms
         * Note: calling this service while a reconnection cycle is in progess will terminate the current cycle 
         * before restarting with the new parmeters.
         * @param[in] sleep time to sleep between successively checking the status of the asynchrounus reconnection
         * @param[in] timeout current time to wait before terminating the current reconnection try, and
         * restarting the reconnection cycle again.
         * @return true is params have been set, false otherwise.
         */
        bool SetReconnectionParams(uint sleep, uint timeout);

        /**
         * @brief Gets the Reconnect Statistics collected by the RTSP source 
         * @param[out] data current Connection Stats and Parameters for the source
         * or since the stats were last cleared by the client.
         */
        void GetConnectionData(dsl_rtsp_connection_data* data);
        
        /**
         * @brief Sets the Reconnect Statistics collected by the RTSP source 
         * Note: this services is to be called by the test services only
         * It is left public for the purposes of test only. 
         * @param[in] data new Connection Stats and Paremters to use under Test.
         */
        void _setConnectionData(dsl_rtsp_connection_data data);
        
        /**
         * @brief Clears the Reconnection Statistics collected by the RTSP source
         */
        void ClearConnectionStats();
        
        /**
         * @brief adds a callback to be notified on change of RTSP source state
         * @param[in] listener pointer to the client's function to call on state change
         * @param[in] userdata opaque pointer to client data passed into the listener function.
         * @return true on successfull add, false otherwise
         */
        bool AddStateChangeListener(dsl_state_change_listener_cb listener, void* userdata);

        /**
         * @brief removes a previously added callback
         * @param[in] listener pointer to the client's function to remove
         * @return true on successfull remove, false otherwise
         */
        bool RemoveStateChangeListener(dsl_state_change_listener_cb listener);

        /**
         * @brief Called periodically on timer experation to Check the status of the RTSP stream
         * and to initiate a reconnection cycle when the last buffer time execeeds timeout
         */
        int StreamManager();
        
        /**
         * @brief Called to manage the reconnection cycle on loss of connection
         */
        int ReconnectionManager();
        
        /**
         * @brief gets the RTSP Source's current state as maintaned by the component.
         * Not to be confussed with the GetState() Bintr base class function 
         * @return current state of the RTSP Source
         */
        GstState GetCurrentState();

        /**
         * @brief sets the RTSP Source's current state variable to newState, one of DSL_STATE_*
         * Changes in state will notifiy all client state-change-listeners, not to be confused
         * with the SetState() Bintr base class function which attempts change the actual state 
         * of the GstElement for this Bintr.
         * @param[in] newState new state to set the current state variable
         */
        void SetCurrentState(GstState newState);
        
        /**
         * @brief implements a timer thread to notify all client listeners in the main loop context.
         * @return false always to self remove timer once clients have been notified. Timer/tread will
         * be restarted on next call to SetCurrentState() that changes the current state.
         */
        int NotifyClientListeners();
        
        /**
         * @brief NOTE: Used for test purposes only, allows access to the Source's Timestamp PPH which 
         * is used to maintain a timestamp of the last buffer received for the source. 
         * @return 
         */
        DSL_PPH_TIMESTAMP_PTR _getTimestampPph(){return m_TimestampPph;};
        
        /**
         * @brief adds a TapBintr to the RTSP Source - one at most
         * @return true if the Source was able to add the Child TapBintr
         */
        bool AddTapBintr(DSL_BASE_PTR pTapBintr);

        /**
         * @brief Removes a TapBintr from the RTSP Source - if it currently has one
         * @return true if the Source was able to remove the Child TapBintr
         */
        bool RemoveTapBintr();
        
        /**
         * @brief call to query the RTSP Source if it has a TapBntr
         * @return true if the Source has a Child TapBintr
         */
        bool HasTapBintr();
        
        bool HandleSelectStream(GstElement* pBin, uint num, GstCaps* pCaps);

        void HandleSourceElementOnPadAdded(GstElement* pBin, GstPad* pPad);

        void HandleDecodeElementOnPadAdded(GstElement* pBin, GstPad* pPad);
        
    private:
    
        /**
         @brief 0x4 for TCP and 0x7 for All (UDP/UDP-MCAST/TCP)
         */
        uint m_rtpProtocols;
        
        /**
         * @brief optional child TapBintr, tapped in pre-decode
         */ 
        DSL_TAP_PTR m_pTapBintr;

        /**
         * @brief H.264 or H.265 RTP Depay for the RtspSourceBintr
         */
        DSL_ELEMENT_PTR m_pDepay;

        /**
         * @brief H.264 or H.265 RTP Parser for the RtspSourceBintr
         */
        DSL_ELEMENT_PTR m_pParser;
        
        /**
         * @brief Pre-decode queue 
         */
        DSL_ELEMENT_PTR m_pPreDecodeQueue;

        /**
         * @brief Pre-decode tee - optional to tap off pre-decode strame for TapBintr
         */
        DSL_ELEMENT_PTR m_pPreDecodeTee;

        /**
         * @brief
         */
        DSL_ELEMENT_PTR m_pDecodeBin;
        
        /**
         * @brief Pad Probe Handler to create a timestamp for the last recieved buffer
         */
        DSL_PPH_TIMESTAMP_PTR m_TimestampPph;

        /**
         * @brief maximim time between successive buffers before determining the connection is lost, 0 to disable 
         */
        uint m_bufferTimeout;
        
        /**
         * @brief gnome timer Id for RTSP stream-status and reconnect management 
         */
        uint m_streamManagerTimerId;
        
        /**
         * @brief mutux to guard the buffer timeout managment read/write attributes.
         */
        GMutex m_streamManagerMutex;
        
        /**
         * @brief active connection data for the RtspSourceBintr.
         */
        dsl_rtsp_connection_data m_connectionData;
        
        /**
         * @brief gnome timer Id for the RTSP reconnection manager
         */
        uint m_reconnectionManagerTimerId;

        /**
         * @brief mutux to guard the reconnection managment read/write attributes.
         */
        GMutex m_reconnectionManagerMutex;
        
        /**
         * @brief will be set to true on reconnection failure to force a mew reconnection cycle
         */
        bool m_reconnectionFailed;
        
        /**
         * @brief remaining time to sleep after a failed reconnection attemp, in seconds. 
         */
        uint m_reconnectionSleep;
        
        /**
         * @brief start time of the most recent reconnection cycle, used for maximum timeout 
         * for async state change completion
         */
        timeval m_reconnectionStartTime;

        /**
         * @brief maintains the current state of the RTSP source bin
         */
        GstState m_currentState;

        /**
         * @brief maintains the previous state of the RTSP source bin
         */
        GstState m_previousState;

        /**
         * @brief mutux to guard the current State read/write access.
         */
        GMutex m_stateChangeMutex;

        /**
         * @brief gnome timer Id for the RTSP reconnection manager
         */
        uint m_listenerNotifierTimerId;
        
        /**
         * @brief map of all currently registered state-change-listeners
         * callback functions mapped with the user provided data
         */
        std::map<dsl_state_change_listener_cb, void*>m_stateChangeListeners;
        
        /**
         * @brief a queue of state changes to process and notify clients asynchronously
         */
        std::queue<std::shared_ptr<DslStateChange>> m_stateChanges;
    };
    
    /**
     * @brief Timer Callback to handle the image display timeout
     * @param pSource (callback user data) pointer to the unique source opject
     * @return 0 always - one shot timer.
     */
    static int ImageSourceDisplayTimeoutHandler(gpointer pSource);
    
    /**
     * @brief 
     * @param[in] pBin
     * @param[in] pPad
     * @param[in] pSource (callback user data) pointer to the unique source opject
     */
    static void UriSourceElementOnPadAddedCB(GstElement* pBin, GstPad* pPad, gpointer pSource);

    /**
     * @brief Called to select the Stream, H264 or H265, based on received caps
     * @param pBin -unused
     * @param num -unused
     * @param pCaps pointer to the caps structure that specifies the Stream to select
     * @param[in] pSource shared pointer to the RTSP Source component.
     * @return true on successful selection, false otherwise
     */
    static boolean RtspSourceSelectStreamCB(GstElement *pBin, uint num, GstCaps *pCaps,
        gpointer pSource);
        
    /**
     * @brief Called on new Pad Added to link the depayload and parser
     * @param pBin pointer to the depayload bin
     * @param pPad Pointer to the new Pad added for linking
     * @param[in] pSource shared pointer to the RTSP Source component.
     */
    static void RtspSourceElementOnPadAddedCB(GstElement* pBin, GstPad* pPad, gpointer pSource);
    
    /**
     * @brief Called on new Pad Added to link the parser and the decoder
     * @param pBin pointer to the parser bin
     * @param pPad Pointer to the new Pad added for linking
     * @param[in] pSource shared pointer to the RTSP Source component.
     */
    static void RtspDecodeElementOnPadAddedCB(GstElement* pBin, GstPad* pPad, gpointer pSource);

    /**
     * @brief 
     * @param[in] pChildProxy
     * @param[in] pObject
     * @param[in] name
     * @param[in] pSource shared pointer to the RTSP Source component.
     */
    static void OnChildAddedCB(GstChildProxy* pChildProxy, GObject* pObject,
        gchar* name, gpointer pSource);

    /**
     * @brief 
     * @param[in] pObject
     * @param[in] arg0
     * @param[in] pSource shared pointer to the RTSP Source component.
     */
    static void OnSourceSetupCB(GstElement* pObject, GstElement* arg0, gpointer pSource);

    /**
     * @brief Probe function to drop certain events to support
     * custom logic of looping of each decode source (file) stream.
     * @param pPad
     * @param pInfo
     * @param[in] pSource shared pointer to the RTSP Source component.
     * @return 
     */
    static GstPadProbeReturn StreamBufferRestartProbCB(GstPad* pPad, 
        GstPadProbeInfo* pInfo, gpointer pSource);

    /**
     * @brief 
     * @param[in] pSource shared pointer to the RTSP Source component.
     * @return 
     */
    static gboolean StreamBufferSeekCB(gpointer pSource);
    
    /**
     * @brief Timer callback handler to invoke the RTSP Source's Stream manager.
     * @param pSource shared pointer to RTSP Source component to check/manage.
     * @return int true to continue, 0 to self remove
     */
    static int RtspStreamManagerHandler(gpointer pSource);
    
    /**
     * @brief Timer callback handler to invoke the RTSP Source's Reconnection Manager.
     * @param[in] pSource shared pointer to RTSP Source component to invoke.
     * @return int true to continue, 0 to self remove
     */
    static int RtspReconnectionMangerHandler(gpointer pSource);
    
    /**
     * @brief Timer callback handler to invoke the RTSP Source's Listerner notification.
     * @param[in] pSource shared pointer to RTSP Source component to invoke.
     * @return int true to continue, 0 to self remove
     */
    static int RtspListenerNotificationHandler(gpointer pSource);

} // DSL
#endif // _DSL_SOURCE_BINTR_H
