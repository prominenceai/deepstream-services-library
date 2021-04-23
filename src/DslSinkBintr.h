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

#ifndef _DSL_SINK_BINTR_H
#define _DSL_SINK_BINTR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslBintr.h"
#include "DslElementr.h"
#include "DslRecordMgr.h"
#include "DslSourceMeter.h"

#include <gst-nvdssr.h>

namespace DSL
{
    #define DSL_SINK_PTR std::shared_ptr<SinkBintr>

    #define DSL_FAKE_SINK_PTR std::shared_ptr<FakeSinkBintr>
    #define DSL_FAKE_SINK_NEW(name) \
        std::shared_ptr<FakeSinkBintr>( \
        new FakeSinkBintr(name))

    #define DSL_METER_SINK_PTR std::shared_ptr<MeterSinkBintr>
    #define DSL_METER_SINK_NEW(name, interval, clientListener, clientData) \
        std::shared_ptr<MeterSinkBintr>( \
        new MeterSinkBintr(name, interval, clientListener, clientData))

    #define DSL_RENDER_SINK_PTR std::shared_ptr<RenderSinkBintr>

    #define DSL_OVERLAY_SINK_PTR std::shared_ptr<OverlaySinkBintr>
    #define DSL_OVERLAY_SINK_NEW(name, overlayId, displayId, depth, offsetX, offsetY, width, height) \
        std::shared_ptr<OverlaySinkBintr>( \
        new OverlaySinkBintr(name, overlayId, displayId, depth, offsetX, offsetY, width, height))

    #define DSL_WINDOW_SINK_PTR std::shared_ptr<WindowSinkBintr>
    #define DSL_WINDOW_SINK_NEW(name, offsetX, offsetY, width, height) \
        std::shared_ptr<WindowSinkBintr>( \
        new WindowSinkBintr(name, offsetX, offsetY, width, height))

    #define DSL_ENCODE_SINK_PTR std::shared_ptr<EncodeSinkBintr>
        
    #define DSL_FILE_SINK_PTR std::shared_ptr<FileSinkBintr>
    #define DSL_FILE_SINK_NEW(name, filepath, codec, container, bitRate, interval) \
        std::shared_ptr<FileSinkBintr>( \
        new FileSinkBintr(name, filepath, codec, container, bitRate, interval))
        
    #define DSL_RECORD_SINK_PTR std::shared_ptr<RecordSinkBintr>
    #define DSL_RECORD_SINK_NEW(name, outdir, codec, container, bitRate, interval, clientListener) \
        std::shared_ptr<RecordSinkBintr>( \
        new RecordSinkBintr(name, outdir, codec, container, bitRate, interval, clientListener))
        
    #define DSL_RTSP_SINK_PTR std::shared_ptr<RtspSinkBintr>
    #define DSL_RTSP_SINK_NEW(name, host, udpPort, rtspPort, codec, bitRate, interval) \
        std::shared_ptr<RtspSinkBintr>( \
        new RtspSinkBintr(name, host, udpPort, rtspPort, codec, bitRate, interval))
        
        
    class SinkBintr : public Bintr
    {
    public: 
    
        SinkBintr(const char* name, bool sync, bool async);

        ~SinkBintr();
  
        bool AddToParent(DSL_BASE_PTR pParentBintr);

        bool IsParent(DSL_BASE_PTR pParentBintr);
        
        bool RemoveFromParent(DSL_BASE_PTR pParentBintr);
        
        /**
         * @brief returns the current sync and async settings for the SinkBintr
         * @param[in] sync current sync setting, true if set, false otherwise.
         * @param[in] async current async setting, true if set, false otherwise.
         */
        void GetSyncSettings(bool* sync, bool* async);
        
        /**
         * @brief sets the current sync and async settings for the SinkBintr
         * @param[in] sync current sync setting, true if set, false otherwise.
         * @param[in] async current async setting, true if set, false otherwise.
         */
        virtual bool SetSyncSettings(bool sync, bool async) = 0;
        
    protected:

        /**
         * @brief Sink element's current synchronous attribute setting.
         */
        boolean m_sync;

        /**
         * @brief Sink element's current asynchronous attribute setting.
         */
        boolean m_async;

        /**
         * @brief Queue element as sink for all Sink Bintrs.
         */
        DSL_ELEMENT_PTR m_pQueue;
    };

    //-------------------------------------------------------------------------

    class FakeSinkBintr : public SinkBintr
    {
    public: 
    
        FakeSinkBintr(const char* name);

        ~FakeSinkBintr();
  
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
         * @brief sets the current sync and async settings for the SinkBintr
         * @param[in] sync current sync setting, true if set, false otherwise.
         * @param[in] async current async setting, true if set, false otherwise.
         * @return true is successful, false otherwise. 
         */
        bool SetSyncSettings(bool sync, bool async);

    private:

        boolean m_qos;
        
        /**
         * @brief Fake Sink element for the Sink Bintr.
         */
        DSL_ELEMENT_PTR m_pFakeSink;
    };

    //-------------------------------------------------------------------------

    class RenderSinkBintr : public SinkBintr
    {
    public: 
    
        RenderSinkBintr(const char* name, 
            uint offsetX, uint offsetY, uint width, uint height, bool sync, bool async);

        ~RenderSinkBintr();
        
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
        virtual bool SetOffsets(uint offsetX, uint offsetY) = 0;

        /**
         * @brief Gets the current width and height settings for this WindowSinkBintr
         * @param[out] width the current width setting in pixels
         * @param[out] height the current height setting in pixels
         */ 
        void GetDimensions(uint* width, uint* height);
        
        /**
         * @brief Sets the current width and height settings for this SinkBintr
         * The caller is required to provide valid width and height values
         * @param[in] width the width value to set in pixels
         * @param[in] height the height value to set in pixels
         * @return false if the sink is currently Linked. True otherwise
         */ 
        virtual bool SetDimensions(uint width, uint hieght) = 0;
        

    protected:

        /**
         * @brief offset from the left edge in uints of pixels
         */
        uint m_offsetX;

        /**
         * @brief offset from the top edge in uints of pixels
         */
        uint m_offsetY;

        /**
         * @brief Width property for the SinkBintr in uints of pixels
         */
        uint m_width;

        /**
         * @brief Height property for the SinkBintr in uints of pixels
         */
        uint m_height;
    };
    
    //-------------------------------------------------------------------------

    class OverlaySinkBintr : public RenderSinkBintr
    {
    public: 
    
        OverlaySinkBintr(const char* name, uint overlayId, uint displayId, uint depth, 
            uint offsetX, uint offsetY, uint width, uint height);

        ~OverlaySinkBintr();
  
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

        int GetDisplayId();

        bool SetDisplayId(int id);

        /**
         * @brief Sets the current X and Y offset settings for this OverlaySinkBintr
         * The caller is required to provide valid width and height values
         * @param[in] offsetX the offset in the X direct to set in pixels
         * @param[in] offsetY the offset in the Y direct to set in pixels
         * @return false if the OverlaySink is currently in Use. True otherwise
         */ 
        bool SetOffsets(uint offsetX, uint offsetY);
        
        /**
         * @brief Sets the current width and height settings for this OverlaySinkBintr
         * The caller is required to provide valid width and height values
         * @param[in] width the width value to set in pixels
         * @param[in] height the height value to set in pixels
         * @return false if the OverlaySink is currently in Use. True otherwise
         */ 
        bool SetDimensions(uint width, uint hieght);

        /**
         * @brief sets the current sync and async settings for the SinkBintr
         * @param[in] sync current sync setting, true if set, false otherwise.
         * @param[in] async current async setting, true if set, false otherwise.
         * @return true is successful, false otherwise. 
         */
        bool SetSyncSettings(bool sync, bool async);
        
    private:

        boolean m_qos;
        uint m_overlayId;
        uint m_displayId;
        uint m_uniqueId;
        uint m_depth;

        DSL_ELEMENT_PTR m_pOverlay;
    };

    //-------------------------------------------------------------------------

    class WindowSinkBintr : public RenderSinkBintr
    {
    public: 
    
        WindowSinkBintr(const char* name, guint offsetX, guint offsetY, guint width, guint height);

        ~WindowSinkBintr();
  
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
         * @brief Sets the current X and Y offset settings for this WindowSinkBintr
         * The caller is required to provide valid width and height values
         * @param[in] offsetX the offset in the X direction to set in pixels
         * @param[in] offsetY the offset in the Y direction to set in pixels
         * @return false if the OverlaySink is currently in Use. True otherwise
         */ 
        bool SetOffsets(uint offsetX, uint offsetY);
        
        /**
         * @brief Sets the current width and height settings for this WindowSinkBintr
         * The caller is required to provide valid width and height values
         * @param[in] width the width value to set in pixels
         * @param[in] height the height value to set in pixels
         * @return false if the OverlaySink is currently in Use. True otherwise
         */ 
        bool SetDimensions(uint width, uint hieght);

        /**
         * @brief sets the current sync and async settings for the SinkBintr
         * @param[in] sync current sync setting, true if set, false otherwise.
         * @param[in] async current async setting, true if set, false otherwise.
         * @return true is successful, false otherwise. 
         */
        bool SetSyncSettings(bool sync, bool async);
        
        /**
         * @brief Gets the current force-aspect-ratio setting for the WindowSinkBintr
         * @return true if forced, false otherwise
         */
        bool GetForceAspectRatio();
        
        /**
         * @brief Set the force-aspect-ration setting for the WindowSinkBinter
         * @param[in] force set true to force-aspect-ration false otherwise
         * @return 
         */
        bool SetForceAspectRatio(bool force);

    private:

        /**
         * @brief Resets (recreates) the EGL-GLES sink element to allow the
         * Pipeline to invoke the "prepare_window_handle" on relink and play
         * @return true if successful, false otherwise.
         */
        void Reset();

        boolean m_qos;
        bool m_forceAspectRatio;

        DSL_ELEMENT_PTR m_pTransform;
        DSL_ELEMENT_PTR m_pEglGles;
    };

    //-------------------------------------------------------------------------

    class EncodeSinkBintr : public SinkBintr
    {
    public: 
    
        EncodeSinkBintr(const char* name, 
            uint codec, uint container, uint bitRate, uint interval);

        /**
         * @brief Gets the current codec and media container formats for FileSinkBintr
         * @param[out] codec the current codec format in use [MPEG, H.264, H.265]
         * @param[out] container the current media container format [MPEG, MK4]
         */ 
        void GetVideoFormats(uint* codec, uint* container);

        /**
         * @brief Gets the current bit-rate and interval settings for the Encoder in use
         * @param[out] bitRate the current bit-rate setting for the encoder in use
         * @param[out] interval the current iframe interval for the encoder in use
         */ 
        void GetEncoderSettings(uint* bitRate, uint* interval);

        /**
         * @brief Sets the current bit-rate and interval settings for the Encoder in use
         * @param[in] bitRate the new bit-rate setting in uints of bits/sec
         * @param[in] interval the new iframe-interval setting
         * @return false if the FileSink is currently in Use. True otherwise
         */ 
        bool SetEncoderSettings(uint bitRate, uint interval);

        /**
         * @brief Sets the GPU ID for all Elementrs
         * @return true if successfully set, false otherwise.
         */
        bool SetGpuId(uint gpuId);
        
    protected:

        uint m_codec;
        uint m_container;
        uint m_bitRate;
        uint m_interval;
 
        DSL_ELEMENT_PTR m_pTransform;
        DSL_ELEMENT_PTR m_pCapsFilter;
        DSL_ELEMENT_PTR m_pEncoder;
        DSL_ELEMENT_PTR m_pParser;
        DSL_ELEMENT_PTR m_pContainer;       
    };

    //-------------------------------------------------------------------------

    class FileSinkBintr : public EncodeSinkBintr
    {
    public: 
    
        FileSinkBintr(const char* name, const char* filepath, 
            uint codec, uint container, uint bitRate, uint interval);

        ~FileSinkBintr();
  
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
         * @brief sets the current sync and async settings for the SinkBintr
         * @param[in] sync current sync setting, true if set, false otherwise.
         * @param[in] async current async setting, true if set, false otherwise.
         * @return true is successful, false otherwise. 
         */
        bool SetSyncSettings(bool sync, bool async);
        
    private:

        DSL_ELEMENT_PTR m_pFileSink;
    };

    //-------------------------------------------------------------------------

    class RecordSinkBintr : public EncodeSinkBintr, public RecordMgr
    {
    public: 
    
        RecordSinkBintr(const char* name, const char* outdir, uint codec, uint container, 
            uint bitRate, uint interval, dsl_record_client_listener_cb clientListener);

        ~RecordSinkBintr();
  
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
         * @brief sets the current sync and async settings for the SinkBintr
         * @param[in] sync current sync setting, true if set, false otherwise.
         * @param[in] async current async setting, true if set, false otherwise.
         * @return true is successful, false otherwise. 
         */
        bool SetSyncSettings(bool sync, bool async);

    private:

        /**
         * @brief Node to wrap NVIDIA's Record Bin
         */
        DSL_NODETR_PTR m_pRecordBin;
    };


    //-------------------------------------------------------------------------

    class RtspSinkBintr : public SinkBintr
    {
    public: 
    
        RtspSinkBintr(const char* name, const char* host, uint udpPort, uint rtspPort,
         uint codec, uint bitRate, uint interval);

        ~RtspSinkBintr();
  
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
         * @brief Gets the current codec and media container formats for RtspSinkBintr
         * @param[out] port the current UDP port number for the RTSP Server
         * @param[out] port the current RTSP port number for the RTSP Server
         * @param[out] codec the current codec format in use [H.264, H.265]
         */ 
        void GetServerSettings(uint* udpPort, uint* rtspPort, uint* codec);

        /**
         * @brief Gets the current bit-rate and interval settings for the Encoder in use
         * @param[out] bitRate the current bit-rate setting for the Encoder in use
         * @param[out] interval the current iframe interval to write to file
         */ 
        void GetEncoderSettings(uint* bitRate, uint* interval);

        /**
         * @brief Sets the current bit-rate and interval settings for the Encoder in use
         * @param[in] bitRate the new bit-rate setting in uints of bits/sec
         * @param[in] interval the new iframe-interval setting
         * @return false if the FileSink is currently in Use. True otherwise
         */ 
        bool SetEncoderSettings(uint bitRate, uint interval);

        /**
         * @brief sets the current sync and async settings for the SinkBintr
         * @param[in] sync current sync setting, true if set, false otherwise.
         * @param[in] async current async setting, true if set, false otherwise.
         * @return true is successful, false otherwise. 
         */
        bool SetSyncSettings(bool sync, bool async);


    private:

        std::string m_host;
        uint m_udpPort;
        uint m_rtspPort;
        uint m_codec;
        uint m_bitRate;
        uint m_interval;
        
        GstRTSPServer* m_pServer;
        uint m_pServerSrcId;
        GstRTSPMediaFactory* m_pFactory;
 
        DSL_ELEMENT_PTR m_pUdpSink;
        DSL_ELEMENT_PTR m_pTransform;
        DSL_ELEMENT_PTR m_pCapsFilter;
        DSL_ELEMENT_PTR m_pEncoder;
        DSL_ELEMENT_PTR m_pParser;
        DSL_ELEMENT_PTR m_pPayloader;  
    };
    
}
#endif // _DSL_SINK_BINTR_H
    