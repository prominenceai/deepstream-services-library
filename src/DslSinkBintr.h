
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
#include "DslApi.h"
#include "DslBintr.h"
#include "DslElementr.h"

#include <gst-nvdssr.h>

namespace DSL
{
    #define DSL_SINK_PTR std::shared_ptr<SinkBintr>

    #define DSL_FAKE_SINK_PTR std::shared_ptr<FakeSinkBintr>
    #define DSL_FAKE_SINK_NEW(name) \
        std::shared_ptr<FakeSinkBintr>( \
        new FakeSinkBintr(name))

    #define DSL_IMAGE_SINK_PTR std::shared_ptr<ImageSinkBintr>
    #define DSL_IMAGE_SINK_NEW(name, outdir) \
        std::shared_ptr<ImageSinkBintr>( \
        new ImageSinkBintr(name, outdir))

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
    
        SinkBintr(const char* name);

        ~SinkBintr();
  
        bool AddToParent(DSL_BASE_PTR pParentBintr);

        bool IsParent(DSL_BASE_PTR pParentBintr);
        
        bool RemoveFromParent(DSL_BASE_PTR pParentBintr);
        
        bool LinkToSource(DSL_NODETR_PTR pTee);

        bool UnlinkFromSource();
        
    protected:

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

    private:

        boolean m_sync;
        boolean m_async;
        boolean m_qos;
        
        /**
         * @brief Fake Sink element for the Sink Bintr.
         */
        DSL_ELEMENT_PTR m_pFakeSink;
    };

    //-------------------------------------------------------------------------

    class OverlaySinkBintr : public SinkBintr
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

    //-------------------------------------------------------------------------

    class WindowSinkBintr : public SinkBintr
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
         * @brief Gets the current X and Y offset settings for this WindowSinkBintr
         * @param[out] offsetX the current offset in the X direction in pixels
         * @param[out] offsetY the current offset in the Y direction setting in pixels
         */ 
        void GetOffsets(uint* offsetX, uint* offsetY);

        /**
         * @brief Sets the current X and Y offset settings for this WindowSinkBintr
         * The caller is required to provide valid width and height values
         * @param[in] offsetX the offset in the X direction to set in pixels
         * @param[in] offsetY the offset in the Y direction to set in pixels
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

    private:

        boolean m_qos;
        boolean m_sync;
        boolean m_async;
        uint m_offsetX;
        uint m_offsetY;
        uint m_width;
        uint m_height;

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
         * @param[in] bitRate the new bit-rate setting in units of bits/sec
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
        boolean m_sync;
        boolean m_async;
 
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

    private:

        boolean m_sync;
        boolean m_async;

        DSL_ELEMENT_PTR m_pFileSink;
    };

    //-------------------------------------------------------------------------

    class RecordSinkBintr : public EncodeSinkBintr
    {
    public: 
    
        RecordSinkBintr(const char* name, const char* outdir, uint codec, uint container, 
            uint bitRate, uint interval, NvDsSRCallbackFunc clientListener);

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
         * @brief Gets the current outdir in use by this Bintr
         * @return relative or absolute pathspec as provided on construction or set call.
         */
        const char* GetOutdir();

        /**
         * @brief Sets the outdir to use by this Bintr
         * @param[in] relative or absolute pathspec to the existing directory to use
         * @return true on successfull set, false otherwise
         */
        bool SetOutdir(const char* outdir);

        /**
         * @brief Gets the Smart Record initialization parameters used by this SmartFileSinkBint
         * @return size of the video cache in seconds 
         * default = DSL_DEFAULT_SINK_VIDEO_CACHE_IN_SEC
         */
        uint GetCacheSize();
        
        /**
         * @brief Sets the Smart Record initialization parameters used by this SmartFileSinkBint
         * @param[in] videoCacheSize size of video cache in seconds 
         * default = DSL_DEFAULT_SINK_VIDEO_CACHE_IN_SEC
         * @param[in] defaultDuration default video recording duration.
         * default = DSL_DEFAULT_SINK_VIDEO_DURATION_IN_SEC
         */
        bool SetCacheSize(uint videoCacheSize);
        
        /**
         * @brief Gets the current width and height settings for this RecordSinkBintr
         * Zero values indicates no transcode
         * @param[out] width the current width setting in pixels
         * @param[out] height the current height setting in pixels
         */ 
        void GetDimensions(uint* width, uint* height);
        
        /**
         * @brief Sets the current width and height settings for this RecordSinkBintr
         * Zero values indicates no transcode
         * The caller is required to provide valid width and height values
         * @param[in] width the width value to set in pixels
         * @param[in] height the height value to set in pixels
         * @return false if the OverlaySink is currently in Use. True otherwise
         */ 
        bool SetDimensions(uint width, uint hieght);
        
        /**
         * @brief Start recording to file
         * @param[out] session unique Id for the new recording session, 
         * @param[in] start seconds before the current time. Should be less than video cache size.
         * @param[in] duration of recording in seconds from start
         * @param[in] clientData returned on call to client callback
         * @return true on succesful start, false otherwise
         */
        bool StartSession(uint* session, uint start, uint duration, void* clientData);
        
        /**
         * @brief Stop recording to file
         * @param[in] session unique sission Id of the recording session to stop
         * @return true on succesful start, false otherwise
         */
        bool StopSession(uint session);

        /**
         * @brief Queries the Record Bin context to check the Key Frame
         * @return true if the Bin has the Key Frame ???
         */
        bool GotKeyFrame();
        
        /**
         * @brief Queires the Record Bin context to check if the recording is on
         * @return true if recording is currently on
         */
        bool IsOn();
        
        /**
         * @brief Queries the Record Bin context to check if reset has been
         * @return true if reset has been done.
         */
        bool ResetDone();

    private:

        /**
         * @brief absolute or relative path 
         */
        std::string m_outdir;

        /**
         * @brief SR context, once created, must be passed to 
         */
        NvDsSRContext* m_pContext;
        
        /**
         * @brief SR context initialization parameters, provided by client
         */
        NvDsSRInitParams m_initParams;
        
        DSL_NODETR_PTR m_pRecordBin;
        
        DSL_ELEMENT_PTR m_pRecordBinQueue;
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
         * @param[in] bitRate the new bit-rate setting in units of bits/sec
         * @param[in] interval the new iframe-interval setting
         * @return false if the FileSink is currently in Use. True otherwise
         */ 
        bool SetEncoderSettings(uint bitRate, uint interval);

    private:

        std::string m_host;
        uint m_udpPort;
        uint m_rtspPort;
        uint m_codec;
        uint m_bitRate;
        uint m_interval;
        boolean m_sync;
        boolean m_async;
        
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
    
    //-------------------------------------------------------------------------
    class CaptureClass
    {
    public:
    
        CaptureClass(uint id, bool fullFrame, uint captureLimit)
            : m_id(id)
            , m_fullFrame(fullFrame)
            , m_captureCount(0)
            , m_captureLimit(captureLimit)
            {}
        
        uint m_id;
        bool m_fullFrame;
        uint m_captureCount;
        uint m_captureLimit;
    };
    
    class ImageSinkBintr : public FakeSinkBintr
    {
    public:
    
        ImageSinkBintr(const char* name, const char* outdir);
        
        ~ImageSinkBintr();
        
        /**
         * @brief Gets the current outdir in use by this Bintr
         * @return relative or absolute pathspec as provided on construction or set call.
         */
        const char* GetOutdir();

        /**
         * @brief Sets the outdir to use by this Bintr
         * @param[in] relative or absolute pathspec to the existing directory to use
         * @return true on successfull set, false otherwise
         */
        bool SetOutdir(const char* outdir);
        
        /**
         * @brief Gets the current interval at which to capture frames
         * @return 0 for every frame, 1 for every other, etc.
         */
        uint GetFrameCaptureInterval();

        /**
         * @brief Sets the current Frame Capture interval
         * @param frameCaptureInterval new interval value to use
         * @return ture if successful, false otherwise
         */
        bool SetFrameCaptureInterval(uint frameCaptureInterval);
        
        /**
         * @brief Gets the current state of the Frame Capture enabled flag
         * @return true if enabled, false otherwise.
         */
        bool GetFrameCaptureEnabled();
        
        /**
         * @brief Sets the current state of the Frame Capture enabled flag
         * @param enabled set true to enable, false to disable
         * @return true if successful, false otherwise
         */
        bool SetFrameCaptureEnabled(bool enabled);

        /**
         * @brief Frame callback handler for the Capture service
         * @param pBuffer input buffer of frame data
         * @return true to stay registered, false for self removal
         */
        bool HandleFrameCapture(GstBuffer* pBuffer);
        
        /**
         * @brief Gets the current state of the Object Capture enabled flag
         * @return true if enabled, false otherwise.
         */
        bool GetObjectCaptureEnabled();

        /**
         * @brief Sets the state of the Object Captue enabled flag
         * @param enabled set to true to enable Object Capture, false to disable
         * @return true if successful, false otherwise
         */
        bool SetObjectCaptureEnabled(bool enabled);
        
        /**
         * @brief Adds an Object class to capture
         * @param classId unique class id of the objects to capture
         * @param fullFrame set to true to capture full-frame, false to capture image within bbox
         * @param captureLimit max number of objects to capture for a specific class id, 0 = no limit
         * @return true if successful, false otherwise
         */
        bool AddObjectCaptureClass(uint classId, boolean fullFrame, uint captureLimit);
        
        /**
         * @brief Removes a previously added capture class
         * @param classId unique classId to remove
         * @return true if successful, false otherwise
         */
        bool RemoveObjectCaptureClass(uint classId);
        
        /**
         * @brief Frame callback handler for the Capture service
         * @param pBuffer input buffer of frame data
         * @return true to stay registered, false for self removal
         */
        bool HandleObjectCapture(GstBuffer* pBuffer);
        
    private:
    
        /**
         * @brief Directory to save image output to
         */
        std::string m_outdir;

        /**
         * @brief The current frame count for the ongoing frame capture
         */
        uint m_frameCaptureframeCount;

        /**
         * @brief The frame interval to tranform and save. 0 = capture every frame
         */
        uint m_frameCaptureInterval;

        /**
         * @brief True if frame buffer should be transformed and output.
         */ 
        bool m_isFrameCaptureEnabled;

        /**
         * @brief The current frame count for the ongoing frame capture
         */
        uint m_objectCaptureFrameCount;

        /**
         * @brief The frame interval to tranform objects and save. 0 = capture every frame
         */
        uint m_objectCaptureInterval;

        /**
         * @brief True if objects in frame buffer should be transformed and output.
         */ 
        bool m_isObjectCaptureEnabled;

        /**
         * @brief map of class Id's to capture and whether to capture full frame or bbox rectangle
         */
        std::map <uint, std::shared_ptr<CaptureClass>> m_captureClasses;
        
        /**
         * @brief mutex for updating Image Capture params
         */
        GMutex m_captureMutex;

    };

    
    static boolean FrameCaptureHandler(void* batch_meta, void* user_data);
    
    static boolean ObjectCaptureHandler(void* batch_meta, void* user_data);
}
#endif // _DSL_SINK_BINTR_H
    