/*
The MIT License

Copyright (c) 2019-2024, Prominence AI, Inc.

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
#include "DslQBintr.h"
#include "DslElementr.h"
#include "DslRecordMgr.h"
#include "DslSourceMeter.h"

namespace DSL
{
    #define DSL_SINK_PTR std::shared_ptr<SinkBintr>

    #define DSL_APP_SINK_PTR std::shared_ptr<AppSinkBintr>
    #define DSL_APP_SINK_NEW(name, dataType, clientHandler, clientData) \
        std::shared_ptr<AppSinkBintr>( \
        new AppSinkBintr(name, dataType, clientHandler, clientData))

    #define DSL_FRAME_CAPTURE_SINK_PTR std::shared_ptr<FrameCaptureSinkBintr>
    #define DSL_FRAME_CAPTURE_SINK_NEW(name, pFrameCaptureAction) \
        std::shared_ptr<FrameCaptureSinkBintr>( \
        new FrameCaptureSinkBintr(name, pFrameCaptureAction))

    #define DSL_FAKE_SINK_PTR std::shared_ptr<FakeSinkBintr>
    #define DSL_FAKE_SINK_NEW(name) \
        std::shared_ptr<FakeSinkBintr>( \
        new FakeSinkBintr(name))

    #define DSL_WINDOW_SINK_PTR std::shared_ptr<WindowSinkBintr>

    #define DSL_3D_SINK_PTR std::shared_ptr<ThreeDSinkBintr>
    #define DSL_3D_SINK_NEW(name, offsetX, offsetY, width, height) \
        std::shared_ptr<ThreeDSinkBintr>( \
        new ThreeDSinkBintr(name, offsetX, offsetY, width, height))

    #define DSL_EGL_SINK_PTR std::shared_ptr<EglSinkBintr>
    #define DSL_EGL_SINK_NEW(name, offsetX, offsetY, width, height) \
        std::shared_ptr<EglSinkBintr>( \
        new EglSinkBintr(name, offsetX, offsetY, width, height))

    #define DSL_ENCODE_SINK_PTR std::shared_ptr<EncodeSinkBintr>
        
    #define DSL_FILE_SINK_PTR std::shared_ptr<FileSinkBintr>
    #define DSL_FILE_SINK_NEW(name, filepath, codec, container, bitrate, interval) \
        std::shared_ptr<FileSinkBintr>( \
        new FileSinkBintr(name, filepath, codec, container, bitrate, interval))
        
    #define DSL_RECORD_SINK_PTR std::shared_ptr<RecordSinkBintr>
    #define DSL_RECORD_SINK_NEW(name, outdir, codec, container, bitrate, interval, clientListener) \
        std::shared_ptr<RecordSinkBintr>( \
        new RecordSinkBintr(name, outdir, codec, container, bitrate, interval, clientListener))
        
    #define DSL_RTMP_SINK_PTR std::shared_ptr<RtmpSinkBintr>
    #define DSL_RTMP_SINK_NEW(name, uri, bitrate, interval) \
        std::shared_ptr<RtmpSinkBintr>( \
        new RtmpSinkBintr(name, uri, bitrate, interval))

    #define DSL_RTSP_SERVER_SINK_PTR std::shared_ptr<RtspServerSinkBintr>
    #define DSL_RTSP_SERVER_SINK_NEW(name, host, udpPort, rtspPort, codec, bitrate, interval) \
        std::shared_ptr<RtspServerSinkBintr>( \
        new RtspServerSinkBintr(name, host, udpPort, rtspPort, codec, bitrate, interval))

    #define DSL_RTSP_CLIENT_SINK_PTR std::shared_ptr<RtspClientSinkBintr>
    #define DSL_RTSP_CLIENT_SINK_NEW(name, uri, codec, bitrate, interval) \
        std::shared_ptr<RtspClientSinkBintr>( \
        new RtspClientSinkBintr(name, uri, codec, bitrate, interval))
                
    #define DSL_MESSAGE_SINK_PTR std::shared_ptr<MessageSinkBintr>
    #define DSL_MESSAGE_SINK_NEW(name, \
            converterConfigFile, payloadType, brokerConfigFile, \
            protocolLib, connectionString, topic) \
        std::shared_ptr<MessageSinkBintr>(new MessageSinkBintr(name, \
            converterConfigFile, payloadType, brokerConfigFile, \
            protocolLib, connectionString, topic))

    #define DSL_LIVEKIT_WEBRTC_SINK_PTR std::shared_ptr<LiveKitWebRtcSinkBintr>
    #define DSL_LIVEKIT_WEBRTC_SINK_NEW(name, \
            url, apiKey, secretKey, room, identity, participant) \
        std::shared_ptr<LiveKitWebRtcSinkBintr>(new LiveKitWebRtcSinkBintr(name, \
            url, apiKey, secretKey, room, identity, participant))

    #define DSL_INTERPIPE_SINK_PTR std::shared_ptr<InterpipeSinkBintr>
    #define DSL_INTERPIPE_SINK_NEW(name, forwardEos, forwardEvents) \
        std::shared_ptr<InterpipeSinkBintr>( \
        new InterpipeSinkBintr(name, forwardEos, forwardEvents))

    #define DSL_MULTI_IMAGE_SINK_PTR std::shared_ptr<MultiImageSinkBintr>
    #define DSL_MULTI_IMAGE_SINK_NEW(name, filepath, width, height, fps_n, fps_d) \
        std::shared_ptr<MultiImageSinkBintr>( \
        new MultiImageSinkBintr(name, filepath, width, height, fps_n, fps_d))

    #define DSL_V4L2_SINK_PTR std::shared_ptr<V4l2SinkBintr>
    #define DSL_V4L2_SINK_NEW(name, deviceLocation) \
        std::shared_ptr<V4l2SinkBintr>( \
        new V4l2SinkBintr(name, deviceLocation))

    //-------------------------------------------------------------------------

    class SinkBintr : public QBintr
    {
    public: 
    
        SinkBintr(const char* name);

        ~SinkBintr();
  
        /**
         * @brief adds this SinkBintr to a parent Branch/Pipeline bintr
         * @param[in] pParentBintr parent bintr to add this sink to
         * @return true on successful add, false otherwise
         */
        bool AddToParent(DSL_BASE_PTR pParentBintr);

        /**
         * @brief checks if a Bintr is the a parent Branch/Pipeline bintr of this sink
         * @param[in] pParentBintr parent bintr to check
         */
        bool IsParent(DSL_BASE_PTR pParentBintr);
        
        /**
         * @brief removes this SinkBintr from a parent Branch/Pipeline bintr.
         * @param[in] pParentBintr parent bintr to remove this sink from.
         * @return true on successful remove, false otherwise
         */
        bool RemoveFromParent(DSL_BASE_PTR pParentBintr);
        
        /**
         * @brief returns the current sync enabled property for the SinkBintr.
         * @return true if the sync property is enabled, false othewise.
         */
        virtual gboolean GetSyncEnabled();
        
        /**
         * @brief sets the sync enabled property for the SinkBintr.
         * @param[in] enabled new sync enabled property value.
         */
        virtual bool SetSyncEnabled(gboolean enabled);

        /**
         * @brief returns the current async enabled property value for the SinkBintr.
         * @return true if the async property is enabled, false othewise.
         */
        virtual gboolean GetAsyncEnabled();
        
        /**
         * @brief sets the async enabled property for the SinkBintr.
         * @param[in] enabled new async property value.
         */
        virtual bool SetAsyncEnabled(gboolean enabled);
        
        /**
         * @brief returns the current max-lateness property value for the SinkBintr.
         * @return current max-lateness (default = -1 unlimited).
         */
        virtual gint64 GetMaxLateness();
        
        /**
         * @brief sets the max-lateness property for the SinkBintr.
         * @param[in] maxLateness new max-lateness proprty value.
         */
        virtual bool SetMaxLateness(gint64 maxLateness);

        /**
         * @brief returns the current qos enabled property value for the SinkBintr.
         * @return true if the qos property is enabled, false othewise.
         */
        virtual gboolean GetQosEnabled();
        
        /**
         * @brief sets the qos enabled property for the SinkBintr.
         * @param[in] enabled new qos enabled property value.
         */
        virtual bool SetQosEnabled(gboolean enabled);

    protected:

        /**
         * @brief Device Properties, used for aarch64/x86_64 conditional logic.
         */
        cudaDeviceProp m_cudaDeviceProp;
        
        /**
         * @brief Sink element's current "sync" property setting.
         * Set to true to Sync on the clock.
         */
        gboolean m_sync;

        /**
         * @brief Sink element's current "async" property setting.
         * set to true to go asynchronously to PAUSED.
         */
        gboolean m_async;

        /**
         * @brief Sink element's current "max-lateness" property setting.
         * Maximum number of nanoseconds that a buffer can be late before it is .
         * dropped (-1 unlimited).
         */
        int64_t m_maxLateness; 

        /**
         * @brief Generate Quality-of-Service events upstream if true.
         */
        gboolean m_qos;
        
        /**
         * @brief Enable/disabled the last-sample property.
         */
        gboolean m_enableLastSample;
 
        /**
         * @brief Actual sink element specific to each Sink Bintr.
         */
        DSL_ELEMENT_PTR m_pSink;
    };

    //-------------------------------------------------------------------------

    class AppSinkBintr : public SinkBintr
    {
    public: 
    
        AppSinkBintr(const char* name, uint dataType, 
            dsl_sink_app_new_data_handler_cb clientHandler, void* clientData);

        ~AppSinkBintr();
  
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
         * @brief sets the sync enabled setting for the SinkBintr
         * @param[in] enabled current sync setting.
         */
        bool SetSyncEnabled(bool enabled);
        
        /**
         * @brief sets the async enabled setting for the SinkBintr
         * @param[in] enabled current sync setting.
         */
        bool SetAsyncEnabled(bool enabled);
        
        /**
         * @brief Handles the new sample on signal call and provides either
         * the sample or the contained buffer to the client by callback.
         * @return either GST_FLOW_OK, or GST_FLOW_EOS on no buffer available.
         */
        GstFlowReturn HandleNewSample();
        
        /**
         * @brief Gets the current data-type setting in use by this AppSinkBintr.
         * @return current data-type in use, either DSL_SINK_APP_DATA_TYPE_SAMPLE
         * or DSL_SINK_APP_DATA_TYPE_BUFFER.
         */
        uint GetDataType();
        
        /**
         * @brief Sets the data type to use for this AppSinkBintr.
         * @param[in] dataType either DSL_SINK_APP_DATA_TYPE_SAMPLE
         * or DSL_SINK_APP_DATA_TYPE_BUFFER.
         */
        void SetDataType(uint dataType);

    protected:
    
        /**
         * @brief opaque pointer to client data to return with the callback.
         * Protected (not private) to allow access by the FrameCaptureSinkBintr.
         */
        void* m_clientData;

    private:
    
        /**
         * @brief either DSL_SINK_APP_DATA_TYPE_SAMPLE or 
         * DSL_SINK_APP_DATA_TYPE_BUFFER
         */
        uint m_dataType;
    
        /**
         * @brief mutex to protect mutual access to the client-data-handler
         */
        DslMutex m_dataHandlerMutex;

        /**
         * @brief client callback function to be called with each new 
         * buffer available.
         */
        dsl_sink_app_new_data_handler_cb m_clientHandler; 
        
    };

    /**
     * @brief callback function registered with with the appsink's "new-sample" signal.
     * The callback wraps the AppSinkBintr's HandleNewSample function.
     * @param pSinkElement appsink element - not used.
     * @param pAppSinkBintr opaque pointer the the AppSinkBintr that triggered the 
     * "new-sample" signal - owner of the appsink element.
     * @return either GST_FLOW_OK, or GST_FLOW_EOS on no buffer available.
     */
    static GstFlowReturn on_new_sample_cb(GstElement* pSinkElement, 
        gpointer pAppSinkBintr);
        
    //-------------------------------------------------------------------------

    /**
     * @class FrameCaptureSinkBintr
     * @brief Implements a Frame-Capture Sink to encode and save a frame-buffer
     * to JPEG file on client invocation.
     */
    class FrameCaptureSinkBintr : public AppSinkBintr
    {
    public:
        
        /**
         * @brief ctor for the FrameCaptureSinkBintr
         * @param[in] name unique name for the FrameCaptureSinkBintr
         * @param[in] pCaptureFrameAction shared pointer to an ODE Capture Frame Action
         */
        FrameCaptureSinkBintr(const char* name, 
            DSL_BASE_PTR pFrameCaptureAction);
        
        /**
         * @brief dtor for the FrameCaptureSinkBintr
         */
        ~FrameCaptureSinkBintr();
        
        /**
         * @brief Function to initiate the capture of the next output
         * frame-buffer as provided by the base AppSinkBintr. 
         * @return false if the Sink is unlinked, true otherwise.
         */
        bool Initiate();

        /**
         * @brief Function to schedule a capture of a specific 
         * frame-buffer to be provided by the base AppSinkBintr. 
         * @param[in] frameNumber unique frame-number of the buffer to capture.
         * @return false if the Sink is unlinked or the frameNumber is invalid,
         * true otherwise.
         */
        bool Schedule(uint64_t frameNumber);

        /**
         * @brief Function to handle each new buffer provided by the AppSinkBintr.
         * @param[in] buffer new buffer to capture if m_captureNextBuffer == true.
         * @return GST_FLOW_OK always.
         */
        uint HandleNewBuffer(void* buffer);
        
    private:

        /**
         * @brief boolean flag used by the client to signal to the HandleNewBuffer
         * function to capture the next frame-buffer.
         */
        bool m_captureNextBuffer;
        
        /**
         * @brief queue of scheduled frame-numbers to capture.
         */
        std::queue<uint64_t> m_captureFrameNumbers;

        /**
         * @brief mutex to protect mutual access to the Sink's capture control 
         * variables.
         */
        DslMutex m_captureMutex;

        /**
         * @brief Shared pointer to a Frame Capture Action.
         * to selectively call to capture the current buffer
         */
        DSL_BASE_PTR m_pFrameCaptureAction;
    };

    /**
     * @brief callback function registered with with the base AppSinkBintr.
     * The callback wraps the FrameCaptureSinkBintr's HandleNewBuffer function.
     * @param[in] buffer new GstBuffer with metadata to process.
     * @param[in] client_data this pointer to the FrameCaptureSinkBintr instance.
     * @return either GST_FLOW_OK, or GST_FLOW_EOS on no buffer available.
     */
    static uint on_new_buffer_cb(uint data_type, 
        void* buffer, void* client_data);    

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
        
    };

    //-------------------------------------------------------------------------

    class WindowSinkBintr : public SinkBintr
    {
    public: 
    
        WindowSinkBintr(const char* name, 
            uint offsetX, uint offsetY, uint width, uint height);

        ~WindowSinkBintr();
        
        /**
         * @brief Gets the current X and Y offset settings for this WindowSinkBintr
         * @param[out] offsetX the current offset in the X direction in pixels
         * @param[out] offsetY the current offset in the Y direction in pixels
         */ 
        void GetOffsets(uint* offsetX, uint* offsetY);

        /**
         * @brief Sets the current X and Y offset settings for this RednerSinkBintr
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
         * @return false if the sink is currently Linked. True otherwise
         */ 
        bool SetDimensions(uint width, uint hieght);
        
        /**
         * @brief Resets the Sink element for this WindowSinkBintr
         * @return false if the sink is currently Linked. True otherwise
         * IMPORTANT! this is now only used by the EGL Window Sink. 
         */
        virtual bool Reset(){return true;};

        /**
         * @brief Gets the current full-screen-enabled setting for the WindowSinkBintr
         * @retrun true if full-screen-mode is currently enabled, false otherwise
         */
        bool GetFullScreenEnabled();
        
        /**
         * @brief Sets the full-screen-enabled setting for the WindowSinkBintr
         * @param enabled if true, sets the window to full-screen on creation
         * @return true if the full-screen-enabled could be set, 
         * false if called after XWindow creation
         */
        bool SetFullScreenEnabled(bool enabled);

        /**
         * @brief Adds a callback to be notified on display/window KeyRelease event
         * @param[in] handler pointer to the client's function to add
         * @param[in] clientData opaque to client data passed back to the handler.
         * @return true on successful add, false otherwise.
         */
        bool AddKeyEventHandler(dsl_sink_window_key_event_handler_cb handler, 
            void* clientData);

        /**
         * @brief removes a callback previously added with AddKeyEventHandler
         * @param[in] handler pointer to the client's function to remove
         * @return true on successful remove, false otherwise.
         */
        bool RemoveKeyEventHandler(dsl_sink_window_key_event_handler_cb handler);
            
        /**
         * @brief adds a callback to be notified on display/window ButtonPress event.
         * @param[in] handler pointer to the client's function to add
         * @param[in] clientData opaque to client data passed back to the handler.
         * @return true on successful add, false otherwise.
         */
        bool AddButtonEventHandler(dsl_sink_window_button_event_handler_cb handler, 
            void* clientData);

        /**
         * @brief removes a previously added callback
         * @param[in] handler pointer to the client's function to remove
         * @return true on successful remove, false otherwise.
         */
        bool RemoveButtonEventHandler(dsl_sink_window_button_event_handler_cb handler);
        /**
         * @brief adds a callback to be notified on display/window delete event.
         * @param[in] handler pointer to the client's function to add
         * @param[in] clientData opaque to client data passed back to the handler.
         * @return true on successful add, false otherwise.
         */
        bool AddDeleteEventHandler(dsl_sink_window_delete_event_handler_cb handler, 
            void* clientData);

        /**
         * @brief removes a previously added callback
         * @param[in] handler pointer to the client's function to remove
         * @return true on successful remove, false otherwise.
         */
        bool RemoveDeleteEventHandler(dsl_sink_window_delete_event_handler_cb handler);
        
        /**
         * @brief handles incoming window KEY & BUTTON events by calling
         * all client installed event handlers for each queued event.
         */
        void HandleXWindowEvents();

        /**
         * @brief Creates a new XWindow for the current XDisplay
         * @param[in] pSharedClientCbMutex shared pointer to a shared
         * mutex - to use when calling XWindow client callbacks.
         * @return true if successfully created, false otherwise.
         * This call will fail if the client has already provided
         * a Window handle for the WindowSinkBintr to use.
         */
        bool PrepareWindowHandle(std::shared_ptr<DslMutex> pSharedClientCbMutex);
        
        /**
         * @brief Determines if the WindowSinkBintr has an XWindow whether
         * provided by the client or created with a call to CreateXWindow()
         * @return true if the WindowSinkBintr has a Window handle.
         */
        bool HasXWindow();
        
        /**
         * @brief queries the WindowSinkBintr to determine if it owns an xwindow
         * @return true if the WindowSinkBintr has ownership of an xwindow, 
         * false otherwise.
         */
        bool OwnsXWindow();
        
        /**
         * @brief returns a handle to this WindowSinkBintr's XWindow
         * @return XWindow handle, NULL untill created
         */
        Window GetHandle();
        
        /**
         * @brief Sets the WindowSinkBintr's XWindow handle. The Pipeline
         * must be in an unlinked state to change XWindow handles. 
         * @return true on successful clear, false otherwise
         */
        bool SetHandle(Window handle);
        
        /**
         * @brief Clears the WindowSinkBintr's XWindow buffer
         * @return true on successful clear, false otherwise..
         */
        bool Clear();
        

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
        
        /**
         * @brief Creates a new XWindow for the current XDisplay
         * @param[in] 
         * @return true if successfully created, false otherwise.
         * This call will fail if the client has already provided
         * a Window handle for the WindowSinkBintr to use.
         */
        bool CreateXWindow();
        
        /**
         * @brief map of all currently registered XWindow-key-event-handlers
         * callback functions mapped with the user provided data
         */
        std::map<dsl_sink_window_key_event_handler_cb, void*> 
            m_xWindowKeyEventHandlers;

        /**
         * @brief map of all currently registered XWindow-button-event-handlers
         * callback functions mapped with the user provided data
         */
        std::map<dsl_sink_window_button_event_handler_cb, void*> 
            m_xWindowButtonEventHandlers;

        /**
         * @brief map of all currently registered XWindow-delete-event-handlers
         * callback functions mapped with the user provided data
         */
        std::map<dsl_sink_window_delete_event_handler_cb, void*> 
            m_xWindowDeleteEventHandlers;
        
        /**
         * @brief Pointer to the XDisplay opened in CreateXWindow().
         */
        Display* m_pXDisplay;
        
        /**
         * @brief Width of the XDisplay's default screen.
         */
        uint m_XDisplayWidth;
        
        /**
         * @brief Height of the XDisplay's default screen.
         */
        uint m_XDisplayHeight;
        
        /**
         * @brief Mutex to ensures mutual exclusion for the m_pXDisplay member
         * accessed by multiple threads.
         */
        DslMutex m_displayMutex;

        /**
         * @brief handle to X Window
         */
        Window m_pXWindow;
        
        /**
         * @brief Flag to determine if the XWindow was created or provided by
         * the client. The WindowSinkBitnr needs to delete the XWindow if created, 
         * but not the client's
         */
        bool m_pXWindowCreated;
        
        /**
         * @brief handle to the X Window event thread, 
         * active for the life of the Pipeline
         */
        GThread* m_pXWindowEventThread;        
        
        /**
         * @brief Mutex for display thread shared by all WindowSinkBintrs
         * currently linked to the same Pipeline. 
         */
        std::shared_ptr<DslMutex> m_pSharedClientCbMutex;
        
        /**
         * @brief if true, the WindowSinkPinter will set its XWindow to 
         * full-screen if one is created.
         */
        bool m_xWindowfullScreenEnabled;

    };

    static gpointer XWindowEventThread(gpointer pWindowSink);
    
    //-------------------------------------------------------------------------

    class ThreeDSinkBintr : public WindowSinkBintr
    {
    public: 
    
        ThreeDSinkBintr(const char* name, 
            uint offsetX, uint offsetY, uint width, uint height);

        ~ThreeDSinkBintr();

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

    };

    //-------------------------------------------------------------------------

    class EglSinkBintr : public WindowSinkBintr
    {
    public: 
    
        EglSinkBintr(const char* name, 
            guint offsetX, guint offsetY, guint width, guint height);

        ~EglSinkBintr();
  
        /**
         * @brief Resets the Sink element for this EglSinkBintr
         * @return false if the sink is currently Linked. True otherwise
         */
        bool Reset();
        
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
         * @brief Gets the current force-aspect-ratio setting for the EglSinkBintr.
         * @return true if forced, false otherwise.
         */
        bool GetForceAspectRatio();
        
        /**
         * @brief Set the force-aspect-ration setting for the EglSinkBintr.
         * @param[in] force set true to force-aspect-ration false otherwise
         * @return true if successfully set, false otherwise.
         */
        bool SetForceAspectRatio(bool force);

        /**
         * @brief Sets the GPU ID for all Elementrs - x86_64 builds only.
         * @return true if successfully set, false otherwise.
         */
        bool SetGpuId(uint gpuId);

        /**
         * @brief Sets the NVIDIA buffer memory type - x86_64 builds only.
         * @param[in] nvbufMemType new memory type to use, one of the 
         * DSL_NVBUF_MEM_TYPE constant values.
         * @return true if successfully set, false otherwise.
         */
        bool SetNvbufMemType(uint nvbufMemType);

    private:

        /**
         * @brief true if force-aspect-ratio is set, false otherwise. 
         */
        bool m_forceAspectRatio;

        /**
         * @brief Caps Filter required for dGPU EglSinkBintr
         */
        DSL_ELEMENT_PTR m_pCapsFilter;

        /**
         * @brief Platform specific Transform element EglSinkBintr
         */
        DSL_ELEMENT_PTR m_pTransform;
        
    };

    //-------------------------------------------------------------------------

    class EncodeSinkBintr : public SinkBintr
    {
    public: 
    
        EncodeSinkBintr(const char* name,
            uint codec, uint bitrate, uint interval);

        /**
         * @brief Gets the current bit-rate and interval settings in use by the 
         * EncoderSinkBintr
         * @param[out] code the currect codec in used
         * @param[out] bitrate the current bit-rate setting in use by the encoder
         * @param[out] interval the current encode-interval in use by the encoder
         */ 
        void GetEncoderSettings(uint* codec, uint* bitrate, uint* interval);

        /**
         * @brief Sets the bit-rate and interval settings for the EncoderSinkBintr 
         * to use.
         * @param[in] codec the new code to use, either DSL_CODEC_H264 or 
         * DSL_CODE_H265
         * @param[in] bitrate the new bit-rate setting in uints of bits/sec
         * @param[in] interval the new encode-interval setting to use
         * @return true if the settings were set successfully, false otherwise
         */ 
        bool SetEncoderSettings(uint codec, uint bitrate, uint interval);

        /**
         * @brief Gets the current width and height settings for 
         * the EncodeSinkBintr's video converter element.
         * @param[out] width the current width setting in pixels.
         * @param[out] height the current height setting in pixels.
         */ 
        void GetConverterDimensions(uint* width, uint* height);
        
        /**
         * @brief Sets the width and height settings for the EncodeSinkBintr's 
         * video converter elementto use. The caller is required to provide valid 
         * width and height values.
         * @param[in] width the width value to set in pixels. Set to 0 for 
         * no scaling.
         * @param[in] height the height value to set in pixels. Set to 0 for 
         * no scaling.
         * @return true if the video converter dimensions were set successfully, 
         * false otherwise
         */ 
        bool SetConverterDimensions(uint width, uint hieght);

        /**
         * @brief Sets the GPU ID for all Elementrs
         * @return true if successfully set, false otherwise.
         */
        bool SetGpuId(uint gpuId);
        
    protected:

        /**
         * @brief Current codec id for the EncodeSinkBintr
         */
        uint m_codec;

        /**
         * @brief Current bitrate for the EncodeSinkBintr. 0 = use default
         */
        uint m_bitrate;
        
        /**
         * @brief Default bitrate for the EncodeSinkBintr.
         */
        uint m_defaultBitrate;
        
        /**
         * @brief Decode interval - other frames are dropped
         */
        uint m_interval;
 
        /**
         * @brief Width property for the Video Converter element in uints of pixels.
         */
        uint m_width;

        /**
         * @brief Height property for the Video Converter element in uints of pixels.
         */
        uint m_height;

        /**
         * @brief Transform (video converter) element for the EncodeSinkBintr
         */
        DSL_ELEMENT_PTR m_pTransform;

        /**
         * @brief Caps Filter element for the EncodeSinkBintr
         */
        DSL_ELEMENT_PTR m_pCapsFilter;

        /**
         * @brief Encoder element for the EncodeSinkBintr
         */
        DSL_ELEMENT_PTR m_pEncoder;

        /**
         * @brief Parser element for the EncodeSinkBintr
         */
        DSL_ELEMENT_PTR m_pParser;
    };

    //-------------------------------------------------------------------------

    class FileSinkBintr : public EncodeSinkBintr
    {
    public: 
    
        FileSinkBintr(const char* name, const char* filepath, 
            uint codec, uint container, uint bitrate, uint interval);

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

        uint m_container;

        DSL_ELEMENT_PTR m_pContainer;       
    };

    //-------------------------------------------------------------------------

    class RecordSinkBintr : public EncodeSinkBintr, public RecordMgr
    {
    public: 
    
        RecordSinkBintr(const char* name, const char* outdir, uint codec, uint container, 
            uint bitrate, uint interval, dsl_record_client_listener_cb clientListener);

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
         * @brief sets the sync enabled setting for the SinkBintr
         * @param[in] enabled current sync setting.
         */
        bool SetSyncEnabled(bool enabled);

    private:

        /**
         * @brief Node to wrap NVIDIA's Record Bin
         */
        DSL_NODETR_PTR m_pRecordBin;

    };

    //-------------------------------------------------------------------------

    class RtmpSinkBintr : public EncodeSinkBintr
    {
    public: 
    
        RtmpSinkBintr(const char* name, 
            const char* uri, uint bitrate, uint interval);

        ~RtmpSinkBintr();
  
        /**
         * @brief Links all Child Elementrs owned by this Bintr
         * @return true if all links were succesful, false otherwise.
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elemntrs owned by this Bintr
         * Calling UnlinkAll when in an unlinked state has no effect.
         */
        void UnlinkAll();
        
        /**
         * @brief returns the current URI (location) for this RtmpSinkBintr.
         * @return const string for the current URI.
         */
        const char* GetUri();
        
        /**
         * @brief Sets the URI (location)for this RtmpSinkBintr.
         * @param uri new URI for the RtmpSinkBintr to use.
         * @return true on successful update, false otherwise
         */
        bool SetUri(const char* uri);

    private:

        /**
         * @brief RTMP URI to stream to.
         */
        std::string m_uri;

        /**
         * @brief flvmux to convert the stream from video/x-h264 to video/x-flv.
         */
        DSL_ELEMENT_PTR m_pFlvmux;
        
        /**
         * @brief rtph264pay plugin for the RTMP SInk
         */ 
        DSL_ELEMENT_PTR m_pPayloader;
        
    };

    //-------------------------------------------------------------------------

    class RtspServerSinkBintr : public EncodeSinkBintr
    {
    public: 
    
        RtspServerSinkBintr(const char* name, 
            const char* host, uint udpPort, uint rtspPort,
            uint codec, uint bitrate, uint interval);

        ~RtspServerSinkBintr();
  
        /**
         * @brief Links all Child Elementrs owned by this Bintr
         * @return true if all links were succesful, false otherwise.
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elemntrs owned by this Bintr
         * Calling UnlinkAll when in an unlinked state has no effect.
         */
        void UnlinkAll();

        /**
         * @brief Gets the current codec and media container formats for the 
         * RtspServerSinkBintr.
         * @param[out] port the current UDP port number for the RTSP Server.
         * @param[out] port the current RTSP port number for the RTSP Server.
         */ 
        void GetServerSettings(uint* udpPort, uint* rtspPort);

    private:

        std::string m_host;
        uint m_udpPort;
        uint m_rtspPort;
        
        /**
         * @brief current UPD-buffer-size for the RTSP Media Factory
         */
        uint m_udpBufferSize;

        /**
         * @brief string representing current codec; "H264" or "H265"
         */
        std::string m_codecString;

        GstRTSPServer* m_pServer;
        uint m_pServerSrcId;
        GstRTSPMediaFactory* m_pFactory;
 
        DSL_ELEMENT_PTR m_pPayloader;
    };

    //-------------------------------------------------------------------------

    class RtspClientSinkBintr : public EncodeSinkBintr
    {
    public: 
    
        RtspClientSinkBintr(const char* name, const char* uri, 
            uint codec, uint bitrate, uint interval);

        ~RtspClientSinkBintr();
  
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
         * @brief Sets the client credentials for the RtspClientSinkBintr.
         * @param[in] userId client user-id to use for credentials
         * @param[in] userPw client user-password to use for credentials
         * @return true if successfully set, false otherwise.
         */
        bool SetCredentials(const char* userId, const char* userPw);
        
        /**
         * @brief Gets the current latency setting for the RtspClientSinkBintr.
         * @return latency in units of ms.
         */
        uint GetLatency();
        
        /**
         * @brief Sets the latency setting for the RtspClientSinkBintr.
         * @param latency new latency setting in units of ms.
         * @return true if successfully set, false otherwise.
         */
        bool SetLatency(uint latency);
        
        /**
         * @brief Gets the current RTSP Profiles for the RtspClientSinkBintr.
         * @return mask of DSL_RTSP_PROFILE constants. 
         * Default = DSL_RTSP_PROFILE_AVP.
         */
        uint GetProfiles();
        
        /**
         * @brief Sets the RTSP Profiles for the RtspClientSinkBintr to use.
         * @param[in] profiles mask of DSL_RTSP_PROFILE constants. 
         * @return true on successful set, false otherwise.
         */
        bool SetProfiles(uint profiles);
        
        /**
         * @brief Gets the current allowed RTSP lower-protocols for the 
         * RtspClientSinkBintr.
         * @return mask of DSL_RTSP_LOWER_TRANS constant values. 
         * Default = DSL_RTSP_LOWER_TRANS_TCP + DSL_RTSP_LOWER_TRANS_UDP_MCAST +
         * DSL_RTSP_LOWER_TRANS_UDP.
         */
        uint GetProtocols();
        
        /**
         * @brief Sets the allowed RTSP lower-protocols for the RtspClientSinkBintr 
         * to use.
         * @param[in] protocols mask of DSL_RTSP_LOWER_TRANS constant values. 
         * @return true on successful set, false otherwise.
         */
        bool SetProtocols(uint protocols);
        
        /**
         * @brief Gets the current tls-validation-flags for the RtspClientSinkBintr.
         * @return mask of DSL_TLS_CERTIFICATE constants. 
         * Default = DSL_TLS_CERTIFICATE_VALIDATE_ALL.
         */
        uint GetTlsValidationFlags();
        
        /**
         * @brief Sets the tls-validation-flags for the RtspClientSinkBintr to use.
         * @param[in] flags mask of DSL_TLS_CERTIFICATE constants. 
         * @return true on successful set, false otherwise.
         */
        bool SetTlsValidationFlags(uint flags);
        
    private:
    
        /**
         * @brief Amount of data to buffer in ms for this RtspClientSinkBintr.
         */
        uint m_latency;
        
        /**
         * @brief mask of currently allowed RTSP profiles for this 
         * RtspClientSinkBintr.
         */
        uint m_profiles;
        
        /**
         * @brief mask of currently allowed RTSP lower-protocols for this 
         * RtspClientSinkBintr.
         */
        uint m_protocols;
        
        /**
         * @brief mask of DSL_TLS_CERTIFICATE flags used to validate the
         * RTSP server certificate.
         */
        uint m_tlsValidationFlags;

    };


    //-------------------------------------------------------------------------

    /**
     * @class MessageSinkBintr 
     * @brief Implements a Message Sink Bin Container Class (Bintr)
     */
    class MessageSinkBintr : public SinkBintr
    {
    public: 
    
        /**
         * @brief Ctor for the MessageSinkBintr class
         */
        MessageSinkBintr(const char* name, const char* converterConfigFile, 
        uint payloadType, const char* brokerConfigFile, const char* protocolLib, 
        const char* connectionString, const char* topic);

        /**
         * @brief Dtor for the MessageSinkBintr class
         */
        ~MessageSinkBintr();
  
        /**
         * @brief Links all Child Elementrs owned by this MessageSinkBintr
         * @return true if all links were successful, false otherwise
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elemntrs owned by this MessageSinkBintr
         * Calling UnlinkAll when in an unlinked state has no effect.
         */
        void UnlinkAll();

        /**
         * @brief Gets the current base_meta.meta_type filter in use by 
         * the MessageSinkBintr.
         * @return the current meta-type id in use, default = NVDS_EVENT_MSG_META.
         */
        uint GetMetaType();

        /**
         * @brief Sets the base_meta.meta_type filter for the MessageSinkBintr to use.
         * @param[in] metaType new meta-type id to use, must be >= NVDS_START_USER_META
         * or = NVDS_EVENT_MSG_META.
         * @return true on successful update, false otherwise.
         */
        bool SetMetaType(uint metaType);
        
        /**
         * @brief Gets the current message converter settings for the MessageSinkBintr.
         * @param[out] converterConfigFile absolute file-path to the current
         * message converter config file in use.
         * @param[out] payloadType current payload type setting.
         */
        void GetConverterSettings(const char** converterConfigFile,
            uint* payloadType);
            
        /**
         * @brief Sets the current message converter settings for the MessageSinkBintr.
         * @param[in] converterConfigFile absolute or relate file-path to a new
         * message converter config file to use.
         * @param[in] payloadType new payload type setting to use.
         * @return true if successful, false otherwise.
         */
        bool SetConverterSettings(const char* converterConfigFile,
            uint payloadType);

        /**
         * @brief Gets the current message broker settings for the MsgSinBintr.
         * @param[out] brokerConfigFile absolute file-path to the current message
         * borker config file in use.
         * @param[out] protocolLib current protocol adapter library in use
         * @param[out] connectionString current connection string in use.
         * @param[out] topic (optional) message topic current in use.
         */
        void GetBrokerSettings(const char** brokerConfigFile, const char** protocolLib,
            const char** connectionString, const char** topic);

        /**
         * @brief Sets the message broker settings for the MsgSinBintr.
         * @param[in] brokerConfigFile absolute or relative file-path to 
         * a new message borker config file to use.
         * @param[in] protocolLib new protocol adapter library to use.
         * @param[in] connectionString new connection string to use.
         * @param[in] topic (optional) new message topic to use.
         * @return true if successful, false otherwise.
         */
        bool SetBrokerSettings(const char* brokerConfigFile, const char* protocolLib, 
            const char* connectionString, const char* topic);

        /**
         * @brief Gets the current payload-debug-directory.
         * @return current payload-debug-directory. Null if unset.
         */
        const char* GetDebugDir();
        
        /**
         * @brief Sets the current payload-debug-directory.
         * @param[in] debugDir new payload-debug-directory to use.
         * @return true if successful, false otherwise.
         */
        bool SetDebugDir(const char* debugDir);

    private:

        /**
         * @brief defines the base_meta.meta_type id filter to use for
         * all message meta to convert and s IN. Default = NVDS_EVENT_MSG_META.
         * Custom values must be greater than NVDS_START_USER_META
         * Both constants are defined in nvdsmeta.h 
         */
        uint m_metaType;

        /**
         * @brief absolute path to the message converter config file is use.
         */
        std::string m_converterConfigFile;
        
        /**
         * @brief payload type, one of the DSL_MSG_PAYLOAD_<*> constants 
         */
        uint m_payloadType; 
        
        /**
         * @brief absolute path to the message broker config file in use.
         */
        std::string m_brokerConfigFile; 
        
        /**
         * @brief connection string used as end-point for communication with server.
         */
        std::string m_connectionString;
        
        /**
         * @brief Absolute pathname to the library that contains the protocol adapter.
         */
        std::string m_protocolLib; 
        
        /**
         * @brief (optional) message topic name.
         */
        std::string m_topic;
    
        /**
         * @brief Directory to dump payload-debug-files into.
         */
        std::string m_debugDir;
    
        /**
         * @brief NVIDIA message-converter element for this MessageSinkBintr 
         */
        DSL_ELEMENT_PTR m_pMsgConverter;

    };

    //-------------------------------------------------------------------------

    /**
     * @class LiveKitWebRtcSinkBintr 
     * @brief Implements a Live Kit WebRTC Sink Bin Container Class (Bintr)
     */
    class LiveKitWebRtcSinkBintr : public SinkBintr
    {
    public: 
    
        /**
         * @brief Ctor for the LiveKitWebRtcSinkBintr class
         */
        LiveKitWebRtcSinkBintr(const char* name, const char* url, 
        const char* apiKey, const char* secretKey, const char* room, 
        const char* identity, const char* participant);

        /**
         * @brief Dtor for the LiveKitWebRtcSinkBintr class
         */
        ~LiveKitWebRtcSinkBintr();
  
        /**
         * @brief Links all Child Elementrs owned by this LiveKitWebRtcSinkBintr
         * @return true if all links were successful, false otherwise
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elemntrs owned by this LiveKitWebRtcSinkBintr
         * Calling UnlinkAll when in an unlinked state has no effect.
         */
        void UnlinkAll();

    private:

        /**
         * @brief LiveKit URL to publish the stream to.
         */
        std::string m_url;
        
        /**
         * @brief LiveKit API Key required to connect.
         */
        std::string  m_apiKey; 
        
        /**
         * @brief LiveKit Secret Key required to connect.
         */
        std::string m_secretKey; 
        
        /**
         * @brief connection string used as end-point for communication with server.
         */
        std::string m_room;
        
        /**
         * @brief Absolute pathname to the library that contains the protocol adapter.
         */
        std::string m_identity; 
        
        /**
         * @brief (optional) message topic name.
         */
        std::string m_participant;
    
        /**
         * @brief LiveKit WebRTC plugin for LiveKitWebRtcSinkBintr 
         */
        DSL_ELEMENT_PTR m_pSink;

    };

    //-------------------------------------------------------------------------

    class InterpipeSinkBintr : public SinkBintr
    {
    public: 
    
        InterpipeSinkBintr(const char* name, 
            bool forwardEos, bool forwardEvents);

        ~InterpipeSinkBintr();
  
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
         * @brief Gets the current forward settings for this SinkBintr 
         * @param[out] forwardEos if true, EOS event will be forwarded to 
         * all listeners. 
         * @param[out] forwardEvents if true, downstream events (except for 
         * EOS) will be forwarded to all listeners.
         */
        void GetForwardSettings(bool* forwardEos, bool* forwardEvents);

        /**
         * @brief Gets the current forward settings for this SinkBintr 
         * @param[in] forwardEos set to true to forward EOS event to 
         * all listeners, false otherwise. 
         * @param[in] forwardEvents set to true to forward downstream events
         * (except for EOS) to all listeners, false otherwise.
         * @returns ture on succesful update, false otherwise.
         */
        bool SetForwardSettings(bool forwardEos, bool forwardEvents);
        
        /**
         * @brief Gets the current numer of Inter-Pipe Sources listening
         * to this SinkBintr.
         * @return number of Sources currently listening.
         */
        uint GetNumListeners();

        /**
         * @brief sets the sync enabled setting for the SinkBintr
         * @param[in] enabled current sync setting.
         */
        bool SetSyncEnabled(bool enabled);

    private:
    
        /**
         * @brief forward the EOS event to all the listeners if true
         */
        bool m_forwardEos;
        
        /**
         * @brief forward downstream events to all the listeners 
         * (except for EOS) if true.
         */
        bool m_forwardEvents;

    };

    //-------------------------------------------------------------------------

    class MultiImageSinkBintr : public SinkBintr
    {
    public: 
    
        MultiImageSinkBintr(const char* name, const char* filepath,
            uint width, uint height, uint fps_n, uint fps_d);

        ~MultiImageSinkBintr();
  
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
         * @brief Sets the file-path for MultiImageSinkBintr 
         * @return Current filepath set for the MultiImageSinkBintr
         */
        const char* GetFilePath();

        /**
         * @brief Sets the file-path for MultiImageSinkBintr 
         * @param[in] filepath printf style %d in an absolute or relative path. 
         * Eample: "./my_images/image.%d04.jpg", will create files in "./my_images/"
         * named "image.0000.jpg", "image.0001.jpg", "image.0002.jpg" etc.
         */
        bool SetFilePath(const char* filepath);

        /**
         * @brief Gets the current width and height settings for this 
         * MultiImageSinkBintr.
         * @param[out] width the current width setting in pixels.
         * @param[out] height the current height setting in pixels.
         */ 
        void GetDimensions(uint* width, uint* height);
        
        /**
         * @brief Sets the width and height settings for the MultiImageSinkBintr
         * to use. The caller is required to provide valid width and height values.
         * @param[in] width the width value to set in pixels. Set to 0 for 
         * no transcode.
         * @param[in] height the height value to set in pixels. Set to 0 for 
         * no transcode.
         * @return false if the sink is currently Linked. True otherwise
         */ 
        bool SetDimensions(uint width, uint hieght);

        /**
         * @brief Gets the current frame-rate setting for this 
         * MultiImageSinkBintr.
         * @param[out] fpsN the current frames/second numerator.
         * @param[out] fpsD the current frames/second denominator.
         */ 
        void GetFrameRate(uint* fpsN, uint* fpsD);
        
        /**
         * @brief Sets the width and height settings for the MultiImageSinkBintr
         * to use. The caller is required to provide valid width and height values.
         * @param[in] fpsN the new frames/second numerator.
         * @param[in] fpsD the new frames/second denominator.
         * @return false if the sink is currently Linked. True otherwise
         */ 
        bool SetFrameRate(uint fpsN, uint fpsD);
        
        /**
         * @brief Gets the current max-files setting for the MultiImageSinkBintr.
         * @return current max-files setting. 0 indicates no maximum.
         */
        uint GetMaxFiles();
        
        /**
         * @brief Sets the max-files setting for the MultiImageSinkBintr.
         * Set to 0 for no maximum.
         * @param[in] max new max for the max-files setting. 
         * @return false if the sink is currently Linked. True otherwise.
         */
        bool SetMaxFiles(uint max);

        /**
         * @brief sets the sync enabled setting for the SinkBintr
         * @param[in] enabled current sync setting.
         */
        bool SetSyncEnabled(bool enabled);
        
    private:

        /**
         * @brief Sets the Caps Filter settings using the current m_width,
         * m_height, m_fpsN, and m_fpsD member variables.
         * @return 
         */
        bool setCaps();
        
        /**
         * @brief Current output filepath ("location") for the MultiImageSinkBintr.
         */
        std::string m_filepath;

        /**
         * @brief Width property for the Video Converter element in uints of pixels.
         */
        uint m_width;

        /**
         * @brief Height property for the Video Converter element in uints of pixels.
         */
        uint m_height;

        /**
         * @brief Frames/second numerator for the Video Rate element.
         */
        uint m_fpsN;

        /**
         * @brief Frames/second denominator for the Video Rate element.
         */
        uint m_fpsD;
        
        /** 
         * @brief maximum number of files to keep on disk. Once the maximum is 
         * reached, old files start to be deleted to make room for new ones.
         */
        uint m_maxFiles;

        /**
         * @brief Video Converter element for the MultiImageSinkBintr.
         */
        DSL_ELEMENT_PTR m_pVideoConv;

        /**
         * @brief Video Rate element for the MultiImageSinkBintr.
         */
        DSL_ELEMENT_PTR m_pVideoRate;

        /**
         * @brief Caps Filter element for the MultiImageSinkBintr.
         */
        DSL_ELEMENT_PTR m_pCapsFilter;

        /**
         * @brief JPEG Encoder element for the MultiImageSinkBintr.
         */
        DSL_ELEMENT_PTR m_pJpegEnc;
    };

    //-------------------------------------------------------------------------

    class V4l2SinkBintr : public SinkBintr
    {
    public: 
    
        V4l2SinkBintr(const char* name, const char* deviceLocation);

        ~V4l2SinkBintr();
  
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
         * @brief Gets the current device-location setting for the V4l2SinkBintr
         * @return current device location.
         */
        const char* GetDeviceLocation();
        
        /**
         * @brief Sets the device-location setting for the V4l2SinkBintr.
         * @param[in] new device location for the V4l2SinkBintr to use.
         * @return true if successfully set, false otherwise.
         */
        bool SetDeviceLocation(const char* deviceLocation);

        /**
         * @brief Gets the current device-name setting for the V4l2SinkBintr
         * Default = "". Updated after negotiation with the V4L2 Device.
         * @return current device location.
         */
        const char* GetDeviceName();
        
        /**
         * @brief Gets the current device-fd (file-descriptor) setting for 
         * the V4l2SinkBintr. Default = -1 (unset). Updated at runtime after
         * negotiation with the V4l2 Device.
         * @return current device location.
         */
        int GetDeviceFd();
        
        /**
         * @brief Gets the current device-flags setting for the V4l2SinkBintr. 
         * Default = 0 (none). Updated at runtime after negotiation with the 
         * V4l2 Device.
         * @return current device location.
         */
        uint GetDeviceFlags();
        
        /**
         * @brief Gets the current buffer-in-format for this V4l2SinkBintr.
         * @return Current buffer-in-format. string version of one of the 
         * DSL_VIDEO_FORMAT constants.
         */
        const char* GetBufferInFormat();
        
        /**
         * @brief Sets the buffer-in-format for the V4l2SinkBintr.
         * @param[in] format string version of one of the DSL_VIDEO_FORMAT constants.
         * @return true if successfully set, false otherwise.
         */
        bool SetBufferInFormat(const char* format);

        /**
         * @brief Gets the current picture settings for this V4l2SinkBintr.
         * @param[out] brightness current brightness (actually darkness) level.
         * @param[out] contrast current picture contrast or luna gain level.
         * @param[out] saturation current color saturation or chroma gain level.
         */
        void GetPictureSettings(int* brightness, int* contrast, int* saturation);
        
        /**
         * @brief Sets the picture settings for the V4l2SinkBintr to use.
         * @param[in] brightness new brightness (actually darkness) level.
         * @param[in] contrast new picture contrast or luna level.
         * @param[in] saturation new color saturation or chroma level.
         * @return true if successfully set, false otherwise.
         */
        bool SetPictureSettings(int brightness, int contrast, int saturation);

    private:

        /**
         * @brief Device location string for this V4l2SinkBintr.
         */
        std::string m_deviceLocation;

        /**
         * @brief Device name string for this V4l2SinkBintr. Default size=0
         */
        std::string m_deviceName;
        
        /**
         * @brief Device file-descriptor for this V4l2SinkBintr. Default = -1
         */
        int m_deviceFd;
        
        /**
         * @brief Device type-flags for this V4l2SinkBintr. 
         * Default = DSL_V4L2_DEVICE_TYPE_NONE
         */
        uint m_deviceFlags;

        /**
         * @brief Buffer format to input into the v4l2sink plugin.
         */
        std::string m_bufferInFormat;
        
        /**
         * @brief Picture brightness level, or more accurately, darkness level. 
         */
        int m_brightness;
        
        /**
         * @brief Picture contrast level, or luma. 
         */
        int m_contrast;
        
        /**
         * @brief Picture color saturation level, or chroma. 
         */
        int m_saturation;
        
        /**
         * @brief Identity module for the  V4l2SinkBintr required to workaround 
         * a bug in the v4l2loopback.
         */
        DSL_ELEMENT_PTR m_pIdentity;

        /**
         * @brief Caps Filter required for for the V4l2SinkBintr.
         */
        DSL_ELEMENT_PTR m_pCapsFilter;

        /**
         * @brief Video converter required for the V4l2SinkBintr.
         */
        DSL_ELEMENT_PTR m_pTransform;
  
    };

}

#endif // _DSL_SINK_BINTR_H
    
