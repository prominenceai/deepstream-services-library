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

namespace DSL
{
    #define DSL_SINK_PTR std::shared_ptr<SinkBintr>

    #define DSL_APP_SINK_PTR std::shared_ptr<AppSinkBintr>
    #define DSL_APP_SINK_NEW(name, dataType, clientHandler, clientData) \
        std::shared_ptr<AppSinkBintr>( \
        new AppSinkBintr(name, dataType, clientHandler, clientData))

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
    #define DSL_OVERLAY_SINK_NEW(name, displayId, depth, offsetX, offsetY, width, height) \
        std::shared_ptr<OverlaySinkBintr>( \
        new OverlaySinkBintr(name, displayId, depth, offsetX, offsetY, width, height))

    #define DSL_WINDOW_SINK_PTR std::shared_ptr<WindowSinkBintr>
    #define DSL_WINDOW_SINK_NEW(name, offsetX, offsetY, width, height) \
        std::shared_ptr<WindowSinkBintr>( \
        new WindowSinkBintr(name, offsetX, offsetY, width, height))

    #define DSL_ENCODE_SINK_PTR std::shared_ptr<EncodeSinkBintr>
        
    #define DSL_FILE_SINK_PTR std::shared_ptr<FileSinkBintr>
    #define DSL_FILE_SINK_NEW(name, filepath, codec, container, bitrate, interval) \
        std::shared_ptr<FileSinkBintr>( \
        new FileSinkBintr(name, filepath, codec, container, bitrate, interval))
        
    #define DSL_RECORD_SINK_PTR std::shared_ptr<RecordSinkBintr>
    #define DSL_RECORD_SINK_NEW(name, outdir, codec, container, bitrate, interval, clientListener) \
        std::shared_ptr<RecordSinkBintr>( \
        new RecordSinkBintr(name, outdir, codec, container, bitrate, interval, clientListener))
        
    #define DSL_RTSP_SINK_PTR std::shared_ptr<RtspSinkBintr>
    #define DSL_RTSP_SINK_NEW(name, host, udpPort, rtspPort, codec, bitrate, interval) \
        std::shared_ptr<RtspSinkBintr>( \
        new RtspSinkBintr(name, host, udpPort, rtspPort, codec, bitrate, interval))
        
    #define DSL_MESSAGE_SINK_PTR std::shared_ptr<MessageSinkBintr>
    #define DSL_MESSAGE_SINK_NEW(name, \
            converterConfigFile, payloadType, brokerConfigFile, \
            protocolLib, connectionString, topic) \
        std::shared_ptr<MessageSinkBintr>(new MessageSinkBintr(name, \
            converterConfigFile, payloadType, brokerConfigFile, \
            protocolLib, connectionString, topic))
        
    #define DSL_INTERPIPE_SINK_PTR std::shared_ptr<InterpipeSinkBintr>
    #define DSL_INTERPIPE_SINK_NEW(name, forwardEos, forwardEvents) \
        std::shared_ptr<InterpipeSinkBintr>( \
        new InterpipeSinkBintr(name, forwardEos, forwardEvents))


    class SinkBintr : public Bintr
    {
    public: 
    
        SinkBintr(const char* name, bool sync);

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
         * @brief removes this SinkBintr from a parent Branch/Pipeline bintr
         * @param[in] pParentBintr parent bintr to remove this sink from
         * @return true on successful remove, false otherwise
         */
        bool RemoveFromParent(DSL_BASE_PTR pParentBintr);
        
        /**
         * @brief returns the current sync enabled setting for the SinkBintr
         * @return true if the sync attribute is enabled, false othewise
         */
        bool GetSyncEnabled();
        
        /**
         * @brief sets the sync enabled setting for the SinkBintr
         * @param[in] enabled current sync setting.
         */
        virtual bool SetSyncEnabled(bool enabled) = 0;
        
    protected:

        /**
         * @brief Device Properties, used for aarch64/x86_64 conditional logic
         */
        cudaDeviceProp m_cudaDeviceProp;
        
        /**
         * @brief Generate Quality-of-Service events upstream if true
         */
        bool m_qos;

        /**
         * @brief Sink element's current synchronous attribute setting.
         */
        bool m_sync;

        /**
         * @brief Queue element as sink for all Sink Bintrs.
         */
        DSL_ELEMENT_PTR m_pQueue;
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

    private:
    
        /**
         * @brief either DSL_SINK_APP_DATA_TYPE_SAMPLE or 
         * DSL_SINK_APP_DATA_TYPE_BUFFER
         */
        uint m_dataType;
    
        /**
         * @brief mutex to protect mutual access to the client-data-handler
         */
        GMutex m_dataHandlerMutex;

        /**
         * @brief client callback function to be called with each new 
         * buffer available.
         */
        dsl_sink_app_new_data_handler_cb m_clientHandler; 
        
        /**
         * @brief opaque pointer to client data to return with the callback.
         */
        void* m_clientData;
        
        /**
         * @brief App Sink element for the Sink Bintr.
         */
        DSL_ELEMENT_PTR m_pAppSink;
        
    };

    /**
     * @brief callback function registered with with the appsink's "new-sample" signal.
     * The callback wraps the AppSinkBintr's HandleNewSample function.
     * @param sink appsink element - not used.
     * @param pAppSinkBintr opaque pointer the the AppSinkBintr that triggered the 
     * "new-sample" signal - owner of the appsink element.
     * @return either GST_FLOW_OK, or GST_FLOW_EOS on no buffer available.
     */
    static GstFlowReturn on_new_sample_cb(GstElement sink, 
        gpointer pAppSinkBintr);
        
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
         * @brief sets the sync enabled setting for the SinkBintr
         * @param[in] enabled current sync setting.
         */
        bool SetSyncEnabled(bool enabled);

    private:
        
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
            uint offsetX, uint offsetY, uint width, uint height, bool sync);

        ~RenderSinkBintr();
        
        /**
         * @brief Gets the current X and Y offset settings for this RenderSinkBintr
         * @param[out] offsetX the current offset in the X direction in pixels
         * @param[out] offsetY the current offset in the Y direction setting in pixels
         */ 
        void GetOffsets(uint* offsetX, uint* offsetY);

        /**
         * @brief Sets the current X and Y offset settings for this RednerSinkBintr
         * The caller is required to provide valid width and height values
         * @param[in] offsetX the offset in the X direct to set in pixels
         * @param[in] offsetY the offset in the Y direct to set in pixels
         * @return false if the OverlaySink is currently in Use. True otherwise
         */ 
        virtual bool SetOffsets(uint offsetX, uint offsetY) = 0;

        /**
         * @brief Gets the current width and height settings for this RenderSinkBintr
         * @param[out] width the current width setting in pixels
         * @param[out] height the current height setting in pixels
         */ 
        void GetDimensions(uint* width, uint* height);
        
        /**
         * @brief Sets the current width and height settings for this RenderSinkBintr
         * The caller is required to provide valid width and height values
         * @param[in] width the width value to set in pixels
         * @param[in] height the height value to set in pixels
         * @return false if the sink is currently Linked. True otherwise
         */ 
        virtual bool SetDimensions(uint width, uint hieght) = 0;
        
        /**
         * @brief Resets the Sink element for this RenderSinkBintr
         * @return false if the sink is currently Linked. True otherwise
         */
        virtual bool Reset() = 0;

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
    
        OverlaySinkBintr(const char* name, uint displayId, uint depth, 
            uint offsetX, uint offsetY, uint width, uint height);

        ~OverlaySinkBintr();

        /**
         * @brief Resets the Sink element for this OverlaySinkBintr
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
         * @brief sets the sync enabled setting for the SinkBintr
         * @param[in] enabled current sync setting.
         */
        bool SetSyncEnabled(bool enabled);

        /**
         * @brief static list of unique Overlay IDs to be used/recycled by all
         * Overlay Sinks cto/dtor
         */
        static std::list<uint> s_uniqueIds;
        
    private:

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
         * @brief Resets the Sink element for this RenderSinkBintr
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
         * @brief sets the sync enabled setting for the SinkBintr
         * @param[in] enabled current sync setting.
         */
        bool SetSyncEnabled(bool enabled);
        
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

        /**
         * @brief Sets the GPU ID for all Elementrs - x86_64 builds only.
         * @return true if successfully set, false otherwise.
         */
        bool SetGpuId(uint gpuId);

        /**
         * @brief Sets the NVIDIA buffer memory type - x86_64 builds only.
         * @brief nvbufMemType new memory type to use, one of the 
         * DSL_NVBUF_MEM_TYPE constant values.
         * @return true if successfully set, false otherwise.
         */
        bool SetNvbufMemType(uint nvbufMemType);

    private:

        bool m_forceAspectRatio;

        /**
         * @brief Caps Filter required for dGPU WindowSinkBintr
         */
        DSL_ELEMENT_PTR m_pCapsFilter;

        /**
         * @brief Platform specific Transform element WindowSinkBintr
         */
        DSL_ELEMENT_PTR m_pTransform;
        
        /**
         * @brief Window Sink Element for the WindowSinkBintr
         */
        DSL_ELEMENT_PTR m_pEglGles;
    };

    //-------------------------------------------------------------------------

    class EncodeSinkBintr : public SinkBintr
    {
    public: 
    
        EncodeSinkBintr(const char* name,
            uint codec, uint bitrate, uint interval);

        /**
         * @brief Gets the current bit-rate and interval settings for the Encoder in use
         * @param[out] code the currect codec in used
         * @param[out] bitrate the current bit-rate setting for the encoder in use
         * @param[out] interval the current iframe interval for the encoder in use
         */ 
        void GetEncoderSettings(uint* codec, uint* bitrate, uint* interval);

        /**
         * @brief Sets the current bit-rate and interval settings for the Encoder in use
         * @param[in] codec the new code to use, either DSL_CODEC_H264 or DSL_CODE_H265
         * @param[in] bitrate the new bit-rate setting in uints of bits/sec
         * @param[in] interval the new iframe-interval setting
         * @return false if the FileSink is currently in Use. True otherwise
         */ 
        bool SetEncoderSettings(uint codec, uint bitrate, uint interval);

        /**
         * @brief Sets the GPU ID for all Elementrs
         * @return true if successfully set, false otherwise.
         */
        bool SetGpuId(uint gpuId);
        
    protected:

        uint m_codec;
        uint m_bitrate;
        uint m_interval;
 
        DSL_ELEMENT_PTR m_pTransform;
        DSL_ELEMENT_PTR m_pCapsFilter;
        DSL_ELEMENT_PTR m_pEncoder;
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

        /**
         * @brief sets the sync enabled setting for the SinkBintr
         * @param[in] enabled current sync setting.
         */
        bool SetSyncEnabled(bool enabled);
        
    private:

        uint m_container;

        DSL_ELEMENT_PTR m_pFileSink;

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

    class RtspSinkBintr : public EncodeSinkBintr
    {
    public: 
    
        RtspSinkBintr(const char* name, const char* host, uint udpPort, uint rtspPort,
         uint codec, uint bitrate, uint interval);

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
         */ 
        void GetServerSettings(uint* udpPort, uint* rtspPort);

        /**
         * @brief sets the sync enabled setting for the SinkBintr
         * @param[in] enabled current sync setting.
         */
        bool SetSyncEnabled(bool enabled);

    private:

        std::string m_host;
        uint m_udpPort;
        uint m_rtspPort;
        
        GstRTSPServer* m_pServer;
        uint m_pServerSrcId;
        GstRTSPMediaFactory* m_pFactory;
 
        DSL_ELEMENT_PTR m_pPayloader;
        DSL_ELEMENT_PTR m_pUdpSink;
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
         * @brief sets the sync enabled setting for the SinkBintr
         * @param[in] enabled current sync setting.
         */
        bool SetSyncEnabled(bool enabled);

    private:

        /**
         * @brief defines the base_meta.meta_type id filter to use for
         * all message meta to convert and send. Default = NVDS_EVENT_MSG_META.
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
         * @brief Tee element for this MessageSinkBintr 
         */
        DSL_ELEMENT_PTR m_pTee;

        /**
         * @brief Tee Src Queue for the message-converter element for this MessageSinkBintr 
         */
        DSL_ELEMENT_PTR m_pMsgConverterQueue;
        
        /**
         * @brief NVIDIA message-converter element for this MessageSinkBintr 
         */
        DSL_ELEMENT_PTR m_pMsgConverter;

        /**
         * @brief NVIDIA message-broker element for this MessageSinkBintr.
         */
        DSL_ELEMENT_PTR m_pMsgBroker;

        /**
         * @brief Tee Src Queue for the Fake Sink element for this MessageSinkBintr 
         */
        DSL_ELEMENT_PTR m_pFakeSinkQueue;

        /**
         * @brief Fake Sink element for the MessageSinkBintr.
         */
        DSL_ELEMENT_PTR m_pFakeSink;

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

        /**
         * @brief Inter-Pipe Sink element for the InterpipeSinkBintr.
         */
        DSL_ELEMENT_PTR m_pSinkElement;
    };

}
#endif // _DSL_SINK_BINTR_H
    