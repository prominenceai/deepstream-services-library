/*
The MIT License

Copyright (c) 2019-2023, Prominence AI, Inc.

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

    #define DSL_VIDEO_SOURCE_PTR std::shared_ptr<VideoSourceBintr>

    #define DSL_APP_SOURCE_PTR std::shared_ptr<AppSourceBintr>
    #define DSL_APP_SOURCE_NEW(name, isLive, bufferInFormat, width, height, fpsN, fpsD) \
        std::shared_ptr<AppSourceBintr>(new AppSourceBintr(name, isLive, \
            bufferInFormat, width, height, fpsN, fpsD))
        
    #define DSL_CSI_SOURCE_PTR std::shared_ptr<CsiSourceBintr>
    #define DSL_CSI_SOURCE_NEW(name, width, height, fpsN, fpsD) \
        std::shared_ptr<CsiSourceBintr>(new CsiSourceBintr(name, width, height, fpsN, fpsD))
        
    #define DSL_V4L2_SOURCE_PTR std::shared_ptr<V4l2SourceBintr>
    #define DSL_V4L2_SOURCE_NEW(name, deviceLocation) \
        std::shared_ptr<V4l2SourceBintr>(new V4l2SourceBintr(name, deviceLocation))

    #define DSL_RESOURCE_SOURCE_PTR std::shared_ptr<ResourceSourceBintr>
        
    #define DSL_URI_SOURCE_PTR std::shared_ptr<UriSourceBintr>
    #define DSL_URI_SOURCE_NEW(name, uri, isLive, skipFrames, dropFrameInterval) \
        std::shared_ptr<UriSourceBintr>(new UriSourceBintr(name, \
            uri, isLive, skipFrames, dropFrameInterval))
        
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
    #define DSL_IMAGE_STREAM_SOURCE_NEW(name, uri, isLive, fpsN, fpsD, timeout) \
        std::shared_ptr<ImageStreamSourceBintr>(new ImageStreamSourceBintr(name, \
            uri, isLive, fpsN, fpsD, timeout))

    #define DSL_INTERPIPE_SOURCE_PTR std::shared_ptr<InterpipeSourceBintr>
    #define DSL_INTERPIPE_SOURCE_NEW(name, listenTo, isLive, acceptEos, acceptEvents) \
        std::shared_ptr<InterpipeSourceBintr>(new InterpipeSourceBintr(name, \
            listenTo, isLive, acceptEos, acceptEvents))

    #define DSL_RTSP_SOURCE_PTR std::shared_ptr<RtspSourceBintr>
    #define DSL_RTSP_SOURCE_NEW(name, uri, protocol, \
        skipFrames, dropFrameInterval, latency, timeout) \
        std::shared_ptr<RtspSourceBintr>(new RtspSourceBintr(name, uri, protocol, \
            skipFrames, dropFrameInterval, latency, timeout))

    #define DSL_DUPLICATE_SOURCE_PTR std::shared_ptr<DuplicateSourceBintr>
    #define DSL_DUPLICATE_SOURCE_NEW(name, original, isLive) \
        std::shared_ptr<DuplicateSourceBintr>(new DuplicateSourceBintr(name, \
            original, isLive))

    /**
     * @brief Utility function to define/set all capabilities (media, 
     * format, width, height, and frame rate) for a given element.
     * @param[in] pElement element to update.
     * @param[in] media ascii version of one of the DSL_MEDIA_TYPE constants.
     * @param[in] format ascii version of one of the DSL_VIDEO_FORMAT constants.
     * @param[in] width frame width in units of pixels.
     * @param[in] height frame height in units of pixels.
     * @param[in] fpsN frames-per-sec numerator.
     * @param[in] fpsD frames-per-sec denominator.
     * @param[in] isNvidia set to true to add memory:NVMM feature.
     * @return on successful set, false otherwise.
     */
    static bool set_full_caps(DSL_ELEMENT_PTR pElement, 
        const char* media, const char* format, uint width, uint height, 
        uint fpsN, uint fpsD, bool isNvidia);

    /**
     * @brief Utility function to define/set the media and 
     * format for a given element.
     * @param[in] pElement element to update.
     * @param[in] media ascii version of one of the DSL_MEDIA_TYPE constants.
     * @param[in] format ascii version of one of the DSL_VIDEO_FORMAT constants.
     * @param[in] isNvidia set to true to add memory:NVMM feature.
     * @return on successful set, false otherwise.
     */
    static bool set_format_caps(DSL_ELEMENT_PTR pElement, 
        const char* media, const char* format, bool isNvidia);

    /**
     * @class SourceBintr
     * @brief Implements a base Source Bintr for all derived Source types.
     */
    class SourceBintr : public Bintr
    {
    public: 
    
        /**
         * @brief ctor for the SourceBintr base class
         * @param[in] name unique name for the new SourceBintr
         */
        SourceBintr(const char* name);

        /**
         * @brief dtor for the SourceBintr base class
         */
        ~SourceBintr();

        /**
         * @brief Adds the SourceBintr to a given Parent Bintr (PipelineSourcesBintr).
         * @param[in] pParentBintr shared pointer to the Parent Bintr to add to.
         * @return true on successful add, false otherwise.
         */
        bool AddToParent(DSL_BASE_PTR pParentBintr);

        /**
         * @brief Tests if a give Bintr is the parent of the SourceBintr.
         * @param[in] pParentBintr shared pointer to the Bintr to test for parenthood.
         * @return true if the Bintr is the parent, false otherwise.
         */
        bool IsParent(DSL_BASE_PTR pParentBintr);
        
        /**
         * @brief Removes the SourceBintr from a give Parent Bintr.
         * @param pParentBintr shared pointer to the parent Bintr to remove from.
         * @return true on successfull remove, false otherwise.
         */
        bool RemoveFromParent(DSL_BASE_PTR pParentBintr);

        /**
         * @brief Function is overridden by all derived SourceBintrs
         */
        void UnlinkAll(){};

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
         * @brief Gets the current FPS numerator and denominator settings for this SourceBintr
         * @param[out] fpsN the FPS numerator
         * @param[out] fpsD the FPS denominator
         */ 
        void GetFrameRate(uint* fpsN, uint* fpsD)
        {
            LOG_FUNC();
            
            *fpsN = m_fpsN;
            *fpsD = m_fpsD;
        }

        /**
         * @brief Gets the current media-type for the SourceBintr.
         * @return Current media type string. 
         */
        const char* GetMediaType()
        {
            LOG_FUNC();
            
            return m_mediaType.c_str();
        }

        /**
         * @brief Returns the current linkable state of the Source. Camera Sources
         * HTTP, and RTSP sources are linkable for the life of the Source.
         * File and Image Sources can be created without a file path and are unlinkable
         * until they are updated with a valid path.
         * @return true if the Source if linkable (able to link), false otherwise
         */
        virtual bool IsLinkable(){return true;};
    
        /**
         * @brief For sources that manage EOS Consumers, this service must.
         * called before sending the source an EOS Event to stop playing.
         */
        virtual void DisableEosConsumer(){};
        
        /**
         * @brief Gets the current unique id for the SourceBintr.
         * @return unique-id, -1 untill assigned by the parent PipelineSourceBintr.
         */
        int GetUniqueId()
        {
            LOG_FUNC();
            
            return m_uniqueId;
        }
        
        /**
         * @brief Sets the current unique id for the SourceBintr.
         * @param[in] id, new unique id to assign set the SourceBintr, 
         * set to -1 to unassign.
         */
        void SetUniqueId(int id)
        {
            LOG_FUNC();
            
            m_uniqueId = id;
        }
        
    protected:
    
        /**
         * @brief Unique, assigned Source-Id for the SourceBintr. -1 when
         * unassigned, set to a unique-id once added to a PipelineSourceBintr 
         * derived from the parent Pipeline's unique-id or'd with the unique 
         * stream/pad-id == guaranted uniqueness.
         */
        int m_uniqueId; 
    
        /**
         * @brief Device Properties, used for aarch64/x86_64 conditional logic
         */
        cudaDeviceProp m_cudaDeviceProp;

        /**
         * @brief media-type for the SourceBintr. String version of one of the
         * DSL_MEDIA_TYPE constant values.
         */
        std::string m_mediaType;

        /**
         * @brief True if the source is live and cannot be paused without losing data, 
         * False otherwise.
         */
        bool m_isLive;
        
        /**
         * @brief current frames-per-second numerator value for the SourceBintr
         */
        uint m_fpsN;

        /**
         * @brief current frames-per-second denominator value for the SourceBintr
         */
        uint m_fpsD;

        /**
         * @brief Soure Element for this SourceBintr
         */
        DSL_ELEMENT_PTR m_pSourceElement;
        
    };

    /**
     * @class VideoSourceBintr
     * @brief Implements a base Video Source Bintr for all derived Video Source types.
     */
    class VideoSourceBintr : public SourceBintr
    {
    public: 
    
        /**
         * @brief ctor for the VideoSourceBintr base class
         * @param[in] name unique name for the new VideoSourceBintr
         */
        VideoSourceBintr(const char* name);

        /**
         * @brief dtor for the VideoSourceBintr base class
         */
        ~VideoSourceBintr();

        /**
         * @brief Gets the current width and height settings for this SourceBintr
         * @param[out] width the current width setting in pixels
         * @param[out] height the current height setting in pixels
         */ 
        void GetDimensions(uint* width, uint* height);

        /**
         * @brief Gets the current buffer-out-format for this SourceBintr.
         * @return Current buffer-out-format. string version of one of the 
         * DSL_VIDEO_FORMAT constants.
         */
        const char* GetBufferOutFormat()
        {
            LOG_FUNC();
            
            return m_bufferOutFormat.c_str();
        };
        
        /**
         * @brief Sets the buffer-out-format for the SourceBintr.
         * @param[in] format string version of one of the DSL_VIDEO_FORMAT constants.
         * @return true if successfully set, false otherwise.
         */
        bool SetBufferOutFormat(const char* format);

        /**
         * @brief Sets the buffer-out-dimensions for the SourceBintr.
         * @param[out] width current width value to scale the output buffer in pixels
         * @param[out] height current height value to scale the output buffer in pixels
         */
        void GetBufferOutDimensions(uint* width, uint* height);
        
        /**
         * @brief Sets the buffer-out-dimensions for the SourceBintr.
         * @param[in] width new width value to scale the output buffer in pixels.
         * @param[in] height new height value to scale the output buffer in pixels.
         * @return true if successfully set, false otherwise.
         */
        bool SetBufferOutDimensions(uint width, uint height);
        
        /**
         * @brief Gets the buffer-out-frame-rate for the SourceBintr.
         * The default value of 0 for fps_n and fps_d indicates no scaling.
         * @param[out] fpsN current fpsN value to scale the output buffer.
         * @param[out] fpsD current fpsD value to scale the output buffer.
         */
        void GetBufferOutFrameRate(uint* fpsN, uint* fpsD);
        
        /**
         * @brief Sets the buffer-out-frame-rate for the SourceBintr.
         * Set fps_n and fps_d to 0 to indicate no scaling.
         * @param[in] fpsN new fpsN value to scale the output buffer.
         * @param[in] fpsD new fpsN value to scale the output buffer.
         * @return true if successfully set, false otherwise.
         */
        bool SetBufferOutFrameRate(uint fpsN, uint fpsD);
        
        /**
         * @brief Gets the buffer-out-crop values for the SourceBintr.
         * @param[in] cropAt either DSL_VIDEO_CROP_AT_SRC or 
         * DSL_VIDEO_CROP_AT_DESTINATION.
         * @param[out] left left coordinate for the crop frame in pixels.
         * @param[out] top top coordinate for the crop frame in pixels.
         * @param[out] width width of the crop frame in pixels.
         * @param[out] height height of the crop frame in pixels.
         * @return true if successfully set, false otherwise.
         */
        void GetBufferOutCropRectangle(uint cropAt, uint* left, uint* top, 
            uint* width, uint* height);

        /**
         * @brief Sets the buffer-out-crop values for the SourceBintr.
         * @param[in] cropAt either DSL_VIDEO_CROP_AT_SRC or 
         * DSL_VIDEO_CROP_AT_DEST.
         * @param[in] left left coordinate for the crop frame in pixels.
         * @param[in] top top coordinate for the crop frame in pixels.
         * @param[in] width width of the crop frame in pixels.
         * @param[in] height height of the crop frame in pixels.
         * @return true if successfully set, false otherwise.
         */
        bool SetBufferOutCropRectangle(uint when, uint left, uint top, 
            uint width, uint height);
        
        /**
         * @brief Gets the current buffer-out-orientation for the VideoCon. 
         * @param[out] orientation current buffer-out-format. One of the 
         * DSL_VIDEO_ORIENTATION constant value. Default = DSL_VIDEO_ORIENTATION_NONE.
         * @return 
         */
        uint GetBufferOutOrientation();
        
        /**
         * @brief Sets the buffer-out-orientation setting. 
         * @param[out] orientation current buffer-out-format. One of the 
         * DSL_VIDEO_ORIENTATION constant value. Default = DSL_VIDEO_ORIENTATION_NONE.
         * @return 
         */
        bool SetBufferOutOrientation(uint orientaion);

        /**
         * @brief Sets the GPU ID for all Elementrs
         * @param[in] gpuId new GPU ID to use. 
         * @return true if successfully set, false otherwise.
         */
        bool SetGpuId(uint gpuId);

        /**
         * @brief Sets the NVIDIA buffer memory type.
         * @param[in] nvbufMemType new memory type to use, one of the 
         * DSL_NVBUF_MEM_TYPE constant values.
         * @return true if successfully set, false otherwise.
         */
        bool SetNvbufMemType(uint nvbufMemType);

        /**
         * @brief Adds a single Dewarper Bintr to this SourceBintr 
         * @param[in] pDewarperBintr shared pointer to Dewarper to add
         * @returns true if the Dewarper could be added, false otherwise
         */
        bool AddDewarperBintr(DSL_BASE_PTR pDewarperBintr);

        /**
         * @brief Removed a previously added DewarperBintr from this SourceBintr.
         * @returns true if the Dewarper could be removed, false otherwise
         */
        bool RemoveDewarperBintr();
        
        /**
         * @brief Call to query the VideoSourceBintr if it has a Dewarper.
         * @return true if the Source has a Child
         */
        bool HasDewarperBintr();
        
        /**
         * @brief Adds a DuplicateSourceBintr to this VideoSourceBintr.
         * @param pDuplicateSource shared pointer to the DuplicateSourceBintr to add.
         * @return true if the DuplicateSourceBintr could be added, false otherwise.
         */
        bool AddDuplicateSource(DSL_VIDEO_SOURCE_PTR pDuplicateSource);

        /**
         * @brief Removes a DuplicateSourceBintr from this VideoSourceBintr.
         * @param pDuplicateSource shared pointer to the DuplicateSourceBintr to add.
         * @return true if the DuplicateSourceBintr could be added, false otherwise.
         */
        bool RemoveDuplicateSource(DSL_VIDEO_SOURCE_PTR pDuplicateSource);

    private:

        /**
         * @brief Private helper function to update the Video Converter's capability filter.
         * @return true if successful, false otherwise.
         */
        bool updateVidConvCaps();

        /**
         * @brief Private helper function to link all DuplicateSourceBintrs to
         * this VideoSourceBintr.
         * @return true on successful link, false otherwise
         */
        bool linkAllDuplicates();

        /**
         * @brief Private helper function to unlink all DuplicateSourceBintrs from
         * this VideoSourceBintr.
         * @return true on successful unlink, false otherwise
         */
        bool unlinkAllDuplicates();
    
    protected:
    
        /**
         * @brief Links the derived Source's last specific element (SrcNodetr)
         * to the common elements shared by all sources.
         * @param[in] pSrcNodetr source specific element to link to the common
         * elements.
         * @return True on success, false otherwise
         */
        bool LinkToCommon(DSL_NODETR_PTR pSrcNodetr);
        
        /**
         * @brief Links a dynamic src-pad to the common elements shared by all sources
         * @param[in] pSrcPad dynamic src-pad to link
         * @return True on success, false otherwise
         */
        bool LinkToCommon(GstPad* pSrcPad);

        /**
         * @brief Common shared code for the two LinkToCommon methods.
         * @return True on success, false otherwise
         */
        bool CompleteLinkToCommon();
        
        /**
         * @brief Unlinks all common Elementrs owned by this VidoSourceBintr.
         */
        void UnlinkCommon();

        /**
         * @brief vector to link/unlink all common elements
         */
        std::vector<DSL_GSTNODETR_PTR> m_linkedCommonElements;

        /**
         * @brief current buffer-out-format. 
         */
        std::string m_bufferOutFormat;
        
        /**
         * @brief current width of the streaming source in Pixels.
         */
        uint m_width;

        /**
         * @brief current height of the streaming source in Pixels.
         */
        uint m_height;

        /**
         * @brief Current scaled width value for the SourceBintr's Output Buffer 
         * Video Converter in units of pixels. Default = 0 for no transcode.
         */
        uint m_bufferOutWidth;
        
        /**
         * @brief Current scaled height setting for the SourceBintr's Output Buffer
         * Video Converter in units of pixels. Default = 0 for no transcode
         */
        uint m_bufferOutHeight;

        /**
         * @brief Current scaled fps-n value for the SourceBintr's Output Buffer 
         * rate controler. Default = 0 for no rate change.
         */
        uint m_bufferOutFpsN;
        
        /**
         * @brief Current scaled height setting for the SourceBintr's Output Buffer
         * rate controler. Default = 0 for no rate change
         */
        uint m_bufferOutFpsD;

        /**
         * @brief Current buffer-out-orientation setting for the SourceBintr
         */
        uint m_bufferOutOrientation;

        /**
         * @brief Output-buffer Video Converter element for this SourceBintr.
         */
        DSL_ELEMENT_PTR m_pBufferOutVidConv;

        /**
         * @brief Output-buffer Video Rate element for this SourceBintr.
         */
        DSL_ELEMENT_PTR m_pBufferOutVidRate;

        /**
         * @brief Caps Filter for the SourceBintr's output-buffe.
         */
        DSL_ELEMENT_PTR m_pBufferOutCapsFilter;

        /**
         * @brief Single, optional dewarper for the DecodeSourceBintr
         */ 
        DSL_DEWARPER_PTR m_pDewarperBintr;
        
        /**
         * @brief Source Queue for SourceBintr - set as ghost-pad for each source
         */
        DSL_ELEMENT_PTR  m_pSourceQueue;

        /**
         * @brief Conditional Tee used if this VideoSourceBintr has 1 or more
         * DuplicateSourceBintrs.
         */
        DSL_ELEMENT_PTR m_pDuplicateSourceTee;
        
        /**
         * @brief Conditional Queue used if this VideoSourceBintr has 1 or more
         * DuplicateSourceBintrs.
         */
        DSL_ELEMENT_PTR m_pDuplicateSourceTeeQueue;
        
        /**
         * @brief map of DuplicateSourceBintrs to duplicate this VideSourceBintr
         */
        std::map <std::string, DSL_VIDEO_SOURCE_PTR> m_duplicateSources;
        
        /**
         * @brief vecotr of requested source pads from m_pDuplicateSourceTee
         */
        std::vector <GstPad*> m_requestedDuplicateSrcPads;
        
    };

    /**
     * @class DuplicateSourceBintr
     * @brief Implements a Source that can be added to any other Video Source
     * to duplicate the original stream.
     */
    class DuplicateSourceBintr : public VideoSourceBintr
    {
    public: 
    
        /**
         * @brief ctor for the DuplicateSourceBintr
         * @param[in] name unique name to give the new DuplicateSourceBintr 
         * @param[in] original unique name of the original source for this
         * @param[in] isLive set to true if original source isLive, false otherwise. 
         */
        DuplicateSourceBintr(const char* name, const char* original,
            bool isLive);

        /**
         * @brief dtor for the DuplicateSourceBintr
         */
        ~DuplicateSourceBintr();

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
         * @brief Gets the unique name of the original Source (VideoSourceBintr)
         * for this DuplicateSourceBintr
         */
        const char* GetOriginal();
        
        /**
         * @brief Sets the the original Source (VideoSourceBintr) by name
         * for this DuplicateSourceBintr
         */
        void SetOriginal(const char* original);
        
    private:
    
        /**
         * @brief name of the Original Source -- currently added to -- to duplicate.
         */
        std::string m_original;
        
        /**
         * @brief Sink (input) queue for this DuplicateSourceBintr.
         */
        DSL_ELEMENT_PTR m_pSinkQueue;
        
    };

    //*********************************************************************************
    /**
     * @class AppSourceBintr
     * @brief 
     */
    class AppSourceBintr : public VideoSourceBintr
    {
    public: 
    
        AppSourceBintr(const char* name, bool isLive, 
            const char* bufferInFormat, uint width, uint height, uint fpsN, uint fpsD);

        ~AppSourceBintr();

        /**
         * @brief Links all Child Elementrs owned by this AppSourceBintr
         * @return True success, false otherwise
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elementrs owned by this AppSourceBintr
         */
        void UnlinkAll();

        /**
         * @brief Adds data-handler callback functions to this AppSourceBintr
         * @param[in] needDataHandler callback function to be called when new data is needed.
         * @param[in] enoughDataHandler callback function to be called when the Source
         * has enough data to process.
         * @param[in] clientData opaque pointer to client data passed back into the 
         * client_handler function.
         * @return true on successful add, false otherwise.
         */
        bool AddDataHandlers(dsl_source_app_need_data_handler_cb needDataHandler, 
            dsl_source_app_enough_data_handler_cb enoughDataHandler, 
            void* clientData);
        
        /**
         * @brief Adds data-handler callback functions from this AppSourceBintr,
         * function previously added with AddDataHandlers
         * @return true on successful remove, false otherwise.
         */
        bool RemoveDataHandlers();
        
        /**
         * @brief Pushes a new buffer to this AppSourceBintr for processing.
         * @param[in] buffer buffer to push to this AppSourceBintr
         * @return true on successful push, false otherwise.
         */
        bool PushBuffer(void* buffer);
        
        /**
         * @brief Pushes a new sample to this AppSourceBintr for processing.
         * @param[in] sample sample to push to this AppSourceBintr
         * @return true on successful push, false otherwise.
         */
        bool PushSample(void* sample);
        
        /**
         * @brief Notifies this AppSourceBintr that there are no more buffers 
         * for processing.
         * @return true on successful Eos, false otherwise.
         */
        bool Eos();
        
        /**
         * @brief Callback handler function to handle the "need-data" signal.
         * @param length the amount of bytes needed. length is just a hint and 
         * when it is set to -1, any number of bytes can be pushed into appsrc.
         */
        void HandleNeedData(uint length);
        
        /**
         * @brief Callback handler function to handle the "enough-data"signal.
         */
        void HandleEnoughData();

        /**
         * @brief Sets the source-dimensions for the AppSourceBintr.
         * @param[in] width new width value for the Source in pixels
         * @param[in] height new height value for the Source in pixels
         * @return true if successfully set, false otherwise.
         */
        bool SetDimensions(uint width, uint height);

        /**
         * @brief Gets the current do-timestamp property setting for the AppSourceBintr.
         * @return If TRUE, the base class will automatically timestamp outgoing 
         * buffers based on the current running_time.
         */
        boolean GetDoTimestamp();

        /**
         * @brief Sets the do-timestamp property settings for the AppSourceBintr
         * @param[in] doTimestamp set to TRUE to have the base class automatically 
         * timestamp outgoing buffers. FALSE otherwise.
         * @return 
         */
        bool SetDoTimestamp(boolean doTimestamp);
        

        /**
         * @brief Gets the current stream-format for this AppSourceBintr
         * @return one of the DSL_STREAM_FORMAT constants.
         */
        uint GetStreamFormat();
        
        /**
         * @brief Sets the stream-format for this AppSourceBintr to use.
         * @param[in] streamFormat one of the DSL_STREAM_FORMAT constants.
         * @return true on successful set, false otherwise.
         */
        bool SetStreamFormat(uint streamFormat);
        
        /**
         * @brief Gets the current block-enabled setting for this AppSourceBintr.
         * @return If true, when max-bytes/buffers/time are queued and after the 
         * enough-data signal has been emitted, the source will block any further 
         * push-buffer calls until the amount of queued bytes drops below the 
         * max-bytes/buffers/time limit.
         */
        boolean GetBlockEnabled();
        
        /**
         * @brief Sets the block-enabled setting for this AppSourceBintr.
         * @param[in] enabled If true, when max-bytes/buffers/time are queued and 
         * after the enough-data signal has been emitted, the source will block any 
         * further push-buffer calls until the amount of queued bytes drops below the 
         * max-bytes/buffers/time limit.
         * @return true on successful set, false otherwise.
         */
        bool SetBlockEnabled(boolean enabled);
        
        /**
         * @brief Gets the current level of queued data in bytes for 
         * this AppSrcBintr.
         * @return current level of queued data in bytes.
         */
        uint64_t GetCurrentLevelBytes();

        /**
         * @brief Gets the max level of queued data in bytes for
         * this AppSrcBintr.
         * @return current maximum level of queued bytes.
         */
        uint64_t GetMaxLevelBytes();
        
        /**
         * @brief Sets the max level of queued data in bytes for 
         * for this AppSrcBintr.
         * @param[in] level new max level in bytes for the App Source to use.
         * @return true on successful set, false otherwise.
         */
        bool SetMaxLevelBytes(uint64_t level);
        
//        /**
//         * @brief Gets the current leaky-type in use by this AppSourceBintr
//         * @return leaky-type one of the DSL_QUEUE_LEAKY_TYPE constant values. 
//         */
//        uint GetLeakyType();  // TODO support GST 1.20 properties
        
//        /**
//         * @brief Sets the leaky-type for the AppSrcBintr to use.
//         * @param leakyType one of the DSL_QUEUE_LEAKY_TYPE constant values. 
//         * @return true on successful set, false otherwise.
//         */
//        bool SetLeakyType(uint leakyType);
        
    private:
    
        /**
         * @brief stream format for the AppSourceBintr - on of the DSL_STREAM_FORMAT constants.
         */
        uint m_streamFormat;

        /**
         * @brief If TRUE, the base class will automatically timestamp outgoing buffers
         * based on the current running_time..
         */
        boolean m_doTimestamp;

        /**
         * @brief buffer-in format for the AppSourceBintr - on of the DSL_VIDEO_FORMAT constants.
         */
        std::string m_bufferInFormat;

        /**
         * @brief client callback function to be called when new data is needed.
         */
        dsl_source_app_need_data_handler_cb m_needDataHandler;
        
        /**
         * @brief client callback function to be called when new data is needed.
         */
        dsl_source_app_enough_data_handler_cb m_enoughDataHandler;
        
        /**
         * @brief opaque pointer to client data to return with the "need-data" and 
         * "enough-data" callback.
         */
        void* m_clientData;
        
        /**
         * @brief mutex to protect mutual access to the client-data-handlers
         */
        DslMutex m_dataHandlerMutex;
        
        /**
         * @brief block-enabled setting for this AppSourceBintr.
         */
        boolean m_blockEnabled;
        
        /**
         * @brief The maximum amount of bytes that can be queued internally. 
         * After the maximum amount of bytes are queued, appsrc will emit 
         * the "enough-data" signal.
         */
        uint64_t m_maxBytes;
        
        /**
         * @brief The maximum amount of buffers that can be queued internally. 
         * After the maximum amount of buffers are queued, appsrc will emit 
         * the "enough-data" signal.
         */
//        uint64_t m_maxBuffers; // TODO support GST 1.20 properties
        
        /**
         * @brief The maximum amount of time that can be queued internally. 
         * After the maximum amount of time is queued, appsrc will emit 
         * the "enough-data" signal.
         */
//        uint64_t m_maxTime;  // TODO support GST 1.20 properties
        
        /**
         * @brief Current Queue leaky-type, one of the DSL_QUEUE_LEAKY_TYPE
         * constant values. 
         */
//        uint m_leakyType;  // TODO support GST 1.20 properties

    };    
    
    /**
     * @brief Callback function for the "need-data" signal
     * @param pSourceElement "appsrc" plugin/element that invoked the signal (unused)
     * @param length the amount of bytes needed. length is just a hint and 
     * when it is set to -1, any number of bytes can be pushed into appsrc.
     * @param pAppSrcBintr pointer to the AppSrcBintr that registered for the
     * "need-data" signal.
     */
    static void on_need_data_cb(GstElement* pSourceElement, uint length,
        gpointer pAppSrcBintr);
    
    /**
     * @brief Callback function for the "enough-data" signal
     * @param pSourceElement "appsrc" plugin/element that invoked the signal (unused)
     * @param pAppSrcBintr pointer to the AppSrcBintr that registered for the
     * "enough-data" signal.
     */
    static void on_enough_data_cb(GstElement* pSourceElement, 
        gpointer pAppSrcBintr);
        
    //*********************************************************************************
    /**
     * @class CsiSourceBintr
     * @brief 
     */
    class CsiSourceBintr : public VideoSourceBintr
    {
    public: 
    
        CsiSourceBintr(const char* name, uint width, uint height, 
            uint fpsN, uint fpsD);

        ~CsiSourceBintr();

        /**
         * @brief Links all Child Elementrs owned by this CsiSourceBintr
         * @return True success, false otherwise
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elementrs owned by this CsiSourceBintr
         */
        void UnlinkAll();
        
        /**
         * @brief Gets the current sensor-id for the CsiSourceBintr
         * @return current unqiue sensor-id starting with 0
         */
        uint GetSensorId();
        
        /**
         * @brief Sets the sensor-id
         * @param[in] sensorId new sensor-id for the CsiSourceBintr
         * @return true if successfull, false otherwise.
         */
        bool SetSensorId(uint sensorId);

    private:

        /**
         * @brief static list of unique sersor IDs to be used/recycled by all
         * CsiSourceBintrs
         */
        static std::list<uint> s_uniqueSensorIds;
    
        /**
         * @brief unique sensorId for the CsiSourceBintr starting with 0
         */
        uint m_sensorId;

        /**
         * @brief If TRUE, the class will automatically timestamp outgoing buffers
         * based on the current running_time..
         */
        boolean m_doTimestamp;

        /**
         * @brief Caps Filter for the CsiSourceBintr's Source Element
         * - nvarguscamerasrc.
         */
        DSL_ELEMENT_PTR m_pSourceCapsFilter;
    };    

    //*********************************************************************************
    /**
     * @class V4l2SourceBintr
     * @brief 
     */
    class V4l2SourceBintr : public VideoSourceBintr
    {
    public: 
    
        V4l2SourceBintr(const char* name, const char* deviceLocation);

        ~V4l2SourceBintr();

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
         * @brief Gets the current device location setting for the Source Bintr
         * @return current device location. Default = /dev/video0
         */
        const char* GetDeviceLocation();
        
        /**
         * @brief Sets the device location setting for the V4l2SourceBintr.
         * @param[in] new device location for the Source Bintr to use.
         * @return true if successfully set, false otherwise.
         */
        bool SetDeviceLocation(const char* deviceLocation);

        /**
         * @brief Sets the dimensions for the V4l2SourceBintr.
         * @param[in] width new width value for the Source in pixels
         * @param[in] height new height value for the Source in pixels
         * @return true if successfully set, false otherwise.
         */
        bool SetDimensions(uint width, uint height);

        /**
         * @brief Sets the frame-rate for the V4l2SourceBintr.
         * Set fps_n and fps_d to 0 to indicate no scaling.
         * @param[in] fpsN new fpsN value for the Source.
         * @param[in] fpsD new fpsN value Source.
         * @return true if successfully set, false otherwise.
         */
        bool SetFrameRate(uint fpsN, uint fpsD);

        /**
         * @brief Gets the current device-name setting for the V4l2SourceBintr
         * Default = "". Updated after negotiation with the V4L2 Device.
         * @return current device location.
         */
        const char* GetDeviceName();
        
        /**
         * @brief Gets the current device-fd (file-descriptor) setting for 
         * the V4l2SourceBintr. Default = -1 (unset). Updated at runtime after
         * negotiation with the V4l2 Device.
         * @return current device location.
         */
        int GetDeviceFd();
        
        /**
         * @brief Gets the current device-flags setting for the V4l2SourceBintr. 
         * Default = 0 (none). Updated at runtime after negotiation with the 
         * V4l2 Device.
         * @return current device location.
         */
        uint GetDeviceFlags();
                
        /**
         * @brief Gets the current picture settings for this V4l2SourceBintr.
         * @param[out] brightness current brightness (actually darkness) level.
         * @param[out] contrast current picture contrast or luna gain level.
         * @param[out] saturation current color saturation or chroma gain level.
         */
        void GetPictureSettings(int* brightness, int* contrast, int* saturation);
        
        /**
         * @brief Sets the picture settings for the V4l2SourceBintr to use.
         * @param[in] brightness new brightness (actually darkness) level.
         * @param[in] contrast new picture contrast or luna level.
         * @param[in] saturation new color saturation or chroma level.
         * @return true if successfully set, false otherwise.
         */
        bool SetPictureSettings(int brightness, int contrast, int saturation);
    private:

        /**
         * @brief current device location for the V4L2 Source
         */
        std::string m_deviceLocation;

        /**
         * @brief Device name string for this V4l2SourceBintr. Default size=0
         */
        std::string m_deviceName;
        
        /**
         * @brief Device file-descriptor for this V4l2SourceBintr. Default = -1
         */
        int m_deviceFd;
        
        /**
         * @brief Device type-flags for this V4l2SourceBintr. 
         * Default = DSL_V4L2_DEVICE_TYPE_NONE
         */
        uint m_deviceFlags;

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
         * @brief If TRUE, the base class will automatically timestamp outgoing buffers
         * based on the current running_time..
         */
        boolean m_doTimestamp;

        /**
         * @brief "v4l2src" caps filter 
         */
         
        DSL_ELEMENT_PTR m_pSourceCapsFilter;
        
        /**
         * @brief Video converter, first of two, for the V4L2 Source if dGPU
         */
        DSL_ELEMENT_PTR m_pdGpuVidConv;

    }; 

    //*********************************************************************************
    
    class ResourceSourceBintr: public VideoSourceBintr
    {
    public:
    
        ResourceSourceBintr(const char* name, const char* uri)
            : VideoSourceBintr(name)
            , m_uri(uri)
        {
            LOG_FUNC();
        };
            
        ResourceSourceBintr(const char* name, const char** uris)
            : VideoSourceBintr(name)
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
         * @brief Virtual method to be implented by each derived Resource Source
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
     * @class UriSourceBintr
     * @brief 
     */
    class UriSourceBintr : public ResourceSourceBintr
    {
    public: 
    
        UriSourceBintr(const char* name, const char* uri, bool isLive,
            uint skipFrames, uint dropFrameInterval);

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

        /**
         * @brief Sets the URL for file decode sources only
         * @param uri relative or absolute path to the file decode source
         */
        bool SetFileUri(const char* uri);
        
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
         * @brief Disables Auto Repeat without updating the RepeatEnabled flag 
         * which will take affect on next Play Pipeline command. This function
         * should be called on non-live sources before sending the source an EOS
         */
        void DisableEosConsumer();

        void HandleSourceElementOnPadAdded(GstElement* pBin, GstPad* pPad);
        
    protected:
    
        /**
         * @brief is set to true, non-live source will restart on EOS
         */
        bool m_repeatEnabled;

    private:
    
        /**
         * @brief The common elements are not linked until after the uridecodebin's
         * pad is ready. We don't want to try and unlink unless fully linked. 
         */
        bool m_isFullyLinked;

        /**
         * @brief Additional number of surfaces in addition to min decode surfaces 
         * given by the v4l2 driver. Default = 1.
         */
        uint m_numExtraSurfaces;
        
        /**
         * @brief Type of frames to skip during decoding.
         *   (0): decode_all       - Decode all frames
         *   (1): decode_non_ref   - Decode non-ref frame
         *   (2): decode_key       - decode key frames
         */
        uint m_skipFrames;
        
        /**
         * @brief Interval to drop the frames. Ex: a value of 5 means every 5th 
         * frame will be delivered by decoder, the rest will all dropped.
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
         * @brief mutual exclusion of the repeat enabled setting.
         */
        DslMutex m_repeatEnabledMutex;
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
         * @brief ctor for the ImageSourceBintr
         * @param[in] name unique name for the Image Source
         * @param[in] uri relative or absolute path to the input file source.
         * @param[in] type on of the DSL_IMAGE_TYPE_* constants
         */
        ImageSourceBintr(const char* name, const char* uri, uint type);
        
        /**
         * @brief dtor for the ImageSourceBintr
         */
        ~ImageSourceBintr();
        
    protected:
        /**
         * @brief one of the DSL_IMAGE_EXTENTION_* constants
         */
        std::string m_ext;

        /**
         * @brief one of the DSL_IMAGE_FORMAT_* constants
         */
        uint m_format;
        
        /**
         * @brief JPEG only, set to true if file source is mjpeg.
         */
        boolean m_mjpeg;
        
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
         * @brief ctor for the SingleImageSourceBintr
         * @param[in] name unique name for the Image Source
         * @param[in] uri relative or absolute path to the input file source.
         */
        SingleImageSourceBintr(const char* name, const char* uri);
        
        /**
         * @brief dtor for the ImageSourceBintr
         */
        ~SingleImageSourceBintr();

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
         * @brief dtor for the MultiImageSourceBintr
         */
        ~MultiImageSourceBintr();

        /**
         * @brief Links all Child Elementrs owned by this Source Bintr
         * @return True success, false otherwise.
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elementrs owned by this Source Bintr.
         */
        void UnlinkAll();
        
        /**
         * @brief Sets the URIs for MultiImageSourceBintr 
         * @param uri relative or absolute path to the input file source.
         */
        bool SetUri(const char* uri);
        
        /**
         * @brief Gets the current loop-enabled setting for the 
         * MultiImageSourceBintr.
         * @return true if loop is enabled, false otherwise.
         */
        bool GetLoopEnabled();
        
        /**
         * @brief Sets the loop-enabled setting for the MultiImageSourceBintr.
         * @param[in] loopEnabled set to true to enable, false otherwise.
         * @return true on successful update, false otherwise.
         */
        bool SetLoopEnabled(bool loopEnabled);

        /**
         * @brief Gets the current start and stop index settings for the
         * MultiImageSourceBintr
         * @param[out] startIndex zero-based index to start on. Default = 0
         * @param[out] stopIndex index to stop on. -1 = no stop, default.
         */
        void GetIndices(int* startIndex, int* stopIndex);
        
        /**
         * @brief Sets the start and stop index settings for the
         * MultiImageSourceBintr
         * @param[in] startIndex zero-based index to start on.
         * @param[in] stopIndex index to stop on. set to -1 for no stop.
         */
        bool SetIndices(int startIndex, int stopIndex);
        
    private:
    
        /**
         * @brief Current loop-enabled setting for the MultiImageSourceBintr
         */
        bool m_loopEnabled;

        /**
         * @brief Current start index for the MultiImageSourceBintr
         * Zero-base, default = 0
         */
        int m_startIndex;

        /**
         * @brief Current stop index for the MultiImageSourceBintr
         * Default = -1, no stop.
         */
        int m_stopIndex;

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
         * @brief Sets the URI (filepath) to use by this ImageStreamSourceBintr
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
        DslMutex m_timeoutTimerMutex;

        /**
         * @brief Caps Filter for the ImageStreamSourceBintr
         */
        DSL_ELEMENT_PTR m_pSourceCapsFilter;

        /**
         * @brief Image Overlay element for the FileSourceBintr
         */
        DSL_ELEMENT_PTR m_pImageOverlay;
    };

    //*********************************************************************************

    /**
     * @class InterpipeSourceBintr
     * @brief 
     */
    class InterpipeSourceBintr : public VideoSourceBintr
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
    class RtspSourceBintr : public ResourceSourceBintr
    {
    public: 
    
        RtspSourceBintr(const char* name, const char* uri, uint protocol, 
            uint skipFrames, uint dropFrameInterval, 
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
         * @brief Gets the current reconnection params in use by the named RTSP Source. 
         * The parameters are set to DSL_RTSP_CONNECTION_SLEEP_TIME_MS and 
         * DSL_RTSP_CONNECTION_TIMEOUT_MS on source creation.
         * @param[out] sleep time, in unit of seconds, to sleep after a failed connection.
         * @param[out] timeout time, in units of seconds, to wait before terminating the 
         * current connection attempt and restarting the connection cycle again.
         */
        void GetConnectionParams(uint* sleep, uint* timeout);
        
        /**
         * @brief Sets the current reconnection params in use by the named RTSP Source. 
         * The parameters are set to DSL_RTSP_CONNECTION_SLEEP_TIME_MS and 
         * DSL_RTSP_CONNECTION_TIMEOUT_MS on source creation.
         * Note: calling this service while a reconnection cycle is in progess will terminate
         * the current cycle before restarting with the new parmeters.
         * @param[in] sleep time, in unit of seconds, to sleep after a failed connection.
         * @param[in] timeout time, in units of seconds, to wait before terminating the 
         * current connection attempt and restarting the connection cycle again.
         * @return true is params have been set, false otherwise.
         */
        bool SetConnectionParams(uint sleep, uint timeout);

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
         * @brief NOTE: Used for test purposes only, allows access to the 
         * Source's Timestamp PPH which 
         * is used to maintain a timestamp of the last buffer received for 
         * the source. 
         * @return 
         */
        DSL_PPH_TIMESTAMP_PTR _getTimestampPph(){return m_TimestampPph;};

        /**
         * @brief Gets the current latency setting for the RtspSourceBintr.
         * @return latency in units of ms.
         */
        uint GetLatency();
        
        /**
         * @brief Sets the latency setting for the RtspSourceBintr.
         * @param latency new latency setting in units of ms.
         * @return true if successfully set, false otherwise.
         */
        bool SetLatency(uint latency);
        
        /**
         * @brief Gets the current drop-on-latency enabled setting for the 
         * RspSourceBintr.
         * @return true if enabled, false otherwise.
         */
        boolean GetDropOnLatencyEnabled();
        
        /**
         * @brief Sets the drop-on-latency enabled setting for the RtspSourceBintr.
         * @return true if successfully set, false otherwise.
         */
        bool SetDropOnLatencyEnabled(boolean dropOnLatency);
        
        /**
         * @brief Gets the current tls-validation-flags for the RtspSourceBintr.
         * @return mask of DSL_TLS_CERTIFICATE constants. 
         * Default = DSL_TLS_CERTIFICATE_VALIDATE_ALL.
         */
        uint GetTlsValidationFlags();
        
        /**
         * @brief Sets the tls-validation-flags for the RtspSourceBintr to use.
         * @param[in] flags mask of DSL_TLS_CERTIFICATE constants. 
         * @return true on successful set, false otherwise.
         */
        bool SetTlsValidationFlags(uint flags);
        
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
         * @brief The common elements are not linked until after the rtspsrc
         * has called the select-stream callback. We don't want to try and 
         * unlink unless fully linked. 
         */
        bool m_isFullyLinked;
        
        /**
         * @brief Amount of data to buffer in ms.
         */
        uint m_latency;
        
        /**
         * @brief If true, tells the jitterbuffer to never exceed the given 
         * latency in size.
         */
        boolean m_dropOnLatency;
    
        /**
         @brief 0x4 for TCP and 0x7 for All (UDP/UDP-MCAST/TCP)
         */
        uint m_rtpProtocols;

        /**
         * @brief Additional number of surfaces in addition to min decode surfaces 
         * given by the v4l2 driver. Default = 1.
         */
        uint m_numExtraSurfaces;
        
        /**
         * @brief Type of frames to skip during decoding.
         *   (0): decode_all       - Decode all frames
         *   (1): decode_non_ref   - Decode non-ref frame
         *   (2): decode_key       - decode key frames
         */
        uint m_skipFrames;
        
        /**
         * @brief Interval to drop the frames. Ex: a value of 5 means every 5th 
         * frame will be delivered by decoder, the rest will all dropped.
         */
        uint m_dropFrameInterval;
        
        /**
         * @brief mask of DSL_TLS_CERTIFICATE flags used to validate the
         * RTSP server certificate.
         */
        uint m_tlsValidationFlags;

        /**
         * @brief optional child TapBintr, tapped in pre-decode
         */ 
        DSL_TAP_PTR m_pTapBintr;

        /**
         * @brief H.264 or H.265 RTP Depay for the RtspSourceBintr
         */
        DSL_ELEMENT_PTR m_pDepay;

        /**
         * @brief Depay capsfilter for the RtspSourceBintr
         */
        DSL_ELEMENT_PTR m_pDepayCapsfilter;
        
        /**
         * @brief Pre-parser queue 
         */
        DSL_ELEMENT_PTR m_pPreParserQueue;

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
         * @brief Decoder based on stream encoding type.
         */
        DSL_ELEMENT_PTR m_pDecoder;

        /**
         * @brief Pad Probe Handler to create a timestamp for the last recieved buffer
         */
        DSL_PPH_TIMESTAMP_PTR m_TimestampPph;

        /**
         * @brief time incremented while waiting for first connection in ms.
         */
        uint m_firstConnectTime;
        
        /**
         * @brief maximim time between successive buffers before determining the 
         * connection is lost, 0 to disable 
         */
        uint m_bufferTimeout;
        
        /**
         * @brief gnome timer Id for RTSP stream-status and reconnect management 
         */
        uint m_streamManagerTimerId;
        
        /**
         * @brief mutux to guard the buffer timeout managment read/write attributes.
         */
        DslMutex m_streamManagerMutex;
        
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
        DslMutex m_reconnectionManagerMutex;
        
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
        DslMutex m_stateChangeMutex;

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
