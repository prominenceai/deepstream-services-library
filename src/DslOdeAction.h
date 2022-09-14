/*
The MIT License

Copyright (c) 2019-2022, Prominence AI, Inc.

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

#ifndef _DSL_ODE_ACTION_H
#define _DSL_ODE_ACTION_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslOdeBase.h"
#include "DslSurfaceTransform.h"
#include "DslDisplayTypes.h"
#include "DslPlayerBintr.h"
#include "DslMailer.h"

namespace DSL
{
    
    /**
     * @brief Constants for indexing "pObjectMeta->misc_obj_info" 
     * Triggers add specific metric data for child Actions to act on.
     */
    #define DSL_OBJECT_INFO_PRIMARY_METRIC              0
    #define DSL_OBJECT_INFO_PERSISTENCE                 1
    #define DSL_OBJECT_INFO_DIRECTION                   2
    
    /**
     * @brief Constants for indexing "pFrameMeta->misc_frame_info" 
     * Triggers add specific metric data for child Actions to act on.
     */
    #define DSL_FRAME_INFO_ACTIVE_INDEX                 0
    #define DSL_FRAME_INFO_OCCURRENCES                  1
    #define DSL_FRAME_INFO_OCCURRENCES_DIRECTION_IN     2
    #define DSL_FRAME_INFO_OCCURRENCES_DIRECTION_OUT    3
    
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_ODE_ACTION_PTR std::shared_ptr<OdeAction>

    #define DSL_ODE_ACTION_CUSTOM_PTR std::shared_ptr<CustomOdeAction>
    #define DSL_ODE_ACTION_CUSTOM_NEW(name, clientHandler, clientData) \
        std::shared_ptr<CustomOdeAction>(new CustomOdeAction(name, \
            clientHandler, clientData))
        
    #define DSL_ODE_ACTION_CATPURE_PTR std::shared_ptr<CaptureOdeAction>
    
    #define DSL_ODE_ACTION_CAPTURE_FRAME_PTR std::shared_ptr<CaptureFrameOdeAction>
    #define DSL_ODE_ACTION_CAPTURE_FRAME_NEW(name, outdir, annotate) \
        std::shared_ptr<CaptureFrameOdeAction>(new CaptureFrameOdeAction( \
            name, outdir, annotate))
        
    #define DSL_ODE_ACTION_CAPTURE_OBJECT_PTR std::shared_ptr<CaptureObjectOdeAction>
    #define DSL_ODE_ACTION_CAPTURE_OBJECT_NEW(name, outdir) \
        std::shared_ptr<CaptureObjectOdeAction>(new CaptureObjectOdeAction( \
            name, outdir))

    #define DSL_ODE_ACTION_DISPLAY_PTR std::shared_ptr<DisplayOdeAction>
    #define DSL_ODE_ACTION_DISPLAY_NEW(name, \
        formatString, offsetX, offsetY, pFont, hasBgColor, pBgColor) \
        std::shared_ptr<DisplayOdeAction>(new DisplayOdeAction(name, \
            formatString, offsetX, offsetY, pFont, hasBgColor, pBgColor))

    #define DSL_ODE_ACTION_DISPLAY_META_ADD_PTR std::shared_ptr<AddDisplayMetaOdeAction>
    #define DSL_ODE_ACTION_DISPLAY_META_ADD_NEW(name, displayType) \
        std::shared_ptr<AddDisplayMetaOdeAction>(new AddDisplayMetaOdeAction( \
            name, displayType))
        
    #define DSL_ODE_ACTION_DISABLE_HANDLER_PTR std::shared_ptr<DisableHandlerOdeAction>
    #define DSL_ODE_ACTION_DISABLE_HANDLER_NEW(name, handler) \
        std::shared_ptr<DisableHandlerOdeAction>(new DisableHandlerOdeAction( \
            name, handler))

    #define DSL_ODE_ACTION_EMAIL_PTR std::shared_ptr<EmailOdeAction>
    #define DSL_ODE_ACTION_EMAIL_NEW(name, pMailer, subject) \
        std::shared_ptr<EmailOdeAction>(new EmailOdeAction(name, pMailer, subject))
        
    #define DSL_ODE_ACTION_FILL_AREA_PTR std::shared_ptr<FillAreaOdeAction>
    #define DSL_ODE_ACTION_FILL_AREA_NEW(name, area, pColor) \
        std::shared_ptr<FillAreaOdeAction>(new FillAreaOdeAction(name, area, pColor))

    #define DSL_ODE_ACTION_FILL_FRAME_PTR std::shared_ptr<FillFrameOdeAction>
    #define DSL_ODE_ACTION_FILL_FRAME_NEW(name, pColor) \
        std::shared_ptr<FillFrameOdeAction>(new FillFrameOdeAction(name, pColor))

    #define DSL_ODE_ACTION_FILL_SURROUNDINGS_PTR std::shared_ptr<FillSurroundingsOdeAction>
    #define DSL_ODE_ACTION_FILL_SURROUNDINGS_NEW(name, pColor) \
        std::shared_ptr<FillSurroundingsOdeAction>(new FillSurroundingsOdeAction(name, pColor))

    #define DSL_ODE_ACTION_BBOX_FORMAT_PTR std::shared_ptr<FormatBBoxOdeAction>
    #define DSL_ODE_ACTION_BBOX_FORMAT_NEW(name, \
        borderWidth, pBorderColor, hasBgColor, pBgColor) \
        std::shared_ptr<FormatBBoxOdeAction>(new FormatBBoxOdeAction(name, \
            borderWidth, pBorderColor, hasBgColor, pBgColor))

    #define DSL_ODE_ACTION_BBOX_SCALE_PTR std::shared_ptr<ScaleBBoxOdeAction>
    #define DSL_ODE_ACTION_BBOX_SCALE_NEW(name, scale) \
        std::shared_ptr<ScaleBBoxOdeAction>(new ScaleBBoxOdeAction(name, scale))

    #define DSL_ODE_ACTION_LABEL_CUSTOMIZE_PTR std::shared_ptr<CustomizeLabelOdeAction>
    #define DSL_ODE_ACTION_LABEL_CUSTOMIZE_NEW(name, contentTypes) \
        std::shared_ptr<CustomizeLabelOdeAction>(new CustomizeLabelOdeAction( \
            name, contentTypes))
        
    #define DSL_ODE_ACTION_LABEL_FORMAT_PTR std::shared_ptr<FormatLabelOdeAction>
    #define DSL_ODE_ACTION_LABEL_FORMAT_NEW(name, pFont, hasBgColor, pBgColor) \
        std::shared_ptr<FormatLabelOdeAction>(new FormatLabelOdeAction(name, \
            pFont, hasBgColor, pBgColor))

    #define DSL_ODE_ACTION_LABEL_OFFSET_PTR std::shared_ptr<OffsetLabelOdeAction>
    #define DSL_ODE_ACTION_LABEL_OFFSET_NEW(name, offsetX, offsetY) \
        std::shared_ptr<OffsetLabelOdeAction>(new OffsetLabelOdeAction(name, \
            offsetX, offsetY))

    #define DSL_ODE_ACTION_LOG_PTR std::shared_ptr<LogOdeAction>
    #define DSL_ODE_ACTION_LOG_NEW(name) \
        std::shared_ptr<LogOdeAction>(new LogOdeAction(name))

    #define DSL_ODE_ACTION_MESSAGE_META_ADD_PTR std::shared_ptr<MessageMetaAddOdeAction>
    #define DSL_ODE_ACTION_MESSAGE_META_ADD_NEW(name) \
        std::shared_ptr<MessageMetaAddOdeAction>(new MessageMetaAddOdeAction(name))

    #define DSL_ODE_ACTION_MONITOR_PTR std::shared_ptr<MonitorOdeAction>
    #define DSL_ODE_ACTION_MONITOR_NEW(name, clientMonitor, clientData) \
        std::shared_ptr<MonitorOdeAction>(new MonitorOdeAction(name, \
            clientMonitor, clientData))

    #define DSL_ODE_ACTION_OBJECT_REMOVE_PTR std::shared_ptr<PauseOdeAction>
    #define DSL_ODE_ACTION_OBJECT_REMOVE_NEW(name) \
        std::shared_ptr<RemoveObjectOdeAction>(new RemoveObjectOdeAction(name))
        
    #define DSL_ODE_ACTION_PAUSE_PTR std::shared_ptr<PauseOdeAction>
    #define DSL_ODE_ACTION_PAUSE_NEW(name, pipeline) \
        std::shared_ptr<PauseOdeAction>(new PauseOdeAction(name, pipeline))
        
    #define DSL_ODE_ACTION_PRINT_PTR std::shared_ptr<PrintOdeAction>
    #define DSL_ODE_ACTION_PRINT_NEW(name, forceFlush) \
        std::shared_ptr<PrintOdeAction>(new PrintOdeAction(name, forceFlush))

    #define DSL_ODE_ACTION_FILE_TEXT_PTR std::shared_ptr<FileTextOdeAction>
    #define DSL_ODE_ACTION_FILE_TEXT_NEW(name, filePath, mode, forceFlush) \
        std::shared_ptr<FileTextOdeAction>(new FileTextOdeAction(name, \
            filePath, mode, forceFlush))
        
    #define DSL_ODE_ACTION_FILE_CSV_PTR std::shared_ptr<FileCsvOdeAction>
    #define DSL_ODE_ACTION_FILE_CSV_NEW(name, filePath, mode, forceFlush) \
        std::shared_ptr<FileCsvOdeAction>(new FileCsvOdeAction(name, \
            filePath, mode, forceFlush))
        
    #define DSL_ODE_ACTION_FILE_MOTC_PTR std::shared_ptr<FileOdeAction>
    #define DSL_ODE_ACTION_FILE_MOTC_NEW(name, filePath, mode, forceFlush) \
        std::shared_ptr<FileMotcOdeAction>(new FileMotcOdeAction(name, \
            filePath, mode, forceFlush))
        
    #define DSL_ODE_ACTION_REDACT_PTR std::shared_ptr<RedactOdeAction>
    #define DSL_ODE_ACTION_REDACT_NEW(name) \
        std::shared_ptr<RedactOdeAction>(new RedactOdeAction(name))

    #define DSL_ODE_ACTION_SINK_ADD_PTR std::shared_ptr<AddSinkOdeAction>
    #define DSL_ODE_ACTION_SINK_ADD_NEW(name, pipeline, sink) \
        std::shared_ptr<AddSinkOdeAction>(new AddSinkOdeAction(name, pipeline, sink))
        
    #define DSL_ODE_ACTION_SINK_REMOVE_PTR std::shared_ptr<RemoveSinkOdeAction>
    #define DSL_ODE_ACTION_SINK_REMOVE_NEW(name, pipeline, sink) \
        std::shared_ptr<RemoveSinkOdeAction>(new RemoveSinkOdeAction(name, pipeline, sink))
        
    #define DSL_ODE_ACTION_SOURCE_ADD_PTR std::shared_ptr<AddSourceOdeAction>
    #define DSL_ODE_ACTION_SOURCE_ADD_NEW(name, pipeline, source) \
        std::shared_ptr<AddSourceOdeAction>(new AddSourceOdeAction(name, pipeline, source))
        
    #define DSL_ODE_ACTION_SOURCE_REMOVE_PTR std::shared_ptr<RemoveSourceOdeAction>
    #define DSL_ODE_ACTION_SOURCE_REMOVE_NEW(name, pipeline, source) \
        std::shared_ptr<RemoveSourceOdeAction>(new RemoveSourceOdeAction(name, pipeline, source))
        
    #define DSL_ODE_ACTION_TRIGGER_DISABLE_PTR std::shared_ptr<DisableTriggerOdeAction>
    #define DSL_ODE_ACTION_TRIGGER_DISABLE_NEW(name, trigger) \
        std::shared_ptr<DisableTriggerOdeAction>(new DisableTriggerOdeAction(name, trigger))
        
    #define DSL_ODE_ACTION_TRIGGER_ENABLE_PTR std::shared_ptr<EnableTriggerOdeAction>
    #define DSL_ODE_ACTION_TRIGGER_ENABLE_NEW(name, trigger) \
        std::shared_ptr<EnableTriggerOdeAction>(new EnableTriggerOdeAction(name, trigger))
        
    #define DSL_ODE_ACTION_TRIGGER_RESET_PTR std::shared_ptr<ResetTriggerOdeAction>
    #define DSL_ODE_ACTION_TRIGGER_RESET_NEW(name, trigger) \
        std::shared_ptr<ResetTriggerOdeAction>(new ResetTriggerOdeAction(name, trigger))
        
    #define DSL_ODE_ACTION_ACTION_DISABLE_PTR std::shared_ptr<DisableActionOdeAction>
    #define DSL_ODE_ACTION_ACTION_DISABLE_NEW(name, trigger) \
        std::shared_ptr<DisableActionOdeAction>(new DisableActionOdeAction(name, trigger))
        
    #define DSL_ODE_ACTION_ACTION_ENABLE_PTR std::shared_ptr<EnableActionOdeAction>
    #define DSL_ODE_ACTION_ACTION_ENABLE_NEW(name, trigger) \
        std::shared_ptr<EnableActionOdeAction>(new EnableActionOdeAction(name, trigger))
        
    #define DSL_ODE_ACTION_AREA_ADD_PTR std::shared_ptr<AddAreaOdeAction>
    #define DSL_ODE_ACTION_AREA_ADD_NEW(name, trigger, area) \
        std::shared_ptr<AddAreaOdeAction>(new AddAreaOdeAction(name, trigger, area))
        
    #define DSL_ODE_ACTION_AREA_REMOVE_PTR std::shared_ptr<RemoveAreaOdeAction>
    #define DSL_ODE_ACTION_AREA_REMOVE_NEW(name, trigger, area) \
        std::shared_ptr<RemoveAreaOdeAction>(new RemoveAreaOdeAction(name, trigger, area))
        
    #define DSL_ODE_ACTION_SINK_RECORD_START_PTR std::shared_ptr<RecordSinkStartOdeAction>
    #define DSL_ODE_ACTION_SINK_RECORD_START_NEW(name, pRecordSink, start, duration, clientData) \
        std::shared_ptr<RecordSinkStartOdeAction>(new RecordSinkStartOdeAction(name, \
            pRecordSink, start, duration, clientData))
        
    #define DSL_ODE_ACTION_SINK_RECORD_STOP_PTR std::shared_ptr<RecordSinkStopOdeAction>
    #define DSL_ODE_ACTION_SINK_RECORD_STOP_NEW(name, pRecordSink) \
        std::shared_ptr<RecordSinkStopOdeAction>(new RecordSinkStopOdeAction(name, pRecordSink))
        
    #define DSL_ODE_ACTION_TAP_RECORD_START_PTR std::shared_ptr<RecordTapStartOdeAction>
    #define DSL_ODE_ACTION_TAP_RECORD_START_NEW(name, pRecordTap, start, duration, clientData) \
        std::shared_ptr<RecordTapStartOdeAction>(new RecordTapStartOdeAction(name, \
            pRecordTap, start, duration, clientData))
        
    #define DSL_ODE_ACTION_TAP_RECORD_STOP_PTR std::shared_ptr<RecordTapStopOdeAction>
    #define DSL_ODE_ACTION_TAP_RECORD_STOP_NEW(name, pRecordTap) \
        std::shared_ptr<RecordTapStopOdeAction>(new RecordTapStopOdeAction(name, pRecordTap))
        
    #define DSL_ODE_ACTION_TILER_SHOW_SOURCE_PTR std::shared_ptr<TilerShowSourceOdeAction>
    #define DSL_ODE_ACTION_TILER_SHOW_SOURCE_NEW(name, tiler, timeout, hasPrecedence) \
        std::shared_ptr<TilerShowSourceOdeAction>(new TilerShowSourceOdeAction(name, \
            tiler, timeout, hasPrecedence))
        
        
        
    // ********************************************************************

    class OdeAction : public OdeBase
    {
    public: 
    
        /**
         * @brief ctor for the ODE Action virtual base class
         * @param[in] name unique name for the ODE Action
         */
        OdeAction(const char* name);

        ~OdeAction();

        /**
         * @brief Virtual function to handle the occurrence of an ODE by taking
         * a specific Action as implemented by the derived class
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        virtual void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta) = 0;
        
    protected:

        std::string Ntp2Str(uint64_t ntp);

    };

    // ********************************************************************

    /**
     * @class FormatBBoxOdeAction
     * @brief Format Bounding Box ODE Action class
     */
    class FormatBBoxOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Format BBox ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] borderWidth line width for the bounding box rectangle
         * @param[in] pBorderColor shared pointer to an RGBA Color for the border
         * @param[in] hasBgColor true to fill the background with an RGBA color
         * @param[in] pBgColor shared pointer to an RGBA fill color to use if 
         * hasBgColor = true
         */
        FormatBBoxOdeAction(const char* name, uint borderWidth,
            DSL_RGBA_COLOR_PTR pBorderColor, bool hasBgColor, 
            DSL_RGBA_COLOR_PTR pBgColor);
        
        /**
         * @brief dtor for the ODE Format BBox Action class
         */
        ~FormatBBoxOdeAction();

        /**
         * @brief Handles the ODE occurrence by formating the bounding box of pObjectMeta
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief line width to use for the rectengle 
         */
        uint m_borderWidth;
         
        /**
         * @brief Color used to Fill the object
         */
        DSL_RGBA_COLOR_PTR m_pBorderColor;

        /**
         * @brief true if the object's bounding box is to be filled with color, 
         * false otherwise.
         */
        bool m_hasBgColor;

        /**
         * @brief Background color used to Fill the object's bounding box
         */
        DSL_RGBA_COLOR_PTR m_pBgColor;

    };

    // ********************************************************************

    /**
     * @class ScaleBBoxOdeAction
     * @brief Scale Bounding Box ODE Action class
     */
    class ScaleBBoxOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Scale BBox ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] scale scale factor to apply to each ObjectMeta
         */
        ScaleBBoxOdeAction(const char* name, uint scale);
        
        /**
         * @brief dtor for the ODE Scale BBox Action class
         */
        ~ScaleBBoxOdeAction();

        /**
         * @brief Handles the ODE occurrence by scalling the bounding box of pObjectMeta
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief scale factor to apply to each ObjectMeta 
         */
        uint m_scale;

    };

    // ********************************************************************

    /**
     * @class CustomOdeAction
     * @brief Custom ODE Action class
     */
    class CustomOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Custom ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] clientHandler client callback function to call on ODE
         * @param[in] clientData opaque pointer to client data t return on callback
         */
        CustomOdeAction(const char* name, 
            dsl_ode_handle_occurrence_cb clientHandler, void* clientData);
        
        /**
         * @brief dtor for the ODE Custom Action class
         */
        ~CustomOdeAction();

        /**
         * @brief Handles the ODE occurrence by calling the client handler
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief Client Callback function to call on ODE occurrence
         */
        dsl_ode_handle_occurrence_cb m_clientHandler;
        
        /**
         * @brief pointer to client's data returned on callback
         */ 
        void* m_clientData;

    };
    
    // ********************************************************************

    /**
     * @class CaptureOdeAction
     * @brief ODE Capture Action class
     */
    class CaptureOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Capture ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] captureType DSL_CAPTURE_TYPE_OBJECT or DSL_CAPTURE_TYPE_FRAME.
         * @param[in] outdir output directory to write captured image files.
         */
        CaptureOdeAction(const char* name, 
            uint captureType, const char* outdir, bool annotate);
        
        /**
         * @brief dtor for the Capture ODE Action class
         */
        ~CaptureOdeAction();

        cv::Mat& AnnotateObject(NvDsObjectMeta* pObjectMeta, cv::Mat& bgr_frame);
        
        /**
         * @brief Handles the ODE occurrence by capturing a frame or object image to file
         * @param[in] pOdeTrigger shared pointer to ODE Type that triggered the event
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, 
            std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
            
        /**
         * @brief adds a callback to be notified on Image Capture complete callback
         * @param[in] listener pointer to the client's function to call on capture complete
         * @param[in] userdata opaque pointer to client data passed into the listener function.
         * @return true on successfull add, false otherwise
         */
        bool AddCaptureCompleteListener(dsl_capture_complete_listener_cb listener, 
            void* userdata);
        
        /**
         * @brief removes a previously added Image Capture Complete callback
         * @param[in] listener pointer to the client's function to remove
         * @return true on successfull remove, false otherwise
         */
        bool RemoveCaptureCompleteListener(dsl_capture_complete_listener_cb listener);
        
        /**
         * @brief adds an Image Player, Render or RTSP type, to this CaptureAction
         * @param pPlayer shared pointer to an Image Player to add
         * @return true on successfull add, false otherwise
         */
        bool AddImagePlayer(DSL_PLAYER_BINTR_PTR pPlayer);
        
        /**
         * @brief removes an Image Player, Render or RTSP type, from this CaptureAction
         * @param pPlayer shared pointer to an Image Player to remove
         * @return true on successfull remove, false otherwise
         */
        bool RemoveImagePlayer(DSL_PLAYER_BINTR_PTR pPlayer);
        
        /**
         * @brief adds a SMTP Mailer to this CaptureAction
         * @param[in] pMailer shared pointer to a Mailer to add
         * @param[in] subject subject line to use for all email
         * @param[in] attach boolean flag to optionally attach the image file
         * @return true on successfull add, false otherwise
         */
        bool AddMailer(DSL_MAILER_PTR pMailer, const char* subject, bool attach);
        
        /**
         * @brief removes a SMPT Mailer to this CaptureAction
         * @param[in] pMailer shared pointer to an Mailer to remove
         * @return true on successfull remove, false otherwise
         */
        bool RemoveMailer(DSL_MAILER_PTR pMailer);
        
        /**
         * @brief removes all child Mailers, Players, and Listeners from this parent Object
         */
        void RemoveAllChildren();
        
        /**
         * @brief Queues capture info and starts the Listener notification timer
         * @param info shared pointer to cv::MAT containing the captured image
         */
        void QueueCapturedImage(std::shared_ptr<cv::Mat> pImageMat);
        
        /**
         * @brief implements a timer callback to complete the capture process 
         * by saving the image to file, notifying all client listeners, and 
         * sending email all in the main loop context.
         * @return false always to self remove timer once clients have been notified. 
         * Timer/tread will be restarted on next Image Capture
         */
        int CompleteCapture();
        
    protected:

        /**
         * @brief static, unique capture id shared by all Capture actions
         */
        static uint64_t s_captureId;
    
        /**
         * @brief either DSL_CAPTURE_TYPE_OBJECT or DSL_CAPTURE_TYPE_FRAME
         */
        uint m_captureType;

        /**
         * @brief relative or absolute path to output directory
         */ 
        std::string m_outdir;
        
        /**
         * @brief annotates the image with bbox and label DSL_CAPTURE_TYPE_FRAME only
         */
        bool m_annotate;

        /**
         * @brief mutux to guard the Capture info read/write access.
         */
        GMutex m_captureCompleteMutex;

        /**
         * @brief gnome timer Id for the capture complete callback
         */
        uint m_captureCompleteTimerId;
        
        /**
         * @brief map of all currently registered capture-complete-listeners
         * callback functions mapped with thier user provided data
         */
        std::map<dsl_capture_complete_listener_cb, void*> m_captureCompleteListeners;
        
        /**
         * @brief map of all Image Players to play captured images.
         */
        std::map<std::string, DSL_PLAYER_BINTR_PTR> m_imagePlayers;
        
        /**
         * @brief map of all Mailers to send email.
         */
        std::map<std::string, std::shared_ptr<MailerSpecs>> m_mailers;
        
        /**
         * @brief a queue of captured Images to save to file and notify clients
         */
        std::queue<std::shared_ptr<cv::Mat>> m_imageMats;
    };

    /**
     * @brief Timer callback handler to complete the capture process
     * by notifying all listeners and sending email with all mailers.
     * @param[in] pSource shared pointer to Capture Action to invoke.
     * @return int true to continue, 0 to self remove
     */
    static int CompleteCaptureHandler(gpointer pAction);
    
    // ********************************************************************

    /**
     * @class CaptureFrameOdeAction
     * @brief ODE Capture Frame Action class
     */
    class CaptureFrameOdeAction : public CaptureOdeAction
    {
    public:
    
        /**
         * @brief ctor for the Capture Frame ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] outdir output directory to write captured image files
         * @param[in] annotate adds bbox and label to one or all objects in the frame.
         * One object in the case of valid pObjectMeta on call to HandleOccurrence
         */
        CaptureFrameOdeAction(const char* name, const char* outdir, bool annotate)
            : CaptureOdeAction(name, DSL_CAPTURE_TYPE_FRAME, outdir, annotate)
        {};

    };

    /**
     * @class CaptureObjectOdeAction
     * @brief ODE Capture Object Action class
     */
    class CaptureObjectOdeAction : public CaptureOdeAction
    {
    public:
    
        /**
         * @brief ctor for the Capture Frame ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] outdir output directory to write captured image files
         */
        CaptureObjectOdeAction(const char* name, const char* outdir)
            : CaptureOdeAction(name, DSL_CAPTURE_TYPE_OBJECT, outdir, false)
        {};

    };

    // ********************************************************************

    /**
     * @class CustomizeLabelOdeAction
     * @brief Customize Object Labels ODE Action class
     */
    class CustomizeLabelOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Customize Label ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] contentTypes NULL terminated list of 
         * DSL_OBJECT_LABEL_<type> values for specific content
         */
        CustomizeLabelOdeAction(const char* name, 
            const std::vector<uint>& contentTypes);
        
        /**
         * @brief dtor for the Customize Label ODE Action class
         */
        ~CustomizeLabelOdeAction();

        /**
         * @brief gets the content types in use by this Customize Label Action
         * @return vector of DSL_OBJECT_LABEL_<type> values
         */
        const std::vector<uint> Get();
        
        /**
         * @brief sets the content types for this Customize Label Action to use
         * @param[in] contentTypes new vector of DSL_OBJECT_LABEL_<type> values to use
         */
        void Set(const std::vector<uint>& contentTypes);
        
        /**
         * @brief Handles the ODE occurrence by customizing the Object's label
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
            
    private:
        
        /**
         * @brief Content types for label customization
         */
        std::vector <uint> m_contentTypes;
    };

    // ********************************************************************

    /**
     * @class DisplayOdeAction
     * @brief ODE Display Ode Action class
     */
    class DisplayOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Display ODE Action class
         * @param[in] name unique name for the ODE Action
        * @param[in] formatString string with format tokens for display
         * @param[in] offsetX horizontal X-offset for the ODE occurrence 
         * data to display
         * @param[in] offsetX vertical Y-offset for the ODE occurrence data to display
         * on ODE class Id if set true
         * @param[in] pFont shared pointer to an RGBA Font to use for display
         * @param[in] hasBgColor true to fill the background with an RGBA color
         * @param[in] pBgColor shared pointer to an RGBA fill color to use if 
         * hasBgColor = true
         */
        DisplayOdeAction(const char* name, const char* formatString,
            uint offsetX, uint offsetY, DSL_RGBA_FONT_PTR pFont, 
            bool hasBgColor, DSL_RGBA_COLOR_PTR pBgColor);
        
        /**
         * @brief dtor for the Display ODE Action class
         */
        ~DisplayOdeAction();
        
        /**
         * @brief Handles the ODE occurrence by adding display info
         * using OSD text overlay
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
            
    private:
    
        /**
         * @brief client defined display string with format tokens
         */
        std::string m_formatString;
        
        /**
         * @brief Horizontal X-offset for the ODE occurrence data to display
         */
        uint m_offsetX;
        
        /**
         * @brief Vertical Y-offset for the ODE occurrence data to display
         */
        uint m_offsetY;
        
        /**
         * @brief Font type to use for the displayed occurrence data.
         */
        DSL_RGBA_FONT_PTR m_pFont;
        
        /**
         * true if the Display text has a background color, false otherwise.
         */
        bool m_hasBgColor;
        
        /**
         * @brief the background color to use for the display text if hasBgColor.
         */
        DSL_RGBA_COLOR_PTR m_pBgColor;
        
    };

    // ********************************************************************

    /**
     * @class DisableHandlerOdeAction
     * @brief ODE Disable Handelr ODE Action class
     */
    class DisableHandlerOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Display ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] handler unique name for the ODE Handler to disable
         */
        DisableHandlerOdeAction(const char* name, const char* handler);
        
        /**
         * @brief dtor for the Display ODE Action class
         */
        ~DisableHandlerOdeAction();
        
        /**
         * @brief Handles the ODE occurrence by disabling a named ODE Handler
         * using OSD text overlay
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
            
    private:
    
        /**
         * @brief Unique name of the ODE handler to disable 
         */
        std::string m_handler;
    
    };
    
    // ********************************************************************

    /**
     * @class EmailOdeAction
     * @brief Email ODE Action class
     */
    class EmailOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the ODE Fill Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] pMailer shared pointer
         * @param[in] subject line to use in all emails
         */
        EmailOdeAction(const char* name, 
            DSL_BASE_PTR pMailer, const char* subject);
        
        /**
         * @brief dtor for the ODE Display Action class
         */
        ~EmailOdeAction();
        
        /**
         * @brief Handles the ODE occurrence by queuing and Email with SMTP API
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
            
    private:
    
        /**
         * @bried shared pointer to Mailer object in use by this Action
         */
        DSL_BASE_PTR m_pMailer;
    
        /**
         * @brief Subject line used for all email messages sent by this action
         */
        std::string m_subject;
    
    };
    
    // ********************************************************************

    /**
     * @class LogOdeAction
     * @brief Log Ode Action class
     */
    class LogOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Log ODE Action class
         * @param[in] name unique name for the ODE Action
         */
        LogOdeAction(const char* name);
        
        /**
         * @brief dtor for the Log ODE Action class
         */
        ~LogOdeAction();
        
        /**
         * @brief Handles the ODE occurrence by adding/calling LOG_INFO 
         * with the ODE occurrence data.data
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event.
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event.
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event.
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

    private:
    
    };
        
    // ********************************************************************

    /**
     * @class MessageMetaAddOdeAction
     * @brief Message ODE Action class
     */
    class MessageMetaAddOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Message ODE Action class.
         * @param[in] name unique name for the ODE Action.
         */
        MessageMetaAddOdeAction(const char* name);
        
        /**
         * @brief dtor for the Log ODE Action class
         */
        ~MessageMetaAddOdeAction();
        
        /**
         * @brief Handles the ODE occurrence by adding NvDsEventMsgMeta with
         * the ODE occurrence data.
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event.
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event.
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event.
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
            
        /**
         * @brief Gets the current base_meta.meta_type identifier in use by 
         * the MessageMetaAddOdeAction.
         * @return the current meta-type id in use, default = NVDS_EVENT_MSG_META
         */
        uint GetMetaType();
        
        /**
         * @brief Sets the base_meta.meta_type identifier to use by 
         * the MessageMetaAddOdeAction.
         * @param[in] metaType new meta-type id to use, must be >= NVDS_START_USER_META
         * or = NVDS_EVENT_MSG_META.
         */
        void SetMetaType(uint metaType);

    private:
    
        /**
         * @brief defines the base_meta.meta_type id to use for
         * all message meta created. Default = NVDS_EVENT_MSG_META
         * Custom values must be greater than NVDS_START_USER_META
         * Both constants are defined in nvdsmeta.h 
         */
        uint m_metaType;
    
    };

    // ********************************************************************

    /**
     * @class MonitorOdeAction
     * @brief Monitor ODE Action class
     */
    class MonitorOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Monitor ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] clientMonitor client callback function to call on ODE
         * @param[in] clientData opaque pointer to client data t return on callback
         */
        MonitorOdeAction(const char* name, 
            dsl_ode_monitor_occurrence_cb clientMonitor, void* clientData);
        
        /**
         * @brief dtor for the ODE Monitor Action class
         */
        ~MonitorOdeAction();

        /**
         * @brief Handles the ODE occurrence by calling the client handler
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief Client Callback function to call on ODE occurrence
         */
        dsl_ode_monitor_occurrence_cb m_clientMonitor;
        
        /**
         * @brief pointer to client's data returned on callback
         */ 
        void* m_clientData;

    };
        
    // ********************************************************************

    /**
     * @class FillFrameOdeAction
     * @brief Fill ODE Action class
     */
    class FillFrameOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the ODE Fill Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] pColor shared pointer to an RGBA Color to fill the Frame
         */
        FillFrameOdeAction(const char* name, DSL_RGBA_COLOR_PTR pColor);
        
        /**
         * @brief dtor for the ODE Display Action class
         */
        ~FillFrameOdeAction();
        
        /**
         * @brief Handles the ODE occurrence by adding a rectangle to Fill the Frame
         * with a set of RGBA color values
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
            
    private:
    
        /**
         * @brief Background color used to Fill the object
         */
        DSL_RGBA_COLOR_PTR m_pColor;
    
    };


    // ********************************************************************

    /**
     * @class FillSurroundingOdeAction
     * @brief Fill Surroundings ODE Action class
     */
    class FillSurroundingsOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the ODE Fill Surroundings Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] pColor shared pointer to an RGBA Color to fill the Object
         */
        FillSurroundingsOdeAction(const char* name, DSL_RGBA_COLOR_PTR pColor);
        
        /**
         * @brief dtor for the ODE Display Action class
         */
        ~FillSurroundingsOdeAction();
        
        /**
         * @brief Handles the ODE occurrence by Filling the object's surrounding 
         * using four adjacent rectangles. 
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
            
    private:
    
        /**
         * @brief Background color used to Fill everything but the object
         */
        DSL_RGBA_COLOR_PTR m_pColor;
    
    };


    // ********************************************************************

    /**
     * @class AddDisplayMetaOdeAction
     * @brief Add Display Meta ODE Action class
     */
    class AddDisplayMetaOdeAction : public OdeAction 
    {
    public:
    
        /**
         * @brief ctor for the AddDisplayMeta ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] shared pointer to the 
         */
        AddDisplayMetaOdeAction(const char* name, DSL_DISPLAY_TYPE_PTR pDisplayType);
        
        /**
         * @brief dtor for the AddDisplayMeta ODE Action class
         */
        ~AddDisplayMetaOdeAction();
        
        /**
         * @brief adds an additional Display Type for adding metadata
         */
        void AddDisplayType(DSL_DISPLAY_TYPE_PTR pDisplayType);
        
        /**
         * @brief Handles the ODE by overlaying the pFrameMeta with the named Display Type
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

    private:
    
        std::vector<DSL_DISPLAY_TYPE_PTR> m_pDisplayTypes;
    
    };

    // ********************************************************************

    /**
     * @class FormatLabelOdeAction
     * @brief Format Object Label ODE Action class
     */
    class FormatLabelOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Format Label ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] pFont shared pointer to an RGBA Font for the object's label
         * @param[in] hasBgColor true to fill the label background with an RGBA color
         * @param[in] pBgColor shared pointer to an RGBA color to use if 
         * hasBgColor = true
         */
        FormatLabelOdeAction(const char* name,
            DSL_RGBA_FONT_PTR pFont, bool hasBgColor, DSL_RGBA_COLOR_PTR pBgColor);
        
        /**
         * @brief dtor for the ODE Format Label Action class
         */
        ~FormatLabelOdeAction();

        /**
         * @brief Handles the ODE occurrence by calling the client handler
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief Font to use for the object's label
         */
        DSL_RGBA_FONT_PTR m_pFont;

        /**
         * @brief true if the object's label is to have a background color, 
         * false otherwise.
         */
        bool m_hasBgColor;

        /**
         * @brief Background color used for the object's label
         */
        DSL_RGBA_COLOR_PTR m_pBgColor;

    };

    // ********************************************************************

    /**
     * @class OffsetLabelOdeAction
     * @brief Offset Object Label ODE Action class
     */
    class OffsetLabelOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Offset Label ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] offsetX horizontal offset from the default top left 
         * bounding box corner.
         * @param[in] offsetY vertical offset from the default top left 
         * bounding box corner.
         */
        OffsetLabelOdeAction(const char* name,
            int offsetX, int offsetY);
        
        /**
         * @brief dtor for the ODE Format Label Action class
         */
        ~OffsetLabelOdeAction();

        /**
         * @brief Handles the ODE occurrence by calling the client handler
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief horizontal offset from the default top left 
         * bounding box corner.
         */
        int m_offsetX;

        /**
         * @brief vertical offset from the default top left 
         * bounding box corner.
         */
        int m_offsetY;

    };
    
    // ********************************************************************

    /**
     * @class RemoveObjectOdeAction
     * @brief Remove Object ODE Action class
     */
    class RemoveObjectOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Remove Object ODE Action class
         * @param[in] name unique name for the ODE Action
         */
        RemoveObjectOdeAction(const char* name);
        
        /**
         * @brief dtor for the Pause ODE Action class
         */
        ~RemoveObjectOdeAction();
        
        /**
         * @brief Handles the ODE occurrence by removing pObjectMeta from pFrameMeta
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

    private:
    
    };
        
    // ********************************************************************

    /**
     * @class PauseOdeAction
     * @brief Pause ODE Action class
     */
    class PauseOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Pause ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] pipeline unique name of the pipeline to pause on ODE occurrence
         */
        PauseOdeAction(const char* name, const char* pipeline);
        
        /**
         * @brief dtor for the Pause ODE Action class
         */
        ~PauseOdeAction();
        
        /**
         * @brief Handles the ODE occurrence by pausing a named Pipeline
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

    private:
    
        std::string m_pipeline;
    
    };
        
    // ********************************************************************

    /**
     * @class PrintOdeAction
     * @brief Print ODE Action class
     */
    class PrintOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the ODE Print Action class
         * @param[in] name unique name for the ODE Action
         */
        PrintOdeAction(const char* name, bool forceFlush);
        
        /**
         * @brief dtor for the Print ODE Action class
         */
        ~PrintOdeAction();
        
        /**
         * @brief Handles the ODE occurrence by printing the  
         * the occurrence data to the console
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

        /**
         * @brief Flushes the stdout buffer. ** To be called by the idle thread only **.
         * @return false to unschedule always - single flush operation.
         */
        bool Flush();

    private:

        /**
         * @brief flag to enable/disable forced stream buffer flushing
         */
        bool m_forceFlush;
    
        /**
         * @brief gnome thread id for the background thread to flush
         */
        uint m_flushThreadFunctionId;

        /**
         * @brief mutex to protect mutual access to m_flushThreadFunctionId
         */
        GMutex m_ostreamMutex;
    };

    /**
     * @brief Idle Thread Function to flush the stdout buffer
     * @param pAction pointer to the Print Action to call flush
     * @return false to unschedule always
     */
    static gboolean PrintActionFlush(gpointer pAction);

    // ********************************************************************

    /**
     * @class FileOdeAction
     * @brief File ODE Action class
     */
    class FileOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the ODE File Action class
         * @param[in] filePath absolute or relative path to the output file.
         * @param[in] mode open/write mode - truncate or append
         * @param[in] forceFlush unique name for the ODE Action
         */
        FileOdeAction(const char* name, 
            const char* filePath, uint mode, bool forceFlush);
        
        /**
         * @brief dtor for the File ODE Action class
         */
        ~FileOdeAction();
        
        /**
         * @brief Flushes the ostream buffer. ** To be called by the idle thread only **.
         * @return false to unschedule always - single flush operation.
         */
        bool Flush();

    protected:
    
        /**
         * @brief relative or absolute path to the file to write to
         */ 
        std::string m_filePath;
        
        /**
         * @brief specifies the file open mode, DSL_WRITE_MODE_APPEND or
         * DSL_WRITE_MODE_OVERWRITE
         */
        uint m_mode;
        
        /**
         * @brief output stream for all file writes
         */
        std::fstream m_ostream;
        
        /**
         * @brief flag to enable/disable forced stream buffer flushing
         */
        bool m_forceFlush;
    
        /**
         * @brief gnome thread id for the background thread to flush
         */
        uint m_flushThreadFunctionId;

        /**
         * @brief mutex to protect mutual access to comms data
         */
        GMutex m_ostreamMutex;
    };

    /**
     * @brief Idle Thread Function to flush the ostream buffer
     * @param pAction pointer to the File Action to call flush
     * @return false to unschedule always
     */
    static gboolean FileActionFlush(gpointer pAction);

    /**
     * @class FileTextOdeAction
     * @brief Text File ODE Action class
     */
    class FileTextOdeAction : public FileOdeAction
    {
    public:
    
        /**
         * @brief ctor for the ODE Text File Action class
         * @param[in] filePath absolute or relative path to the output file.
         * @param[in] mode open/write mode - truncate or append
         * @param[in] forceFlush unique name for the ODE Action
         */
        FileTextOdeAction(const char* name, 
            const char* filePath, uint mode, bool forceFlush);
        
        /**
         * @brief dtor for the ODE Text Action class
         */
        ~FileTextOdeAction();
        
        /**
         * @brief Handles the ODE occurrence by writing the occurrence data to file
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
    
    };

    /**
     * @class FileCsvOdeAction
     * @brief CSV File ODE Action class
     */
    class FileCsvOdeAction : public FileOdeAction
    {
    public:
    
        /**
         * @brief ctor for the ODE Text File Action class
         * @param[in] filePath absolute or relative path to the output file.
         * @param[in] mode open/write mode - truncate or append
         * @param[in] forceFlush unique name for the ODE Action
         */
        FileCsvOdeAction(const char* name, 
            const char* filePath, uint mode, bool forceFlush);
        
        /**
         * @brief dtor for the ODE Text Action class
         */
        ~FileCsvOdeAction();
        
        /**
         * @brief Handles the ODE occurrence by writing the occurrence data to file
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
    
    };

    /**
     * @class FileMotcOdeAction
     * @brief MOT Challenge File ODE Action class
     */
    class FileMotcOdeAction : public FileOdeAction
    {
    public:
    
        /**
         * @brief ctor for the ODE MOT Challenge File Action class
         * @param[in] filePath absolute or relative path to the output file.
         * @param[in] mode open/write mode - truncate or append
         * @param[in] forceFlush unique name for the ODE Action
         */
        FileMotcOdeAction(const char* name, 
            const char* filePath, uint mode, bool forceFlush);
        
        /**
         * @brief dtor for the ODE MOT Challenge Action class
         */
        ~FileMotcOdeAction();
        
        /**
         * @brief Handles the ODE occurrence by writing the occurrence data to file.
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event.
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event.
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event.
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
    
    };
        
    // ********************************************************************

    /**
     * @class RedactOdeAction
     * @brief Redact ODE Action class
     */
    class RedactOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the ODE Redact Action class
         * @param[in] name unique name for the ODE Action
         */
        RedactOdeAction(const char* name);
        
        /**
         * @brief dtor for the ODE Display Action class
         */
        ~RedactOdeAction();
        
        /**
         * @brief Handles the ODE occurrence by redacting the object 
         * with the boox coordinates in the ODE occurrence data
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
            
    private:
    
    };

    // ********************************************************************

    /**
     * @class AddSinkOdeAction
     * @brief Add Sink ODE Action class
     */
    class AddSinkOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Add Sink ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] pipeline unique name of the Pipeline to add the Sink to
         * @param[in] sink unique name of the Sink to add to the Pipeline
         */
        AddSinkOdeAction(const char* name, const char* pipeline, const char* sink);
        
        /**
         * @brief dtor for the Add Sink ODE Action class
         */
        ~AddSinkOdeAction();

        /**
         * @brief Handles the ODE occurrence by adding a named Sink to a named Pipeline
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief Pipeline to add the Sink to on ODE occurrence
         */
        std::string m_pipeline;
        
        /**
         * @brief Sink to add to the Pipeline on ODE occurrence
         */ 
        std::string m_sink;

    };
    
    // ********************************************************************

    /**
     * @class RemoveSinkOdeAction
     * @brief Remove Sink ODE Action class
     */
    class RemoveSinkOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Remove Sink ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] pipeline unique name of the Pipeline to add the Sink to
         * @param[in] sink unique name of the Sink to add to the Pipeline
         */
        RemoveSinkOdeAction(const char* name, const char* pipeline, const char* sink);
        
        /**
         * @brief dtor for the Remove Sink ODE Action class
         */
        ~RemoveSinkOdeAction();

        /**
         * @brief Handles the ODE occurrence by removing a named Sink from a named Pipeline
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief Pipeline to remove the Sink from on ODE occurrence
         */
        std::string m_pipeline;
        
        /**
         * @brief Sink to from the Pipeline on ODE occurrence
         */ 
        std::string m_sink;

    };
    
        // ********************************************************************

    /**
     * @class AddSourceOdeAction
     * @brief Add Source ODE Action class
     */
    class AddSourceOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Add Source ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] pipeline unique name of the Pipeline to add the Source to
         * @param[in] source unique name of the Source to add to the Pipeline
         */
        AddSourceOdeAction(const char* name, const char* pipeline, const char* source);
        
        /**
         * @brief dtor for the Add Source ODE Action class
         */
        ~AddSourceOdeAction();

        /**
         * @brief Handles the ODE occurrence by adding a named Source to a named Pipeline
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief Pipeline to add the Source to on ODE occurrence
         */
        std::string m_pipeline;
        
        /**
         * @brief Source to add to the Pipeline on ODE occurrence
         */ 
        std::string m_source;

    };
    
    // ********************************************************************

    /**
     * @class RemoveSourceOdeAction
     * @brief Remove Source ODE Action class
     */
    class RemoveSourceOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Remove Source ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] pipeline unique name of the Pipeline to add the Sink to
         * @param[in] souce unique name of the Sink to add to the Pipeline
         */
        RemoveSourceOdeAction(const char* name, const char* pipeline, const char* source);
        
        /**
         * @brief dtor for the Remove Source ODE Action class
         */
        ~RemoveSourceOdeAction();

        /**
         * @brief Handles the ODE occurrence by removing a named Soure from a named Pipeline
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief Pipeline to remove the source from on ODE occurrence
         */
        std::string m_pipeline;
        
        /**
         * @brief Source to remove from the Pipeline on ODE occurrence
         */ 
        std::string m_source;

    };
    
    // ********************************************************************

    /**
     * @class DisableTriggerOdeAction
     * @brief Disable Trigger ODE Action class
     */
    class DisableTriggerOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Disable Trigger ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] trigger ODE Trigger to disable on ODE occurrence
         */
        DisableTriggerOdeAction(const char* name, const char* trigger);
        
        /**
         * @brief dtor for the Disable Trigger ODE Action class
         */
        ~DisableTriggerOdeAction();

        /**
         * @brief Handles the ODE occurrence by disabling a named ODE Trigger
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pBaseTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief ODE Trigger to disable on ODE occurrence
         */
        std::string m_trigger;

    };
    
    // ********************************************************************

    /**
     * @class EnableTriggerOdeAction
     * @brief Enable Trigger ODE Action class
     */
    class EnableTriggerOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Enable Trigger ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] trigger ODE Trigger to disable on ODE occurrence
         */
        EnableTriggerOdeAction(const char* name, const char* trigger);
        
        /**
         * @brief dtor for the Enable Trigger ODE Action class
         */
        ~EnableTriggerOdeAction();

        /**
         * @brief Handles the ODE occurrence by enabling a named ODE Trigger
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief ODE Trigger to enable on ODE occurrence
         */
        std::string m_trigger;

    };
    
    // ********************************************************************

    /**
     * @class ResetTriggerOdeAction
     * @brief Reset Trigger ODE Action class
     */
    class ResetTriggerOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Reset Trigger ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] trigger ODE Trigger to Rest on ODE occurrence
         */
        ResetTriggerOdeAction(const char* name, const char* trigger);
        
        /**
         * @brief dtor for the Reset Trigger ODE Action class
         */
        ~ResetTriggerOdeAction();

        /**
         * @brief Handles the ODE occurrence by reseting a named ODE Trigger
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pBaseTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief ODE Trigger to reset on ODE occurrence
         */
        std::string m_trigger;

    };
    
    // ********************************************************************

    /**
     * @class DisableActionOdeAction
     * @brief Disable Action ODE Action class
     */
    class DisableActionOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Disable Action ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] trigger ODE Trigger to disable on ODE occurrence
         */
        DisableActionOdeAction(const char* name, const char* action);
        
        /**
         * @brief dtor for the Disable Action ODE Action class
         */
        ~DisableActionOdeAction();

        /**
         * @brief Handles the ODE occurrence by disabling a named ODE Action
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief ODE Action to disable on ODE occurrence
         */
        std::string m_action;

    };
    
    // ********************************************************************

    /**
     * @class EnableActionOdeAction
     * @brief Enable Action ODE Action class
     */
    class EnableActionOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Enable Action ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] action ODE Action to enabled on ODE occurrence
         */
        EnableActionOdeAction(const char* name, const char* action);
        
        /**
         * @brief dtor for the Enable Action ODE class
         */
        ~EnableActionOdeAction();

        /**
         * @brief Handles the ODE occurrence by enabling a named ODE Trigger
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief ODE Action to enable on ODE occurrence
         */
        std::string m_action;

    };
    
    // ********************************************************************

    /**
     * @class AddAreaOdeAction
     * @brief Add Area ODE Action class
     */
    class AddAreaOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Add Area ODE Action class
         * @param[in] name unique name for the Add Area ODE Action
         * @param[in] trigger ODE Trigger to add the ODE Area to
         * @param[in] action ODE Area to add on ODE occurrence
         */
        AddAreaOdeAction(const char* name, const char* trigger, const char* area);
        
        /**
         * @brief dtor for the Add Area ODE Action class
         */
        ~AddAreaOdeAction();

        /**
         * @brief Handles the ODE occurrence by adding an ODE Area to an ODE Trigger
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief ODE Trigger to add the ODE Area to
         */ 
        std::string m_trigger;

        /**
         * @brief ODE Area to add to the ODE Trigger on ODE occurrence
         */
        std::string m_area;
    };

    // ********************************************************************

    /**
     * @class RemoveAreaOdeAction
     * @brief Remove Area ODE Action class
     */
    class RemoveAreaOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Remove Area ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] trigger ODE Trigger to add the ODE Area to
         * @param[in] action ODE Area to add on ODE occurrence
         */
        RemoveAreaOdeAction(const char* name, const char* trigger, const char* area);
        
        /**
         * @brief dtor for the Remove Area ODE Action class
         */
        ~RemoveAreaOdeAction();

        /**
         * @brief Handles the ODE occurrence by removing an ODE Area from an ODE Trigger
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief ODE Trigger to remove the ODE Action from
         */ 
        std::string m_trigger;

        /**
         * @brief ODE Area to remove from the ODE Trigger on ODE occurrence
         */
        std::string m_area;
    };

    // ********************************************************************

    /**
     * @class RecordSinkStartOdeAction
     * @brief Start Record Sink ODE Action class
     */
    class RecordSinkStartOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Start Record Sink ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] pRecordSink shared pointer to Record Sink to Start on ODE
         * @param[in] start time before current time in secs
         * @param[in] duration for recording unless stopped before completion
         */
        RecordSinkStartOdeAction(const char* name, 
            DSL_BASE_PTR pRecordSink, uint start, uint duration, void* clientData);
        
        /**
         * @brief dtor for the Start Record ODE Action class
         */
        ~RecordSinkStartOdeAction();

        /**
         * @brief Handles the ODE occurrence by Starting a Video Recording Session
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief Record Sink to start the recording session
         */ 
         DSL_BASE_PTR m_pRecordSink;

        /**
         * @brief Start time before current time in seconds
         */
        uint m_start;

        /**
         * @brief Duration for recording in seconds
         */
        uint m_duration;
        
        /**
         * @brief client Data for client listening for recording session complete/stopped
         */
        void* m_clientData;
        
    };

    // ********************************************************************

    /**
     * @class RecordSinkStopOdeAction
     * @brief Stop Record Sink ODE Action class
     */
    class RecordSinkStopOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Stop Record Sink ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] pRecordSink shared pointer to a Record Sink to Stop on ODE
         * @param[in] duration for recording unless stopped before completion
         */
        RecordSinkStopOdeAction(const char* name, DSL_BASE_PTR pRecordSink);
        
        /**
         * @brief dtor for the Stop Record ODE Action class
         */
        ~RecordSinkStopOdeAction();

        /**
         * @brief Handles the ODE occurrence by Stoping a Video Recording Session
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief Record Sink to start the recording session
         */ 
        DSL_BASE_PTR m_pRecordSink;
        
    };

    // ********************************************************************

    /**
     * @class RecordTapOdeAction
     * @brief Start Record Tap ODE Action class
     */
    class RecordTapStartOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Start Record Tap ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] pRecordTap shared pointer to a Record Tap to Start on ODE
         * @param[in] start time before current time in seconds
         * @param[in] duration for recording unless stopped before completion
         */
        RecordTapStartOdeAction(const char* name, 
            DSL_BASE_PTR pRecordTap, uint start, uint duration, void* clientData);
        
        /**
         * @brief dtor for the Start Record ODE Action class
         */
        ~RecordTapStartOdeAction();

        /**
         * @brief Handles the ODE occurrence by Starting a Video Recording Session
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief Record Tap to start the recording session
         */ 
        DSL_BASE_PTR m_pRecordTap;

        /**
         * @brief Start time before current time in seconds
         */
        uint m_start;

        /**
         * @brief Duration for recording in seconds
         */
        uint m_duration;
        
        /**
         * @brief client Data for client listening for recording session complete/stopped
         */
        void* m_clientData;
    };

    // ********************************************************************

    /**
     * @class RecordTapOdeAction
     * @brief Stop Record Tap ODE Action class
     */
    class RecordTapStopOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Stop Record Tap ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] pRecordSink shared pointer to Record Sink to Stop on ODE
         * @param[in] start time before current time in secs
         * @param[in] duration for recording unless stopped before completion
         */
        RecordTapStopOdeAction(const char* name, DSL_BASE_PTR pRecordTap);
        
        /**
         * @brief dtor for the Stop Record ODE Action class
         */
        ~RecordTapStopOdeAction();

        /**
         * @brief Handles the ODE occurrence by Stoping a Video Recording Session
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief Record Tap to start the recording session
         */ 
        DSL_BASE_PTR m_pRecordTap;

    };

    // ********************************************************************

    /**
     * @class TilerShowSourceOdeAction
     * @brief Tiler Show Source ODE Action class
     */
    class TilerShowSourceOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Tiler Show Source ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] tiler name of the tiler to call on to show source on ODE occurrence
         * @param[in] timeout show source timeout to pass to the Tiler, in units of seconds
         */
        TilerShowSourceOdeAction(const char* name, const char* tiler, uint timeout, bool hasPrecedence);
        
        /**
         * @brief dtor for the Enable Action ODE class
         */
        ~TilerShowSourceOdeAction();

        /**
         * @brief Handles the ODE occurrence by calling a named tiler to show the source for the frame
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, 
            GstBuffer* pBuffer, std::vector<NvDsDisplayMeta*>& displayMetaData,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief Tiler to call to show source on ODE occurrence
         */
        std::string m_tiler;
        
        /**
         * @brief show source timeout to pass to the Tiler in units of seconds
         */
        uint m_timeout;
        
        /**
         * @brief if true, the show source action will take precedence over a currently shown single source
         */
        bool m_hasPrecedence;

    };

}

#endif // _DSL_ODE_ACTION_H
