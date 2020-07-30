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

#ifndef _DSL_ODE_ACTION_H
#define _DSL_ODE_ACTION_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslBase.h"
#include "DslDisplayTypes.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_ODE_ACTION_PTR std::shared_ptr<OdeAction>

    #define DSL_ODE_ACTION_CALLBACK_PTR std::shared_ptr<CallbackOdeAction>
    #define DSL_ODE_ACTION_CALLBACK_NEW(name, clientHandler, clientData) \
        std::shared_ptr<CallbackOdeAction>(new CallbackOdeAction(name, clientHandler, clientData))
        
    #define DSL_ODE_ACTION_CAPTURE_FRAME_PTR std::shared_ptr<CaptureFrameOdeAction>
    #define DSL_ODE_ACTION_CAPTURE_FRAME_NEW(name, outdir) \
        std::shared_ptr<CaptureFrameOdeAction>(new CaptureFrameOdeAction(name, outdir))
        
    #define DSL_ODE_ACTION_CAPTURE_OBJECT_PTR std::shared_ptr<CaptureObjectOdeAction>
    #define DSL_ODE_ACTION_CAPTURE_OBJECT_NEW(name, outdir) \
        std::shared_ptr<CaptureObjectOdeAction>(new CaptureObjectOdeAction(name, outdir))
        
    #define DSL_ODE_ACTION_DISPLAY_PTR std::shared_ptr<DisplayOdeAction>
    #define DSL_ODE_ACTION_DISPLAY_NEW(name, offsetX, offsetY, offsetYWithClassId, pFont, hasBgColor, pBgColor) \
        std::shared_ptr<DisplayOdeAction>(new DisplayOdeAction(name, \
            offsetX, offsetY, offsetYWithClassId, pFont, hasBgColor, pBgColor))
        
    #define DSL_ODE_ACTION_DISABLE_HANDLER_PTR std::shared_ptr<DisableHandlerOdeAction>
    #define DSL_ODE_ACTION_DISABLE_HANDLER_NEW(name, handler) \
        std::shared_ptr<DisableHandlerOdeAction>(new DisableHandlerOdeAction(name, handler))
        
    #define DSL_ODE_ACTION_FILL_AREA_PTR std::shared_ptr<FillAreaOdeAction>
    #define DSL_ODE_ACTION_FILL_AREA_NEW(name, area, pColor) \
        std::shared_ptr<FillAreaOdeAction>(new FillAreaOdeAction(name, area, pColor))

    #define DSL_ODE_ACTION_FILL_FRAME_PTR std::shared_ptr<FillFrameOdeAction>
    #define DSL_ODE_ACTION_FILL_FRAME_NEW(name, pColor) \
        std::shared_ptr<FillFrameOdeAction>(new FillFrameOdeAction(name, pColor))

    #define DSL_ODE_ACTION_FILL_OBJECT_PTR std::shared_ptr<FillObjectOdeAction>
    #define DSL_ODE_ACTION_FILL_OBJECT_NEW(name, pColor) \
        std::shared_ptr<FillObjectOdeAction>(new FillObjectOdeAction(name, pColor))

    #define DSL_ODE_ACTION_FILL_SURROUNDINGS_PTR std::shared_ptr<FillSurroundingsOdeAction>
    #define DSL_ODE_ACTION_FILL_SURROUNDINGS_NEW(name, pColor) \
        std::shared_ptr<FillSurroundingsOdeAction>(new FillSurroundingsOdeAction(name, pColor))

    #define DSL_ODE_ACTION_HIDE_PTR std::shared_ptr<HideOdeAction>
    #define DSL_ODE_ACTION_HIDE_NEW(name, text, border) \
        std::shared_ptr<HideOdeAction>(new HideOdeAction(name, text, border))
        
    #define DSL_ODE_ACTION_LOG_PTR std::shared_ptr<LogOdeAction>
    #define DSL_ODE_ACTION_LOG_NEW(name) \
        std::shared_ptr<LogOdeAction>(new LogOdeAction(name))
        
    #define DSL_ODE_ACTION_OVERLAY_FRAME_PTR std::shared_ptr<OverlayFrameOdeAction>
    #define DSL_ODE_ACTION_OVERLAY_FRAME_NEW(name, displayType) \
        std::shared_ptr<OverlayFrameOdeAction>(new OverlayFrameOdeAction(name, displayType))
        
    #define DSL_ODE_ACTION_PAUSE_PTR std::shared_ptr<PauseOdeAction>
    #define DSL_ODE_ACTION_PAUSE_NEW(name, pipeline) \
        std::shared_ptr<PauseOdeAction>(new PauseOdeAction(name, pipeline))
        
    #define DSL_ODE_ACTION_PRINT_PTR std::shared_ptr<PrintOdeAction>
    #define DSL_ODE_ACTION_PRINT_NEW(name) \
        std::shared_ptr<PrintOdeAction>(new PrintOdeAction(name))
        
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
    #define DSL_ODE_ACTION_SINK_RECORD_START_NEW(name, recordSink, start, duration, clientData) \
        std::shared_ptr<RecordSinkStartOdeAction>(new RecordSinkStartOdeAction(name, recordSink, start, duration, clientData))
        
    #define DSL_ODE_ACTION_TAP_RECORD_START_PTR std::shared_ptr<RecordTapStartOdeAction>
    #define DSL_ODE_ACTION_TAP_RECORD_START_NEW(name, recordTap, start, duration, clientData) \
        std::shared_ptr<RecordTapStartOdeAction>(new RecordTapStartOdeAction(name, recordTap, start, duration, clientData))
        
        
    // ********************************************************************

    class OdeAction : public Base
    {
    public: 
    
        /**
         * @brief ctor for the ODE virtual base class
         * @param[in] name unique name for the ODE Action
         */
        OdeAction(const char* name);

        ~OdeAction();

        /**
         * @brief Gets the current Enabled setting, default = true
         * @return the current Enabled setting
         */
        bool GetEnabled();
        
        /**
         * @brief Sets the Enabled setting for ODE Action
         * @param[in] the new value to use
         */
        void SetEnabled(bool enabled);
        
        /**
         * @brief Virtual function to handle the occurrence of an ODE by taking
         * a specific Action as implemented by the derived class
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        virtual void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta) = 0;
        
    protected:

        /**
         * @brief enabled flag.
         */
        bool m_enabled;
    };

    // ********************************************************************

    /**
     * @class CallbackOdeAction
     * @brief Callback ODE Action class
     */
    class CallbackOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Callback ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] clientHandler client callback function to call on ODE
         * @param[in] clientData opaque pointer to client data t return on callback
         */
        CallbackOdeAction(const char* name, dsl_ode_handle_occurrence_cb clientHandler, void* clientData);
        
        /**
         * @brief dtor for the ODE Callback Action class
         */
        ~CallbackOdeAction();

        /**
         * @brief Handles the ODE occurrence by calling the client handler
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta,
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
         * @param[in] captureType DSL_CAPTURE_TYPE_OBJECT or DSL_CAPTURE_TYPE_FRAME
         * @param[in] outdir output directory to write captured image files
         */
        CaptureOdeAction(const char* name, uint captureType, const char* outdir);
        
        /**
         * @brief dtor for the Capture ODE Action class
         */
        ~CaptureOdeAction();

        /**
         * @brief Handles the ODE occurrence by capturing a frame or object image to file
         * @param[in] pOdeTrigger shared pointer to ODE Type that triggered the event
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    protected:
    
        /**
         * @brief either DSL_CAPTURE_TYPE_OBJECT or DSL_CAPTURE_TYPE_FRAME
         */
        uint m_captureType;
        
        /**
         * @brief relative or absolute path to output directory
         */ 
        std::string m_outdir;

    };
    
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
         */
        CaptureFrameOdeAction(const char* name, const char* outdir)
            : CaptureOdeAction(name, DSL_CAPTURE_TYPE_FRAME, outdir)
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
            : CaptureOdeAction(name, DSL_CAPTURE_TYPE_OBJECT, outdir)
        {};

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
         * @param[in] offsetX horizontal X-offset for the ODE occurrence data to display
         * @param[in] offsetX vertical Y-offset for the ODE occurrence data to display
         * @param[in] offsetYWithClassId adds an additional offset based on ODE class Id if set true
         */
        DisplayOdeAction(const char* name, uint offsetX, uint offsetY, bool offsetYWithClassId,
            DSL_RGBA_FONT_PTR pFont, bool hasBgColor, DSL_RGBA_COLOR_PTR pBgColor);
        
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
            
    private:
    
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
        
        /**
         * @brief Adds an additional offset based on ODE class Id if set true
         */
        bool m_offsetYWithClassId;
    
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
            
    private:
    
        /**
         * @brief Unique name of the ODE handler to disable 
         */
        std::string m_handler;
    
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
         * with the ODE occurrence data data
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

    private:
    
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
            
    private:
    
        /**
         * @brief Background color used to Fill the object
         */
        DSL_RGBA_COLOR_PTR m_pColor;
    
    };

    // ********************************************************************

    /**
     * @class FillObjectOdeAction
     * @brief Fill ODE Action class
     */
    class FillObjectOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the ODE Fill Object Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] pColor shared pointer to an RGBA Color to fill the Object
         */
        FillObjectOdeAction(const char* name, DSL_RGBA_COLOR_PTR pColor);
        
        /**
         * @brief dtor for the ODE Display Action class
         */
        ~FillObjectOdeAction();
        
        /**
         * @brief Handles the ODE occurrence by Filling the object's rectangle background 
         * with a set of RGBA color values
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta,
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
            
    private:
    
        /**
         * @brief Background color used to Fill everything but the object
         */
        DSL_RGBA_COLOR_PTR m_pColor;
    
    };

    // ********************************************************************

    /**
     * @class HideOdeAction
     * @brief Hide Ode Action class
     */
    class HideOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Hide ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] text if true, hides the Object's Display Text on HandleOccurrence
         * @param[in] border if true, hides the Object Rectangle Boarder on HandlerOccurrence 
         */
        HideOdeAction(const char* name, bool text, bool border);
        
        /**
         * @brief dtor for the Log ODE Action class
         */
        ~HideOdeAction();
        
        /**
         * @brief Handles the ODE occurrence by hiding the Object's Display Text and/or Rectangle Border 
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

    private:
    
        /**
         * @brief If true, hides the Object's Display Text on HandleOccurrence
         */
        bool m_hideText;
        
        /**
         * @brief If true, hides the Object's Rectangle Border on HandlerOccurrence
         */
        bool m_hideBorder;
    
    };
        

    // ********************************************************************

    /**
     * @class OverlayFrameOdeAction
     * @brief Frame Overlay ODE Action class
     */
    class OverlayFrameOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the OverlayFrame ODE Action class
         * @param[in] name unique name for the ODE Action
         * @param[in] shared pointer to the 
         */
        OverlayFrameOdeAction(const char* name, DSL_DISPLAY_TYPE_PTR pDisplayType);
        
        /**
         * @brief dtor for the OverlayFrame ODE Action class
         */
        ~OverlayFrameOdeAction();
        
        /**
         * @brief Handles the ODE by overlaying the pFrameMeta with the named Display Type
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

    private:
    
        DSL_DISPLAY_TYPE_PTR m_pDisplayType;
    
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta, 
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
        PrintOdeAction(const char* name);
        
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

    private:
    
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta,
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta,
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta,
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta,
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta,
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
        void HandleOccurrence(DSL_BASE_PTR pBaseTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta,
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta,
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
        void HandleOccurrence(DSL_BASE_PTR pBaseTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta,
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta,
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta,
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta,
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta,
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
         * @param[in] recordSink Record Sink component name to Start on ODE
         * @param[in] start time before current time in secs
         * @param[in] duration for recording unless stopped before completion
         */
        RecordSinkStartOdeAction(const char* name, 
            const char* recordSink, uint start, uint duration, void* clientData);
        
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief Record Sink to start the recording session
         */ 
        std::string m_recordSink;

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
        
        /**
         * @brief unique recording session id aquired on Start Record
         */
        uint m_session;
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
         * @param[in] recordSink Record Sink component name to Start on ODE
         * @param[in] start time before current time in secs
         * @param[in] duration for recording unless stopped before completion
         */
        RecordTapStartOdeAction(const char* name, 
            const char* tapSink, uint start, uint duration, void* clientData);
        
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, NvDsBatchMeta* pBatchMeta,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief Record Tap to start the recording session
         */ 
        std::string m_recordTap;

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
        
        /**
         * @brief unique recording session id aquired on Start Record
         */
        uint m_session;
    };
}

#endif // _DSL_ODE_ACTION_H
