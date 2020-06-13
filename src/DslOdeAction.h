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
//#include "DslOdeOccurrence.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_ODE_ACTION_PTR std::shared_ptr<OdeAction>

    #define DSL_ODE_ACTION_CALLBACK_PTR std::shared_ptr<CallbackOdeAction>
    #define DSL_ODE_ACTION_CALLBACK_NEW(name, clientHandler, clientData) \
        std::shared_ptr<CallbackOdeAction>(new CallbackOdeAction(name, clientHandler, clientData))
        
    #define DSL_ODE_ACTION_CAPTURE_PTR std::shared_ptr<CaptureOdeAction>
    #define DSL_ODE_ACTION_CAPTURE_NEW(name, captureType, outdir) \
        std::shared_ptr<CaptureOdeAction>(new CaptureOdeAction(name, captureType, outdir))
        
    #define DSL_ODE_ACTION_DISPLAY_PTR std::shared_ptr<DisplayOdeAction>
    #define DSL_ODE_ACTION_DISPLAY_NEW(name, offsetX, offsetY, offsetYWithClassId) \
        std::shared_ptr<DisplayOdeAction>(new DisplayOdeAction(name, offsetX, offsetY, offsetYWithClassId))
        
    #define DSL_ODE_ACTION_LOG_PTR std::shared_ptr<LogOdeAction>
    #define DSL_ODE_ACTION_LOG_NEW(name) \
        std::shared_ptr<LogOdeAction>(new LogOdeAction(name))
        
    #define DSL_ODE_ACTION_PAUSE_PTR std::shared_ptr<PauseOdeAction>
    #define DSL_ODE_ACTION_PAUSE_NEW(name, pipeline) \
        std::shared_ptr<PauseOdeAction>(new PauseOdeAction(name, pipeline))
        
    #define DSL_ODE_ACTION_PRINT_PTR std::shared_ptr<PrintOdeAction>
    #define DSL_ODE_ACTION_PRINT_NEW(name) \
        std::shared_ptr<PrintOdeAction>(new PrintOdeAction(name))
        
    #define DSL_ODE_ACTION_REDACT_PTR std::shared_ptr<RedactOdeAction>
    #define DSL_ODE_ACTION_REDACT_NEW(name, red, green, blue, alpha) \
        std::shared_ptr<RedactOdeAction>(new RedactOdeAction(name, red, green, blue, alpha))

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
        
    #define DSL_ODE_ACTION_TRIGGER_ADD_PTR std::shared_ptr<AddTriggerOdeAction>
    #define DSL_ODE_ACTION_TRIGGER_ADD_NEW(name, handler, trigger) \
        std::shared_ptr<AddTriggerOdeAction>(new AddTriggerOdeAction(name, handler, trigger))
        
    #define DSL_ODE_ACTION_TRIGGER_DISABLE_PTR std::shared_ptr<DisableTriggerOdeAction>
    #define DSL_ODE_ACTION_TRIGGER_DISABLE_NEW(name, trigger) \
        std::shared_ptr<DisableTriggerOdeAction>(new DisableTriggerOdeAction(name, trigger))
        
    #define DSL_ODE_ACTION_TRIGGER_ENABLE_PTR std::shared_ptr<EnableTriggerOdeAction>
    #define DSL_ODE_ACTION_TRIGGER_ENABLE_NEW(name, trigger) \
        std::shared_ptr<EnableTriggerOdeAction>(new EnableTriggerOdeAction(name, trigger))
        
    #define DSL_ODE_ACTION_TRIGGER_REMOVE_PTR std::shared_ptr<RemoveTriggerOdeAction>
    #define DSL_ODE_ACTION_TRIGGER_REMOVE_NEW(name, handler, trigger) \
        std::shared_ptr<RemoveTriggerOdeAction>(new RemoveTriggerOdeAction(name, handler, trigger))
        
    #define DSL_ODE_ACTION_ACTION_ADD_PTR std::shared_ptr<AddActionOdeAction>
    #define DSL_ODE_ACTION_ACTION_ADD_NEW(name, trigger, action) \
        std::shared_ptr<AddActionOdeAction>(new AddActionOdeAction(name, trigger, action))
        
    #define DSL_ODE_ACTION_ACTION_DISABLE_PTR std::shared_ptr<DisableActionOdeAction>
    #define DSL_ODE_ACTION_ACTION_DISABLE_NEW(name, trigger) \
        std::shared_ptr<DisableActionOdeAction>(new DisableActionOdeAction(name, trigger))
        
    #define DSL_ODE_ACTION_ACTION_ENABLE_PTR std::shared_ptr<EnableActionOdeAction>
    #define DSL_ODE_ACTION_ACTION_ENABLE_NEW(name, trigger) \
        std::shared_ptr<EnableActionOdeAction>(new EnableActionOdeAction(name, trigger))
        
    #define DSL_ODE_ACTION_ACTION_REMOVE_PTR std::shared_ptr<RemoveActionOdeAction>
    #define DSL_ODE_ACTION_ACTION_REMOVE_NEW(name, trigger, action) \
        std::shared_ptr<RemoveActionOdeAction>(new RemoveActionOdeAction(name, trigger, action))
        
    // ********************************************************************

    class OdeAction : public Base
    {
    public: 
    
        /**
         * @brief ctor for the ODE virtual base class
         * @param[in] name unique name for the ODE action
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
        virtual void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer,
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
         * @brief ctor for the ODE Callback Action class
         * @param[in] name unique name for the ODE action
         * @param[in] clientHandler client callback function to call on ODE
         * @param[in] clientData opaque pointer to client data t return on callback
         */
        CallbackOdeAction(const char* name, dsl_ode_occurrence_handler_cb clientHandler, void* clientData);
        
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief Client Callback function to call on ODE occurrence
         */
        dsl_ode_occurrence_handler_cb m_clientHandler;
        
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
         * @brief ctor for the ODE Capture Action class
         * @param[in] name unique name for the ODE action
         * @param[in] captureType DSL_CAPTURE_TYPE_OBJECT or DSL_CAPTURE_TYPE_FRAME
         * @param[in] outdir output directory to write captured image files
         */
        CaptureOdeAction(const char* name, uint captureType, const char* outdir);
        
        /**
         * @brief dtor for the ODE Capture Action class
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
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
     * @class DisplayOdeAction
     * @brief ODE Display Ode Action class
     */
    class DisplayOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the ODE Display Action class
         * @param[in] name unique name for the ODE action
         * @param[in] offsetX horizontal X-offset for the ODE occurrence data to display
         * @param[in] offsetX vertical Y-offset for the ODE occurrence data to display
         * @param[in] offsetYWithClassId adds an additional offset based on ODE class Id if set true
         */
        DisplayOdeAction(const char* name, uint offsetX, uint offsetY, bool offsetYWithClassId);
        
        /**
         * @brief dtor for the ODE Display Action class
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer,
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
         * @brief Adds an additional offset based on ODE class Id if set true
         */
        bool m_offsetYWithClassId;
    
    };

    // ********************************************************************

    /**
     * @class LogOdeAction
     * @brief ODE Log Ode Action class
     */
    class LogOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the ODE Log Action class
         * @param[in] name unique name for the ODE action
         */
        LogOdeAction(const char* name);
        
        /**
         * @brief dtor for the ODE Display Action class
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer,
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
         * @brief ctor for the ODE Pause Action class
         * @param[in] name unique name for the ODE action
         * @param[in] pipeline unique name of the pipeline to pause on ODE occurrence
         */
        PauseOdeAction(const char* name, const char* pipeline);
        
        /**
         * @brief dtor for the ODE Pause Action class
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, 
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
         * @param[in] name unique name for the ODE action
         */
        PrintOdeAction(const char* name);
        
        /**
         * @brief dtor for the ODE Display Action class
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, 
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
         * @brief ctor for the ODE Log Action class
         * @param[in] name unique name for the ODE action
         * @param[in] red red level for the redaction background color [0..1]
         * @param[in] blue blue level for the redaction background color [0..1]
         * @param[in] green green level for the redaction background color [0..1]
         * @param[in] alpha alpha level for the redaction background color [0..1]
         */
        RedactOdeAction(const char* name, double red, double green, double blue, double alpha);
        
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
            
    private:
    
        /**
         * @brief Background color used to Redact the object
         */
        NvOSD_ColorParams m_backgroundColor;
    
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
         * @param[in] name unique name for the ODE action
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer,
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
         * @param[in] name unique name for the ODE action
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer,
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
         * @param[in] name unique name for the ODE action
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer,
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
         * @param[in] name unique name for the ODE action
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer,
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
     * @class AddTriggerOdeAction
     * @brief Add ODE Action class
     */
    class AddTriggerOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the ODE Add Action class
         * @param[in] name unique name for the ODE action
         * @param[in] handler ODE Handler component to add the ODE type to
         * @param[in] trigger ODE Trigger to add on ODE occurrence
         */
        AddTriggerOdeAction(const char* name, const char* handler, const char* trigger);
        
        /**
         * @brief dtor for the ODE Add Action class
         */
        ~AddTriggerOdeAction();

        /**
         * @brief Handles the ODE occurrence by adding an ODE Trigger to and ODE Handler
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief ODE Handler to add the ODE Trigger to
         */ 
        std::string m_handler;

        /**
         * @brief ODE Trigger to add to the ODE Handler on ODE occurrence
         */
        std::string m_trigger;
    };
    
    /**
     * @class DisableTriggerOdeAction
     * @brief Disable Trigger ODE Action class
     */
    class DisableTriggerOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Disable Trigger ODE Action class
         * @param[in] name unique name for the ODE action
         * @param[in] trigger ODE Trigger to disable on ODE occurrence
         */
        DisableTriggerOdeAction(const char* name, const char* trigger);
        
        /**
         * @brief dtor for the ODE Add Action class
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
        void HandleOccurrence(DSL_BASE_PTR pBaseTrigger, GstBuffer* pBuffer,
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
         * @param[in] name unique name for the ODE action
         * @param[in] trigger ODE Trigger to disable on ODE occurrence
         */
        EnableTriggerOdeAction(const char* name, const char* trigger);
        
        /**
         * @brief dtor for the ODE Add Action class
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief ODE Trigger to enable on ODE occurrence
         */
        std::string m_trigger;

    };
    
    // ********************************************************************
    /**
     * @class RemoveTriggerOdeAction
     * @brief Remove ODE Action class
     */
    class RemoveTriggerOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the ODE Remove Action class
         * @param[in] name unique name for the ODE action
         * @param[in] handler ODE Handler component to add the ODE type to
         * @param[in] trigger ODE Trigger to add on ODE occurrence
         */
        RemoveTriggerOdeAction(const char* name, const char* handler, const char* trigger);
        
        /**
         * @brief dtor for the ODE Add Action class
         */
        ~RemoveTriggerOdeAction();

        /**
         * @brief Handles the ODE occurrence by removing an ODE Trigger from an ODE Handler
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief ODE Handler to remove the ODE Trigger from
         */ 
        std::string m_handler;

        /**
         * @brief ODE Trigger to remove from the ODE Handler on ODE occurrence
         */
        std::string m_trigger;
    };

    // ********************************************************************

    /**
     * @class AddActionOdeAction
     * @brief Add Action ODE Action class
     */
    class AddActionOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Add Action ODE Action class
         * @param[in] name unique name for the ODE action
         * @param[in] trigger ODE Trigger to add the ODE action to
         * @param[in] action ODE Action to add on ODE occurrence
         */
        AddActionOdeAction(const char* name, const char* trigger, const char* action);
        
        /**
         * @brief dtor for the Add Action class
         */
        ~AddActionOdeAction();

        /**
         * @brief Handles the ODE occurrence by adding an ODE Action to an ODE Trigger
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief ODE Trigger to add the ODE Action to
         */ 
        std::string m_trigger;

        /**
         * @brief ODE Action to add to the ODE Trigger on ODE occurrence
         */
        std::string m_action;
    };
    
    /**
     * @class DisableActionOdeAction
     * @brief Disable Action ODE Action class
     */
    class DisableActionOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Disable Action ODE Action class
         * @param[in] name unique name for the ODE action
         * @param[in] trigger ODE Trigger to disable on ODE occurrence
         */
        DisableActionOdeAction(const char* name, const char* trigger);
        
        /**
         * @brief dtor for the ODE Add Action class
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer,
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
         * @param[in] name unique name for the ODE action
         * @param[in] action ODE Action to enabled on ODE occurrence
         */
        EnableActionOdeAction(const char* name, const char* action);
        
        /**
         * @brief dtor for the ODE Add Action class
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
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief ODE Action to enable on ODE occurrence
         */
        std::string m_action;

    };
    
    // ********************************************************************
    /**
     * @class RemoveActionOdeAction
     * @brief Remove Action ODE Action class
     */
    class RemoveActionOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the Remove Action ODE Action class
         * @param[in] name unique name for the ODE action
         * @param[in] trigger ODE Trigger to add the ODE Action to
         * @param[in] action ODE Action to add on ODE occurrence
         */
        RemoveActionOdeAction(const char* name, const char* handler, const char* trigger);
        
        /**
         * @brief dtor for the ODE Add Action class
         */
        ~RemoveActionOdeAction();

        /**
         * @brief Handles the ODE occurrence by removing an ODE Action from an ODE Handler
         * @param[in] pBuffer pointer to the batched stream buffer that triggered the event
         * @param[in] pOdeTrigger shared pointer to ODE Trigger that triggered the event
         * @param[in] pFrameMeta pointer to the Frame Meta data that triggered the event
         * @param[in] pObjectMeta pointer to Object Meta if Object detection event, 
         * NULL if Frame level absence, total, min, max, etc. events.
         */
        void HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief ODE Trigger to remove the ODE Action from
         */ 
        std::string m_trigger;

        /**
         * @brief ODE Action to remove from the ODE Trigger on ODE occurrence
         */
        std::string m_action;
    };
}


#endif // _DSL_ODE_ACTION_H
