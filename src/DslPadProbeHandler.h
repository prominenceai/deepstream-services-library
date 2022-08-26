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

#ifndef _DSL_PAD_PROBE_HANDLER_H
#define _DSL_PAD_PROBE_HANDLER_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslElementr.h"
#include "DslOdeTrigger.h"
#include "DslSourceMeter.h"


namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_PPH_PTR std::shared_ptr<PadProbeHandler>

    #define DSL_PPH_CUSTOM_PTR std::shared_ptr<CustomPadProbeHandler>
    #define DSL_PPH_CUSTOM_NEW(name, clientHandler, clientData) \
        std::shared_ptr<CustomPadProbeHandler>(new CustomPadProbeHandler(name, \
            clientHandler, clientData))
        
    #define DSL_PPH_METER_PTR std::shared_ptr<MeterPadProbeHandler>
    #define DSL_PPH_METER_NEW(name, interval, clientHandler, clientData) \
        std::shared_ptr<MeterPadProbeHandler>(new MeterPadProbeHandler(name, \
            interval, clientHandler, clientData))
        
    #define DSL_PPH_ODE_PTR std::shared_ptr<OdePadProbeHandler>
    #define DSL_PPH_ODE_NEW(name) \
        std::shared_ptr<OdePadProbeHandler>(new OdePadProbeHandler(name))

    #define DSL_PPH_TIMESTAMP_PTR std::shared_ptr<TimestampPadProbeHandler>
    #define DSL_PPH_TIMESTAMP_NEW(name) \
        std::shared_ptr<TimestampPadProbeHandler>(new TimestampPadProbeHandler(name))

    #define DSL_PPEH_EOS_CONSUMER_PTR std::shared_ptr<EosConsumerPadProbeEventHandler>
    #define DSL_PPEH_EOS_CONSUMER_NEW(name) \
        std::shared_ptr<EosConsumerPadProbeEventHandler>( \
            new EosConsumerPadProbeEventHandler(name))
        
    #define DSL_PPEH_EOS_HANDLER_PTR std::shared_ptr<EosHandlerPadProbeEventHandler>
    #define DSL_PPEH_EOS_HANDLER_NEW(name, clientHandler, clientData) \
        std::shared_ptr<EosHandlerPadProbeEventHandler>( \
            new EosHandlerPadProbeEventHandler(name, clientHandler, clientData))

    #define DSL_PPEH_FRAME_NUMBER_ADDER_PTR std::shared_ptr \
        <FrameNumberAdderPadProbeEventHandler>
    #define DSL_PPEH_FRAME_NUMBER_ADDER_NEW(name) \
        std::shared_ptr<FrameNumberAdderPadProbeEventHandler>( \
            new FrameNumberAdderPadProbeEventHandler(name))

    #define DSL_PAD_PROBE_PTR std::shared_ptr<PadProbetr>

    #define DSL_PAD_BUFFER_PROBE_PTR std::shared_ptr<PadBufferProbetr>
    #define DSL_PAD_BUFFER_PROBE_NEW(name, factoryName, parentElement) \
        std::shared_ptr<PadBufferProbetr>(new PadBufferProbetr(name, \
            factoryName, parentElement))    

    #define DSL_PAD_EVENT_DOWNSTREAM_PROBE_PTR std::shared_ptr<PadEventDownStreamProbetr>
    #define DSL_PAD_EVENT_DOWNSTREAM_PROBE_NEW(name, factoryName, parentElement) \
        std::shared_ptr<PadEventDownStreamProbetr>(new PadEventDownStreamProbetr( \
            name, factoryName, parentElement))    

    /**
     * @brief Pad Probe Handler Callback type
     */
    typedef boolean (*dsl_pph_client_handler_cb)(void* buffer, void* client_data);

        
    class PadProbeHandler : public Base
    {
    public: 
    
        PadProbeHandler(const char* name);

        ~PadProbeHandler();

        bool AddToParent(DSL_BASE_PTR pParent, uint pad);

        bool RemoveFromParent(DSL_BASE_PTR pParent, uint pad);
        
        /**
         * @brief Gets the current state of the Handler enabled flag
         * @return true if Handler is current enabled, false otherwise
         */
        bool GetEnabled();

        /**
         * @brief Sets the current state of the Handler enabled flag. 
         * The default state on creation is True
         * @param[in] enabled set to true if Repororting is to be enabled, false otherwise
         */
        virtual bool SetEnabled(bool enabled);
        
        /**
         * @brief Handler specific
         * @param[in]pBuffer Pad buffer
         * @return GstPadProbeReturn see GST reference, one of [GST_PAD_PROBE_DROP, GST_PAD_PROBE_OK,
         * GST_PAD_PROBE_REMOVE, GST_PAD_PROBE_PASS, GST_PAD_PROBE_HANDLED]
         */
        virtual GstPadProbeReturn HandlePadData(GstPadProbeInfo* pInfo){return GST_PAD_PROBE_OK;};

    protected:
    
        /**
         * @brief Handler enabled setting, default = true (enabled), 
         */ 
        bool m_isEnabled;

        /**
         * @brief mutex to protect mutual access to probe data
         */
        GMutex m_padHandlerMutex;
        
    };
    
    //----------------------------------------------------------------------------------------------

    /**
     * @class CustomPadProbeHandler
     * @brief 
     */
    class CustomPadProbeHandler : public PadProbeHandler
    {
    public: 
    
        /**
         * @brief ctor for the Custom Pad Probe Handler
         * @param[in] name unique name for the PPH
         * @param[in] clientHandler client callback function to handle the EOS event
         * @param[in] clientData return to the client when the handler is called
         */
        CustomPadProbeHandler(const char* name, dsl_pph_client_handler_cb clientHandler, void* clientData);

        /**
         * @brief dtor for the ODE Pad Probe Handler
         */
        ~CustomPadProbeHandler();

        /**
         * @brief Custom Pad Probe Handler
         * @param[in]pBuffer Pad buffer
         * @return GstPadProbeReturn see GST reference, one of [GST_PAD_PROBE_DROP, GST_PAD_PROBE_OK,
         * GST_PAD_PROBE_REMOVE, GST_PAD_PROBE_PASS, GST_PAD_PROBE_HANDLED]
         */
        GstPadProbeReturn HandlePadData(GstPadProbeInfo* pInfo);

    private:
    
        /**
         * @brief client callback funtion, called on each HandlePadData
         */
        dsl_pph_custom_client_handler_cb m_clientHandler;
        
        /**
         * @brief opaue pointer to client data, returned on callback
         */
        void* m_clientData;
        
    };

    //----------------------------------------------------------------------------------------------

    /**
     * @class EosConsumerPadProbeEventHandler
     * @brief Pad Probe Handler to consume all downstream EOS events that cross the pad
     */
    class EosConsumerPadProbeEventHandler : public PadProbeHandler
    {
    public: 
    
        /**
         * @brief ctor for the EOS Consumer Pad Probe Handler
         * @param[in] name unique name for the PPH
         */
        EosConsumerPadProbeEventHandler(const char* name);

        /**
         * @brief dtor for the EOS Consumer Pad Probe Handler
         */
        ~EosConsumerPadProbeEventHandler();

        /**
         * @brief ODE Pad Probe Handler
         * @param[in] pBuffer Pad buffer
         * @return GstPadProbeReturn see GST reference, one of [GST_PAD_PROBE_DROP, GST_PAD_PROBE_OK,
         * GST_PAD_PROBE_REMOVE, GST_PAD_PROBE_PASS, GST_PAD_PROBE_HANDLED]
         */
        GstPadProbeReturn HandlePadData(GstPadProbeInfo* pInfo);
    };
    
    //----------------------------------------------------------------------------------------------

    /**
     * @class EosHandlerPadProbeEventHandler
     * @brief Pad Probe Handler to call a client handler callback function on downstream EOS event.
     */
    class EosHandlerPadProbeEventHandler : public PadProbeHandler
    {
    public: 
    
        /**
         * @brief ctor for the EOS Handler Pad Probe Handler
         * @param[in] name unique name for the PPH
         * @param[in] clientHandler client callback function to handle the EOS event
         * @param[in] clientData return to the client when the handler is called
         */
        EosHandlerPadProbeEventHandler(const char* name, dsl_pph_client_handler_cb clientHandler, void* clientData);

        /**
         * @brief dtor for the EOS Consumer Pad Probe Handler
         */
        ~EosHandlerPadProbeEventHandler();

        /**
         * @brief Custom Pad Probe Handler
         * @param[in]pBuffer Pad buffer
         * @return GstPadProbeReturn see GST reference, one of [GST_PAD_PROBE_DROP, GST_PAD_PROBE_OK,
         * GST_PAD_PROBE_REMOVE, GST_PAD_PROBE_PASS, GST_PAD_PROBE_HANDLED]
         */
        GstPadProbeReturn HandlePadData(GstPadProbeInfo* pInfo);

    private:
    
        /**
         * @brief client callback funtion, called on End-of-Stream event
         */
        dsl_pph_custom_client_handler_cb m_clientHandler;
        
        /**
         * @brief opaue pointer to client data, returned on callback
         */
        void* m_clientData;
        
    };

    //----------------------------------------------------------------------------------------------

    /**
     * @class FrameNumberAdderPadProbeEventHandler
     * @brief Pad Probe Handler to add an incremental/unique frame number
     * to each frame_meta. This PPH will be (conditionally) added to the
     * source pad of 2D Tiler component. Why? Because the Tiler sets every
     * frame_muber to 0 removing any frame reference.
     */
    class FrameNumberAdderPadProbeEventHandler : public PadProbeHandler
    {
    public: 
    
        /**
         * @brief ctor for the Frame Number Adder Pad Probe Handler
         * @param[in] name unique name for the PPH
         */
        FrameNumberAdderPadProbeEventHandler(const char* name);

        /**
         * @brief dtor for the Frame Number Adder Pad Probe Handler
         */
        ~FrameNumberAdderPadProbeEventHandler();
        
        /**
         * @brief resets m_currentFrameNumber to 0.
         */
        void ResetFrameNumber();
        
        /**
         * @brief gets the current value of m_currentFrameNumber;
         * @return the value of m_currentFrameNumber;
         */
        uint64_t GetFrameNumber();

        /**
         * @brief ODE Pad Probe Handler
         * @param[in] pBuffer Pad buffer
         * @return GstPadProbeReturn see GST reference, one of [GST_PAD_PROBE_DROP, 
         * GST_PAD_PROBE_OK, GST_PAD_PROBE_REMOVE, GST_PAD_PROBE_PASS, 
         * GST_PAD_PROBE_HANDLED]
         */
        GstPadProbeReturn HandlePadData(GstPadProbeInfo* pInfo);
        
    private:
    
        /**
         * @brief current frame number to increment and assign to each
         * frame within each batch-metadata received. 
         */
        uint64_t m_currentFrameNumber;
    };
    
    //----------------------------------------------------------------------------------------------

    /**
     * @class OdePadProbeHandler
     * @brief Pad Probe Handler to Handle a collection ODE triggers
     * Note: ODE Triggers are added using the base AddChild function
     */
    class OdePadProbeHandler : public PadProbeHandler
    {
    public: 
    
        /**
         * @brief ctor for the ODE Pad Probe Handler
         */
        OdePadProbeHandler(const char* name);

        /**
         * @brief dtor for the ODE Pad Probe Handler
         */
        ~OdePadProbeHandler();

        /**
         * @brief adds an ODE Trigger to this ODE Pad Probe Handler.
         * @param[in] pChild child Object to add to this parent Obejct. 
         */
        bool AddChild(DSL_BASE_PTR pChild);

        /**
         * @brief removes a child ODE Trigger from this ODE Pad Probe Handler.
         * @param[in] pChild to remove
         */
        bool RemoveChild(DSL_BASE_PTR pChild);

        /**
         * @brief removes all childred ODE Triggers from this ODE Pad Probe Handler.
         */
        void RemoveAllChildren();
        
        /**
         * @brief Gets the current Display Meta Allocation per frame size.
         * @return the allocation size, default = 1
         */
        uint GetDisplayMetaAllocSize();
        
        /**
         * @brief Gets the current Display Meta Allocation per frame size.
         * @return the allocation size, default = 1
         */
        void SetDisplayMetaAllocSize(uint count);

        /**
         * @brief ODE Pad Probe Handler
         * @param[in] pBuffer Pad buffer
         * @return GstPadProbeReturn see GST reference, one of [GST_PAD_PROBE_DROP,
         *  GST_PAD_PROBE_OK,GST_PAD_PROBE_REMOVE, GST_PAD_PROBE_PASS, GST_PAD_PROBE_HANDLED]
         */
        GstPadProbeReturn HandlePadData(GstPadProbeInfo* pInfo);
        
    private:
    
        /**
         * @brief specifies how many Display Meta structures are allocated for each frame
         */
        uint m_displayMetaAllocSize;
        
        /**
         * @brief Index variable to incremment/assign on ODE Action add.
         */
        uint m_nextTriggerIndex;
        
        /**
         * @brief Map of child ODE Triggers indexed by their add-order for execution
         */
        std::map <uint, DSL_BASE_PTR> m_pChildrenIndexed; 
        
    };
    
    //----------------------------------------------------------------------------------------------
    /**
     * @class MeterPadProbeHandler
     * @brief 
     */
    class MeterPadProbeHandler : public PadProbeHandler
    {
    public: 

    
        MeterPadProbeHandler(const char* name, 
            uint interval, dsl_pph_meter_client_handler_cb clientHandler, void* clientData);

        /**
         * @brief dtor for the Meter Consumer Pad Probe Handler
         */
        ~MeterPadProbeHandler();

        /**
         * @brief Sets the current state of the Handler enabled flag. 
         * The default state on creation is True
         * @param[in] enabled set to true if Repororting is to be enabled, false otherwise
         */
        bool SetEnabled(bool enabled);

        /**
         * @brief Handler specific Pad BufferHandler
         * @param[in]pBuffer Pad buffer
         * @return GstPadProbeReturn see GST reference, one of [GST_PAD_PROBE_DROP, GST_PAD_PROBE_OK,
         * GST_PAD_PROBE_REMOVE, GST_PAD_PROBE_PASS, GST_PAD_PROBE_HANDLED]
         */
        GstPadProbeReturn HandlePadData(GstPadProbeInfo* pInfo);
        
         /**
         * @brief gets the current reporting interval for the MeterPadProbeHandler
         * @return the current reporting interval in units of seconds
         */
        uint GetInterval();

        /**
         * @brief sets the current reporting interval for the MeterPadProbeHandler
         * @param[in] interval the new reporting interval to use
         * @return true if successful, false otherwise
         */
        bool SetInterval(uint interval);
        
        /**
         * @brief Interval Timer experation handler
         * @return non-zero (true) to continue, 0 (false) otherwise 
         */
        int HandleIntervalTimeout();
    
    private:
    
        /**
         * @brief measurement reporting interval in seconds.
         */
        uint m_interval;
        
        /**
         * @brief gnome timer Id for peformance calculation interval timer
         */
        uint m_timerId;
        
        /**
         * @brief client callback funtion, called on reporting interval
         */
        dsl_pph_meter_client_handler_cb m_clientHandler;
        
        /**
         * @brief opaue pointer to client data, returned on callback
         */
        void* m_clientData;
        
        /**
         * @brief map of all current source meters, one per source_id
         */
        std::map<uint, DSL_SOURCE_METER_PTR> m_sourceMeters;
    };

    //----------------------------------------------------------------------------------------------

    static int MeterIntervalTimeoutHandler(void* user_data);

    //----------------------------------------------------------------------------------------------

    /**
     * @class TimestampPadProbeHandler
     * @brief implements a timestamp that is updated on each call to handle buffer
     */
    class TimestampPadProbeHandler : public PadProbeHandler
    {
    public: 
    
        TimestampPadProbeHandler(const char* name);

        ~TimestampPadProbeHandler();

        /**
         * @brief returns the time of the last buffer
         * @param[in] timestamp timestamp struct to fil in
         */
        void GetTime(struct timeval& timestamp);
        
        /**
         * @brief sets the time of the last buffer
         * @param[in] timestamp timevalue to set this Timestamp
         */
        void SetTime(struct timeval& timestamp);
        
        /**
         * @brief Timestamp Pad Probe Handler. Updates the Timestamp to current-time on each buffer
         * @param[out] pBuffer Pad buffer
         * @return GstPadProbeReturn see GST reference, one of [GST_PAD_PROBE_DROP, GST_PAD_PROBE_OK,
         * GST_PAD_PROBE_REMOVE, GST_PAD_PROBE_PASS, GST_PAD_PROBE_HANDLED]
         */
        GstPadProbeReturn HandlePadData(GstPadProbeInfo* pInfo);

    private:
    
        /**
         * @brief updated on each call to HandlePadData which provides a time stamp for the last.
         */
        timeval m_timestamp;
        
    };
    
    //----------------------------------------------------------------------------------------------
    /**
     * @class PadProbetr
     * @brief Implements a container class for GST Pad Probe
     */
    class PadProbetr : public Base
    {
    public:
        
        /**
         * @brief ctor for the PadProbetr class
         * @param[in] name name for the new PadProbetr
         * @param[in] factoryNme "sink" or "src" Pad Probe type
         */
        PadProbetr(const char* name, const char* factoryName, DSL_ELEMENT_PTR parentElement, 
            GstPadProbeType padProbeType);
        
        /**
         * @brief dtor for the PadProbetr base class
         */
        ~PadProbetr();

        /**
         * @brief Adds a Pad Probe Handler callback function to the PadProbetr
         * @param[in] pHandler shared pointer to the handler to add
         * @return false if the PadProbetr has an existing Pad Probe Handler
         */
        bool AddPadProbeHandler(DSL_BASE_PTR pHandler);
            
        /**
         * @brief Removes the Handler function from the PadProbetr
         * @param[in]pClientPadProbeHandler callback function pointer to remove
         * @param[in] pHandler shared pointer to the handler to add
         */
        bool RemovePadProbeHandler(DSL_BASE_PTR pHandler);
        
        
        /**
         * @brief 
         * @param[in]pPad
         * @param[in]pInfo
         * @return 
         */
        virtual GstPadProbeReturn HandlePadProbe(
            GstPad* pPad, GstPadProbeInfo* pInfo) = 0;

    protected:
    
        /**
         * @brief unique name for this PadProbetr
         */
        std::string m_name;
        
        /**
         * @brief factory name for this PadProbetr
         */
        std::string m_factoryName;
        
        /**
         * @brief pointer to parent bintr's GST object
         */
        GstElement* m_pParentGstElement;

        /**
         * @brief mutex fo the Pad Probe handler
         */
        GMutex m_padProbeMutex;
        
        /**
         * @brief unique sink/src pad probe id (handle)
         */
        uint m_padProbeId;
        
        /**
         * @brief type identifer for the pad probe 
         */
        GstPadProbeType m_padProbeType;
        
        /**
         * @brief Static Pad to attach the Probe to
         */
        GstPad* m_pStaticPad;

    };

    //----------------------------------------------------------------------------------------------
    /**
     * @class PadBufferProbetr PadProbetr of type GST_PAD_PROBE_TYPE_BUFFER
     * @brief Implements a container class for GST Pad Probe
     */
    class PadBufferProbetr : public PadProbetr
    {
    public:
        
        /**
         * @brief ctor for the PadBufferProbetr class
         * @param[in] name name for the new PadBufferProbetr
         * @param[in] factoryNme "sink" or "src" Pad Probe type
         * @param[in] parentElement parent bin for pad creation.
         */
        PadBufferProbetr(const char* name, const char* factoryName, DSL_ELEMENT_PTR parentElement);
        
        /**
         * @brief dtor for the PadBufferProbetr base class
         */
        ~PadBufferProbetr();
        
        /**
         * @brief Called to handle the specific Pad Probe of type BUFFER
         * @param[in]pPad pointer to the Pad that produced the buffer
         * @param[in]pInfo pointer to pad info with pInfo->data for processing
         * @return one of the GST_PAD_PROBE return types,
         */
        GstPadProbeReturn HandlePadProbe(
            GstPad* pPad, GstPadProbeInfo* pInfo);
    };

    //----------------------------------------------------------------------------------------------
    /**
     * @class PadEventDownStreamProbetr PadProbetr of type GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM
     * @brief Implements a container class for GST Pad Probe
     */
    class PadEventDownStreamProbetr : public PadProbetr
    {
    public:
        
        /**
         * @brief ctor for the PadEventDownStreamProbetr class
         * @param[in] name name for the new PadEventDownStreamProbetr
         * @param[in] factoryName "sink" or "src" Pad Probe type
         * @param[in] parentElement parent bin for pad creation.
         */
        PadEventDownStreamProbetr(const char* name, const char* factoryName, DSL_ELEMENT_PTR parentElement);
        
        /**
         * @brief dtor for the PadBufferProbetr base class
         */
        ~PadEventDownStreamProbetr();

        /**
         * @brief Called to handle the specific Pad Probe of type BUFFER
         * @param[in]pPad pointer to the GstPad that produced the buffer
         * @param[in]pInfo pointer to pad info with pInfo->data for processing
         * @return one of the GST_PAD_PROBE return types,
         */
        GstPadProbeReturn HandlePadProbe(
            GstPad* pPad, GstPadProbeInfo* pInfo);
    };
    
    //----------------------------------------------------------------------------------------------
    static GstPadProbeReturn PadProbeCB(GstPad* pPad, 
        GstPadProbeInfo* pInfo, gpointer pPadProbetr);

}

#endif // _DSL_PAD_PROBE_HANDLER_H
