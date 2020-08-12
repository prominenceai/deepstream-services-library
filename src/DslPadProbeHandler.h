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

#ifndef _DSL_ODE_HANDLER_H
#define _DSL_ODE_HANDLER_H

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

    #define DSL_PPH_ODE_PTR std::shared_ptr<OdePadProbeHandler>
    #define DSL_PPH_ODE_NEW(name) \
        std::shared_ptr<OdePadProbeHandler>(new OdePadProbeHandler(name))

    #define DSL_PPH_METER_PTR std::shared_ptr<MeterPadProbeHandler>
    #define DSL_PPH_METER_NEW(name, interval, clientHandler, clientData) \
        std::shared_ptr<MeterPadProbeHandler>(new MeterPadProbeHandler(name, interval, clientHandler, clientData))
        
    #define DSL_PPH_CUSTOM_PTR std::shared_ptr<CustomPadProbeHandler>
    #define DSL_PPH_CUSTOM_NEW(name, clientHandler, clientData) \
        std::shared_ptr<CustomPadProbeHandler>(new CustomPadProbeHandler(name, clientHandler, clientData))
        
    #define DSL_PAD_PROBE_PTR std::shared_ptr<PadProbetr>
    #define DSL_PAD_PROBE_NEW(name, factoryName, parentElement) \
        std::shared_ptr<PadProbetr>(new PadProbetr(name, factoryName, parentElement))    

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
         * @param pBuffer Pad buffer
         * @return true to continue handling, false to stop and self remove callback
         */
        virtual bool HandlePadBuffer(GstBuffer* pBuffer){return true;};

    protected:
    
        /**
         * @brief Handler enabled setting, default = true (enabled), 
         */ 
        bool m_isEnabled;
        
    };
    
    //----------------------------------------------------------------------------------------------

    /**
     * @class CustomPadProbeHandler
     * @brief 
     */
    class CustomPadProbeHandler : public PadProbeHandler
    {
    public: 
    
        CustomPadProbeHandler(const char* name, dsl_pph_client_handler_cb clientHandler, void* clientData);

        ~CustomPadProbeHandler();

        /**
         * @brief Custom Pad Probe Handler
         * @param pBuffer Pad buffer
         * @return true to continue handling, false to stop and self remove callback
         */
        bool HandlePadBuffer(GstBuffer* pBuffer);

    private:
    
        /**
         * @brief client callback funtion, called on each HandlePadBuffer
         */
        dsl_pph_custom_client_handler_cb m_clientHandler;
        
        /**
         * @brief opaue pointer to client data, returned on callback
         */
        void* m_clientData;
        
    };
    
    //----------------------------------------------------------------------------------------------

    /**
     * @class OdePadProbeHandler
     * @brief 
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
         * @brief ODE Pad Probe Handler
         * @param pBuffer Pad buffer
         * @return true to continue handling, false to stop and self remove callback
         */
        bool HandlePadBuffer(GstBuffer* pBuffer);
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

        ~MeterPadProbeHandler();

        /**
         * @brief Sets the current state of the Handler enabled flag. 
         * The default state on creation is True
         * @param[in] enabled set to true if Repororting is to be enabled, false otherwise
         */
        bool SetEnabled(bool enabled);

        /**
         * @brief Handler specific Pad BufferHandler
         * @param pBuffer Pad buffer
         * @return true to continue handling, false to stop and self remove callback
         */
        bool HandlePadBuffer(GstBuffer* pBuffer);
        
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
         * @brief 
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
         * @brief mutex to prevent callback reentry
         */
        GMutex m_meterMutex;
        
        /**
         * @brief map of all current source meters, one per source_id
         */
        std::map<uint, DSL_SOURCE_METER_PTR> m_sourceMeters;
    };

    //----------------------------------------------------------------------------------------------

    static int MeterIntervalTimeoutHandler(void* user_data);

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
        PadProbetr(const char* name, const char* factoryName, DSL_ELEMENT_PTR parentElement);
        
        /**
         * @brief dtor for the PadProbetr base class
         */
        ~PadProbetr();

        /**
         * @brief Adds a Pad Probe Handler callback function to the PadProbetr
         * @param pClientPadProbeHandler callback function pointer to add
         * @param pClientUserData user data to return on callback
         * @return false if the PadProbetr has an existing Pad Probe Handler
         */
        bool AddPadProbeHandler(DSL_BASE_PTR pHandler);
            
        /**
         * @brief Removes the current Pad Probe Handler callback function from the PadProbetr
         * @param pClientPadProbeHandler callback function pointer to remove
         * @return false if the PadProbetr does not have a Pad Probe Handler to remove.
         */
        bool RemovePadProbeHandler(DSL_BASE_PTR pHandler);
        
        
        /**
         * @brief 
         * @param pPad
         * @param pInfo
         * @return 
         */
        GstPadProbeReturn HandlePadProbe(
            GstPad* pPad, GstPadProbeInfo* pInfo);

    private:
    
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
         * @brief sink/src pad probe handle
         */
        uint m_padProbeId;
        
        /**
         * @brief Static Pad to attach the Probe to
         */
        GstPad* m_pStaticPad;

    };
    
    static GstPadProbeReturn PadProbeCB(GstPad* pPad, 
        GstPadProbeInfo* pInfo, gpointer pPadProbetr);

}

#endif // _DSL_ODE_HANDLER_H
