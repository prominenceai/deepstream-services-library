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

#ifndef _DSL_BINTR_H
#define _DSL_BINTR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslNodetr.h"
#include "DslElementr.h"
#include "DslPadProbeHandler.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_BINTR_PTR std::shared_ptr<Bintr>
    #define DSL_BINTR_NEW(name) \
        std::shared_ptr<Bintr>(new Bintr(name))    

    /**
     * @class Bintr
     * @brief Implements a base container class for a GST Bin
     */
    class Bintr : public GstNodetr
    {
    public:

        /**
         * @brief named container ctor with new Bin 
         */
        Bintr(const char* name, bool isPipeline = false)
            : GstNodetr(name)
            , m_isPipeline(isPipeline)
            , m_requestPadId(-1)
            , m_linkMethod(DSL_PIPELINE_LINK_METHOD_DEFAULT)
            , m_isLinked(false)
            , m_batchSize(0)
            , m_gpuId(0)
            , m_nvbufMemType(DSL_NVBUF_MEM_TYPE_DEFAULT)
        { 
            LOG_FUNC(); 

            if (m_isPipeline)
            {
                m_pGstObj = GST_OBJECT(gst_pipeline_new(name));
            }
            else
            {
                m_pGstObj = GST_OBJECT(gst_bin_new(name));
            }
            if (!m_pGstObj)
            {
                LOG_ERROR("Failed to create a new GST bin for Bintr '" << name << "'");
                throw std::exception();  
            }
        }
        
        Bintr(const char* name, GstObject* GstObj)
            : GstNodetr(name)
            , m_isPipeline(false)
            , m_requestPadId(-1)
            , m_linkMethod(DSL_PIPELINE_LINK_METHOD_DEFAULT)
            , m_isLinked(false)
            , m_batchSize(0)
            , m_gpuId(0)
            , m_nvbufMemType(DSL_NVBUF_MEM_TYPE_DEFAULT)
        { 
            LOG_FUNC(); 

            SetGstObjAsProxy(GstObj);
        }
        
        /**
         * @brief Bintr dtor to release all GST references
         */
        ~Bintr()
        {
            LOG_FUNC();
        }

        /**
         * @brief returns the current sink or src request pad-id -- as managed by the 
         * multi-component Parent Bintr -- for this bintr if used (i.e connected a 
         * streammuxer, demuxer, or splitter).
         * @return -1 when id is not assigned, i.e. bintr is not currently in use
         */
        int GetRequestPadId()
        {
            LOG_FUNC();
            
            return m_requestPadId;
        }
        
        /**
         * @brief Sets the the sink or src request pad-id -- as managed by the 
         * multi-component Parent Bintr -- for this bintr if used (i.e connected a 
         * streammuxer, demuxer, or splitter).
         * @param request pad-id value to assign. use -1 for unassigned. 
         */
        void SetRequestPadId(int id)
        {
            LOG_FUNC();

            m_requestPadId = id;
        }
        
        /**
         * @brief Allows a client to determined derived type from base pointer
         * @param[in] typeInfo to compare against
         * @return true if this Bintr is of typeInfo, false otherwise
         */
        bool IsType(const std::type_info& typeInfo)
        {
            LOG_FUNC();
            
            return (typeInfo.hash_code() == typeid(*this).hash_code());
        }

        /**
         * @brief Adds this Bintr as a child to a ParentBinter
         * @param[in] pParentBintr to add to
         */
        virtual bool AddToParent(DSL_BASE_PTR pParent)
        {
            LOG_FUNC();
                
            return (bool)pParent->AddChild(shared_from_this());
        }
        
        /**
         * @brief removes this Bintr from the provided pParentBintr
         * @param[in] pParentBintr Bintr to remove from
         */
        virtual bool RemoveFromParent(DSL_BASE_PTR pParentBintr)
        {
            LOG_FUNC();
                
            return pParentBintr->RemoveChild(shared_from_this());
        }

        /**
         * @brief virtual function for derived classes to implement
         * a bintr type specific function to link all children.
         */
        virtual bool LinkAll() = 0;
        
        /**
         * @brief virtual function for derived classes to implement
         * a bintr type specific function to unlink all child elements.
         */
        virtual void UnlinkAll() = 0;
        
        /**
         * @brief Returns the current link method for this bintr
         * @return one of DSL_PIPELINE_LINK_COMPONENTS_BY_POSITION or
         * DSL_PIPELINE_LINK_COMPONENTS_BY_ORDER
         */
        uint GetLinkMethod()
        {
            LOG_FUNC();
            
            return m_linkMethod;
        }
        
        /**
         * @brief Sets the link method for this bintr to use
         * @return one of DSL_PIPELINE_LINK_COMPONENTS_BY_POSITION or
         * DSL_PIPELINE_LINK_COMPONENTS_BY_ORDER
         */
        void SetLinkMethod(uint linkMethod)
        {
            LOG_FUNC();
            
            m_linkMethod = linkMethod;
        }
        
        /**
         * @brief called to determine if a Bintr's Child Elementrs are Linked
         * @return true if Child Elementrs are currently Linked, false otherwise
         */
        bool IsLinked()
        {
            LOG_FUNC();
            
            return m_isLinked;
        }

        /**
         * @brief gets the current batchSize in use by this Bintr
         * @return the current batchSize
         */
        virtual uint GetBatchSize()
        {
            LOG_FUNC();
            
            return m_batchSize;
        };
        
        /**
         * @brief sets the batch size for this Bintr
         * @param[in] batchSize the new batchSize to use.
         */
        virtual bool SetBatchSize(uint batchSize)
        {
            LOG_FUNC();
            LOG_INFO("Setting batch size to '" << batchSize 
                << "' for Bintr '" << GetName() << "'");
            
            m_batchSize = batchSize;
            return true;
        };

        /**
         * @brief sets the interval for this Bintr
         * @param the new interval to use
         */
        bool SetInterval(uint interval, uint timeout);

        /**
         * @brief Adds the "buffer" and "downstream event" Pad Probes to the sink-pad
         * of a given Element.
         * @param the Parent Element to add the Probes to.
         */
        void AddSinkPadProbes(DSL_ELEMENT_PTR parentElement)
        {
            LOG_FUNC();
            
            std::string padProbeName = GetName() + "-sink-pad-buffer-probe";
            m_pSinkPadBufferProbe = DSL_PAD_BUFFER_PROBE_NEW(
                padProbeName.c_str(), "sink", parentElement);
                
            padProbeName = GetName() + "-sink-pad-event-probe";
            m_pSinkPadDsEventProbe = DSL_PAD_EVENT_DS_PROBE_NEW(
                padProbeName.c_str(), "sink", parentElement);
                
        }
        
        /**
         * @brief Adds the "buffer" and "downstream event" Pad Probes to the src-pad
         * of a given Element.
         * @param the Parent Element to add the Probes to.
         */
        void AddSrcPadProbes(DSL_ELEMENT_PTR parentElement)
        {
            LOG_FUNC();
            
            std::string padProbeName = GetName() + "-src-pad-buffer-probe";
            m_pSrcPadBufferProbe = DSL_PAD_BUFFER_PROBE_NEW(
                padProbeName.c_str(), "src", parentElement);
                
            padProbeName = GetName() + "-src-pad-event-probe";
            m_pSrcPadDsEventProbe = DSL_PAD_EVENT_DS_PROBE_NEW(
                padProbeName.c_str(), "src", parentElement);
        }
        
        /**
         * @brief Adds a Pad Probe Buffer Handler to the Bintr
         * @param[in] pPadProbeHandler shared pointer to the PPBH to add
         * @param[in] pad pad to add the handler to; DSL_PAD_SINK | DSL_PAD SRC
         * @return true if successful, false otherwise
         */
        bool AddPadProbeBufferHandler(DSL_BASE_PTR pPadProbeHandler, uint pad)
        {
            LOG_FUNC();
            
            if (pPadProbeHandler->IsInUse())
            {
                LOG_ERROR("Can't add Pad Probe Handler = '" 
                << pPadProbeHandler->GetName() 
                    << "' as it is currently in use");
                return false;
            }
            bool result(false);

            if (pad == DSL_PAD_SINK)
            {
                result = m_pSinkPadBufferProbe->AddPadProbeHandler(
                    pPadProbeHandler);
            }
            else if (pad == DSL_PAD_SRC)
            {
                result = m_pSrcPadBufferProbe->AddPadProbeHandler(
                    pPadProbeHandler);
            }
            else
            {
                LOG_ERROR("Invalid Pad type = " << pad << " for Bintr '" 
                    << GetName() << "'");
            }
            if (result)
            {
                pPadProbeHandler->AssignParentName(GetName());
            }
            return result;
        }
        
        /**
         * @brief Removes a Pad Probe Buffer Handler from the Bintr
         * @param[in] pPadProbeHandler shared pointer to the PPBH to remove
         * @param[in] pad pad to remove the handler from; DSL_PAD_SINK | DSL_PAD SRC
         * @return false if the Bintr does not own the Handler to remove.
         */
        bool RemovePadProbeBufferHandler(DSL_BASE_PTR pPadProbeHandler, uint pad)
        {
            LOG_FUNC();
            
            bool result(false);

            if (pad == DSL_PAD_SINK)
            {
                result = m_pSinkPadBufferProbe->RemovePadProbeHandler(
                    pPadProbeHandler);
            }
            else if (pad == DSL_PAD_SRC)
            {
                result = m_pSrcPadBufferProbe->RemovePadProbeHandler(
                    pPadProbeHandler);
            }
            else
            {
                LOG_ERROR("Invalid Pad type = " << pad << " for Bintr '" 
                    << GetName() << "'");
            }
            if (result)
            {
                pPadProbeHandler->ClearParentName();
            }
            return result;
        }

        /**
         * @brief Adds a Pad Probe Event Handler to the Bintr
         * @param[in] pPadProbeHandler shared pointer to the PPEH to add
         * @param[in] pad pad to add the handler to; DSL_PAD_SINK | DSL_PAD SRC
         * @return true if successful, false otherwise
         */
        bool AddPadProbeEventHandler(DSL_BASE_PTR pPadProbeHandler, uint pad)
        {
            LOG_FUNC();
            
            if (pPadProbeHandler->IsInUse())
            {
                LOG_ERROR("Can't add Pad Probe Handler = '" 
                    << pPadProbeHandler->GetName() 
                    << "' as it is currently in use");
                return false;
            }
            bool result(false);

            if (pad == DSL_PAD_SINK)
            {
                result = m_pSinkPadDsEventProbe->AddPadProbeHandler(pPadProbeHandler);
            }
            else if (pad == DSL_PAD_SRC)
            {
                result = m_pSrcPadDsEventProbe->AddPadProbeHandler(pPadProbeHandler);
            }
            else
            {
                LOG_ERROR("Invalid Pad type = " << pad << " for Bintr '" << GetName() << "'");
            }
            if (result)
            {
                pPadProbeHandler->AssignParentName(GetName());
            }
            return result;
        }
            
        /**
         * @brief Removes a Pad Probe Event Handler from the Bintr
         * @param[in] pPadProbeHandler shared pointer to the PPBH to remove
         * @param[in] pad pad to remove the handler from; DSL_PAD_SINK | DSL_PAD SRC
         * @return false if the Bintr does not own the Handler to remove.
         */
        bool RemovePadProbeEventHandler(DSL_BASE_PTR pPadProbeHandler, uint pad)
        {
            LOG_FUNC();
            
            bool result(false);

            if (pad == DSL_PAD_SINK)
            {
                result = m_pSinkPadDsEventProbe->RemovePadProbeHandler(pPadProbeHandler);
            }
            else if (pad == DSL_PAD_SRC)
            {
                result = m_pSrcPadDsEventProbe->RemovePadProbeHandler(pPadProbeHandler);
            }
            else
            {
                LOG_ERROR("Invalid Pad type = " << pad << " for Bintr '" << GetName() << "'");
            }
            if (result)
            {
                pPadProbeHandler->ClearParentName();
            }
            return result;
        }
        
        /**
         * @brief Gets the current GPU ID used by this Bintr
         * @return the ID for the current GPU in use.
         */
        virtual uint GetGpuId()
        {
            LOG_FUNC();

            LOG_DEBUG("Returning a GPU ID of " << m_gpuId <<"' for Bintr '" << GetName() << "'");
            return m_gpuId;
        }

        /**
         * @brief Bintr type specific implementation to set the GPU ID.
         * @return true if successfully set, false otherwise.
         */
        virtual bool SetGpuId(uint gpuId)
        {
            LOG_FUNC();
            
            if (IsLinked())
            {
                LOG_ERROR("Unable to set GPU ID for Bintr '" << GetName() 
                    << "' as it's currently linked");
                return false;
            }
            m_gpuId = gpuId;
            return true;
        }

        /**
         * @brief Gets the current NVIDIA buffer memory type used by this Bintr
         * @return one of the DSL_NVBUF_MEM_TYPE constant values.
         */
        uint GetNvbufMemType()
        {
            LOG_FUNC();

            LOG_DEBUG("Returning NVIDIA buffer memory type of " << m_nvbufMemType 
                <<"' for Bintr '" << GetName() << "'");
            return m_nvbufMemType;
        }

        /**
         * @brief Bintr type specific implementation to set the memory type.
         * @brief nvbufMemType new memory type to use
         * @return true if successfully set, false otherwise.
         */
        virtual bool SetNvbufMemType(uint nvbufMemType)
        {
            LOG_FUNC();
            
            if (IsInUse())
            {
                LOG_ERROR("Unable to set NVIDIA buffer memory type for Bintr '" << GetName() 
                    << "' as it's currently linked");
                return false;
            }
            m_nvbufMemType = nvbufMemType;
            return true;
        }

    protected:
    
        /**
         * @brief flag to specify if derived as Pipeline or other Bintr.
         */
        bool m_isPipeline;

        /**
         * @brief unique request pad id managed by the 
         * parent from the point of add until removed
         */
        int m_requestPadId;
    
        /**
         * @brief one of DSL_PIPELINE_LINK_METHOD_BY_POSITION or
         * DSL_PIPELINE_LINK_METHOD_BY_ORDER
         */
        bool m_linkMethod;
        
        /**
         * @brief current is-linked state for this Bintr
         */
        bool m_isLinked;
        
        /**
         * @brief current batch size for this Bintr
         */
        uint m_batchSize;

        /**
         * @brief current GPU Id in used by this Bintr
         */
        uint m_gpuId;

        /**
         * @brief current Memory Type used by this Bintr
         */
        uint m_nvbufMemType;

        /**
         * @brief Sink Buffer PadProbetr for this Bintr
         */
        DSL_PAD_BUFFER_PROBE_PTR m_pSinkPadBufferProbe;

        /**
         * @brief Source Buffer PadProbetr for this Bintr
         */
        DSL_PAD_BUFFER_PROBE_PTR m_pSrcPadBufferProbe;

        /**
         * @brief Sink Event PadProbetr for this Bintr
         */
        DSL_PAD_EVENT_DS_PROBE_PTR m_pSinkPadDsEventProbe;

        /**
         * @brief Source PadProbetr for this Bintr
         */
        DSL_PAD_EVENT_DS_PROBE_PTR m_pSrcPadDsEventProbe;
    };

} // DSL

#endif // _DSL_BINTR_H
