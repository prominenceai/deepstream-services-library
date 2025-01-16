/*
The MIT License

Copyright (c) 2019-2025, Prominence AI, Inc.

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

#ifndef _DSL_NODETR_H
#define _DSL_NODETR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslBase.h"

#include "DslPadProbeHandler.h"
namespace DSL
{

    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_NODETR_PTR std::shared_ptr<Nodetr>
    #define DSL_NODETR_NEW(name) \
        std::shared_ptr<Nodetr>(new Nodetr(name))    

    #define DSL_GSTNODETR_PTR std::shared_ptr<GstNodetr>
    #define DSL_GSTNODETR_NEW(name) \
        std::shared_ptr<GstNodetr>(new GstNodetr(name))    

    /**
     * @class Nodetr
     * @brief Implements a container class for all DSL Tree Node types
     */
    class Nodetr : public Base
    {
    public:
        
        /**
         * @brief ctor for the Nodetr base class
         * @param[in] name for the new Nodetr
         */
        Nodetr(const char* name)
            : Base(name)
            , m_pGstObj(NULL)
            , m_pParentGstObj(NULL)
        {
            LOG_FUNC();

            LOG_DEBUG("New Nodetr '" << m_name << "' created");
        }

        /**
         * @brief dtor for the Nodetr base class
         */
        ~Nodetr()
        {
            LOG_FUNC();
            
            LOG_DEBUG("Nodetr '" << m_name << "' deleted");
        }
        
        /**
         * @brief Links this Noder, becoming a source, to a sink Nodre
         * @param[in] pSink Sink Nodre to link this Source Nodre to
         * @return True if this source Nodetr is successfully linked 
         * to the provided sink Nodetr
         */
        virtual bool LinkToSink(DSL_NODETR_PTR pSink)
        {
            LOG_FUNC();

            if (m_pSink)
            {
                 LOG_ERROR("Nodetr '" << GetName() 
                    << "' is currently in a linked to Sink");
                return false;
            }
            m_pSink = pSink;
            LOG_DEBUG("Source '" << GetName() 
                << "' linked to Sink '" << pSink->GetName() << "'");
            
            return true;
        }
        
        /**
         * @brief Unlinks this Source Nodetr from its Sink Nodetr
         * @return True if this source Nodetr is successfully unlinked 
         * the sink Nodetr
         */
        virtual bool UnlinkFromSink()
        {
            LOG_FUNC();

            if (!m_pSink)
            {
                LOG_ERROR("Nodetr '" << GetName() 
                    << "' is not currently linked to Sink");
                return false;
            }
            LOG_DEBUG("Unlinking Source '" << GetName() 
                << "' from Sink '" << m_pSink->GetName() << "'");
            m_pSink = nullptr; 

            return true;
        }

        /**
         * @brief Links this Noder, becoming a source, to a sink Nodre
         * @param[in] pSrc Nodre to link this Sink Nodre back to
         */
        virtual bool LinkToSource(DSL_NODETR_PTR pSrc)
        {
            LOG_FUNC();

            if (m_pSrc)
            {
                LOG_ERROR("Nodetr '" << GetName() 
                    << "' is currently in a linked to a Source");
                return false;
            }
            m_pSrc = pSrc;
            LOG_DEBUG("Source '" << pSrc->GetName() 
                << "' linked to Sink '" << GetName() << "'");
            
            return true;
        }

        /**
         * @brief returns the currently linked to Sink Nodetr
         * @return shared pointer to Sink Nodetr, nullptr if Unlinked to sink
         */
        DSL_NODETR_PTR GetSource()
        {
            LOG_FUNC();
            
            return m_pSrc;
        }

        /**
         * @brief Unlinks this Sink Nodetr from its Source Nodetr
         */
        virtual bool UnlinkFromSource()
        {
            LOG_FUNC();

            if (!m_pSrc)
            {
                LOG_ERROR("Nodetr '" << GetName() 
                    << "' is not currently linked to Source");
                return false;
            }
            LOG_DEBUG("Unlinking self '" << GetName() 
                << "' as a Sink from '" << m_pSrc->GetName() << "' Source");
            m_pSrc = nullptr;
            
            return true;
        }
        
        /**
         * @brief returns the Source-to-Sink linked state for this Nodetr
         * @return true if this Nodetr is linked to a Sink Nodetr
         */
        bool IsLinkedToSink()
        {
            LOG_FUNC();
            
            return bool(m_pSink);
        }
        
        /**
         * @brief returns the Sink-to-Source linked state for this Nodetr
         * @return true if this Nodetr is linked to a Source Nodetr
         */
        bool IsLinkedToSource()
        {
            LOG_FUNC();
            
            return bool(m_pSrc);
        }
              
        /**
         * @brief returns the Sink Nodetr that this Nodetr is.urrently linked to
         * @return shared pointer to Sink Nodetr, nullptr if Unlinked from Sink
         */
        DSL_NODETR_PTR GetSink()
        {
            LOG_FUNC();
            
            return m_pSink;
        }
        
        /**
         * @brief returns this Nodetr's GStreamer bin as a GST_OBJECT
         * @return this Nodetr's bin cast to GST_OBJECT
         */
        GstObject* GetGstObject()
        {
            LOG_FUNC();
            
            return GST_OBJECT(m_pGstObj);
        }
        
        /**
         * @brief returns this Nodetr's GStreamer bin as a GST_ELEMENT
         * @return this Nodetr's bin cast to GST_ELEMENT
         */
        GstElement* GetGstElement()
        {
            LOG_FUNC();
            
            return GST_ELEMENT(m_pGstObj);
        }
        
        /**
         * @brief returns this Nodetr's GStreamer bin as a G_OBJECT
         * @return this Nodetr's bin cast to G_OBJECT
         */
        GObject* GetGObject()
        {
            LOG_FUNC();
            
            return G_OBJECT(m_pGstObj);
        }

        /**
         * @brief returns this Nodetr's Parent's GStreamer bin as a GST_OBJECT
         * @return this Nodetr's Parent's bin cast to GST_OBJECT
         */
        GstObject* GetParentGstObject()
        {
            LOG_FUNC();
            
            return GST_OBJECT(m_pParentGstObj);
        }

        /**
         * @brief returns this Nodetr's Parent's GStreamer bin as a GST_OBJECT
         * @return this Nodetr's Parent's bin cast to GST_OBJECT
         */
        GstElement* GetParentGstElement()
        {
            LOG_FUNC();
            
            return GST_ELEMENT(m_pParentGstObj);
        }
        
        /**
         * @brief Sets this Nodetr's GStreamer bin to a new GST_OBJECT
         */
        void SetGstObject(GstObject* pGstObj)
        {
            LOG_FUNC();
            
            m_pGstObj = pGstObj;
        }

        /**
         * @brief called to determine if a Bintr is currently in use - has a Parent
         * @return true if the Nodetr has a Parent, false otherwise
         */
        bool IsInUse()
        {
            LOG_FUNC();
            
            return (bool)GetParentGstElement();
        }

    public:
    
        /**
         * @brief Parent of this Nodetr if one exists. NULL otherwise
         */
        GstObject* m_pParentGstObj;
        
        
    protected:

        /**
         * @brief Gst object wrapped by the Nodetr
         */
        GstObject* m_pGstObj;

        /**
         * @brief defines the relationship between a Source Nodetr
         * linked to this Nodetr, making this Nodetr a Sink
         */
        DSL_NODETR_PTR m_pSrc;

        /**
         * @brief defines the relationship this Nodetr linked to
         * a Sink Nodetr making this Nodetr a Source. 
         */
        DSL_NODETR_PTR m_pSink;

        /**
         * @brief defines the relationship between a Source Nodetr
         * linked to this Nodetr (audio stream specifcally) making 
         * this Nodetr a Audio Sink.
         */
        DSL_NODETR_PTR m_pAudioSrc;

        /**
         * @brief defines the relationship this Nodetr (audio stream 
         * specifcally) linked to a Sink Nodetr making this Nodetr a 
         * Audio Source.
         */
        DSL_NODETR_PTR m_pAudioSink;

        /**
         * @brief defines the relationship between a Source Nodetr
         * linked to this Nodetr (video stream specifcally) making 
         * this Nodetr a Video Sink.
         */
        DSL_NODETR_PTR m_pVideoSrc;

        /**
         * @brief defines the relationship this Nodetr (audio stream 
         * specifcally) linked to a Sink Nodetr making this Nodetr a 
         * Audio Source.
         */
        DSL_NODETR_PTR m_pVideoSink;
    };

    static GstPadProbeReturn complete_unlink_from_source_tee_cb(GstPad* pad, 
        GstPadProbeInfo *info, gpointer pNoder);

   /**
     * @class GstNodetr
     * @brief Overrides the Base Class Virtual functions, adding the actuall 
     * GstObject* management. This allows the Nodetr class, and all its relational 
     * behavior, to be tested independent from GStreamer. Each method of this class
     * calls the base class to complete its behavior.
     */
    class GstNodetr : public Nodetr
    {
    public:
        
        /**
         * @brief ctor for the GstNodetr base class
         * @param[in] name for the new GstNodetr
         */
        GstNodetr(const char* name)
            : Nodetr(name)
            , m_isProxy(false)
            , m_releaseRequestedPadOnUnlink(false)
        {
            LOG_FUNC();

            LOG_DEBUG("New GstNodetr '" << GetName() << "' created");
        }

        /**
         * @brief dtor for the GstNodetr base class
         */
        ~GstNodetr()
        {
            LOG_FUNC();
        
            if (!GetGstElement())
            {
                LOG_WARN("GstElement for GstNodetr '" << GetName() 
                    << "' has not been instantiated");
            }
            else
            {
                // Set the State to NULL to free up all resource before 
                // removing children
                LOG_DEBUG("Setting GstElement for GstNodetr '" 
                    << GetName() << "' to GST_STATE_NULL");
                gst_element_set_state(GetGstElement(), GST_STATE_NULL);

                // Remove all child references 
                RemoveAllChildren();
                
                // Don't delete the gstreamer bin if currently owned by Parent
                // or if acting as a proxy for the Parent.
                if (!m_pParentGstObj and !m_isProxy)
                {
                    LOG_DEBUG("Unreferencing GST Object contained by this GstNodetr '" 
                        << GetName() << "'");
                    gst_object_unref(m_pGstObj);
                }
            }
            LOG_DEBUG("Nodetr '" << GetName() << "' deleted");
        }
        
        void SetGstObjAsProxy(GstObject* pGstObj)
        {
            LOG_FUNC();
            
            if (m_pGstObj)
            {
                LOG_ERROR("Failed to set GstObj as proxy for GstNodetr '"
                    << GetName() << "' as it's currently set");
                throw std::exception();
            }
            // Set the bin object pointer and isProxy flag (don't unref on delete).
            m_pGstObj = pGstObj;
            m_isProxy = true;
        }

        /**
         * @brief adds a child GstNodetr to this parent Bintr
         * @param[in] pChild to add. Once added, calling InUse()
         *  on the Child Bintr will return true
         * @return true if pChild was added successfully, false otherwise
         */
        virtual bool AddChild(DSL_BASE_PTR pChild)
        {
            LOG_FUNC();
            
            DSL_NODETR_PTR pChildNodetr = std::dynamic_pointer_cast<Nodetr>(pChild);

            if (!gst_bin_add(GST_BIN(m_pGstObj), pChildNodetr->GetGstElement()))
            {
                LOG_ERROR("Failed to add " << pChildNodetr->GetName() 
                    << " to " << GetName() <<"'");
                throw std::exception();
            }
            pChildNodetr->m_pParentGstObj = m_pGstObj;
            return Nodetr::AddChild(pChild);
        }
        
        /**
         * @brief removes a child from this parent GstNodetr
         * @param[in] pChild to remove. Once removed, calling InUse()
         *  on the Child Bintr will return false
         */
        virtual bool RemoveChild(DSL_BASE_PTR pChild)
        {
            LOG_FUNC();
            
            if (!IsChild(pChild))
            {
                LOG_ERROR("'" << pChild->GetName() 
                    << "' is not a child of '" << GetName() <<"'");
                return false;
            }

            DSL_NODETR_PTR pChildNodetr = std::dynamic_pointer_cast<Nodetr>(pChild);
            
            // Increase the reference count so the child is not destroyed.
            gst_object_ref_sink(pChildNodetr->GetGstObject());
            
            if (!gst_bin_remove(GST_BIN(m_pGstObj), pChildNodetr->GetGstElement()))
            {
                LOG_ERROR("Failed to remove " << pChildNodetr->GetName() 
                    << " from " << GetName() <<"'");
                return false;
            }
            pChildNodetr->m_pParentGstObj = NULL;
            return Nodetr::RemoveChild(pChildNodetr);
        }

        /**
         * @brief removed a child Nodetr of this parent Nodetr
         * @param[in] pChild to remove
         */
        void RemoveAllChildren()
        {
            LOG_FUNC();

            for (auto &imap: m_pChildren)
            {
                LOG_DEBUG("Removing Child GstNodetr'" << imap.second->GetName() 
                    << "' from Parent GST BIn'" << GetName() <<"'");
                
                DSL_NODETR_PTR pChildNodetr = 
                    std::dynamic_pointer_cast<Nodetr>(imap.second);

                // Increase the reference count so the child is not destroyed.
                gst_object_ref_sink(pChildNodetr->GetGstObject());

                if (!gst_bin_remove(GST_BIN(m_pGstObj), 
                    pChildNodetr->GetGstElement()))
                {
                    LOG_ERROR("Failed to remove GstNodetr " 
                        << pChildNodetr->GetName() << " from " << GetName() <<"'");
                }
                pChildNodetr->m_pParentGstObj = NULL;
            }
            Nodetr::RemoveAllChildren();
        }
        
        /**
         * @brief Creates a new Ghost pad for this Nodetr and adds it
         * to its paranet. 
         * @param[in] padname which pad to create the ghostpad for, and
         * name to use for the ghostpad. 
         * @throws a general exception on failure.
         */
        void AddGhostPadToParent(const char* padname)
        {
            LOG_FUNC();

            // create a new ghost pad with the static Sink pad retrieved from 
            // this Elementr's pGstObj and adds it to the the Elementr's Parent 
            // Bintr's pGstObj.
            GstPad* pStaticPad = gst_element_get_static_pad(
                GetGstElement(), padname);   
            
            if (!gst_element_add_pad(GST_ELEMENT(GetParentGstObject()), 
                gst_ghost_pad_new(padname, pStaticPad)))
            {
                LOG_ERROR("Failed to add Pad '" << padname 
                    << "' to parent of element'" << GetName() << "'");
                throw std::exception();
            }
            gst_object_unref(pStaticPad);
        }

        /**
         * @brief Creates a new Ghost Pad for this Gst Nodetr and adds it to 
         * its parent.
         * @param[in] padname which pad to create the ghostpad for.
         * @param[in] ghostPadname name to give the ghostpad to add.
         * @throws a general exception on failure.
         */
        void AddGhostPadToParent(const char* padname, const char* ghostPadname)
        {
            LOG_FUNC();

            // create a new ghost pad with the static Sink pad retrieved from 
            // this Elementr's pGstObj and adds it to the the Elementr's Parent 
            // Bintr's pGstObj.
            GstPad* pStaticPad = gst_element_get_static_pad(
                GetGstElement(), padname);   
            
            if (!gst_element_add_pad(GST_ELEMENT(GetParentGstObject()), 
                gst_ghost_pad_new(ghostPadname, pStaticPad)))
            {
                LOG_ERROR("Failed to add Pad '" << padname 
                    << "' to parent of element'" << GetName() << "'");
                throw std::exception();
            }
            gst_object_unref(pStaticPad);
        }

        /**
         * @brief Removed a Ghost Pad that was previously added to this Notetr's 
         * Parent.
         * @param[in] ghostPadname name of the ghostPad to remove.
         * @throws a general exception on failure.
         */
        void RemoveGhostPadFromParent(const char* ghostPadName)
        {
            LOG_FUNC();

            GstPad* pStaticPad = gst_element_get_static_pad(
                GST_ELEMENT(GetParentGstObject()), ghostPadName);       

            if (!pStaticPad)
            {
                LOG_ERROR("Failed to get Static Pad '" << ghostPadName 
                    << "' for parent of element'" << GetName() << "'");
                throw std::exception();
            }
                
            if (!gst_element_remove_pad(GST_ELEMENT(GetParentGstObject()), 
                pStaticPad))
            {
                LOG_ERROR("Failed to remove Pad '" << ghostPadName 
                    << "' for element'" << GetName() << "'");
                throw std::exception();
            }
            gst_object_unref(pStaticPad);
        }

        /**
         * @brief links this Elementr as Source to a given Sink Elementr
         * @param[in] pSink to link to
         */
        bool LinkToSink(DSL_NODETR_PTR pSink)
        { 
            LOG_FUNC();
            
            // Call the base class to setup the relationship first
            // Then call GST to Link Source Element to Sink Element 
            if (!Nodetr::LinkToSink(pSink) or 
                !gst_element_link(GetGstElement(), m_pSink->GetGstElement()))
            {
                LOG_ERROR("Failed to link " << GetName() 
                    << " to " << pSink->GetName());
                return false;
            }
            return true;
        }

        /**
         * @brief unlinks this Nodetr from a previously linked-to Sink Notetr
         */
        bool UnlinkFromSink()
        { 
            LOG_FUNC();

            // Need to check here first, as we're calling the parent class 
            // last when unlinking
            if (!IsLinkedToSink())
            {
                LOG_ERROR("GstNodetr '" << GetName() 
                    << "' is not in a linked state");
                return false;
            }
            if (!GetGstElement() or !m_pSink->GetGstElement())
            {
                LOG_ERROR("Invalid GstElements for  '" << GetName());
                return false;
            }
            gst_element_unlink(GetGstElement(), m_pSink->GetGstElement());
            return Nodetr::UnlinkFromSink();
        }

        /**
         * @brief links this Elementr as Source to a given Sink Elementr
         * @param[in] pSink to link to
         */
        bool LinkAudioToSink(DSL_NODETR_PTR pSink)
        { 
            LOG_FUNC();
            
            if (m_pAudioSink)
            {
                LOG_ERROR("Can't link GstNodetr" << GetName() 
                    << " as it's currently linked");
                return false;
            }

            // Get a reference to this GstNodetr's source pad
            GstPad* pStaticAudioSrcPad = gst_element_get_static_pad(GetGstElement(), 
                "audio_src");
            if (!pStaticAudioSrcPad)
            {
                LOG_ERROR("Failed to get Static Audio Src Pad for GstNodetr '" 
                    << GetName() << "'");
                return false;
            }

            // Get a reference to the sink element's static sink pad
            GstPad* pStaticSinkPad = gst_element_get_static_pad(
                    pSink->GetGstElement(), "sink");
            if (!pStaticSinkPad)
            {
                LOG_ERROR("Failed to get Static sink Pad for GstNodetr '" 
                    << GetName() << "'");
                return false;
            }
            if (gst_pad_link(pStaticAudioSrcPad, 
                pStaticSinkPad) != GST_PAD_LINK_OK)
            {
                LOG_ERROR("GstNodetr '" << GetName() 
                    << "' failed to link to Muxer");
                return false;
            }
            
            // unreference both static pads
            gst_object_unref(pStaticAudioSrcPad);
            gst_object_unref(pStaticSinkPad);

            // persist the relationship.
            m_pAudioSink = pSink;
            return true;
        }

        /**
         * @brief unlinks this Nodetr from a previously linked-to Sink Notetr
         */
        bool UnlinkAudioFromSink()
        { 
            LOG_FUNC();

            if (!m_pAudioSink)
            {
                LOG_ERROR("GstNodetr '" << GetName() 
                    << "' is not in a linked state");
                return false;
            }

            if (!GetGstElement() or !m_pAudioSink->GetGstElement())
            {
                LOG_ERROR("Invalid GstElements for  '" << GetName());
                return false;
            }
            gst_element_unlink(GetGstElement(), m_pAudioSink->GetGstElement());

            // clear the relationship
            m_pAudioSink = nullptr;
            
            return true;
        }

        /**
         * @brief links this Elementr as Source to a given Sink Elementr
         * @param[in] pSink to link to
         */
        bool LinkVideoToSink(DSL_NODETR_PTR pSink)
        { 
            LOG_FUNC();
            
            // Call the base class to setup the relationship
            if (m_pVideoSink)
            {
                LOG_ERROR("Can't link GstNodetr" << GetName() 
                    << " as it's currently linked");
                return false;
            }

            // Get a reference to this GstNodetr's source pad
            GstPad* pStaticVideoSrcPad = gst_element_get_static_pad(GetGstElement(), 
                "video_src");
            if (!pStaticVideoSrcPad)
            {
                LOG_ERROR("Failed to get Static Audio Src Pad for GstNodetr '" 
                    << GetName() << "'");
                return false;
            }

            // Get a reference to this static sink pad
            GstPad* pStaticSinkPad = gst_element_get_static_pad(
                    pSink->GetGstElement(), "sink");
            if (!pStaticSinkPad)
            {
                LOG_ERROR("Failed to get Static sink Pad for GstNodetr '" 
                    << GetName() << "'");
                return false;
            }
            if (gst_pad_link(pStaticVideoSrcPad, 
                pStaticSinkPad) != GST_PAD_LINK_OK)
            {
                LOG_ERROR("GstNodetr '" << GetName() 
                    << "' failed to link to Muxer");
                return false;
            }
            
            // unreference both static pads
            gst_object_unref(pStaticVideoSrcPad);
            gst_object_unref(pStaticSinkPad);

            // persist the relationship.
            m_pVideoSink = pSink;
            
            return true;
        }

        /**
         * @brief unlinks this Nodetr from a previously linked-to Sink Notetr
         */
        bool UnlinkVideoFromSink()
        { 
            LOG_FUNC();

            if (!m_pVideoSink)
            {
                LOG_ERROR("GstNodetr '" << GetName() 
                    << "' is not in a linked state");
                return false;
            }
            if (!GetGstElement() or !m_pVideoSink->GetGstElement())
            {
                LOG_ERROR("Invalid GstElements for  '" << GetName());
                return false;
            }
            gst_element_unlink(GetGstElement(), m_pVideoSink->GetGstElement());

            // clear the relationship
            m_pVideoSink = nullptr;

            return true;
        }

        /**
         * @brief links this Elementr as Sink to a given Source Nodetr
         * @param[in] pSinkBintr to link to
         */
        bool LinkToSource(DSL_NODETR_PTR pSrc)
        { 
            LOG_FUNC();
            
            DSL_NODETR_PTR pSrcNodetr = std::dynamic_pointer_cast<Nodetr>(pSrc);

            // Call the base class to setup the relationship first
            // Then call GST to Link Source Element to Sink Element 
            if (!Nodetr::LinkToSource(pSrcNodetr) or 
                !gst_element_link(pSrcNodetr->GetGstElement(), GetGstElement()))
            {
                LOG_ERROR("Failed to link Source '" << pSrcNodetr->GetName() 
                    << " to Sink" << GetName());
                return false;
            }
            return true;
        }

        /**
         * @brief returns the currently linked to Sink Nodetr
         * @return shared pointer to Sink Nodetr, nullptr if Unlinked to sink
         */
        DSL_GSTNODETR_PTR GetGstSource()
        {
            LOG_FUNC();
            
            return std::dynamic_pointer_cast<GstNodetr>(m_pSrc);
        }

        /**
         * @brief unlinks this Elementr from a previously linked-to Source Element
         */
        bool UnlinkFromSource()
        { 
            LOG_FUNC();

            // Need to check here first, as we're calling the parent class 
            // last when unlinking
            if (!IsLinkedToSource())
            {
                LOG_ERROR("GstNodetr '" << GetName() << "' is not in a linked state");
                return false;
            }
            if (!m_pSrc->GetGstElement() or !GetGstElement())
            {
                LOG_ERROR("Invalid GstElements for  '" << m_pSrc->GetName() 
                    << "' and '" << GetName() << "'");
                return false;
            }
            gst_element_unlink(m_pSrc->GetGstElement(), GetGstElement());

            return Nodetr::UnlinkFromSource();
        }

        /**
         * @brief links this Noder to the Sink Pad of Muxer
         * @param[in] pMuxer nodetr to link to
         * @param[in] sinkPadName name to give the requested Sink Pad
         * @return true if able to successfully link with Muxer Sink Pad
         */
        virtual bool LinkToSinkMuxer(DSL_NODETR_PTR pMuxer, 
            const char* srcPadName, const char* sinkPadName)
        {
            LOG_FUNC();
            
            // Get a reference to this GstNodetr's source pad
            GstPad* pStaticSrcPad = gst_element_get_static_pad(GetGstElement(), 
                srcPadName);
            if (!pStaticSrcPad)
            {
                LOG_ERROR("Failed to get Static Src Pad for GstNodetr '" 
                    << GetName() << "'");
                return false;
            }

            // Request a new sink pad from the Muxer to connect to this 
            // GstNodetr's source pad
            GstPad* pRequestedSinkPad = gst_element_get_request_pad(
                pMuxer->GetGstElement(), sinkPadName);
            if (!pRequestedSinkPad)
            {
                LOG_ERROR("Failed to get requested Tee Sink Pad for GstNodetr '" 
                    << GetName() <<"'");
                return false;
            }
            
            LOG_INFO("Linking requested Sink Pad'" << pRequestedSinkPad 
                << "' for GstNodetr '" << GetName() << "'");
                
            if (gst_pad_link(pStaticSrcPad, 
                pRequestedSinkPad) != GST_PAD_LINK_OK)
            {
                LOG_ERROR("GstNodetr '" << GetName() 
                    << "' failed to link to Muxer");
                return false;
            }
            
            // unreference both the static source pad and requested sink
            gst_object_unref(pStaticSrcPad);
            gst_object_unref(pRequestedSinkPad);

            // call the parent class to complete the link-to-sink
            return true;
        }

        /**
         * @brief Sets the state of the Src to NULL and then sends flush-start, 
         * flush-stop, EOS events to the muxers Sink Pad connected to this GstNoder.
         * @return true if able to successfully EOS the Sink Pad
         */
        virtual bool NullSrcEosSinkMuxer(const char* srcPadName)
        {
            LOG_FUNC();
            
            GstState currState, nextState;
            GstStateChangeReturn result = gst_element_get_state(GetGstElement(), 
                &currState, &nextState, 1);

            // TODO - removing for now until Audio shutdown is better understood
            // This function will be called twice if Audio and Video Sources. The
            // current state will be null on second call. 
            // if (currState < GST_STATE_PLAYING)
            // {
            //     LOG_ERROR("GstNodetr '" << GetName() 
            //         << "' is not in a PLAYING state");
            //     return false;
            // }

            // Get a reference to this GstNodetr's source pad
            GstPad* pStaticSrcPad = gst_element_get_static_pad(GetGstElement(),
                srcPadName);
            if (!pStaticSrcPad)
            {
                LOG_ERROR("Failed to get static source pad for GstNodetr '" 
                    << GetName() << "'");
                return false;
            }
            
            // Get a reference to the Muxer's sink pad that is connected
            // to this GstNodetr's source pad
            GstPad* pRequestedSinkPad = gst_pad_get_peer(pStaticSrcPad);
            if (!pRequestedSinkPad)
            {
                LOG_ERROR("Failed to get requested sink pad peer for GstNodetr '" 
                    << GetName() << "'");
                return false;
            }

            GstStateChangeReturn changeResult = gst_element_set_state(
                GetGstElement(), GST_STATE_NULL);
                
            switch (changeResult)
            {
            case GST_STATE_CHANGE_FAILURE:
                LOG_ERROR("GstNodetr '" << GetName() 
                    << "' failed to set state to NULL");
                return false;

            case GST_STATE_CHANGE_ASYNC:
                LOG_INFO("GstNodetr '" << GetName() 
                    << "' changing state to NULL async");
                    
                // block on get state until change completes. 
                if (gst_element_get_state(GetGstElement(), 
                    NULL, NULL, GST_CLOCK_TIME_NONE) == GST_STATE_CHANGE_FAILURE)
                {
                    LOG_ERROR("GstNodetr '" << GetName() 
                        << "' failed to set state to NULL");
                    return false;
                }
                // drop through on success - DO NOT BREAK

            case GST_STATE_CHANGE_SUCCESS:
                LOG_INFO("GstNodetr '" << GetName() 
                    << "' changed state to NULL successfully");
                    
                // Send flush-start and flush-stop events downstream to the muxer 
                // followed by an end-of-stream for this GstNodetr's stream
                gst_pad_send_event(pRequestedSinkPad, 
                    gst_event_new_flush_start());
                gst_pad_send_event(pRequestedSinkPad, 
                    gst_event_new_flush_stop(TRUE));
                gst_pad_send_event(pRequestedSinkPad, 
                    gst_event_new_eos());

                break;
            default:
                LOG_ERROR("Unknown state change for Bintr '" << GetName() << "'");
                return false;
            }

            // unreference both the static source pad and requested sink
            gst_object_unref(pStaticSrcPad);
            gst_object_unref(pRequestedSinkPad);
            
            // Call the parent class to complete the unlink from sink
            return true;
        }
        
        /**
         * @brief unlinks this Nodetr from a previously linked Muxer Sink Pad
         * @param[in] pMuxer sink muxer to unlink from
         * @param[in] srcPadName name of the Src Pad to unlink
         * @return true if able to successfully unlink from Muxer Sink Pad
         */
        virtual bool UnlinkFromSinkMuxer(DSL_NODETR_PTR pMuxer, 
            const char* srcPadName)
        {
            LOG_FUNC();
            
            // Get a reference to this GstNodetr's source pad
            GstPad* pStaticSrcPad = gst_element_get_static_pad(GetGstElement(), 
                srcPadName);
            if (!pStaticSrcPad)
            {
                LOG_ERROR("Failed to get static source pad for GstNodetr '" 
                    << GetName() << "'");
                return false;
            }
            
            // Get a reference to the Muxer's sink pad that is connected
            // to this GstNodetr's source pad
            GstPad* pRequestedSinkPad = gst_pad_get_peer(pStaticSrcPad);
            if (!pRequestedSinkPad)
            {
                LOG_ERROR("Failed to get requested sink pad peer for GstNodetr '" 
                    << GetName() << "'");
                return false;
            }

            GstState currState, nextState;
            GstStateChangeReturn result = gst_element_get_state(GetGstElement(), 
                &currState, &nextState, 1);

            if (currState > GST_STATE_NULL)
            { 
                GstStateChangeReturn changeResult = gst_element_set_state(
                    GetGstElement(), GST_STATE_NULL);
                    
                switch (changeResult)
                {
                case GST_STATE_CHANGE_FAILURE:
                    LOG_ERROR("GstNodetr '" << GetName() 
                        << "' failed to set state to NULL");
                    return false;

                case GST_STATE_CHANGE_ASYNC:
                    LOG_INFO("GstNodetr '" << GetName() 
                        << "' changing state to NULL async");
                        
                    // block on get state until change completes. 
                    if (gst_element_get_state(GetGstElement(), 
                        NULL, NULL, GST_CLOCK_TIME_NONE) == GST_STATE_CHANGE_FAILURE)
                    {
                        LOG_ERROR("GstNodetr '" << GetName() 
                            << "' failed to set state to NULL");
                        return false;
                    }
                    // drop through on success - DO NOT BREAK

                case GST_STATE_CHANGE_SUCCESS:
                    LOG_INFO("GstNodetr '" << GetName() 
                        << "' changed state to NULL successfully");
                        
                    // Send flush-start and flush-stop events downstream to the muxer 
                    // followed by an end-of-stream for this GstNodetr's stream
                    gst_pad_send_event(pRequestedSinkPad, 
                        gst_event_new_flush_start());
                    gst_pad_send_event(pRequestedSinkPad, 
                        gst_event_new_flush_stop(TRUE));
                    gst_pad_send_event(pRequestedSinkPad, 
                        gst_event_new_eos());

                    break;
                default:
                    LOG_ERROR("Unknown state change for Bintr '" << GetName() << "'");
                    return false;
                }
            }
            LOG_INFO("Unlinking and releasing requested sink pad '" 
                << pRequestedSinkPad << "' for GstNodetr '" << GetName() << "'");

            // It should now be safe to unlink this GstNodetr from the Muxer
            if (!gst_pad_unlink(pStaticSrcPad, pRequestedSinkPad))
            {
                LOG_ERROR("GstNodetr '" << GetName() 
                    << "' failed to unlink from Muxer");
                return false;
            }
            // Need to release the previously requested sink pad
            gst_element_release_request_pad(pMuxer->GetGstElement(), 
                pRequestedSinkPad);

            // unreference both the static source pad and requested sink
            gst_object_unref(pStaticSrcPad);
            gst_object_unref(pRequestedSinkPad);

            return true;
        }
        
        /**
         * @brief links this Nodetr as a Sink to the Source Pad of a Splitter Tee
         * @param[in] pTee to link to
         * @param[in] padName name to give the requested Src Pad
         * @return true if able to successfully link with Tee Src Pad
         */
        virtual bool LinkToSourceTee(DSL_NODETR_PTR pTee, const char* padName)
        {
            LOG_FUNC();
            
            // Get a reference to the static sink pad for this GstNodetr
            GstPad* pStaticSinkPad = gst_element_get_static_pad(
                GetGstElement(), "sink");
            if (!pStaticSinkPad)
            {
                LOG_ERROR("Failed to get static sink pad for GstNodetr '" 
                    << GetName() << "'");
                return false;
            }

            // Request a new source pad from the Tee 
            GstPad* pRequestedSrcPad = gst_element_get_request_pad(
                pTee->GetGstElement(), padName);
            if (!pRequestedSrcPad)
            {
                LOG_ERROR("Failed to get a requested source pad for Tee '" 
                    << pTee->GetName() <<"'");
                return false;
            }
            m_releaseRequestedPadOnUnlink = true;

            LOG_INFO("Linking requested source pad'" << pRequestedSrcPad 
                << "' for GstNodetr '" << GetName() << "'");

            if (gst_pad_link(pRequestedSrcPad, pStaticSinkPad) != GST_PAD_LINK_OK)
            {
                LOG_ERROR("GstNodetr '" << GetName() 
                    << "' failed to link to Source Tee");
                return false;
            }

            // Unreference both the static sink pad and requested source pad
            gst_object_unref(pStaticSinkPad);
            gst_object_unref(pRequestedSrcPad);
            
            // Call the parent class to complete the link to source
            return Nodetr::LinkToSource(pTee);
        }

        /**
         * @brief links this Nodetr as a Sink to the Source Pad of a Splitter Tee
         * @param[in] pTee to link to
         * @param[in] pRequestedSrcPad requested source pad for the Tee to link with
         * @return true if able to successfully link with Tee Src Pad
         */
        virtual bool LinkToSourceTee(DSL_NODETR_PTR pTee, GstPad* pRequestedSrcPad)
        {
            LOG_FUNC();
            m_releaseRequestedPadOnUnlink = false;
            
            // Get a reference to the static sink pad for this GstNodetr
            GstPad* pStaticSinkPad = gst_element_get_static_pad(
                GetGstElement(), "sink");
            if (!pStaticSinkPad)
            {
                LOG_ERROR("Failed to get static sink pad for GstNodetr '" 
                    << GetName() << "'");
                return false;
            }

            LOG_INFO("Linking requested source pad'" << pRequestedSrcPad 
                << "' for GstNodetr '" << GetName() << "'");

            if (gst_pad_link(pRequestedSrcPad, pStaticSinkPad) != GST_PAD_LINK_OK)
            {
                LOG_ERROR("GstNodetr '" << GetName() 
                    << "' failed to link to Source Tee");
                return false;
            }

            // Unreference just the static sink pad and not the requested source pad
            gst_object_unref(pStaticSinkPad);
            
            // Call the parent class to complete the link to source
            return Nodetr::LinkToSource(pTee);
        }
        
        /**
         * @brief unlinks this Nodetr from a previously linked Source Tee
         * @return true if able to successfully unlink from Source Tee
         */
        virtual bool UnlinkFromSourceTee()
        {
            LOG_FUNC();
            
            if (!IsLinkedToSource())
            {
                return false;
            }

            // Get a reference to this GstNodetr's sink pad
            GstPad* pStaticSinkPad = gst_element_get_static_pad(GetGstElement(), "sink");
            if (!pStaticSinkPad)
            {
                LOG_ERROR("Failed to get static sink pad for GstNodetr '" 
                    << GetName() << "'");
                return false;
            }
            
            // Get a reference to the Tee's source pad that is connected
            // to this GstNodetr's sink pad
            GstPad* pRequestedSrcPad = gst_pad_get_peer(pStaticSinkPad);
            if (!pRequestedSrcPad)
            {
                LOG_ERROR("Failed to get requested source pad peer for GstNodetr '"
                    << GetName() << "'");
                return false;
            }

            LOG_INFO("Unlinking requested source pad '" 
                << pRequestedSrcPad << "' for GstNodetr '" << GetName() << "'");

            // It should now be safe to unlink this GstNodetr from the Muxer
            if (!gst_pad_unlink(pRequestedSrcPad, pStaticSinkPad))
            {
                LOG_ERROR("GstNodetr '" << GetName() 
                    << "' failed to unlink from source Tee");
                Nodetr::UnlinkFromSource();
                return false;
            }
            if (m_releaseRequestedPadOnUnlink)
            {
                LOG_INFO("Releasing requested source pad '" 
                    << pRequestedSrcPad << "' for GstNodetr '" << GetName() << "'");
                // Need to release the previously requested sink pad
                gst_element_release_request_pad(GetSource()->GetGstElement(), 
                    pRequestedSrcPad);
            }
            gst_object_unref(pStaticSinkPad);
            gst_object_unref(pRequestedSrcPad);

            return Nodetr::UnlinkFromSource();
        }
        
        /**
         * @brief Returns the current State of this GstNodetr's Parent
         * @return the current state of the Parenet, GST_STATE_NULL if the
         * GstNodetr is currently an orphen. 
         */
        uint GetParentState()
        {
            LOG_FUNC();
            
            if (!m_pParentGstObj)
            {
                return GST_STATE_NULL;
            }
            GstState currentState;
            
            gst_element_get_state(GetParentGstElement(), &currentState, NULL, 1);
            
            LOG_DEBUG("Returning a state of '" 
                << gst_element_state_get_name(currentState) 
                << "' for GstNodetr '" << GetName());
            
            return currentState;
        }

        bool SendEos()
        {
            LOG_FUNC();

            return gst_element_send_event(GetGstElement(), gst_event_new_eos());
        }

        uint GetState(GstState& state, GstClockTime timeout)
        {
            LOG_FUNC();

            uint retval = gst_element_get_state(GetGstElement(), 
                &state, NULL, timeout);
            LOG_DEBUG("Get state returned '" << gst_element_state_get_name(state) 
                << "' for GstNodetr '" << GetName() << "'");
            
            return retval;
        }
        
        /**
         * @brief Attempts to set the state of this GstNodetr's GST Element
         * @return true if successful transition, false on failure
         */
        bool SetState(GstState state, GstClockTime timeout)
        {
            LOG_FUNC();
            LOG_INFO("Changing state to '" << gst_element_state_get_name(state) 
                << "' for GstNodetr '" << GetName() << "'");

            GstStateChangeReturn returnVal = gst_element_set_state(GetGstElement(), 
                state);
            switch (returnVal) 
            {
                case GST_STATE_CHANGE_SUCCESS:
                    LOG_INFO("State change completed synchronously for GstNodetr'" 
                        << GetName() << "'");
                    return true;
                case GST_STATE_CHANGE_FAILURE:
                    LOG_ERROR("FAILURE occured when trying to change state to '" 
                        << gst_element_state_get_name(state) << "' for GstNodetr '" 
                        << GetName() << "'");
                    return false;
                case GST_STATE_CHANGE_NO_PREROLL:
                    LOG_INFO("Set state for GstNodetr '" << GetName() 
                        << "' returned GST_STATE_CHANGE_NO_PREROLL");
                    return true;
                case GST_STATE_CHANGE_ASYNC:
                    LOG_INFO("State change will complete asynchronously for GstNodetr '" 
                        << GetName() << "'");
                    break;
                default:
                    break;
            }
            
            // Wait until state change or failure, no timeout.
            if (gst_element_get_state(GetGstElement(), NULL, NULL, timeout) == 
                GST_STATE_CHANGE_FAILURE)
            {
                LOG_ERROR("FAILURE occured waiting for state to change to '" 
                    << gst_element_state_get_name(state) 
                        << "' for GstNodetr '" << GetName() << "'");
                return false;
            }
            LOG_INFO("State change completed asynchronously for GstNodetr'" 
                << GetName() << "'");
            return true;
        }

        uint SyncStateWithParent(GstState& parentState, GstClockTime timeout)
        {
            LOG_FUNC();
            
            uint returnVal = gst_element_sync_state_with_parent(GetGstElement());

            switch (returnVal) 
            {
                case GST_STATE_CHANGE_SUCCESS:
                    LOG_INFO("State change completed synchronously for GstNodetr'" 
                        << GetName() << "'");
                    return returnVal;
                case GST_STATE_CHANGE_FAILURE:
                    LOG_ERROR("FAILURE occured when trying to sync state with Parent for GstNodetr '" 
                        << GetName() << "'");
                    return returnVal;
                case GST_STATE_CHANGE_NO_PREROLL:
                    LOG_INFO("Set state for GstNodetr '" << GetName() 
                        << "' return GST_STATE_CHANGE_NO_PREROLL");
                    return returnVal;
                case GST_STATE_CHANGE_ASYNC:
                    LOG_INFO("State change will complete asynchronously for GstNodetr '" 
                        << GetName() << "'");
                    break;
                default:
                    break;
            }
            uint retval = gst_element_get_state(GST_ELEMENT_PARENT(GetGstElement()), 
                &parentState, NULL, timeout);
            LOG_INFO("Get state returned '" << gst_element_state_get_name(parentState) 
                << "' for Parent of GstNodetr '" << GetName() << "'");
            return retval;
        }
        
        /**
         * @brief Adds the "buffer" and "downstream event" Pad Probes to the sink-pad
         * of a given Element.
         * @param the Parent Element to add the Probes to.
         */
        void AddSinkPadProbes(GstElement* parentElement)
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
        void AddSrcPadProbes(GstElement* parentElement)
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
         * @brief Removes the "buffer" and "downstream event" Pad Probes from the 
         * src-pad of a given Element.
         * @param the Parent Element to add the Probes to.
         */
        void RemoveSrcPadProbes(GstElement* parentElement)
        {
            LOG_FUNC();
            
            m_pSrcPadBufferProbe = nullptr;
            m_pSrcPadDsEventProbe = nullptr;
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
        
    protected:
        
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
        
    private:
    
        /**
         * @brief true if the GstNodetr acts as a proxy for it's parent
         * using it's parent's bin i.e. m_pGstObj
         */
        bool m_isProxy;
        
        bool m_releaseRequestedPadOnUnlink;
    };

} // DSL namespace    

#endif // _DSL_NODETR_H
