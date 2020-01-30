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

#ifndef _DSL_NODETR_H
#define _DSL_NODETR_H

#include "Dsl.h"

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
     * @brief Implements a base container class for all DSL Tree Node types
     */
    class Nodetr : public std::enable_shared_from_this<Nodetr>
    {
    public:
        
        /**
         * @brief ctor for the Nodetr base class
         * @param[in] name for the new Nodetr
         */
        Nodetr(const char* name)
        : m_name(name)
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
         * @brief return the name given to this Nodetr on creation
         * @return const std::string name given to this Nodetr
         */
        const std::string& GetName()
        {
            LOG_FUNC();
            
            return m_name;
        }
        
        /**
         * @brief return the name given to this Nodetr on creation
         * @return const c_str name given to this Nodetr
         */
        const char* GetCStrName()
        {
            LOG_FUNC();
            
            return m_name.c_str();
        }
        
        /**
         * @brief adds a child Nodetr to this parent Nodetr
         * @param[in] pChild to add to this parent Nodetr. 
         */
        virtual bool AddChild(DSL_NODETR_PTR pChild)
        {
            LOG_FUNC();
            
            if (IsChild(pChild))
            {
                LOG_ERROR("Child '" << pChild->m_name << "' is not unique for Parent '" <<m_name << "'");
                return false;
            }
            m_pChildren[pChild->m_name] = pChild;
            pChild->m_pParentGstObj = m_pGstObj;   
                            
            LOG_DEBUG("Child '" << pChild->m_name <<"' added to Parent '" << m_name << "'");
            
            return true;
        }
        
        /**
         * @brief removed a child Nodetr of this parent Nodetr
         * @param[in] pChild to remove
         */
        virtual bool RemoveChild(DSL_NODETR_PTR pChild)
        {
            LOG_FUNC();
            
            if (!IsChild(pChild))
            {
                LOG_WARN("'" << pChild->m_name <<"' is not a child of Parent '" << m_name <<"'");
                return false;
            }
            pChild->m_pParentGstObj = NULL;
            m_pChildren[pChild->m_name] = nullptr;
            m_pChildren.erase(pChild->m_name);
                            
            LOG_DEBUG("Child '" << pChild->m_name <<"' removed from Parent '" << m_name <<"'");
            
            return true;
        }

        /**
         * @brief removes all child Nodetrs from this parent Nodetr
         */
        virtual void RemoveAllChildren()
        {
            LOG_FUNC();

            for (auto &imap: m_pChildren)
            {
                LOG_DEBUG("Removing Child '" << imap.second->GetName() <<"' from Parent '" << GetName() <<"'");
                imap.second->m_pParentGstObj = NULL;
            }
            m_pChildren.clear();
        }
        
        /**
         * @brief function to determine if a Nodetr is a child of this Nodetr
         * @param[in] pChild Nodetr to test for the child relationship
         * @return true if pChild is a child of this Nodetr
         */
        virtual bool IsChild(DSL_NODETR_PTR pChild)
        {
            LOG_FUNC();
            
            return (m_pChildren.find(pChild->GetName()) != m_pChildren.end());
        }
        
        /**
         * @brief determines whether this Nodetr is a child of a given pParent
         * @param[in] pParent the Nodetr to check for a Parental relationship
         * @return True if the provided Nodetr is this Nodetr's Parent
         */
        virtual bool IsParent(DSL_NODETR_PTR pParent)
        {
            LOG_FUNC();
            
            return (m_pParentGstObj == pParent->m_pGstObj);
        }
        
        /**
         * @brief Links this Noder, becoming a source, to a sink Nodre
         * @param[in] pSink Sink Nodre to link this Source Nodre to
         */
        virtual bool LinkToSink(DSL_NODETR_PTR pSink)
        {
            LOG_FUNC();

            if (m_pSink)
            {
                LOG_ERROR("Nodetr '" << GetName() << "' is currently in a linked to Sink");
                return false;
            }
            m_pSink = pSink;
//            pSink->m_pSource = shared_from_this();   
            LOG_DEBUG("Source '" << GetName() << "' linked to Sink '" << pSink->GetName() << "'");
            
            return true;
        }
        
        /**
         * @brief Unlinks this Source Nodetr from its Sink Nodetr
         */
        virtual bool UnlinkFromSink()
        {
            LOG_FUNC();

            if (!m_pSink)
            {
                LOG_ERROR("Nodetr '" << GetName() << "' is not currently linked to Sink");
                return false;
            }
            LOG_DEBUG("Unlinking Source '" << GetName() <<"' from Sink '" << m_pSink->GetName() << "'");
//            m_pSink->m_pSource = nullptr;
            m_pSink = nullptr; 

            return true;
        }
        
        /**
         * @brief Links this Noder, becoming a source, to a sink Nodre
         * @param[in] pSource Nodre to link this Sink Nodre back to
         */
        virtual bool LinkToSource(DSL_NODETR_PTR pSource)
        {
            LOG_FUNC();

            if (m_pSource)
            {
                LOG_ERROR("Nodetr '" << GetName() << "' is currently in a linked to a Source");
                return false;
            }
            m_pSource = pSource;
//            pSink->m_pSource = shared_from_this();   
            LOG_DEBUG("Source '" << pSource->GetName() << "' linked to Sink '" << GetName() << "'");
            
            return true;
        }
        
        /**
         * @brief Unlinks this Sink Nodetr from its Source Nodetr
         */
        virtual bool UnlinkFromSource()
        {
            LOG_FUNC();

            if (!m_pSource)
            {
                LOG_ERROR("Nodetr '" << GetName() << "' is not currently linked to Source");
                return false;
            }
            LOG_DEBUG("Unlinking self '" << GetName() <<"' as a Sink from '" << m_pSource->GetName() << "' Source");
//            m_pSource->m_pSink = nullptr;
            m_pSource = nullptr;
            
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
            
            return bool(m_pSource);
        }
        
        /**
         * @brief returns the currently linked to Sink Nodetr
         * @return shared pointer to Sink Nodetr, nullptr if Unlinked to sink
         */
        DSL_NODETR_PTR GetSink()
        {
            LOG_FUNC();
            
            return m_pSink;
        }
        
        /**
         * @brief returns the currently linked to Sink Nodetr
         * @return shared pointer to Sink Nodetr, nullptr if Unlinked to sink
         */
        DSL_NODETR_PTR GetSource()
        {
            LOG_FUNC();
            
            return m_pSource;
        }
        
        /**
         * @brief get the current number of children for this Nodetr 
         * @return the number of Child Nodetrs held by this Parent Nodetr
         */
        uint GetNumChildren()
        {
            LOG_FUNC();
            
            return m_pChildren.size();
        }

        GstObject* GetGstObject()
        {
            LOG_FUNC();
            
            return GST_OBJECT(m_pGstObj);
        }
        
        GstElement* GetGstElement()
        {
            LOG_FUNC();
            
            return GST_ELEMENT(m_pGstObj);
        }
        
        GObject* GetGObject()
        {
            LOG_FUNC();
            
            return G_OBJECT(m_pGstObj);
        }

        GstObject* GetParentGstObject()
        {
            LOG_FUNC();
            
            return GST_OBJECT(m_pParentGstObj);
        }

        GstElement* GetParentGstElement()
        {
            LOG_FUNC();
            
            return GST_ELEMENT(m_pParentGstObj);
        }
        
        /**
         * @brief backdoor for testing purposes only, untill GST test doubles can be used.
         */
        void __setGstObject__(GstObject* pGstObj)
        {
            LOG_FUNC();
            
            m_pGstObj = pGstObj;
        }

    public:
    
        /**
         * @brief Parent of this Nodetr if one exists. NULL otherwise
         */
        GstObject * m_pParentGstObj;
        
        
    protected:

        /**
         * @brief unique name for this Nodetr
         */
        std::string m_name;

        /**
         * @brief Gst object wrapped by the Nodetr
         */
        GstObject * m_pGstObj;

        /**
         * @brief map of Child Nodetrs in-use by this Nodetr
         */
        std::map<std::string, DSL_NODETR_PTR> m_pChildren;
        
        /**
         * @brief defines the relationship between a Source Nodetr
         * linked to this Nodetr, making this Nodetr a Sink
         */
        DSL_NODETR_PTR m_pSource;

        /**
         * @brief defines the relationship this Nodetr linked to
         * a Sink Nodetr making this Nodetr a Source
         */
        DSL_NODETR_PTR m_pSink;
    };

   /**
     * @class GstNodetr
     * @brief Overrides the Base Class Virtual functions, adding the actuall GstObject* management
     * This allows the Nodetr class, and all its relational behavior, to be tested independent from GStreamer
     * Each method of this class calls the base class to complete its behavior.
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
                LOG_WARN("GstElement for GstNodetr '" << GetName() << "' has not been instantiated");
            }
            else
            {
            // Remove all child references 
                RemoveAllChildren();

                LOG_DEBUG("Setting GstElement for GstNodetr '" << GetName() << "' to GST_STATE_NULL");
                gst_element_set_state(GetGstElement(), GST_STATE_NULL);
                
                if (!m_pParentGstObj)
                {
                    LOG_DEBUG("Unreferencing GST Object contained by this Bintr '" << GetName() << "'");
                    gst_object_unref(m_pGstObj);
                }
            }
            LOG_DEBUG("Nodetr '" << GetName() << "' deleted");
        }

        /**
         * @brief adds a child Bintr to this parent Bintr
         * @param[in] pChildBintr to add. Once added, calling InUse()
         *  on the Child Bintr will return true
         * @return true if pChild was added successfully, false otherwise
         */
        bool AddChild(DSL_NODETR_PTR pChild)
        {
            LOG_FUNC();
            
            LOG_DEBUG("Adding Child element to Bin");
            if (!gst_bin_add(GST_BIN(m_pGstObj), pChild->GetGstElement()))
            {
                LOG_ERROR("Failed to add " << pChild->GetName() << " to " << GetName() <<"'");
                throw;
            }
            return Nodetr::AddChild(pChild);
        }
        
        /**
         * @brief removes a child Bintr from this parent Bintr
         * @param[in] pChildBintr to remove. Once removed, calling InUse()
         *  on the Child Bintr will return false
         */
        bool RemoveChild(DSL_NODETR_PTR pChild)
        {
            LOG_FUNC();
            
            if (!IsChild(pChild))
            {
                LOG_ERROR("'" << pChild->GetName() << "' is not a child of '" << GetName() <<"'");
                return false;
            }

            // Increase the reference count so the child is not destroyed.
            gst_object_ref(pChild->GetGstElement());
            
            if (!gst_bin_remove(GST_BIN(m_pGstObj), pChild->GetGstElement()))
            {
                LOG_ERROR("Failed to remove " << pChild->GetName() << " from " << GetName() <<"'");
                return false;
            }
            return Nodetr::RemoveChild(pChild);
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
                LOG_DEBUG("Removing Child '" << imap.second->GetName() <<"' from Parent '" << GetName() <<"'");
                
                // Increase the reference count so the child is not destroyed.
                gst_object_ref(imap.second->GetGstElement());

                if (!gst_bin_remove(GST_BIN(m_pGstObj), imap.second->GetGstElement()))
                {
                    LOG_ERROR("Failed to remove " << imap.second->GetName() << " from " << GetName() <<"'");
                }
                imap.second->m_pParentGstObj = NULL;
            }
            m_pChildren.clear();
        }
        
        /**
         * @brief Creates a new Ghost Sink pad for this Gst Element
         * @param[in] name unique name for the Ghost Pad
         * and adds it to the parent Gst Bin.
         * @throws a general exception on failure
         */
        void AddGhostPadToParent(const char* name)
        {
            LOG_FUNC();

            // create a new ghost pad with the static Sink pad retrieved from this Elementr's 
            // pGstObj and adds it to the the Elementr's Parent Bintr's pGstObj.
            if (!gst_element_add_pad(GST_ELEMENT(GetParentGstObject()), 
                gst_ghost_pad_new(name, gst_element_get_static_pad(GetGstElement(), name))))
            {
                LOG_ERROR("Failed to add Pad '" << name << "' for element'" << GetName() << "'");
                throw;
            }
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
            if (!Nodetr::LinkToSink(pSink) or !gst_element_link(GetGstElement(), m_pSink->GetGstElement()))
            {
                LOG_ERROR("Failed to link " << GetName() << " to " << pSink->GetName());
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

            // Need to check here first, as we're calling the base class last when unlinking
            if (!IsLinkedToSink())
            {
                LOG_ERROR("GstNodetr '" << GetName() << "' is not in a linked state");
                return false;
            }
            if (!GetGstElement() or !m_pSink->GetGstElement())
            {
//                LOG_ERROR("Invalid GstElements for  '" << GetName() << "' and '" << m_pSink>GetName() << "'");
                LOG_ERROR("Invalid GstElements for  '" << GetName());
                return false;
            }
            gst_element_unlink(GetGstElement(), m_pSink->GetGstElement());
            
            return Nodetr::UnlinkFromSink();
        }

        /**
         * @brief links this Elementr as Sink to a given Source Nodetr
         * @param[in] pSinkBintr to link to
         */
        bool LinkToSource(DSL_NODETR_PTR pSource)
        { 
            LOG_FUNC();
            
            // Call the base class to setup the relationship first
            // Then call GST to Link Source Element to Sink Element 
            if (!Nodetr::LinkToSource(pSource) or !gst_element_link(m_pSource->GetGstElement(), GetGstElement()))
            {
                LOG_ERROR("Failed to link Source '" << pSource->GetName() << " to Sink" << GetName());
                return false;
            }
            return true;
        }

        /**
         * @brief unlinks this Elementr from a previously linked-to Source Element
         */
        bool UnlinkFromSource()
        { 
            LOG_FUNC();

            // Need to check here first, as we're calling the base class last when unlinking
            if (!IsLinkedToSource())
            {
                LOG_ERROR("GstNodetr '" << GetName() << "' is not in a linked state");
                return false;
            }
            if (!m_pSource->GetGstElement() or !GetGstElement())
            {
                LOG_ERROR("Invalid GstElements for  '" << m_pSource->GetName() << "' and '" << GetName() << "'");
                return false;
            }
            gst_element_unlink(m_pSource->GetGstElement(), GetGstElement());

            return Nodetr::UnlinkFromSource();
        }
        
        /**
         * @brief Returns the current State of this GstNodetr
         * @return the current state of the GstNodetr. 
         */
        uint GetState()
        {
            LOG_FUNC();
            
            GstState currentState;
            
            if (gst_element_get_state(GetGstElement(), &currentState, NULL, 1) == GST_STATE_CHANGE_ASYNC)
            {
                return DSL_STATE_IN_TRANSITION;
            }
            
            LOG_DEBUG("Returning a state of '" << gst_element_state_get_name(currentState)
                << "' for Nodetr '" << GetName());
            
            return currentState;
        }
        
        /**
         * @brief Returns the current State of this Bintr's Parent
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
            
            if (gst_element_get_state(GetParentGstElement(), &currentState, NULL, 1) == GST_STATE_CHANGE_ASYNC)
            {
                return DSL_STATE_IN_TRANSITION;
            }
            
            LOG_DEBUG("Returning a state of '" << gst_element_state_get_name(currentState) 
                << "' for Nodetr '" << GetName());
            
            return currentState;
        }
    };

} // DSL namespace    

#endif // _DSL_NODETR_H