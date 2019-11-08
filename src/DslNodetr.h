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

    /**
     * @class Padtr
     * @brief Implements a base container class for all DSL nodes types
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

            LOG_INFO("New Nodetr '" << m_name << "' created");
        }

        /**
         * @brief dtor for the Nodetr base class
         */
        ~Nodetr()
        {
            LOG_FUNC();
            
            // Remove all child references 
            RemoveAllChildren();
            
            LOG_INFO("Nodetr '" << m_name << "' deleted");
        }
        
        /**
         * @brief returns whether the Nodetr object is in-use 
         * @return True if the Nodetr has a relationship with another Nodetr
         */
        bool IsInUse()
        {
            LOG_FUNC();

            return (bool)(m_pParentGstObj or m_pChildren.size() or m_pSink or m_pSource);
        }

        /**
         * @brief adds a child Nodetr to this parent Nodetr
         * @param pChild to add to this parent Nodetr. 
         */
        virtual DSL_NODETR_PTR AddChild(DSL_NODETR_PTR pChild)
        {
            LOG_FUNC();
            
            if (m_pChildren[pChild->m_name])
            {
                LOG_ERROR("Child '" << pChild->m_name << "' is not unique for Parent '" <<m_name << "'");
                throw;
            }
            m_pChildren[pChild->m_name] = pChild;
            pChild->m_pParentGstObj = m_pGstObj;   
                            
            LOG_INFO("Child '" << pChild->m_name <<"' added to Parent '" << m_name << "'");
            
            return pChild;
        }
        
        /**
         * @brief removed a child Nodetr of this parent Nodetr
         * @param pChild to remove
         */
        virtual void RemoveChild(DSL_NODETR_PTR pChild)
        {
            LOG_FUNC();
            
            if (!IsChild(pChild))
            {
                LOG_INFO("'" << pChild->m_name <<"' is not a child of Parent '" << m_name <<"'");
                throw;
            }
            pChild->m_pParentGstObj = NULL;
            m_pChildren[pChild->m_name] = nullptr;
            m_pChildren.erase(pChild->m_name);
                            
            LOG_INFO("Child '" << pChild->m_name <<"' removed from Parent '" << m_name <<"'");
        }

        /**
         * @brief removed a child Nodetr of this parent Nodetr
         * @param pChild to remove
         */
        virtual void RemoveAllChildren()
        {
            LOG_FUNC();

            for (auto &imap: m_pChildren)
            {
                LOG_INFO("Removing Child '" << imap.second->m_name <<"' from Parent '" << m_name <<"'");
                imap.second->m_pParentGstObj = NULL;
            }
            m_pChildren.clear();
        }
        
        /**
         * @brief function to determine if a Nodetr is a child of this Nodetr
         * @param pChild Nodetr to test for the child relationship
         * @return true if pChild is a child of this Nodetr
         */
        bool IsChild(DSL_NODETR_PTR pChild)
        {
            LOG_FUNC();
            
            return bool(m_pChildren[pChild->m_name]);
        }
        
        /**
         * @brief determines whether this Nodetr is a child of a given pParent
         * @param pParent the Nodetr to check for a Parental relationship
         * @return True if the provided Nodetr is this Nodetr's Parent
         */
        bool IsParent(DSL_NODETR_PTR pParent)
        {
            LOG_FUNC();
            
            return (m_pParentGstObj == pParent->m_pGstObj);
        }
        
        /**
         * @brief Links this Noder, becoming a source, to a sink Nodre
         * @param pSink Sink Nodre to link this Source Nodre to
         */
        virtual void LinkTo(DSL_NODETR_PTR pSink)
        {
            LOG_FUNC();

            m_pSink = pSink;
            pSink->m_pSource = shared_from_this();   
            LOG_INFO("Source '" << m_name <<"' linked to Sink '" << pSink->m_name << "'");
        }
        
        /**
         * @brief Unlinks this Source Nodetr from its Sink Nodetr
         */
        virtual void Unlink()
        {
            LOG_FUNC();

            if (m_pSink)
            {
                LOG_INFO("Unlinking Source '" << m_name <<"' from Sink '" << m_pSink->m_name <<"'");
                m_pSink->m_pSource = nullptr;
                m_pSink = nullptr;   
            }
        }
        
        /**
         * @brief returns the Source-to-Sink linked state for this Nodetr
         * @return true if this Nodetr is linked to a Sink Nodetr
         */
        bool IsLinked()
        {
            LOG_FUNC();
            
            return bool(m_pSink);
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
        
    public:

        /**
         * @brief unique name for this Nodetr
         */
        std::string m_name;

        /**
         * @brief Gst object wrapped by the Nodetr
         */
        GstObject * m_pGstObj;

        /**
         * @brief Parent of this Nodetr if one exists. NULL otherwise
         */
        GstObject * m_pParentGstObj;
        
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

} // DSL namespace    

#endif // _DSL_NODETR_H