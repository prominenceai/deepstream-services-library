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

#ifndef _DSL_BINTR_H
#define _DSL_BINTR_H

#include "Dsl.h"

namespace DSL
{
    /**
     * @class Bintr
     * @brief Implements a base container class for a GST Bin
     */
    class Bintr : public std::enable_shared_from_this<Bintr>
    {
    public:

        /**
         * @brief basic container ctor without name and Bin initialization
         */
        Bintr()
            : m_gpuId(0)
            , m_nvbufMemoryType(0)
//            , m_pParentBintr(nullptr)
//            , m_pSourceBintr(nullptr)
//            , m_pSinkBintr(nullptr)
        { 
            LOG_FUNC(); 
        };

        /**
         * @brief named container ctor with new Bin 
         */
        Bintr(const char* name)
            : m_name(name)
            , m_gpuId(0)
            , m_nvbufMemoryType(0)
//            , m_pParentBintr(nullptr)
//            , m_pSourceBintr(nullptr)
//            , m_pSinkBintr(nullptr)
        { 
            LOG_FUNC(); 
            LOG_INFO("New bintr:: " << name);

            m_pBin = gst_bin_new((gchar*)name);
            if (!m_pBin)
            {
                LOG_ERROR("Failed to create new bin for component'" << name << "'");
                throw;  
            }
            
        };
        
        /**
         * @brief Bintr dtor to release all GST references
         */
        ~Bintr()
        {
            LOG_FUNC();
            LOG_INFO("Delete bintr:: " << m_name);

//            if (m_pBin and GST_OBJECT_REFCOUNT_VALUE(m_pBin))
//            {
//                gst_object_unref(m_pBin);
//            }
        };

        /**
         * @brief virtual function for derived classes to implement
         * a Bintr type specific function to link all child elements.
         */
        virtual void LinkAll()
        {
            LOG_FUNC();
        };
        
        /**
         * @brief virtual function for derived classes to implement
         * a Bintr type specific function to unlink all child elements.
         */
        virtual void UnlinkAll()
        {
            LOG_FUNC();
        };
        
        /**
         * @brief returns whether the Bintr object is in-use 
         * @return True if the Bintr has a parent 
         */
        bool IsInUse()
        {
            LOG_FUNC();
            
            return (
                (m_pParentBintr != nullptr) or
                (m_pChildBintrs.size() != 0) or
                (m_pSinkBintr != nullptr) or
                (m_pSourceBintr != nullptr));
        }

        
        /**
         * @brief returns the current number of child Bintrs in-use
         * @return number of children 
         */
        uint GetNumChildren()
        {
            LOG_FUNC();
            
            return m_pChildBintrs.size();
        }
        
        /**
         * @brief links this Bintr as source to a destination Bintr as sink
         * @param pSinkBintr to link to
         */
        void LinkTo(std::shared_ptr<Bintr> pSinkBintr)
        { 
            LOG_FUNC();
            
            m_pSinkBintr = pSinkBintr;

            pSinkBintr->m_pSourceBintr = 
                std::dynamic_pointer_cast<Bintr>(shared_from_this());
            
            if (!gst_element_link(m_pBin, m_pSinkBintr->m_pBin))
            {
                LOG_ERROR("Failed to link " << m_name << " to "
                    << pSinkBintr->m_name);
                throw;
            }
        };

        void Unlink()
        { 
            LOG_FUNC();
            
            gst_element_unlink(m_pBin, m_pSinkBintr->m_pBin);
            m_pSinkBintr->m_pSourceBintr = nullptr;
            m_pSinkBintr = nullptr;
        };

        /**
         * @brief adds a child Bintr to this parent Bintr
         * @param pChildBintr to add. Once added, calling InUse()
         *  on the Child Bintr will return true
         */
        virtual void AddChild(std::shared_ptr<Bintr> pChildBintr)
        {
            LOG_FUNC();
            
            pChildBintr->m_pParentBintr = shared_from_this();   

            m_pChildBintrs[pChildBintr->m_name] = pChildBintr;
                            
            if (!gst_bin_add(GST_BIN(m_pBin), pChildBintr->m_pBin))
            {
                LOG_ERROR("Failed to add " << pChildBintr->m_name << " to " << m_name <<"'");
                throw;
            }
            LOG_INFO("Child bin '" << pChildBintr->m_name <<"' added to '" << m_name <<"'");
        };
        
        /**
         * @brief removes a child Bintr from this parent Bintr
         * @param pChildBintr to remove. Once removed, calling InUse()
         *  on the Child Bintr will return false
         */
        virtual void RemoveChild(std::shared_ptr<Bintr> pChildBintr)
        {
            LOG_FUNC();
            
            if (m_pChildBintrs[pChildBintr->m_name] != pChildBintr)
            {
                LOG_ERROR("'" << pChildBintr->m_name << "' is not a child of '" << m_name <<"'");
                throw;
            }
                            
            pChildBintr->m_pParentBintr = nullptr;

            if (!gst_bin_remove(GST_BIN(m_pBin), pChildBintr->m_pBin))
            {
                LOG_ERROR("Failed to remove " << pChildBintr->m_name << " from " << m_name <<"'");
                throw;
            }
            m_pChildBintrs.erase(pChildBintr->m_name);
            
            LOG_INFO("Child bin '" << pChildBintr->m_name <<"' removed from '" << m_name <<"'");
        };

        /**
         * @brief Adds this Bintr as a child to a ParentBinter
         * @param pParentBintr to add to
         */
        virtual void AddToParent(std::shared_ptr<Bintr> pParentBintr)
        {
            LOG_FUNC();
                
            pParentBintr->AddChild(shared_from_this());
        }
        
        /**
         * @brief determines whether this Bintr is a child of pParentBintr
         * @param pParentBintr the Bintr to check for a Parental relationship
         * @return True if the provided Bintr is this Bintr's Parent
         */
        virtual bool IsMyParent(std::shared_ptr<Bintr> pParentBintr)
        {
            LOG_FUNC();
            
            return (m_pParentBintr == pParentBintr);
        }        
        
        /**
         * @brief removes this Bintr from the provided pParentBintr
         * @param pParentBintr Bintr to remove from
         */
        virtual void RemoveFromParent(std::shared_ptr<Bintr> pParentBintr)
        {
            LOG_FUNC();
                
            pParentBintr->RemoveChild(shared_from_this());
        }
        

    public:

        /**
         @brief unique name for this Bintr
         */
        std::string m_name;

        /**
         @brief pointer to the contained GST Bin for the Bintr
         */
        GstElement* m_pBin;
        
        /**
         @brief
         */
        guint m_gpuId;

        /**
         @brief
         */
        guint m_nvbufMemoryType;

        /**
         @brief Parent of this Bintr if one exists. NULL otherwise
         */
        std::shared_ptr<Bintr> m_pParentBintr;
        
        /**
         @brief map of Child Bintrs in-use by this Bintr
         */
        std::map<std::string, std::shared_ptr<Bintr>> m_pChildBintrs;
        
        /**
         @brief 
         */
        std::shared_ptr<Bintr> m_pSourceBintr;

        /**
         @brief
         */
        std::shared_ptr<Bintr> m_pSinkBintr;
        
    };

} // DSL

#endif // _DSL_BINTR_H