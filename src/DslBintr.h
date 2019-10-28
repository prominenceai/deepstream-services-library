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
#include "DslPadtr.h"

//#define INIT_MEMORY(m) memset(&m, 0, sizeof(m));
#define LINK_TRUE true
#define LINK_FALSE false

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
            , m_pSinkPad(NULL)
            , m_pSourcePad(NULL)
            , m_pParentBintr(nullptr)
            , m_pSourceBintr(nullptr)
            , m_pDestBintr(nullptr)
        { 
            LOG_FUNC(); 
        };

        /**
         * @brief named container ctor with new Bin 
         */
        Bintr(const char* name)
            : m_gpuId(0)
            , m_nvbufMemoryType(0)
            , m_pSinkPad(NULL)
            , m_pSourcePad(NULL)
            , m_pParentBintr(nullptr)
            , m_pSourceBintr(nullptr)
            , m_pDestBintr(nullptr)
        { 
            LOG_FUNC(); 
            LOG_INFO("New bintr:: " << name);
            
            m_name = name;

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

            // Clean up all resources
            if (m_pSinkPad)
            {
                gst_object_unref(m_pSinkPad);
            }

            if (m_pSourcePad)
            {
                gst_object_unref(m_pSourcePad);
            }

            if (GST_OBJECT_REFCOUNT_VALUE(m_pBin))
            {
                gst_object_unref(m_pBin);
            }
        };
        
        /**
         * @brief returns whether the Bintr object is in-use 
         * @return True if the Bintr has a parent 
         */
        bool IsInUse()
        {
            LOG_FUNC();
            
            return (m_pParentBintr != nullptr);
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
         * @param pDestBintr to link to
         */
        void LinkTo(std::shared_ptr<Bintr> pDestBintr)
        { 
            LOG_FUNC();
            
            m_pDestBintr = pDestBintr;

            pDestBintr->m_pSourceBintr = 
                std::dynamic_pointer_cast<Bintr>(shared_from_this());
            
            if (!gst_element_link(m_pBin, pDestBintr->m_pBin))
            {
                LOG_ERROR("Failed to link " << m_name << " to "
                    << pDestBintr->m_name);
                throw;
            }
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

        /**
         * @brief Creates a new GST Element and adds it to This Bintr's 
         * ordered list of child elements
         * @param factoryname defines the type of element to create
         * @param name name to give the new GST element
         * @param linkToPrev if true, this Element is linked to the 
         * previously created Element that was linked
         * @return a handle to the new GST Element
         */
        GstElement* MakeElement(const gchar * factoryname, const gchar * name, bool linkToPrev)
        {
            LOG_FUNC();

            GstElement* pElement = gst_element_factory_make(factoryname, name);
            if (!pElement)
            {
                LOG_ERROR("Failed to create new Element '" << name << "'");
                throw;  
            }

            if (!gst_bin_add(GST_BIN(m_pBin), pElement))
            {
                LOG_ERROR("Failed to add " << name << " to " << m_name);
                throw;
            }
            
            if (linkToPrev)
            {
                // If not the first element
                if (m_pLinkedChildElements.size())
                {
                    // link the previous to the new element 
                    if (!gst_element_link(m_pLinkedChildElements.back(), pElement))
                    {
                        LOG_ERROR("Failed to link new element " << name << " for " << m_name);
                        throw;
                    }
                    LOG_DEBUG("Successfully linked new element " << name << " for " << m_name);
                }
                m_pLinkedChildElements.push_back(pElement);
            }
            return pElement;
        };
        
        /**
         * @brief Creates a new Ghost Sink pad for the first Gst Element
         * added to this Bintr's Gst Bin.
         * @throws a general exception on failure
         */
        virtual void AddSinkGhostPad()
        {
            LOG_FUNC();
            
            if (!m_pLinkedChildElements.size())
            {
                LOG_ERROR("Failed to add Sink Pad for '" << m_name <<"' No Elements");
                throw;
            }

            // get Sink pad for first child element in the ordered list
            StaticPadtr SinkPadtr(m_pLinkedChildElements.front(), "sink");

            // create a new ghost pad with the Sink pad and add to this bintr's bin
            if (!gst_element_add_pad(m_pBin, gst_ghost_pad_new("sink", SinkPadtr.m_pPad)))
            {
                LOG_ERROR("Failed to add Sink Pad for '" << m_name);
                throw;
            }
        };
        
        /**
         * @brief Creates a new Ghost Source pad for the last Gst element
         * added to this Bintr's Gst Bin.
         * @throws a general exception on failure
         */
        virtual void AddSourceGhostPad()
        {
            LOG_FUNC();
            
            if (!m_pLinkedChildElements.size())
            {
                LOG_ERROR("Failed to add Source Pad for '" << m_name <<"' No Elements");
                throw;
            }
            
            // get Source pad for last child element in the ordered list
            StaticPadtr SourcePadtr(m_pLinkedChildElements.back(), "src");

            // create a new ghost pad with the Source pad and add to this bintr's bin
            if (!gst_element_add_pad(m_pBin, gst_ghost_pad_new("src", SourcePadtr.m_pPad)))
            {
                LOG_ERROR("Failed to add Source Pad for '" << m_name);
                throw;
            }
        };

        /**
         * @brief bundles both AddSink and AddSource into a single call for convenience
         */
        void AddGhostPads()
        {
            AddSinkGhostPad();
            AddSourceGhostPad();
        };
            
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
         @brief vector of created and linked child elements
         */
        std::vector<GstElement*> m_pLinkedChildElements;
        
        /**
         @brief
         */
        guint m_gpuId;

        /**
         @brief
         */
        guint m_nvbufMemoryType;

        /**
         @brief
         */
        GstPad *m_pSinkPad;
        
        /**
         @brief
         */
        GstPad *m_pSourcePad; 
        
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
        std::shared_ptr<Bintr> m_pDestBintr;
        
    };

} // DSL

#endif // _DSL_BINTR_H