/*
The MIT License

Copyright (c) 2019-2021, Prominence AI, Inc.

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

#ifndef _DSL_ELEMENTR_H
#define _DSL_ELEMENTR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslNodetr.h"

namespace DSL
{

    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_ELEMENT_PTR std::shared_ptr<Elementr>
    #define DSL_ELEMENT_NEW(factoryName, name) \
        std::shared_ptr<Elementr>(new Elementr(factoryName, name))   

    #define DSL_ELEMENT_EXT_NEW(factoryName, name, suffix) \
        std::shared_ptr<Elementr>(new Elementr(factoryName, name, suffix))

    /**
     * @class Elementr
     * @brief Implements a container class for a GST Element
     */
    class Elementr : public GstNodetr
    {
    public:

        /**
         * @brief ctor for the container class
         * @brief[in] factoryname unique GST factory name to create from
         * @brief[in] name unique name for the Elementr
         */
        Elementr(const char* factoryName, const char* name)
            : GstNodetr(name)
            , m_factoryName(factoryName)
        { 
            LOG_FUNC(); 

            // Create a unique name by appending the plugin name
            AppendSuffix(factoryName);
            
            m_pGstObj = GST_OBJECT(gst_element_factory_make(factoryName, 
                GetCStrName()));
            if (!m_pGstObj)
            {
                LOG_ERROR("Failed to create new Element '" << name << "'");
                throw;  
            }
        };
        
        /**
         * @brief ctor for the container class
         * @brief[in] factoryname unique GST factory name to create from
         * @brief[in] name unique name for the Elementr
         */
        Elementr(const char* factoryName, const char* name, const char* suffix)
            : GstNodetr(name)
            , m_factoryName(factoryName)
        { 
            LOG_FUNC(); 

            // Create a unique name by appending the plugin name and additional suffix
            AppendSuffix(factoryName);
            AppendSuffix(suffix);
            
            m_pGstObj = GST_OBJECT(gst_element_factory_make(factoryName, 
                GetCStrName()));
            if (!m_pGstObj)
            {
                LOG_ERROR("Failed to create new Element '" << name << "'");
                throw;  
            }
        };
        
        /**
         * @brief ctor for the GST Element container class
         */
        ~Elementr()
        {
            LOG_FUNC();
        };

        /**
         * @brief Gets a GST Element's attribute of type int, 
         *  owned by this Elementr
         * @param[in] name name of the attribute to get
         * @param[out] value signed integer value to get the attribute
         */
        void GetAttribute(const char* name, int* value)
        {
            LOG_FUNC();
            
            g_object_get(GetGObject(), name, value, NULL);

            LOG_DEBUG("Attribute '" << name 
                << "' returned int '" << *value << "'");
        }

        /**
         * @brief Gets a GST Element's attribute of type int, 
         * owned by this Elementr
         * @param[in] name name of the attribute to get
         * @param[out] value unsigned integer value to get the attribute
         */
        void GetAttribute(const char* name, uint* value)
        {
            LOG_FUNC();
            
            g_object_get(GetGObject(), name, value, NULL);

            LOG_DEBUG("Attribute '" << name 
                << "' returned uint '" << *value << "'");
        }

        /**
         * @brief Gets a GST Element's attribute of type int, 
         * owned by this Elementr
         * @param[in] name name of the attribute to get
         * @param[out] value const char* value to set the attribute
         */
        void GetAttribute(const char* name, const char** value)
        {
            LOG_FUNC();
            
            g_object_get(GetGObject(), name, value, NULL);

            LOG_DEBUG("Attribute '" << name 
                << "' returned string '" << *value << "'");
        }

        /**
         * @brief Gets a GST Element's attribute of type uint64_t, 
         * owned by this Elementr
         * @param[in] name name of the attribute to get
         * @param[out] value uint64_t value to get the attribute
         */
        void GetAttribute(const char* name, uint64_t* value)
        {
            LOG_FUNC();
            
            g_object_get(GetGObject(), name, value, NULL);

            LOG_DEBUG("Attribute '" << name 
                << "' returned string '" << *value << "'");
        }

        /**
         * @brief Sets a GST Element's attribute, owned by this Elementr to a 
         * value of int
         * @param[in] name name of the attribute to set
         * @param[in] value unsigned integer value to set the attribute
         */
        void SetAttribute(const char* name, int value)
        {
            LOG_FUNC();
            
            LOG_DEBUG("Setting attribute '" << name 
                << "' to uint value '" << value << "'");
            
            g_object_set(GetGObject(), name, value, NULL);
        }

        /**
         * @brief Sets a GST Element's attribute, owned by this Elementr to a 
         * value of uint
         * @param[in] name name of the attribute to set
         * @param[in] value unsigned integer value to set the attribute
         */
        void SetAttribute(const char* name, uint value)
        {
            LOG_FUNC();
            
            LOG_DEBUG("Setting attribute '" << name 
                << "' to uint value '" << value << "'");
            
            g_object_set(GetGObject(), name, value, NULL);
        }
        
        /**
         * @brief Sets a GST Element's attribute, owned by this Elementr to a 
         * null terminated array of characters (char*)
         * @param[in] name name of the attribute to set
         * @param[in] value char* string value to set the attribute
         */
        void SetAttribute(const char* name, const char* value)
        {
            LOG_FUNC();
            
            LOG_DEBUG("Setting attribute '" << name 
                << "' to char* value '" << value << "'");
            
            g_object_set(GetGObject(), name, value, NULL);
        }
        
        /**
         * @brief Sets a GST Element's attribute, owned by this Elementr to a 
         * value of uint64_t
         * @param[in] name name of the attribute to set
         * @param[in] value unsigned integer value to set the attribute
         */
        void SetAttribute(const char* name, uint64_t value)
        {
            LOG_FUNC();
            
            LOG_DEBUG("Setting attribute '" << name 
                << "' to uint value '" << value << "'");
            
            g_object_set(GetGObject(), name, value, NULL);
        }

        /**
         * @brief Sets a GST Element's attribute, owned by this Elementr to a 
         * value of type GstCaps, created with one of gst_caps_new_* 
         * @param[in] name name of the attribute to set
         * @param[in] value char* string value to set the attribute
         */
        void SetAttribute(const char* name, const GstCaps * value)
        {
            LOG_FUNC();
            
            LOG_DEBUG("Setting attribute '" << name 
                << "' to GstCaps* value '" << value << "'");
            
            g_object_set(GetGObject(), name, value, NULL);
        }
        
        bool IsFactoryName(const char* factoryName)
        {
            LOG_FUNC();
            
            LOG_INFO("commparing expected factory'" << factoryName << "' with actual '" 
                << m_factoryName.c_str() << "' for element '" << GetName() << "'");
            
            std::string expectedName(factoryName);
            return (expectedName == m_factoryName);
        }
        
    private:
    
        std::string m_factoryName;
    };
}

#endif // _DSL_ELEMENTR_H   