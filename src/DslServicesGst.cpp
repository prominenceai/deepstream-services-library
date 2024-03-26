
/*
The MIT License

Copyright (c)   2024, Prominence AI, Inc.

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

#include "Dsl.h"
#include "DslApi.h"
#include "DslServices.h"
#include "DslServicesValidate.h"
#include "DslInferBintr.h"
#include "DslCaps.h"

namespace DSL
{
    DslReturnType Services::GstElementNew(const char* name, 
        const char* factoryName)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        if (m_gstElements[name])
        {   
            LOG_ERROR("Element name '" << name << "' is not unique");
            return DSL_RESULT_GST_ELEMENT_NAME_NOT_UNIQUE;
        }
        try
        {
            m_gstElements[name] = DSL_ELEMENT_NEW(factoryName, name);
        }
        catch(...)
        {
            LOG_ERROR("New GST Element '" << name << "' threw exception on create");
            return DSL_RESULT_GST_ELEMENT_THREW_EXCEPTION;
        }
        LOG_INFO("New GST Element '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::GstElementGet(const char* name, void** element)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        try
        {
            DSL_RETURN_IF_ELEMENT_NAME_NOT_FOUND(m_gstElements, name);

            *element = m_gstElements[name]->GetGstElement();

            LOG_INFO("Element '" << name 
                << "' returned element pointer = '" << *element << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Element '" << name 
                << "' threw an exception getting element pointer");
            return DSL_RESULT_GST_ELEMENT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::GstElementPropertyBooleanGet(const char* name, 
        const char* property, boolean* value)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        try
        {
            DSL_RETURN_IF_ELEMENT_NAME_NOT_FOUND(m_gstElements, name);

            m_gstElements[name]->GetAttribute(property, value);

            LOG_INFO("Element '" << name 
                << "' returned boolean value = '" << *value << "' for property '"
                << property << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Element '" << name 
                << "' threw an exception getting boolean property");
            return DSL_RESULT_GST_ELEMENT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::GstElementPropertyBooleanSet(const char* name, 
        const char* property, boolean value)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        try
        {
            DSL_RETURN_IF_ELEMENT_NAME_NOT_FOUND(m_gstElements, name);

            m_gstElements[name]->SetAttribute(property, value);

            LOG_INFO("Element '" << name 
                << "' set boolean value = '" << value << "' for property '"
                << property << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Element '" << name 
                << "' threw an exception setting boolean property");
            return DSL_RESULT_GST_ELEMENT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::GstElementPropertyFloatGet(const char* name, 
        const char* property, float* value)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        try
        {
            DSL_RETURN_IF_ELEMENT_NAME_NOT_FOUND(m_gstElements, name);

            m_gstElements[name]->GetAttribute(property, value);

            LOG_INFO("Element '" << name 
                << "' returned float value = '" << *value << "' for property '"
                << property << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Element '" << name 
                << "' threw an exception getting float property");
            return DSL_RESULT_GST_ELEMENT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::GstElementPropertyFloatSet(const char* name, 
        const char* property, float value)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        try
        {
            DSL_RETURN_IF_ELEMENT_NAME_NOT_FOUND(m_gstElements, name);

            m_gstElements[name]->SetAttribute(property, value);

            LOG_INFO("Element '" << name 
                << "' set float value = '" << value << "' for property '"
                << property << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Element '" << name 
                << "' threw an exception setting float property");
            return DSL_RESULT_GST_ELEMENT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::GstElementPropertyUintGet(const char* name, 
        const char* property, uint* value)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        try
        {
            DSL_RETURN_IF_ELEMENT_NAME_NOT_FOUND(m_gstElements, name);

            m_gstElements[name]->GetAttribute(property, value);

            LOG_INFO("Element '" << name 
                << "' returned uint value = '" << *value << "' for property '"
                << property << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Element '" << name 
                << "' threw an exception getting uint property");
            return DSL_RESULT_GST_ELEMENT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::GstElementPropertyUintSet(const char* name, 
        const char* property, uint value)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        try
        {
            DSL_RETURN_IF_ELEMENT_NAME_NOT_FOUND(m_gstElements, name);

            m_gstElements[name]->SetAttribute(property, value);

            LOG_INFO("Element '" << name 
                << "' set uint value = '" << value << "' for property '"
                << property << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Element '" << name 
                << "' threw an exception setting uint property");
            return DSL_RESULT_GST_ELEMENT_THREW_EXCEPTION;
        }
    }
    
   DslReturnType Services::GstElementPropertyIntGet(const char* name, 
        const char* property, int* value)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        try
        { 
            DSL_RETURN_IF_ELEMENT_NAME_NOT_FOUND(m_gstElements, name);

            m_gstElements[name]->GetAttribute(property, value);

            LOG_INFO("Element '" << name 
                << "' returned int value = '" << *value << "' for property '"
                << property << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Element '" << name 
                << "' threw an exception getting int property");
            return DSL_RESULT_GST_ELEMENT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::GstElementPropertyIntSet(const char* name, 
        const char* property, int value)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        try
        {
            DSL_RETURN_IF_ELEMENT_NAME_NOT_FOUND(m_gstElements, name);

            m_gstElements[name]->SetAttribute(property, value);

            LOG_INFO("Element '" << name 
                << "' set int value = '" << value << "' for property '"
                << property << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Element '" << name 
                << "' threw an exception setting int property");
            return DSL_RESULT_GST_ELEMENT_THREW_EXCEPTION;
        }
    }
 
   DslReturnType Services::GstElementPropertyUint64Get(const char* name, 
        const char* property, uint64_t* value)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        try
        { 
            DSL_RETURN_IF_ELEMENT_NAME_NOT_FOUND(m_gstElements, name);

            m_gstElements[name]->GetAttribute(property, value);

            LOG_INFO("Element '" << name 
                << "' returned uint64 value = '" << *value << "' for property '"
                << property << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Element '" << name 
                << "' threw an exception getting uint64 property");
            return DSL_RESULT_GST_ELEMENT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::GstElementPropertyUint64Set(const char* name, 
        const char* property, uint64_t value)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        try
        {
            DSL_RETURN_IF_ELEMENT_NAME_NOT_FOUND(m_gstElements, name);

            m_gstElements[name]->SetAttribute(property, value);

            LOG_INFO("Element '" << name 
                << "' set uint64 value = '" << value << "' for property '"
                << property << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Element '" << name 
                << "' threw an exception setting uint64 property");
            return DSL_RESULT_GST_ELEMENT_THREW_EXCEPTION;
        }
    }
 
   DslReturnType Services::GstElementPropertyInt64Get(const char* name, 
        const char* property, int64_t* value)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        try
        { 
            DSL_RETURN_IF_ELEMENT_NAME_NOT_FOUND(m_gstElements, name);

            m_gstElements[name]->GetAttribute(property, value);

            LOG_INFO("Element '" << name 
                << "' returned int64 value = '" << *value << "' for property '"
                << property << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Element '" << name 
                << "' threw an exception getting int64 property");
            return DSL_RESULT_GST_ELEMENT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::GstElementPropertyInt64Set(const char* name, 
        const char* property, int64_t value)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        try
        {
            DSL_RETURN_IF_ELEMENT_NAME_NOT_FOUND(m_gstElements, name);

            m_gstElements[name]->SetAttribute(property, value);

            LOG_INFO("Element '" << name 
                << "' set int64 value = '" << value << "' for property '"
                << property << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Element '" << name 
                << "' threw an exception setting int64 property");
            return DSL_RESULT_GST_ELEMENT_THREW_EXCEPTION;
        }
    }
 
    DslReturnType Services::GstElementPropertyStringGet(const char* name, 
        const char* property, const char** value)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        try
        {
            DSL_RETURN_IF_ELEMENT_NAME_NOT_FOUND(m_gstElements, name);

            m_gstElements[name]->GetAttribute(property, value);

            LOG_INFO("Element '" << name 
                << "' returned string value = '" << *value << "' for property '"
                << property << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Element '" << name 
                << "' threw an exception getting string property");
            return DSL_RESULT_GST_ELEMENT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::GstElementPropertyStringSet(const char* name, 
        const char* property, const char* value)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        try
        {
            DSL_RETURN_IF_ELEMENT_NAME_NOT_FOUND(m_gstElements, name);

            m_gstElements[name]->SetAttribute(property, value);

            LOG_INFO("Element '" << name 
                << "' set string value = '" << value << "' for property '"
                << property << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Element '" << name 
                << "' threw an exception setting string property");
            return DSL_RESULT_GST_ELEMENT_THREW_EXCEPTION;
        }
    }
    
}
