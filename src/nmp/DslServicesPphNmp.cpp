/*
The MIT License

Copyright (c)   2021-2022, Prominence AI, Inc.

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
#include "DslPadProbeHandlerNmp.h"

namespace DSL
{
    DslReturnType Services::PphNmpNew(const char* name, const char* labelFile, 
        uint processMethod, uint matchMethod, float matchThreshold)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure handler name uniqueness 
            if (m_padProbeHandlers.find(name) != m_padProbeHandlers.end())
            {   
                LOG_ERROR("NMP Pad Probe Handler name '" << name 
                    << "' is not unique");
                return DSL_RESULT_PPH_NAME_NOT_UNIQUE;
            }
            m_padProbeHandlers[name] = DSL_PPH_NMP_NEW(name, labelFile, 
                processMethod, matchMethod, matchThreshold);

            LOG_INFO("New NMP Pad Probe Handler '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New NMP Pad Probe handler '" << name 
                << "' threw exception on create");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PphNmpLabelFileGet(const char* name, 
        const char** labelFile)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_padProbeHandlers, name, 
                NmpPadProbeHandler);
            
            DSL_PPH_NMP_PTR pNmpPph = 
                std::dynamic_pointer_cast<NmpPadProbeHandler>(
                    m_padProbeHandlers[name]);

            *labelFile = pNmpPph->GetLabelFile();

            LOG_INFO("NMP Pad Probe handler '" << name 
                << "' returned label file = '"
                << *labelFile << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("NMP Pad Probe handler '" << name 
                << "' threw exception getting label file");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PphNmpLabelFileSet(const char* name, 
        const char* labelFile)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_padProbeHandlers, name, 
                NmpPadProbeHandler);
            
            DSL_PPH_NMP_PTR pNmpPph = 
                std::dynamic_pointer_cast<NmpPadProbeHandler>(
                    m_padProbeHandlers[name]);

            if (!pNmpPph->SetLabelFile(labelFile))
            {
                LOG_ERROR("NMP Pad Probe handler '" << name 
                    << "' failed to set the lable file");
                return DSL_RESULT_PPH_SET_FAILED;
            }
            LOG_INFO("NMP Pad Probe handler '" << name 
                << "' set label file = '"
                << labelFile << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("NMP Pad Probe handler '" << name 
                << "' threw exception setting label file");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PphNmpProcessMethodSet(const char* name, 
            uint processMethod)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_padProbeHandlers, name, 
                NmpPadProbeHandler);
            
            DSL_PPH_NMP_PTR pNmpPph = 
                std::dynamic_pointer_cast<NmpPadProbeHandler>(
                    m_padProbeHandlers[name]);

            if (processMethod > DSL_NMP_PROCESS_METHOD_MERGE)
            {
                LOG_ERROR("Invalid process method = " << processMethod <<
                    " for NMP Pad Probe handler '" << name << "'");
                return DSL_RESULT_PPH_SET_FAILED;
            }
            pNmpPph->SetProcessMethod(processMethod);

            LOG_INFO("NMP Pad Probe handler '" << name 
                << "' set process method = " << processMethod 
                << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("NMP Pad Probe handler '" << name 
                << "' threw exception setting process method");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PphNmpProcessMethodGet(const char* name, uint* processMethod)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_padProbeHandlers, name, 
                NmpPadProbeHandler);
            
            DSL_PPH_NMP_PTR pNmpPph = 
                std::dynamic_pointer_cast<NmpPadProbeHandler>(
                    m_padProbeHandlers[name]);

            *processMethod = pNmpPph->GetProcessMethod();

            LOG_INFO("NMP Pad Probe handler '" << name 
                << "' returned process method = " << *processMethod 
                << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("NMP Pad Probe handler '" << name 
                << "' threw exception getting process method");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PphNmpMatchSettingsGet(const char* name, 
            uint* matchMethod, float* matchThreshold)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_padProbeHandlers, name, 
                NmpPadProbeHandler);
            
            DSL_PPH_NMP_PTR pNmpPph = 
                std::dynamic_pointer_cast<NmpPadProbeHandler>(
                    m_padProbeHandlers[name]);

            pNmpPph->GetMatchSettings(matchMethod, matchThreshold);

            LOG_INFO("NMP Pad Probe handler '" << name 
                << "' returned match method = " << *matchMethod 
                << " and match threshold = " << *matchThreshold 
                << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("NMP Pad Probe handler '" << name 
                << "' threw exception getting match settings");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PphNmpMatchSettingsSet(const char* name, 
            uint matchMethod, float matchThreshold)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_padProbeHandlers, name, 
                NmpPadProbeHandler);
            
            DSL_PPH_NMP_PTR pNmpPph = 
                std::dynamic_pointer_cast<NmpPadProbeHandler>(
                    m_padProbeHandlers[name]);

            if (matchMethod > DSL_NMP_MATCH_METHOD_IOS)
            {
                LOG_ERROR("Invalid match method = " << matchMethod <<
                    " for NMP Pad Probe handler '" << name << "'");
                return DSL_RESULT_PPH_SET_FAILED;
            }
            if (matchThreshold > 1.0)
            {
                LOG_ERROR("Invalid match threshold = " << matchThreshold <<
                    " for NMP Pad Probe handler '" << name << "'");
                return DSL_RESULT_PPH_SET_FAILED;
            }
            pNmpPph->SetMatchSettings(matchMethod, matchThreshold);

            LOG_INFO("NMP Pad Probe handler '" << name 
                << "' set match method = " << matchMethod 
                << " and match threshold = " << matchThreshold 
                << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("NMP Pad Probe handler '" << name 
                << "' threw exception setting match settings");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }
}

