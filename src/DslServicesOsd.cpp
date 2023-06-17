/*
The MIT License

Copyright (c)   2021, Prominence AI, Inc.

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
#include "DslOsdBintr.h"

namespace DSL
{
    DslReturnType Services::OsdNew(const char* name, 
        boolean textEnabled, boolean clockEnabled,
        boolean bboxEnabled, boolean maskEnabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {   
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("OSD name '" << name << "' is not unique");
                return DSL_RESULT_OSD_NAME_NOT_UNIQUE;
            }
            m_components[name] = std::shared_ptr<Bintr>(new OsdBintr(
                name, textEnabled, clockEnabled, bboxEnabled, maskEnabled));
                    
            LOG_INFO("New OSD '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New OSD '" << name << "' threw exception on create");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OsdTextEnabledGet(const char* name, boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            DSL_OSD_PTR pOsdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            pOsdBintr->GetTextEnabled(enabled);

            LOG_INFO("OSD '" << name << "' returned text displah enabled = "
                << *enabled << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name 
                << "' threw an exception getting test display enabled");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OsdTextEnabledSet(const char* name, boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);


            DSL_OSD_PTR pOsdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            // TODO verify args before calling
            if (!pOsdBintr->SetTextEnabled(enabled))
            {
                LOG_ERROR("OSD '" << name 
                    << "' failed to set text display enabled");
                return DSL_RESULT_OSD_SET_FAILED;
            }
            LOG_INFO("OSD '" << name << "' set text diplay enabled = " 
                << enabled << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name 
                << "' threw an exception setting text display enabled");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OsdClockEnabledGet(const char* name, boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            DSL_OSD_PTR pOsdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            pOsdBintr->GetClockEnabled(enabled);

            LOG_INFO("OSD '" << name << "' returned clock display enabeld = "
                << *enabled << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name 
                << "' threw an exception getting clock display enabled");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OsdClockEnabledSet(const char* name, boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            DSL_OSD_PTR pOsdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            // TODO verify args before calling
            if (!pOsdBintr->SetClockEnabled(enabled))
            {
                LOG_ERROR("OSD '" << name << "' failed to set clock display enabled");
                return DSL_RESULT_OSD_SET_FAILED;
            }
            LOG_INFO("OSD '" << name << "' set clock diplay enabled = "
                << enabled << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name 
                << "' threw an exception setting clock display enabled");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OsdClockOffsetsGet(const char* name, uint* offsetX, uint* offsetY)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            DSL_OSD_PTR pOsdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            pOsdBintr->GetClockOffsets(offsetX, offsetY);

            LOG_INFO("OSE '" << name << "' returned offeset X = " 
                << *offsetX << " and offeset Y = " << *offsetY << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception getting clock offsets");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OsdClockOffsetsSet(const char* name, uint offsetX, uint offsetY)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            DSL_OSD_PTR pOsdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            // TODO verify args before calling
            if (!pOsdBintr->SetClockOffsets(offsetX, offsetY))
            {
                LOG_ERROR("OSD '" << name << "' failed to set Clock offsets");
                return DSL_RESULT_OSD_SET_FAILED;
            }
            LOG_INFO("OSE '" << name << "' set offeset X = " 
                << offsetX << " and offeset Y = " << offsetY << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception setting Clock offsets");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OsdClockFontGet(const char* name, const char** font, uint* size)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            DSL_OSD_PTR pOsdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            pOsdBintr->GetClockFont(font, size);
            
            LOG_INFO("OSE '" << name << "' returned Clock Font = " 
                << *font << " and Size = " << *size << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception getting clock font");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OsdClockFontSet(const char* name, const char* font, uint size)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            DSL_OSD_PTR pOsdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            if (!pOsdBintr->SetClockFont(font, size))
            {
                LOG_ERROR("OSD '" << name << "' failed to set Clock font");
                return DSL_RESULT_OSD_SET_FAILED;
            }
            LOG_INFO("OSE '" << name << "' set Clock Font = " 
                << font << " and Size = " << size << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception setting Clock offsets");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OsdClockColorGet(const char* name, double* red, double* green, double* blue, double* alpha)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            DSL_OSD_PTR pOsdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            pOsdBintr->GetClockColor(red, green, blue, alpha);

            LOG_INFO("OSE '" << name << "' returned Color Red = " 
                << *red << ", Green = " << *green << ", Blue = " << *blue 
                << ", and Alpha = " << *alpha <<"' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception getting clock font");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OsdClockColorSet(const char* name, double red, double green, double blue, double alpha)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            DSL_OSD_PTR pOsdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            if (!pOsdBintr->SetClockColor(red, green, blue, alpha))
            {
                LOG_ERROR("OSD '" << name << "' failed to set Clock RGB colors");
                return DSL_RESULT_OSD_SET_FAILED;
            }
            LOG_INFO("OSE '" << name << "' set Color Red = " 
                << red << ", Green = " << green << ", Blue = " << blue 
                << ", and Alpha = " << alpha <<"' successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception setting Clock offsets");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OsdBboxEnabledGet(const char* name, boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            DSL_OSD_PTR pOsdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            pOsdBintr->GetBboxEnabled(enabled);

            LOG_INFO("OSD '" << name << "' returned bounding-box display enabled = "
                << *enabled << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name 
                << "' threw an exception getting bounding-box display enabled");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OsdBboxEnabledSet(const char* name, boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);


            DSL_OSD_PTR pOsdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            if (!pOsdBintr->SetBboxEnabled(enabled))
            {
                LOG_ERROR("OSD '" << name 
                    << "' failed to set bounding-box display enabled");
                return DSL_RESULT_OSD_SET_FAILED;
            }
            LOG_INFO("OSD '" << name << "' set bounding-box display enabled = " 
                << enabled << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name 
                << "' threw an exception setting bounding-box display enabled");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OsdMaskEnabledGet(const char* name, boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            DSL_OSD_PTR pOsdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            pOsdBintr->GetMaskEnabled(enabled);

            LOG_INFO("OSD '" << name << "' returned mask display enabled = "
                << *enabled << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name 
                << "' threw an exception getting mask display enabled");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OsdMaskEnabledSet(const char* name, boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);


            DSL_OSD_PTR pOsdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            // TODO verify args before calling
            if (!pOsdBintr->SetMaskEnabled(enabled))
            {
                LOG_ERROR("OSD '" << name << "' failed to set mask display enabled");
                return DSL_RESULT_OSD_SET_FAILED;
            }
            LOG_INFO("OSD '" << name << "' set mask display enabled = " 
                << enabled << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name 
                << "' threw an exception setting mask display enabled");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OsdProcessModeGet(const char* name, uint* mode)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, OsdBintr);

            DSL_OSD_PTR pOsdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            pOsdBintr->GetProcessMode(mode);

            LOG_INFO("OSD '" << name << "' returned process-mode = "
                << *mode << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name 
                << "' threw an exception getting process-mode");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OsdProcessModeSet(const char* name, uint mode)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            if (mode > DSL_OSD_PROCESS_MODE_GPU)
            {
                LOG_ERROR("Invalid process-mode = " << mode 
                    << " for OSD '" << name);
                return DSL_RESULT_OSD_SET_FAILED;
            }
            DSL_OSD_PTR pOsdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            if (!pOsdBintr->SetProcessMode(mode))
            {
                LOG_ERROR("OSD '" << name 
                    << "' failed to set process-mode = " << mode);
                return DSL_RESULT_OSD_SET_FAILED;
            }
            LOG_INFO("OSD '" << name << "' set process-mode = " 
                << mode << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name 
                << "' threw an exception setting process-mode");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }
   
    DslReturnType Services::OsdPphAdd(const char* name, const char* handler, uint pad)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            if (pad > DSL_PAD_SRC)
            {
                LOG_ERROR("Invalid Pad type = " << pad << " for OSD '" << name << "'");
                return DSL_RESULT_PPH_PAD_TYPE_INVALID;
            }

            // call on the Handler to add itself to the Osd as a PadProbeHandler
            if (!m_padProbeHandlers[handler]->AddToParent(m_components[name], pad))
            {
                LOG_ERROR("OSD '" << name << "' failed to add Pad Probe Handler");
                return DSL_RESULT_OSD_HANDLER_ADD_FAILED;
            }
            LOG_INFO("OSD '" << name << "' added Pad Probe Handler successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception adding Pad Probe Handler");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }
   
    DslReturnType Services::OsdPphRemove(const char* name, const char* handler, uint pad) 
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            if (pad > DSL_PAD_SRC)
            {
                LOG_ERROR("Invalid Pad type = " << pad << " for OSD '" << name << "'");
                return DSL_RESULT_PPH_PAD_TYPE_INVALID;
            }

            // call on the Handler to remove itself from the Osd
            if (!m_padProbeHandlers[handler]->RemoveFromParent(m_components[name], pad))
            {
                LOG_ERROR("Pad Probe Handler '" << handler << "' is not a child of OSD '" << name << "'");
                return DSL_RESULT_OSD_HANDLER_REMOVE_FAILED;
            }

            LOG_INFO("Tracker '" << name << "' removed Pad Probe Handler successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Osd '" << name << "' threw an exception removing ODE Handlre");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }

}