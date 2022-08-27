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
#include "DslOdeAction.h"

namespace DSL
{
    DslReturnType Services::OdeActionCaptureFrameNew(const char* name,
        const char* outdir, boolean annotate)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }

            // ensure outdir exists
            struct stat info;
            if ((stat(outdir, &info) != 0) or !(info.st_mode & S_IFDIR))
            {
                LOG_ERROR("Unable to access outdir '" << outdir 
                    << "' for Capture Action '" << name << "'");
                return DSL_RESULT_ODE_ACTION_FILE_PATH_NOT_FOUND;
            }
            m_odeActions[name] = DSL_ODE_ACTION_CAPTURE_FRAME_NEW(name, 
                outdir, annotate);

            LOG_INFO("New Capture Frame ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Capture Frame ODE Action '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeActionCaptureObjectNew(const char* name,
        const char* outdir)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            
            // ensure outdir exists
            struct stat info;
            if ((stat(outdir, &info) != 0) or !(info.st_mode & S_IFDIR))
            {
                LOG_ERROR("Unable to access outdir '" << outdir 
                    << "' for Capture Action '" << name << "'");
                return DSL_RESULT_ODE_ACTION_FILE_PATH_NOT_FOUND;
            }
            m_odeActions[name] = DSL_ODE_ACTION_CAPTURE_OBJECT_NEW(name, outdir);

            LOG_INFO("New Capture Object ODE Action '" 
                << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Capture Object ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionCaptureCompleteListenerAdd(const char* name, 
        dsl_capture_complete_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            DSL_RETURN_IF_ODE_ACTION_IS_NOT_CAPTURE_TYPE(m_odeActions, name);   

            DSL_ODE_ACTION_CATPURE_PTR pOdeAction = 
                std::dynamic_pointer_cast<CaptureOdeAction>(m_odeActions[name]);

            if (!pOdeAction->AddCaptureCompleteListener(listener, clientData))
            {
                LOG_ERROR("ODE Capture Action '" << name 
                    << "' failed to add a Capture Complete Listener");
                return DSL_RESULT_ODE_ACTION_CALLBACK_ADD_FAILED;
            }
            LOG_INFO("ODE Capture Action '" << name << "' added Listener successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Capture Action '" << name 
                << "' threw an exception adding a Capture Complete Lister");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::OdeActionCaptureCompleteListenerRemove(const char* name, 
        dsl_capture_complete_listener_cb listener)
    {
        LOG_FUNC();
    
        try
        {
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            DSL_RETURN_IF_ODE_ACTION_IS_NOT_CAPTURE_TYPE(m_odeActions, name);   

            DSL_ODE_ACTION_CATPURE_PTR pOdeAction = 
                std::dynamic_pointer_cast<CaptureOdeAction>(m_odeActions[name]);

            if (!pOdeAction->RemoveCaptureCompleteListener(listener))
            {
                LOG_ERROR("Capture Action '" << name 
                    << "' failed to add a Capture Complete Listener");
                return DSL_RESULT_ODE_ACTION_CALLBACK_REMOVE_FAILED;
            }
            LOG_INFO("ODE Capture Action '" << name << "' added Listener successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Capture Action '" << name 
                << "' threw an exception adding a Capture Complete Lister");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeActionCaptureImagePlayerAdd(const char* name, 
        const char* player)
    {
        LOG_FUNC();
    
        try
        {
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            DSL_RETURN_IF_ODE_ACTION_IS_NOT_CAPTURE_TYPE(m_odeActions, name);
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, player);
            DSL_RETURN_IF_PLAYER_IS_NOT_IMAGE_PLAYER(m_players, player)

            DSL_ODE_ACTION_CATPURE_PTR pOdeAction = 
                std::dynamic_pointer_cast<CaptureOdeAction>(m_odeActions[name]);

            if (!pOdeAction->AddImagePlayer(m_players[player]))
            {
                LOG_ERROR("Capture Action '" << name 
                    << "' failed to add Player '" << player << "'");
                return DSL_RESULT_ODE_ACTION_PLAYER_ADD_FAILED;
            }
            LOG_INFO("ODE Capture Action '" << name << "' added Image Player '"
                << player << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Capture Action '" << name 
                << "' threw an exception adding Player '" << player << "'");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionCaptureImagePlayerRemove(const char* name, 
        const char* player)
    {
        LOG_FUNC();
    
        try
        {
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            DSL_RETURN_IF_ODE_ACTION_IS_NOT_CAPTURE_TYPE(m_odeActions, name);
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, player);
            DSL_RETURN_IF_PLAYER_IS_NOT_IMAGE_PLAYER(m_players, player)

            DSL_ODE_ACTION_CATPURE_PTR pOdeAction = 
                std::dynamic_pointer_cast<CaptureOdeAction>(m_odeActions[name]);

            if (!pOdeAction->RemoveImagePlayer(m_players[player]))
            {
                LOG_ERROR("Capture Action '" << name 
                    << "' failed to remove Player '" << player << "'");
                return DSL_RESULT_ODE_ACTION_PLAYER_REMOVE_FAILED;
            }
            LOG_INFO("ODE Capture Action '" << name << "' Removed Image Player '"
                << player << "' successfully");
        }
        catch(...)
        {
            LOG_ERROR("ODE Capture Action '" << name 
                << "' threw an exception removeing Player '" << player << "'");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::OdeActionCaptureMailerAdd(const char* name, 
        const char* mailer, const char* subject, boolean attach)
    {
        LOG_FUNC();
    
        try
        {
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            DSL_RETURN_IF_ODE_ACTION_IS_NOT_CAPTURE_TYPE(m_odeActions, name);
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, mailer);

            DSL_ODE_ACTION_CATPURE_PTR pOdeAction = 
                std::dynamic_pointer_cast<CaptureOdeAction>(m_odeActions[name]);

            if (!pOdeAction->AddMailer(m_mailers[mailer], subject, attach))
            {
                LOG_ERROR("Capture Action '" << name 
                    << "' failed to add Mailer '" << mailer << "'");
                return DSL_RESULT_ODE_ACTION_MAILER_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("ODE Capture Action '" << name 
                << "' threw an exception adding Mailer '" << mailer << "'");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::OdeActionCaptureMailerRemove(const char* name, 
        const char* mailer)
    {
        LOG_FUNC();
    
        try
        {
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            DSL_RETURN_IF_ODE_ACTION_IS_NOT_CAPTURE_TYPE(m_odeActions, name);
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, mailer);

            DSL_ODE_ACTION_CATPURE_PTR pOdeAction = 
                std::dynamic_pointer_cast<CaptureOdeAction>(m_odeActions[name]);

            if (!pOdeAction->RemoveMailer(m_mailers[mailer]))
            {
                LOG_ERROR("Capture Action '" << name 
                    << "' failed to remove Mailer '" << mailer << "'");
                return DSL_RESULT_ODE_ACTION_MAILER_REMOVE_FAILED;
            }
            LOG_INFO("ODE Capture Action '" << name << "' removed Mailer '"
                << mailer << "' successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Capture Action '" << name 
                << "' threw an exception removeing Player '" << mailer << "'");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionCustomNew(const char* name,
        dsl_ode_handle_occurrence_cb clientHandler, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_CUSTOM_NEW(
                name, clientHandler, clientData);

            LOG_INFO("New ODE Custom Action '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Custom Action '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionBBoxScaleNew(const char* name, uint scale)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            if (scale <= 100)
            {
                LOG_ERROR("Invalid scale factor = " << scale << " for ODE Action '"
                    << name << "', must be greater than 100%");
                return DSL_RESULT_ODE_ACTION_PARAMETER_INVALID;
            }
            m_odeActions[name] = DSL_ODE_ACTION_BBOX_SCALE_NEW(
                name, scale);

            LOG_INFO("New ODE Scale BBox Action '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Scale BBox Action '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeActionLabelCustomizeNew(const char* name, 
        const uint* contentTypes, uint size)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            
            if (size > DSL_METRIC_OBJECT_PERSISTENCE+1)
            {
                LOG_ERROR("Invalid array size for new Customize Label ODE Action '" 
                    << name << "'");
                return DSL_RESULT_ODE_ACTION_PARAMETER_INVALID;
            }
            
            // convert array of Content Type constants into a vector
            std::vector <uint> contentTypesCopy;
            
            uint count(size);
            for (const uint* contentType = contentTypes; count; contentType++)
            {
                if (*contentType > DSL_METRIC_OBJECT_PERSISTENCE)
                {
                    LOG_ERROR("Invalid Content Type = " << *contentType 
                        << " for new Customize Label ODE Action" << name << "'");
                    return DSL_RESULT_ODE_ACTION_PARAMETER_INVALID;
                }
                contentTypesCopy.push_back(*contentType);
                count--;
            }    
            m_odeActions[name] = DSL_ODE_ACTION_LABEL_CUSTOMIZE_NEW(
                name, contentTypesCopy);

            LOG_INFO("New ODE Customize Label Action '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Customize Label Action '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
        
    }

    DslReturnType  Services::OdeActionLabelCustomizeGet(const char* name, 
        uint* contentTypes, uint* size)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_odeActions, 
                name, CustomizeLabelOdeAction);
            
            DSL_ODE_ACTION_LABEL_CUSTOMIZE_PTR pOdeAction = 
                std::dynamic_pointer_cast<CustomizeLabelOdeAction>(m_odeActions[name]);
                
            std::vector <uint> contentTypesCopy = pOdeAction->Get();
            
            if (*size < contentTypesCopy.size())
            {
                LOG_ERROR("Insufficient 'content_types' array size = " << *size 
                    << " to Get the ODE Customize Label Action '" << name << "' settings");
                return DSL_RESULT_ODE_ACTION_PARAMETER_INVALID;
            }
            std::copy(contentTypesCopy.begin(), contentTypesCopy.end(), contentTypes);
            *size = contentTypesCopy.size();

            LOG_INFO("ODE Customize Label Action '" << name 
                << "' updated successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Customize Label Action '" << name 
                << "' threw exception on set");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType  Services::OdeActionLabelCustomizeSet(const char* name, 
        const uint* contentTypes, uint size)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_odeActions, 
                name, CustomizeLabelOdeAction);
            
            if (size > DSL_METRIC_OBJECT_PERSISTENCE+1)
            {
                LOG_ERROR("Invalid array size for Customize Label ODE Action '" 
                    << name << "'");
                return DSL_RESULT_ODE_ACTION_PARAMETER_INVALID;
            }
            
            // convert array of Content Type constants into a vector
            std::vector <uint> contentTypesCopy;
            
            uint count(size);
            for (const uint* contentType = contentTypes; count; contentType++)
            {
                if (*contentType > DSL_METRIC_OBJECT_PERSISTENCE)
                {
                    LOG_ERROR("Invalid Content Type = " << *contentType 
                        << " for new Customize Label ODE Action" << name << "'");
                    return DSL_RESULT_ODE_ACTION_PARAMETER_INVALID;
                }
                contentTypesCopy.push_back(*contentType);
                count--;
            }
            DSL_ODE_ACTION_LABEL_CUSTOMIZE_PTR pOdeAction = 
                std::dynamic_pointer_cast<CustomizeLabelOdeAction>(m_odeActions[name]);
                
            pOdeAction->Set(contentTypesCopy);

            LOG_INFO("ODE Customize Label Action '" << name 
                << "' updated successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Customize Label Action '" << name 
                << "' threw exception on set");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionLabelOffsetNew(const char* name, 
        int offsetX, int offsetY)
{
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            
            m_odeActions[name] = DSL_ODE_ACTION_LABEL_OFFSET_NEW(name,
                offsetX, offsetY);

            LOG_INFO("New ODE Offset Label Action '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Offset Label Action '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
        
    }
    
    DslReturnType Services::OdeActionDisplayNew(const char* name, 
        const char* formatString, uint offsetX, uint offsetY, 
        const char* font, boolean hasBgColor, const char* bgColor)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, font);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(
                m_displayTypes, font, RgbaFont);
            
            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(
                    m_displayTypes, bgColor, RgbaColor);

                pBgColor = std::dynamic_pointer_cast<RgbaColor>(
                    m_displayTypes[bgColor]);
            }
            else
            {
                pBgColor = std::dynamic_pointer_cast<RgbaColor>
                    (m_intrinsicDisplayTypes[DISPLAY_TYPE_NO_COLOR.c_str()]);
            }

            DSL_RGBA_FONT_PTR pFont = 
                std::dynamic_pointer_cast<RgbaFont>(m_displayTypes[font]);

            m_odeActions[name] = DSL_ODE_ACTION_DISPLAY_NEW(name, 
                formatString, offsetX, offsetY, pFont, hasBgColor, pBgColor);
                
            LOG_INFO("New Display ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Display ODE Action '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionEmailNew(const char* name, 
        const char* mailer, const char* subject)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, mailer)
            
            m_odeActions[name] = DSL_ODE_ACTION_EMAIL_NEW(name, 
                m_mailers[mailer], subject);

            LOG_INFO("New ODE Email Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Email Action '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionFileNew(const char* name, 
        const char* filePath, uint mode, uint format, boolean forceFlush)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            if (mode > DSL_WRITE_MODE_TRUNCATE)
            {
                LOG_ERROR("File open mode " << mode 
                    << " is invalid for ODE Action '" << name << "'");
                return DSL_RESULT_ODE_ACTION_PARAMETER_INVALID;
            }
            switch (format)
            {
            case DSL_EVENT_FILE_FORMAT_TEXT :
                m_odeActions[name] = DSL_ODE_ACTION_FILE_TEXT_NEW(name, 
                    filePath, mode, forceFlush);
                break;
            case DSL_EVENT_FILE_FORMAT_CSV :
                m_odeActions[name] = DSL_ODE_ACTION_FILE_CSV_NEW(name, 
                    filePath, mode, forceFlush);
                break;
            case DSL_EVENT_FILE_FORMAT_MOTC :
                m_odeActions[name] = DSL_ODE_ACTION_FILE_MOTC_NEW(name, 
                    filePath, mode, forceFlush);
                break;
            default :
                LOG_ERROR("File format " << format 
                    << " is invalid for ODE Action '" << name << "'");
                return DSL_RESULT_ODE_ACTION_PARAMETER_INVALID;
            }

            LOG_INFO("New ODE File Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE File Action '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeActionFillSurroundingsNew(const char* 
        name, const char* color)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, 
                color, RgbaColor);

            DSL_RGBA_COLOR_PTR pColor = 
                std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[color]);
                
            m_odeActions[name] = DSL_ODE_ACTION_FILL_SURROUNDINGS_NEW(name, pColor);

            LOG_INFO("New ODE Fill Surroundings Action '" 
                << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Fill Surroundings Action '" 
                << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionFillFrameNew(const char* name, const char* color)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, color, RgbaColor);

            DSL_RGBA_COLOR_PTR pColor = 
                std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[color]);
                
            m_odeActions[name] = DSL_ODE_ACTION_FILL_FRAME_NEW(name, pColor);

            LOG_INFO("New ODE Fill Frame Action '" 
                << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Fill Frame Action '" 
                << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }


    DslReturnType Services::OdeActionBBoxFormatNew(const char* name,
        uint borderWidth, const char* borderColor, boolean hasBgColor, const char* bgColor)  
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }

            DSL_RGBA_COLOR_PTR pBorderColor(nullptr);
            if (borderWidth)
            {
                DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, borderColor);
                DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_COLOR(m_displayTypes, borderColor);

                pBorderColor = std::dynamic_pointer_cast<RgbaColor>
                    (m_displayTypes[borderColor]);
            }
            else
            {
                pBorderColor = std::dynamic_pointer_cast<RgbaColor>
                    (m_intrinsicDisplayTypes[DISPLAY_TYPE_NO_COLOR.c_str()]);
            }

            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_COLOR(m_displayTypes, bgColor);

                pBgColor = std::dynamic_pointer_cast<RgbaColor>
                    (m_displayTypes[bgColor]);
            }
            else
            {
                pBgColor = std::dynamic_pointer_cast<RgbaColor>
                    (m_intrinsicDisplayTypes[DISPLAY_TYPE_NO_COLOR.c_str()]);
            }
            m_odeActions[name] = DSL_ODE_ACTION_BBOX_FORMAT_NEW(name, 
                borderWidth, pBorderColor, hasBgColor, pBgColor);
                
            LOG_INFO("New Format Bounding Box ODE Action '" 
                << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Format Bounding Box ODE Action '" 
                << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }
    

    DslReturnType Services::OdeActionLabelFormatNew(const char* name,
        const char* font, boolean hasBgColor, const char* bgColor)  
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }

            DSL_RGBA_FONT_PTR pFont(nullptr);
            std::string fontString(font);
            if (fontString.size())
            {
                DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, font);
                DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, 
                    font, RgbaFont);

                pFont = std::dynamic_pointer_cast<RgbaFont>(m_displayTypes[font]);
            }
            else
            {
                pFont = std::dynamic_pointer_cast<RgbaFont>(
                    m_intrinsicDisplayTypes[DISPLAY_TYPE_NO_FONT.c_str()]);
            }

            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_COLOR(m_displayTypes, bgColor);

                pBgColor = std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[bgColor]);
            }
            else
            {
                pBgColor = std::dynamic_pointer_cast<RgbaColor>
                    (m_intrinsicDisplayTypes[DISPLAY_TYPE_NO_COLOR.c_str()]);
            }
            m_odeActions[name] = DSL_ODE_ACTION_LABEL_FORMAT_NEW(name, 
                pFont, hasBgColor, pBgColor);
                
            LOG_INFO("New Format Label ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Format Bounding Box ODE Action '" 
                << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionHandlerDisableNew(const char* name, const char* handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_DISABLE_HANDLER_NEW(name, handler);

            LOG_INFO("New ODE Disable Handler Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Disable Handler Action '" 
                << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeActionLogNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_LOG_NEW(name);

            LOG_INFO("New ODE Log Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Log Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionMessageMetaAddNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_MESSAGE_META_ADD_NEW(name);

            LOG_INFO("New ODE Message Meta Add Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Message Meta Add Action '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionMessageMetaTypeGet(const char* name,
        uint* metaType) 
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            DSL_RETURN_IF_ODE_ACTION_IS_NOT_CORRECT_TYPE(m_odeActions, 
                name, MessageMetaAddOdeAction);

            DSL_ODE_ACTION_MESSAGE_META_ADD_PTR pAction = 
                std::dynamic_pointer_cast<MessageMetaAddOdeAction>(m_odeActions[name]);

            *metaType = pAction->GetMetaType();
            
            LOG_INFO("ODE Message Meta Add Action '" << name 
                << "' returned meta_type = " << *metaType << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Message Meta Add Action '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionMessageMetaTypeSet(const char* name,
        uint metaType)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            DSL_RETURN_IF_ODE_ACTION_IS_NOT_CORRECT_TYPE(m_odeActions, 
                name, MessageMetaAddOdeAction);

            DSL_ODE_ACTION_MESSAGE_META_ADD_PTR pAction = 
                std::dynamic_pointer_cast<MessageMetaAddOdeAction>(m_odeActions[name]);

            if (metaType < NVDS_START_USER_META and
                metaType != NVDS_EVENT_MSG_META)
            {
                LOG_ERROR("meta_type = " << metaType 
                    << "' is invalid for ODE Add Message Meta Action '" << name << "'");
                return DSL_RESULT_ODE_ACTION_SET_FAILED;
            }
            pAction->SetMetaType(metaType);
            
            LOG_INFO("ODE Message Meta Add Action '" << name << "' set meta_type = "
                << metaType << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Message Meta Add Action '" 
                << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionDisplayMetaAddNew(const char* name, 
        const char* displayType)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, displayType);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_BASE_TYPE(m_displayTypes, displayType);
            
            DSL_DISPLAY_TYPE_PTR pDisplayType = std::dynamic_pointer_cast<DisplayType>(m_displayTypes[displayType]);
            
            m_odeActions[name] = DSL_ODE_ACTION_DISPLAY_META_ADD_NEW(name, pDisplayType);

            LOG_INFO("New Add Display Meta Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Add Display Meta Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeActionDisplayMetaAddDisplayType(const char* name, const char* displayType)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            DSL_RETURN_IF_ODE_ACTION_IS_NOT_CORRECT_TYPE(m_odeActions, name, AddDisplayMetaOdeAction);
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, displayType);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_BASE_TYPE(m_displayTypes, displayType);
            
            DSL_DISPLAY_TYPE_PTR pDisplayType = std::dynamic_pointer_cast<DisplayType>(m_displayTypes[displayType]);
            
            DSL_ODE_ACTION_DISPLAY_META_ADD_PTR pAction = 
                std::dynamic_pointer_cast<AddDisplayMetaOdeAction>(m_odeActions[name]);

            pAction->AddDisplayType(pDisplayType);
            
            LOG_INFO("Display Type '" << displayType << "' added to Action '" << name << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Overlay Frame Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeActionMonitorNew(const char* name,
        dsl_ode_monitor_occurrence_cb clientMonitor, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_MONITOR_NEW(name, 
                clientMonitor, clientData);

            LOG_INFO("New ODE Monitor Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Monitor Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeActionPauseNew(const char* name, const char* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_PAUSE_NEW(name, pipeline);

            LOG_INFO("New ODE Pause Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Pause Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeActionPrintNew(const char* name,
        boolean forceFlush)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_PRINT_NEW(name, forceFlush);

            LOG_INFO("New ODE Print Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Print Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeActionRedactNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_REDACT_NEW(name);

            LOG_INFO("New ODE Redact Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Redact Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionSinkAddNew(const char* name, 
        const char* pipeline, const char* sink)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_SINK_ADD_NEW(name, pipeline, sink);

            LOG_INFO("New Sink Add ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Sink ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionSinkRemoveNew(const char* name, 
        const char* pipeline, const char* sink)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_SINK_REMOVE_NEW(name, pipeline, sink);

            LOG_INFO("New Sink Remove ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Sink Remove ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionSinkRecordStartNew(const char* name,
        const char* recordSink, uint start, uint duration, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }

            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, recordSink);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, recordSink, RecordSinkBintr);

            DSL_RECORD_SINK_PTR pRecordSinkBintr = 
                std::dynamic_pointer_cast<RecordSinkBintr>(m_components[recordSink]);
            
            m_odeActions[name] = DSL_ODE_ACTION_SINK_RECORD_START_NEW(name,
                pRecordSinkBintr, start, duration, clientData);

            LOG_INFO("New ODE Record Sink Start Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Record Start Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionSinkRecordStopNew(const char* name,
        const char* recordSink)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }

            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, recordSink);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, recordSink, RecordSinkBintr);

            DSL_RECORD_SINK_PTR pRecordSinkBintr = 
                std::dynamic_pointer_cast<RecordSinkBintr>(m_components[recordSink]);
            
            m_odeActions[name] = DSL_ODE_ACTION_SINK_RECORD_STOP_NEW(name,
                pRecordSinkBintr);

            LOG_INFO("New ODE Record Sink Stop Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Record Stop Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeActionSourceAddNew(const char* name, 
        const char* pipeline, const char* source)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_SOURCE_ADD_NEW(name, pipeline, source);

            LOG_INFO("New Source Add ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Source Add ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionSourceRemoveNew(const char* name, 
        const char* pipeline, const char* source)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_SOURCE_REMOVE_NEW(name, pipeline, source);

            LOG_INFO("New Source Remove ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Source Remove ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionTapRecordStartNew(const char* name,
        const char* recordTap, uint start, uint duration, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, recordTap);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, recordTap, RecordTapBintr);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[recordTap]);

            m_odeActions[name] = DSL_ODE_ACTION_TAP_RECORD_START_NEW(name,
                pRecordTapBintr, start, duration, clientData);

            LOG_INFO("New ODE Record Tap Start Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Record Tap Start Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionTapRecordStopNew(const char* name,
        const char* recordTap)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, recordTap);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, recordTap, RecordTapBintr);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[recordTap]);
            
            m_odeActions[name] = DSL_ODE_ACTION_TAP_RECORD_STOP_NEW(name, pRecordTapBintr);

            LOG_INFO("New ODE Record Tap Stop Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Record Tap Stop Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeActionActionDisableNew(const char* name, const char* action)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_ACTION_DISABLE_NEW(name, action);

            LOG_INFO("New Action Disable ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Action Disable ODE Action'" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionActionEnableNew(const char* name, const char* action)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_ACTION_ENABLE_NEW(name, action);

            LOG_INFO("New Action Enable ODE Action'" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Action Enable ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionTilerShowSourceNew(const char* name, 
        const char* tiler, uint timeout, bool hasPrecedence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_TILER_SHOW_SOURCE_NEW(name, tiler, timeout, hasPrecedence);

            LOG_INFO("New Tiler Show Source ODE Action'" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Tiler Show Source ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionTriggerResetNew(const char* name, const char* trigger)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_TRIGGER_RESET_NEW(name, trigger);

            LOG_INFO("New Trigger Reset ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Trigger Reset ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionTriggerDisableNew(const char* name, const char* trigger)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_TRIGGER_DISABLE_NEW(name, trigger);

            LOG_INFO("New Trigger Disable ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Trigger Disable ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionTriggerEnableNew(const char* name, const char* trigger)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_TRIGGER_ENABLE_NEW(name, trigger);

            LOG_INFO("New Trigger Enable ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Trigger Enable ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionAreaAddNew(const char* name, 
        const char* trigger, const char* area)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_AREA_ADD_NEW(name, trigger, area);

            LOG_INFO("New Area Add ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Area Add ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionAreaRemoveNew(const char* name, 
        const char* trigger, const char* area)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_AREA_REMOVE_NEW(name, trigger, area);

            LOG_INFO("New Area Remove ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Area Remove ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionEnabledGet(const char* name, boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            
            DSL_ODE_ACTION_PTR pOdeAction = 
                std::dynamic_pointer_cast<OdeAction>(m_odeActions[name]);
         
            *enabled = pOdeAction->GetEnabled();

            LOG_INFO("ODE Action '" << name << "' returned Enabed = " 
                << *enabled  << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Action '" << name << "' threw exception getting Enabled setting");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeActionEnabledSet(const char* name, boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            
            DSL_ODE_ACTION_PTR pOdeAction = 
                std::dynamic_pointer_cast<OdeAction>(m_odeActions[name]);
         
            pOdeAction->SetEnabled(enabled);

            LOG_INFO("ODE Action '" << name << "' set Enabed = " 
                << enabled  << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Action '" << name << "' threw exception setting Enabled");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeActionEnabledStateChangeListenerAdd(const char* name,
        dsl_ode_enabled_state_change_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            
            DSL_ODE_ACTION_PTR pOdeAction = 
                std::dynamic_pointer_cast<OdeAction>(m_odeActions[name]);
         
            if (!pOdeAction->AddEnabledStateChangeListener(listener, clientData))
            {
                LOG_ERROR("ODE Action '" << name 
                    << "' failed to add an Enabled State Change Listener");
                return DSL_RESULT_ODE_ACTION_CALLBACK_ADD_FAILED;
            }
            LOG_INFO("ODE Action '" << name 
                << "' successfully added an Enabled State Change Listener");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Action '" << name 
                << "' threw exception adding an Enabled State Change Listener");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionEnabledStateChangeListenerRemove(const char* name,
        dsl_ode_enabled_state_change_listener_cb listener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            
            DSL_ODE_ACTION_PTR pOdeAction = 
                std::dynamic_pointer_cast<OdeAction>(m_odeActions[name]);
         
            if (!pOdeAction->RemoveEnabledStateChangeListener(listener))
            {
                LOG_ERROR("ODE Action '" << name 
                    << "' failed to remove an Enabled State Change Listener");
                return DSL_RESULT_ODE_ACTION_CALLBACK_REMOVE_FAILED;
            }
            LOG_INFO("ODE Action '" << name 
                << "' successfully removed an Enabled State Change Listener");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Action '" << name 
                << "' threw exception removing an Enabled State Change Listener");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionDelete(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            
            if (m_odeActions[name].use_count() > 1)
            {
                LOG_INFO("ODE Action'" << name << "' is in use");
                return DSL_RESULT_ODE_ACTION_IN_USE;
            }
            m_odeActions.erase(name);

            LOG_INFO("ODE Action '" << name << "' deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Action '" << name << "' threw exception on deletion");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeActionDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            if (m_odeActions.empty())
            {
                return DSL_RESULT_SUCCESS;
            }
            for (auto const& imap: m_odeActions)
            {
                // In the case of Delete all
                if (imap.second.use_count() > 1)
                {
                    LOG_ERROR("ODE Action '" << imap.second->GetName() << "' is currently in use");
                    return DSL_RESULT_ODE_ACTION_IN_USE;
                }
            }
            m_odeActions.clear();

            LOG_INFO("All ODE Actions deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Action threw exception on delete all");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    uint Services::OdeActionListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_odeActions.size();
    }
}