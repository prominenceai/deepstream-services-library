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
#include "DslTilerBintr.h"

namespace DSL
{
    DslReturnType Services::TilerNew(const char* name, uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Tiler name '" << name << "' is not unique");
                return DSL_RESULT_TILER_NAME_NOT_UNIQUE;
            }
            m_components[name] = std::shared_ptr<Bintr>(new TilerBintr(
                name, width, height));
                
            LOG_INFO("New Tiler '" << name << "' created successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Tiler'" << name << "' threw exception on create");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TilerDimensionsGet(const char* name, uint* width, uint* height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);

            DSL_TILER_PTR tilerBintr = 
                std::dynamic_pointer_cast<TilerBintr>(m_components[name]);

            tilerBintr->GetDimensions(width, height);
            LOG_INFO("New Tiler '" << name << "' created successfully");
            
            LOG_INFO("Width = " << *width << " height = " << *height << 
                " returned successfully for Tiler '" << name << "'");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception getting dimensions");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TilerDimensionsSet(const char* name, uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);


            DSL_TILER_PTR tilerBintr = 
                std::dynamic_pointer_cast<TilerBintr>(m_components[name]);

            // TODO verify args before calling
            if (!tilerBintr->SetDimensions(width, height))
            {
                LOG_ERROR("Tiler '" << name << "' failed to settin dimensions");
                return DSL_RESULT_TILER_SET_FAILED;
            }
            LOG_INFO("Width = " << width << " height = " << height << 
                " set successfully for Tiler '" << name << "'");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception setting dimensions");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TilerTilesGet(const char* name, uint* columns, uint* rows)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);

            DSL_TILER_PTR tilerBintr = 
                std::dynamic_pointer_cast<TilerBintr>(m_components[name]);

            // TODO verify args before calling
            tilerBintr->GetTiles(columns, rows);

            LOG_INFO("Columns = " << *columns << " rows = " << *rows << 
                " returned successfully for Tiler '" << name << "'");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception getting Tiles");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TilerTilesSet(const char* name, uint columns, uint rows)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);

            DSL_TILER_PTR tilerBintr = 
                std::dynamic_pointer_cast<TilerBintr>(m_components[name]);

            // TODO verify args before calling
            if (!tilerBintr->SetTiles(columns, rows))
            {
                LOG_ERROR("Tiler '" << name << "' failed to set Tiles");
                return DSL_RESULT_TILER_SET_FAILED;
            }
            LOG_INFO("Columns = " << columns << " rows = " << rows << 
                " set successfully for Tiler '" << name << "'");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception setting Tiles");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TilerFrameNumberingEnabledGet(const char* name,
            boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                TilerBintr);

            DSL_TILER_PTR tilerBintr = 
                std::dynamic_pointer_cast<TilerBintr>(m_components[name]);

            // TODO verify args before calling
            *enabled = tilerBintr->GetFrameNumberingEnabled();

            LOG_INFO("Tiler '" << name 
                << "' successfully return frame-number enabled = " <<
                *enabled);
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name 
                << "' threw an exception getting frame-number enabled");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TilerFrameNumberingEnabledSet(const char* name,
            boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                TilerBintr);

            DSL_TILER_PTR tilerBintr = 
                std::dynamic_pointer_cast<TilerBintr>(m_components[name]);

            // TODO verify args before calling
            if (!tilerBintr->SetFrameNumberingEnabled(enabled))
            {
                LOG_ERROR("Tiler '" << name << "' failed to set Tiles");
                return DSL_RESULT_TILER_SET_FAILED;
            }
            LOG_INFO("Tiler '" << name 
                << "' successfully set frame-number enable = " << enabled);
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name 
                << "' threw an exception setting frame-number enabled");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TilerSourceShowGet(const char* name, 
        const char** source, uint* timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);

            DSL_TILER_PTR tilerBintr = 
                std::dynamic_pointer_cast<TilerBintr>(m_components[name]);

            int sourceId(-1);
            tilerBintr->GetShowSource(&sourceId, timeout);
            
            if (sourceId == -1)
            {
                *source = NULL;
                return DSL_RESULT_SUCCESS;
            }
            if (m_sourceNamesById.find(sourceId) == m_sourceNamesById.end())
            {
                *source = NULL;
                LOG_ERROR("Tiler '" << name << "' failed to get Source name from Id");
                return DSL_RESULT_SOURCE_NAME_NOT_FOUND;
            }
            *source = m_sourceNamesById[sourceId].c_str();
            
            LOG_INFO("Source = " << *source 
                << " returned successfully for Tiler '" << name << "'");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception setting Tiles");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TilerSourceShowSet(const char* name, 
        const char* source, uint timeout, bool hasPrecedence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);

            DSL_TILER_PTR pTilerBintr = 
                std::dynamic_pointer_cast<TilerBintr>(m_components[name]);

            if (!pTilerBintr->IsLinked())
            {
                LOG_ERROR("Tiler '" << name 
                    << "' must be in a linked state to show a specific source");
                return DSL_RESULT_TILER_SET_FAILED;
            }

            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, source);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, source);

            DSL_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<SourceBintr>(m_components[source]);
                    
            if (!pTilerBintr->SetShowSource(pSourceBintr->GetId(), timeout, hasPrecedence))
            {
                LOG_ERROR("Tiler '" << name << "' failed to show specific source");
                return DSL_RESULT_TILER_SET_FAILED;
            }
            LOG_INFO("Source = " << source << " timeout = " << timeout << 
                " has precedence = " << hasPrecedence 
                << " set successfully for Tiler '" << name << "'");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name 
                << "' threw an exception showing a specific source");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }

    // Note this instance called internally, i.e. not exposed to client 
    DslReturnType Services::TilerSourceShowSet(const char* name, 
        uint sourceId, uint timeout, bool hasPrecedence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);

            DSL_TILER_PTR pTilerBintr = 
                std::dynamic_pointer_cast<TilerBintr>(m_components[name]);

            if (!pTilerBintr->IsLinked())
            {
                LOG_ERROR("Tiler '" << name 
                    << "' must be in a linked state to show a specific source");
                return DSL_RESULT_TILER_SET_FAILED;
            }

            // called by automation - so set hasPrecedence to false always
            if (!pTilerBintr->SetShowSource(sourceId, timeout, hasPrecedence))
            {
                // Don't log error as this can happen with the ODE actions frequently
                LOG_DEBUG("Tiler '" << name << "' failed to show specific source");
                return DSL_RESULT_TILER_SET_FAILED;
            }
            
            LOG_INFO("Source Id = " << sourceId << ", timeout = " << timeout << 
                ", and Has Precedence = " << hasPrecedence << 
                " set successfully for Tiler '" << name << "'");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name 
                << "' threw an exception showing a specific source");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TilerSourceShowSelect(const char* name, 
        int xPos, int yPos, uint windowWidth, uint windowHeight, uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);

            DSL_TILER_PTR pTilerBintr = 
                std::dynamic_pointer_cast<TilerBintr>(m_components[name]);

            if (!pTilerBintr->IsLinked())
            {
                LOG_ERROR("Tiler '" << name 
                    << "' must be in a linked state to show a specific source");
                return DSL_RESULT_TILER_SET_FAILED;
            }

            int sourceId(0);
            uint currentTimeout(0);
            pTilerBintr->GetShowSource(&sourceId, &currentTimeout);
            
            // if currently showing all sources
            if (sourceId == -1)
            {
                uint cols(0), rows(0);
                pTilerBintr->GetTiles(&cols, &rows);
                if (rows*cols == 1)
                {
                    // single source, noting to do
                    return DSL_RESULT_SUCCESS;
                }
                float xRel((float)xPos/windowWidth), yRel((float)yPos/windowHeight);
                sourceId = (int)(xRel*cols);
                sourceId += ((int)(yRel*rows))*cols;
                
                if (sourceId > pTilerBintr->GetBatchSize())
                {
                    // clicked on empty tile, noting to do
                    return DSL_RESULT_SUCCESS;
                }

                if (!pTilerBintr->SetShowSource(sourceId, timeout, true))
                {
                    LOG_ERROR("Tiler '" << name << "' failed to select specific source");
                    return DSL_RESULT_TILER_SET_FAILED;
                }
                LOG_INFO("xPos = " << xPos << " yPos = " << yPos 
                    << " window width = " << windowWidth 
                    << " window hidth = " << windowHeight
                    << " timeout = " << timeout << "selected successfully for Tiler '" 
                    << name << "'");
            }
            // else, showing a single source so return to all sources. 
            else
            {
                pTilerBintr->ShowAllSources();
                LOG_INFO("Return to show all set successfully for Tiler '" << name << "'");
            }
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception showing a specific source");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TilerSourceShowAll(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);

            DSL_TILER_PTR pTilerBintr = 
                std::dynamic_pointer_cast<TilerBintr>(m_components[name]);

            pTilerBintr->ShowAllSources();
            
            LOG_INFO("Show all sources set successfully for Tiler '" << name << "'");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception showing all sources");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TilerSourceShowCycle(const char* name, uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);

            DSL_TILER_PTR pTilerBintr = 
                std::dynamic_pointer_cast<TilerBintr>(m_components[name]);

            if (!pTilerBintr->CycleAllSources(timeout))
            {
                    LOG_ERROR("Tiler '" << name << "' failed to set Cycle all sources");
                    return DSL_RESULT_TILER_SET_FAILED;
            }
            LOG_INFO("Cycle all sources with timeout " << timeout 
                << " set successfully for Tiler '" << name << "'");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception showing all sources");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::TilerPphAdd(const char* name, const char* handler, uint pad)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            if (pad > DSL_PAD_SRC)
            {
                LOG_ERROR("Invalid Pad type = " << pad << " for Tiler '" << name << "'");
                return DSL_RESULT_PPH_PAD_TYPE_INVALID;
            }

            // call on the Handler to add itself to the Tiler as a PadProbeHandler
            if (!m_padProbeHandlers[handler]->AddToParent(m_components[name], pad))
            {
                LOG_ERROR("Tiler '" << name << "' failed to add Pad Probe Handler");
                return DSL_RESULT_TILER_HANDLER_ADD_FAILED;
            }
            LOG_INFO("Pad Probe Handler '" << handler 
                << "' added to Tiler '" << name << "' successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name 
                << "' threw an exception adding Pad Probe Handler");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }
   
    DslReturnType Services::TilerPphRemove(const char* name, const char* handler, uint pad) 
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            if (pad > DSL_PAD_SRC)
            {
                LOG_ERROR("Invalid Pad type = " << pad << " for Tiler '" << name << "'");
                return DSL_RESULT_PPH_PAD_TYPE_INVALID;
            }

            // call on the Handler to remove itself from the Tiler
            if (!m_padProbeHandlers[handler]->RemoveFromParent(m_components[name], pad))
            {
                LOG_ERROR("Pad Probe Handler '" << handler 
                    << "' is not a child of Tiler '" << name << "'");
                return DSL_RESULT_TILER_HANDLER_REMOVE_FAILED;
            }
            LOG_INFO("Pad Probe Handler '" << handler 
                << "' removed from Tiler '" << name << "' successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception removing ODE Handle");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }

}