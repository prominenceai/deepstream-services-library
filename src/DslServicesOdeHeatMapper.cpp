/*
The MIT License

Copyright (c)   2022, Prominence AI, Inc.

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

namespace DSL
{

    DslReturnType Services::OdeHeatMapperNew(const char* name,
        uint cols, uint rows, const char* colorPalette)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure HeatMapper name uniqueness 
            if (m_odeHeatMappers.find(name) != m_odeHeatMappers.end())
            {   
                LOG_ERROR("ODE Heat-Mapper name '" << name 
                    << "' is not unique");
                return DSL_RESULT_ODE_HEAT_MAPPER_NAME_NOT_UNIQUE;
            }
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, colorPalette);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, 
                colorPalette, RgbaColorPalette);

            DSL_RGBA_COLOR_PALETTE_PTR pColorPalette = 
                std::dynamic_pointer_cast<RgbaColorPalette>(
                    m_displayTypes[colorPalette]);
            
            m_odeHeatMappers[name] = DSL_ODE_HEAT_MAPPER_NEW(name,
                cols, rows, pColorPalette);
            
            LOG_INFO("New ODE Heat-Mapper '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Heat-Mapper '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_HEAT_MAPPER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeHeatMapperDelete(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_HEAT_MAPPER_NAME_NOT_FOUND(m_odeHeatMappers, name);
            
            if (m_odeHeatMappers[name]->IsInUse())
            {
                LOG_INFO("ODE HeatMapper '" << name << "' is in use");
                return DSL_RESULT_ODE_HEAT_MAPPER_IN_USE;
            }
            m_odeHeatMappers.erase(name);

            LOG_INFO("ODE Heat-Mapper '" << name << "' deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE HeatMapper '" << name 
                << "' threw an exception on deletion");
            return DSL_RESULT_ODE_HEAT_MAPPER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeHeatMapperDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            if (m_odeHeatMappers.empty())
            {
                return DSL_RESULT_SUCCESS;
            }
            for (auto const& imap: m_odeHeatMappers)
            {
                // In the case of Delete all
                if (imap.second->IsInUse())
                {
                    LOG_ERROR("ODE Heat-Mapper '" << imap.second->GetName() 
                        << "' is currently in use");
                    return DSL_RESULT_ODE_HEAT_MAPPER_IN_USE;
                }
            }
            m_odeHeatMappers.clear();

            LOG_INFO("All ODE Heat-Mappers deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Heat-Mapper API threw an exception on delete all");
            return DSL_RESULT_ODE_HEAT_MAPPER_THREW_EXCEPTION;
        }
    }

    uint Services::OdeHeatMapperListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_odeHeatMappers.size();
    }
    
}