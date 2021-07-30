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
#include "DslOdeArea.h"

namespace DSL
{
    DslReturnType Services::OdeAreaInclusionNew(const char* name, 
        const char* polygon, boolean show, uint bboxTestPoint)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure ODE Area name uniqueness 
            if (m_odeAreas.find(name) != m_odeAreas.end())
            {   
                LOG_ERROR("ODE Area name '" << name << "' is not unique");
                return DSL_RESULT_ODE_AREA_NAME_NOT_UNIQUE;
            }
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, polygon);
            
            // Interim ... only supporting rectangles at this
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, polygon, RgbaPolygon);
            
            if (bboxTestPoint > DSL_BBOX_POINT_ANY)
            {
                LOG_ERROR("Bounding box test point value of '" << bboxTestPoint << 
                    "' is invalid when creating ODE Inclusion Area '" << name << "'");
                return DSL_RESULT_ODE_AREA_PARAMETER_INVALID;
            }
            
            DSL_RGBA_POLYGON_PTR pPolygon = 
                std::dynamic_pointer_cast<RgbaPolygon>(m_displayTypes[polygon]);
            
            m_odeAreas[name] = DSL_ODE_AREA_INCLUSION_NEW(name, 
                pPolygon, show, bboxTestPoint);
         
            LOG_INFO("New ODE Inclusion Area '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Inclusion Area '" << name << "' threw exception on creation");
            return DSL_RESULT_ODE_AREA_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeAreaExclusionNew(const char* name, 
        const char* polygon, boolean show, uint bboxTestPoint)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure ODE Area name uniqueness 
            if (m_odeAreas.find(name) != m_odeAreas.end())
            {   
                LOG_ERROR("ODE Area name '" << name << "' is not unique");
                return DSL_RESULT_ODE_AREA_NAME_NOT_UNIQUE;
            }
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, polygon);
            
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, polygon, RgbaPolygon);

            if (bboxTestPoint > DSL_BBOX_POINT_ANY)
            {
                LOG_ERROR("Bounding box test point value of '" << bboxTestPoint << 
                    "' is invalid when creating ODE Exclusion Area '" << name << "'");
                return DSL_RESULT_ODE_AREA_PARAMETER_INVALID;
            }

            DSL_RGBA_POLYGON_PTR pPolygon = 
                std::dynamic_pointer_cast<RgbaPolygon>(m_displayTypes[polygon]);
            
            m_odeAreas[name] = DSL_ODE_AREA_EXCLUSION_NEW(name, 
                pPolygon, show, bboxTestPoint);
         
            LOG_INFO("New ODE Exclusion Area '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Exclusion Area '" << name << "' threw exception on creation");
            return DSL_RESULT_ODE_AREA_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::OdeAreaLineNew(const char* name, 
        const char* line, boolean show, uint bboxTestEdge)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure ODE Area name uniqueness 
            if (m_odeAreas.find(name) != m_odeAreas.end())
            {   
                LOG_ERROR("ODE Area name '" << name << "' is not unique");
                return DSL_RESULT_ODE_AREA_NAME_NOT_UNIQUE;
            }
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, line);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, line, RgbaLine);
            
            if (bboxTestEdge > DSL_BBOX_EDGE_RIGHT)
            {
                LOG_ERROR("Bounding box test edge value of '" << bboxTestEdge << 
                    "' is invalid when creating ODE Line Area '" << name << "'");
                return DSL_RESULT_ODE_AREA_PARAMETER_INVALID;
            }
            
            DSL_RGBA_LINE_PTR pLine = 
                std::dynamic_pointer_cast<RgbaLine>(m_displayTypes[line]);
            
            m_odeAreas[name] = DSL_ODE_AREA_LINE_NEW(name, pLine, show, bboxTestEdge);
         
            LOG_INFO("New ODE Line Area '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Line Area '" << name << "' threw exception on creation");
            return DSL_RESULT_ODE_AREA_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::OdeAreaDelete(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeAreas, name);
            
            if (m_odeAreas[name].use_count() > 1)
            {
                LOG_INFO("ODE Area'" << name << "' is in use");
                return DSL_RESULT_ODE_ACTION_IN_USE;
            }
            m_odeAreas.erase(name);

            LOG_INFO("ODE Area '" << name << "' deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Area '" << name << "' threw exception on deletion");
            return DSL_RESULT_ODE_AREA_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeAreaDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            if (m_odeAreas.empty())
            {
                return DSL_RESULT_SUCCESS;
            }
            for (auto const& imap: m_odeAreas)
            {
                // In the case of Delete all
                if (imap.second.use_count() > 1)
                {
                    LOG_ERROR("ODE Area '" << imap.second->GetName() << "' is currently in use");
                    return DSL_RESULT_ODE_AREA_IN_USE;
                }
            }
            m_odeAreas.clear();

            LOG_INFO("All ODE Areas deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Area threw exception on delete all");
            return DSL_RESULT_ODE_AREA_THREW_EXCEPTION;
        }
    }

    uint Services::OdeAreaListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_odeAreas.size();
    }
        
}    
