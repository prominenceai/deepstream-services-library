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

namespace DSL
{
    DslReturnType Services::OdeTriggerAlwaysNew(const char* name, const char* source, uint when)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE;
            }
            if (when > DSL_ODE_POST_OCCURRENCE_CHECK)
            {   
                LOG_ERROR("Invalid 'when' parameter for ODE Trigger name '" << name << "'");
                return DSL_RESULT_ODE_TRIGGER_PARAMETER_INVALID;
            }
            m_odeTriggers[name] = DSL_ODE_TRIGGER_ALWAYS_NEW(name, source, when);
            
            LOG_INFO("New Always ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Always ODE Trigger '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerOccurrenceNew(const char* name, 
        const char* source, uint classId, uint limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE;
            }
            m_odeTriggers[name] = DSL_ODE_TRIGGER_OCCURRENCE_NEW(name, 
                source, classId, limit);
            
            LOG_INFO("New Occurrence ODE Trigger '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Occurrence ODE Trigger '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerAbsenceNew(const char* name, 
        const char* source, uint classId, uint limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE;
            }
            m_odeTriggers[name] = DSL_ODE_TRIGGER_ABSENCE_NEW(name, 
                source, classId, limit);
            
            LOG_INFO("New Absence ODE Trigger '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Absence ODE Trigger '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    
    DslReturnType Services::OdeTriggerInstanceNew(const char* name, 
        const char* source, uint classId, uint limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE;
            }
            m_odeTriggers[name] = DSL_ODE_TRIGGER_INSTANCE_NEW(name, 
                source, classId, limit);
            
            LOG_INFO("New Instance ODE Trigger '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Instance ODE Trigger '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerIntersectionNew(const char* name, 
        const char* source, uint classIdA, uint classIdB, uint limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE;
            }
            m_odeTriggers[name] = DSL_ODE_TRIGGER_INTERSECTION_NEW(name, 
                source, classIdA, classIdB, limit);
            
            LOG_INFO("New Intersection ODE Trigger '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Intersection ODE Trigger '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerSummationNew(const char* name, 
        const char* source, uint classId, uint limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE;
            }
            m_odeTriggers[name] = DSL_ODE_TRIGGER_SUMMATION_NEW(name, source, classId, limit);
            
            LOG_INFO("New Summation ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Summation ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerCustomNew(const char* name, const char* source, 
        uint classId, uint limit,  dsl_ode_check_for_occurrence_cb client_checker, 
        dsl_ode_post_process_frame_cb client_post_processor, void* client_data)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE;
            }
            
            m_odeTriggers[name] = DSL_ODE_TRIGGER_CUSTOM_NEW(name, source,
                classId, limit, client_checker, client_post_processor, client_data);
            
            LOG_INFO("New Custom ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Custon ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeTriggerCountNew(const char* name, const char* source, 
        uint classId, uint limit, uint minimum, uint maximum)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE;
            }
            // check for no maximum
            maximum = (maximum == 0) ? UINT32_MAX : maximum;
            
            m_odeTriggers[name] = DSL_ODE_TRIGGER_COUNT_NEW(name, 
                source, classId, limit, minimum, maximum);
            
            LOG_INFO("New Count ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Count ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeTriggerCountRangeGet(const char* name, 
        uint* minimum, uint* maximum)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_odeTriggers, name, CountOdeTrigger);
            
            DSL_ODE_TRIGGER_COUNT_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<CountOdeTrigger>(m_odeTriggers[name]);

            pOdeTrigger->GetRange(minimum, maximum);
            
            // check for no maximum
            *maximum = (*maximum == UINT32_MAX) ? 0 : *maximum;

            LOG_INFO("ODE Count Trigger '" << name << "' returned range from mimimum " 
                << *minimum << " to maximum " << *maximum << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Count Trigger '" << name 
                << "' threw exception getting range");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::OdeTriggerCountRangeSet(const char* name, 
        uint minimum, uint maximum)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_odeTriggers, name, CountOdeTrigger);
            
            DSL_ODE_TRIGGER_COUNT_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<CountOdeTrigger>(m_odeTriggers[name]);

            // check for no maximum
            maximum = (maximum == 0) ? UINT32_MAX : maximum;
         
            pOdeTrigger->SetRange(minimum, maximum);
            
            LOG_INFO("ODE Count Trigger '" << name << "' set new range from mimimum " 
                << minimum << " to maximum " << maximum << " successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Count Trigger '" << name 
                << "' threw exception setting range");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::OdeTriggerDistanceNew(const char* name, const char* source, 
        uint classIdA, uint classIdB, uint limit, uint minimum, uint maximum, 
        uint testPoint, uint testMethod)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE;
            }
            // check for no maximum
            maximum = (maximum == 0) ? UINT32_MAX : maximum;
            
            m_odeTriggers[name] = DSL_ODE_TRIGGER_DISTANCE_NEW(name, 
                source, classIdA, classIdB, limit, minimum, maximum, 
                testPoint, testMethod);
            
            LOG_INFO("New Distance ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Distance ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
            
    DslReturnType Services::OdeTriggerDistanceRangeGet(const char* name, 
        uint* minimum, uint* maximum)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_odeTriggers, name, DistanceOdeTrigger);
            
            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<DistanceOdeTrigger>(m_odeTriggers[name]);

            pOdeTrigger->GetRange(minimum, maximum);
            
            // check for no maximum
            *maximum = (*maximum == UINT32_MAX) ? 0 : *maximum;

            LOG_INFO("ODE Distance Trigger '" << name << "' returned range from mimimum " 
                << *minimum << " to maximum " << *maximum << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Distance Trigger '" << name 
                << "' threw exception getting range");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::OdeTriggerDistanceRangeSet(const char* name, 
        uint minimum, uint maximum)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_odeTriggers, name, DistanceOdeTrigger);
            
            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<DistanceOdeTrigger>(m_odeTriggers[name]);

            // check for no maximum
            maximum = (maximum == 0) ? UINT32_MAX : maximum;
         
            pOdeTrigger->SetRange(minimum, maximum);
            
            LOG_INFO("ODE Distance Trigger '" << name << "' set new range from mimimum " 
                << minimum << " to maximum " << maximum << " successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Distance Trigger '" << name 
                << "' threw exception setting range");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerDistanceTestParamsGet(const char* name, 
        uint* testPoint, uint* testMethod)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_odeTriggers, name, DistanceOdeTrigger);
            
            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<DistanceOdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->GetTestParams(testPoint, testMethod);
            
            LOG_INFO("ODE Distance Trigger '" << name << "' returned Test Parameters Test-Point = " 
                << *testPoint << " and Test-Method = " << *testMethod << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Distance Trigger '" << name 
                << "' threw exception getting test parameters");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::OdeTriggerDistanceTestParamsSet(const char* name, 
        uint testPoint, uint testMethod)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_odeTriggers, name, DistanceOdeTrigger);
            
            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<DistanceOdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->SetTestParams(testPoint, testMethod);

            LOG_INFO("ODE Distance Trigger '" << name << "' set new Test Parameters Test-Point " 
                << testPoint << " and Test-Method " << testMethod << " successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Distance Trigger '" << name 
                << "' threw exception setting test parameters");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::OdeTriggerSmallestNew(const char* name, 
        const char* source, uint classId, uint limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE;
            }
            m_odeTriggers[name] = DSL_ODE_TRIGGER_SMALLEST_NEW(name, source, classId, limit);
            
            LOG_INFO("New Smallest ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Smallest ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerLargestNew(const char* name, 
        const char* source, uint classId, uint limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE;
            }
            m_odeTriggers[name] = DSL_ODE_TRIGGER_LARGEST_NEW(name, source, classId, limit);
            
            LOG_INFO("New Largest ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Largest ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeTriggerNewHighNew(const char* name, 
        const char* source, uint classId, uint limit, uint preset)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE;
            }
            m_odeTriggers[name] = DSL_ODE_TRIGGER_NEW_HIGH_NEW(name, 
                source, classId, limit, preset);
            
            LOG_INFO("New New-High ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New New-High ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }

    
    DslReturnType Services::OdeTriggerNewLowNew(const char* name, 
        const char* source, uint classId, uint limit, uint preset)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE;
            }
            m_odeTriggers[name] = DSL_ODE_TRIGGER_NEW_LOW_NEW(name, 
                source, classId, limit, preset);
            
            LOG_INFO("New New-Low ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New New-Low ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerCrossNew(const char* name, 
        const char* source, uint classId, uint limit, 
        uint minFrameCount, uint maxFrameCount, uint testMethod)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE;
            }
            if (testMethod > DSL_OBJECT_TRACE_TEST_METHOD_ALL_POINTS)
            {
                LOG_ERROR("Invalid test method = " << testMethod 
                    << " for ODE Cross Trigger '" << name << "'");
                return DSL_RESULT_ODE_TRIGGER_PARAMETER_INVALID;
            }
            if (minFrameCount >= maxFrameCount)
            {
                LOG_ERROR("Invalid parameters - max_trace_points = " << maxFrameCount 
                    << "must be greater than min_frame_count = " << minFrameCount
                    << "for ODE Cross Trigger '" << name << "'");
                return DSL_RESULT_ODE_TRIGGER_PARAMETER_INVALID;
            }
            DSL_RGBA_COLOR_PTR pColor = std::dynamic_pointer_cast<RgbaColor>
                (m_intrinsicDisplayTypes[DISPLAY_TYPE_NO_COLOR.c_str()]);
            
            m_odeTriggers[name] = DSL_ODE_TRIGGER_CROSS_NEW(name, 
                source, classId, limit, minFrameCount, maxFrameCount, 
                testMethod, pColor);
            
            LOG_INFO("New Cross ODE Trigger '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Cross ODE Trigger '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeTriggerCrossTestSettingsGet(const char* name, 
        uint* minFrameCount, uint* maxFrameCount, uint* testMethod)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_odeTriggers, name, 
                CrossOdeTrigger);
            
            DSL_ODE_TRIGGER_CROSS_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<CrossOdeTrigger>(m_odeTriggers[name]);

            pOdeTrigger->GetTestSettings(minFrameCount, 
                maxFrameCount, testMethod);

            LOG_INFO("ODE Tracking Trigger '" << name 
                << "' returned test settings successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Tracking Trigger '" << name 
                << "' threw exception getting test settings");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::OdeTriggerCrossTestSettingsSet(const char* name, 
        uint minFrameCount, uint maxFrameCount, uint testMethod)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_odeTriggers, name, 
                CrossOdeTrigger);
            
            if (testMethod > DSL_OBJECT_TRACE_TEST_METHOD_ALL_POINTS)
            {
                LOG_ERROR("Invalid test method = " << testMethod 
                    << " for ODE Tracking Trigger '" << name << "'");
                return DSL_RESULT_ODE_TRIGGER_PARAMETER_INVALID;
            }
            if (minFrameCount >= maxFrameCount)
            {
                LOG_ERROR("Invalid parameters - max_frame_count = " << maxFrameCount
                    << "must be greater than min_frame_count = " << minFrameCount
                    << "for ODE Cross Trigger '" << name << "'");
                return DSL_RESULT_ODE_TRIGGER_PARAMETER_INVALID;
            }
            DSL_ODE_TRIGGER_CROSS_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<CrossOdeTrigger>(m_odeTriggers[name]);

            pOdeTrigger->SetTestSettings(minFrameCount, 
                maxFrameCount, testMethod);

            LOG_INFO("ODE Cross Trigger '" << name 
                << "' set test settings successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Cross Trigger '" << name 
                << "' threw exception getting test settings");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::OdeTriggerCrossViewSettingsGet(const char* name, 
        boolean* enabled, const char** color, uint* lineWidth)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_odeTriggers, name, 
                CrossOdeTrigger);
            
            DSL_ODE_TRIGGER_CROSS_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<CrossOdeTrigger>(m_odeTriggers[name]);

            bool bEnabled;
            pOdeTrigger->GetViewSettings(&bEnabled, color, lineWidth);
            *enabled = bEnabled;

            LOG_INFO("ODE Tracking Trigger '" << name 
                << "' returned view settings successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Tracking Trigger '" << name 
                << "' threw exception getting view settings");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::OdeTriggerCrossViewSettingsSet(const char* name, 
        boolean enabled, const char* color, uint lineWidth)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_odeTriggers, name, 
                CrossOdeTrigger);
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_COLOR(m_displayTypes, color);
            
            DSL_ODE_TRIGGER_CROSS_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<CrossOdeTrigger>(m_odeTriggers[name]);
                
            DSL_RGBA_COLOR_PTR pColor = 
                std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[color]);

            pOdeTrigger->SetViewSettings(enabled, pColor, lineWidth);

            LOG_INFO("ODE Track Trigger '" << name 
                << "' set view settings successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Track Trigger '" << name 
                << "' threw exception setting view settings");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerPersistenceNew(const char* name, const char* source, 
        uint classId, uint limit, uint minimum, uint maximum)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE;
            }
            // check for no maximum
            maximum = (maximum == 0) ? UINT32_MAX : maximum;

            m_odeTriggers[name] = DSL_ODE_TRIGGER_PERSISTENCE_NEW(name, 
                source, classId, limit, minimum, maximum);
            
            LOG_INFO("New Persistence ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Persistence ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeTriggerPersistenceRangeGet(const char* name, 
        uint* minimum, uint* maximum)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_odeTriggers, name, 
                PersistenceOdeTrigger);
            
            DSL_ODE_TRIGGER_PERSISTENCE_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<PersistenceOdeTrigger>(m_odeTriggers[name]);

            pOdeTrigger->GetRange(minimum, maximum);
            
            // check for no maximum
            *maximum = (*maximum == UINT32_MAX) ? 0 : *maximum;

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Persistence Trigger '" << name 
                << "' threw exception getting range");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::OdeTriggerPersistenceRangeSet(const char* name, 
        uint minimum, uint maximum)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_odeTriggers, name, 
                PersistenceOdeTrigger);
            
            DSL_ODE_TRIGGER_PERSISTENCE_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<PersistenceOdeTrigger>(m_odeTriggers[name]);

            // check for no maximum
            maximum = (maximum == 0) ? UINT32_MAX : maximum;
         
            pOdeTrigger->SetRange(minimum, maximum);
            
            LOG_INFO("ODE Persistence Trigger '" << name << "' set new range from mimimum " 
                << minimum << " to maximum " << maximum << " successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Persistence Trigger '" << name 
                << "' threw exception setting range");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerLatestNew(const char* name, 
        const char* source, uint classId, uint limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE;
            }

            m_odeTriggers[name] = DSL_ODE_TRIGGER_LATEST_NEW(name, 
                source, classId, limit);
            
            LOG_INFO("New Latest ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Latest ODE Trigger '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerEarliestNew(const char* name, 
        const char* source, uint classId, uint limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE;
            }

            m_odeTriggers[name] = DSL_ODE_TRIGGER_EARLIEST_NEW(name, 
                source, classId, limit);
            
            LOG_INFO("New Earliest ODE Trigger '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Earliest ODE Trigger '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerReset(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->Reset();
            
            LOG_INFO("ODE Trigger '" << name << "' Reset successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting Enabled setting");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerResetTimeoutGet(const char* name, uint* timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            *timeout = pOdeTrigger->GetResetTimeout();
            
            LOG_INFO("Trigger '" << name << "' returned Timeout = " 
                << *timeout << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting Reset Timer");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerResetTimeoutSet(const char* name, uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->SetResetTimeout(timeout);

            LOG_INFO("Trigger '" << name << "' set Timeout = " 
                << timeout << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception setting Reset Timer");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerLimitStateChangeListenerAdd(const char* name,
        dsl_ode_trigger_limit_state_change_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            if (!pOdeTrigger->AddLimitStateChangeListener(listener, clientData))
            {
                LOG_ERROR("ODE Trigger '" << name 
                    << "' failed to add a Limit State Change Listener");
                return DSL_RESULT_ODE_TRIGGER_CALLBACK_ADD_FAILED;
            }
            LOG_INFO("ODE Trigger '" << name 
                << "' successfully added a Limit State Change Listener");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception adding a Limit State Change Listener");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeTriggerLimitStateChangeListenerRemove(const char* name,
        dsl_ode_trigger_limit_state_change_listener_cb listener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            if (!pOdeTrigger->RemoveLimitStateChangeListener(listener))
            {
                LOG_ERROR("ODE Trigger '" << name 
                    << "' failed to remove a Limit State Change Listener");
                return DSL_RESULT_ODE_TRIGGER_CALLBACK_REMOVE_FAILED;
            }
            LOG_INFO("ODE Trigger '" << name 
                << "' successfully removed a Limit State Change Listener");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception removing a Limit State Change Listener");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerEnabledGet(const char* name, boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            *enabled = pOdeTrigger->GetEnabled();
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting Enabled setting");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerEnabledSet(const char* name, boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->SetEnabled(enabled);
            
            LOG_INFO("Trigger '" << name << "' returned Enabled = "
                << enabled << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception setting Enabled");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerEnabledStateChangeListenerAdd(const char* name,
        dsl_ode_enabled_state_change_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            if (!pOdeTrigger->AddEnabledStateChangeListener(listener, clientData))
            {
                LOG_ERROR("ODE Trigger '" << name 
                    << "' failed to add an Enabled State Change Listener");
                return DSL_RESULT_ODE_TRIGGER_CALLBACK_ADD_FAILED;
            }
            LOG_INFO("ODE Trigger '" << name 
                << "' successfully added an Enabled State Change Listener");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception adding an Enabled State Change Listener");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeTriggerEnabledStateChangeListenerRemove(const char* name,
        dsl_ode_enabled_state_change_listener_cb listener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            if (!pOdeTrigger->RemoveEnabledStateChangeListener(listener))
            {
                LOG_ERROR("ODE Trigger '" << name 
                    << "' failed to remove an Enabled State Change Listener");
                return DSL_RESULT_ODE_TRIGGER_CALLBACK_REMOVE_FAILED;
            }
            LOG_INFO("ODE Trigger '" << name 
                << "' successfully removed an Enabled State Change Listener");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception removing an Enabled State Change Listener");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerSourceGet(const char* name, const char** source)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            *source = pOdeTrigger->GetSource();
            
            LOG_INFO("Trigger '" << name << "' returned Source = " 
                << *source << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting source name");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerSourceSet(const char* name, const char* source)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);

            pOdeTrigger->SetSource(source);
            
            LOG_INFO("Trigger '" << name << "' set Source = " 
                << source << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception setting source name");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerInferGet(const char* name, const char** infer)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            *infer = pOdeTrigger->GetInfer();
            
            LOG_INFO("Trigger '" << name << "' returned inference component name = " 
                << *infer << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting inference component name");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerInferSet(const char* name, const char* infer)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);

            pOdeTrigger->SetInfer(infer);
            
            LOG_INFO("Trigger '" << name << "' set inference component name = " 
                << infer << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting inference component name");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerClassIdGet(const char* name, uint* classId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            *classId = pOdeTrigger->GetClassId();
            
            LOG_INFO("Trigger '" << name << "' returned Class Id = " 
                << *classId << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting class id");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerClassIdSet(const char* name, uint classId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->SetClassId(classId);
            
            LOG_INFO("Trigger '" << name << "' set Class Id = " 
                << classId << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting class id");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerClassIdABGet(const char* name, 
        uint* classIdA, uint* classIdB)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_ODE_TRIGGER_IS_NOT_AB_TYPE(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_AB_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<ABOdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->GetClassIdAB(classIdA, classIdB);
            
            LOG_INFO("AB Trigger '" << name << "' returned Class Id A = " 
                << *classIdA << " and Class Id B = " << *classIdB << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting class id");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerClassIdABSet(const char* name, 
        uint classIdA, uint classIdB)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_ODE_TRIGGER_IS_NOT_AB_TYPE(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_AB_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<ABOdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->SetClassIdAB(classIdA, classIdB);

            LOG_INFO("AB Trigger '" << name << "' set Class Id A = " 
                << classIdA << " and Class Id B = " << classIdB << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting class id");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                
    DslReturnType Services::OdeTriggerLimitEventGet(const char* name, uint* limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            *limit = pOdeTrigger->GetEventLimit();

            LOG_INFO("Trigger '" << name << "' returned Event Limit = " 
                << *limit << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting Event Limit");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerLimitEventSet(const char* name, uint limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->SetEventLimit(limit);
            
            LOG_INFO("Trigger '" << name << "' set Evemt Limit = " 
                << limit << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting Event Limit");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }    
            
    DslReturnType Services::OdeTriggerLimitFrameGet(const char* name, uint* limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            *limit = pOdeTrigger->GetFrameLimit();

            LOG_INFO("Trigger '" << name << "' returned Frame Limit = " 
                << *limit << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting Frame Limit");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerLimitFrameSet(const char* name, uint limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->SetFrameLimit(limit);
            
            LOG_INFO("Trigger '" << name << "' set Frame Limit = " 
                << limit << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting limit");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }    
            
    DslReturnType Services::OdeTriggerConfidenceMinGet(const char* 
        name, float* minConfidence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            *minConfidence = pOdeTrigger->GetMinConfidence();
            
            LOG_INFO("Trigger '" << name << "' returned minimum confidence = " 
                << *minConfidence << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting minimum confidence");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerConfidenceMinSet(const char* name, 
        float minConfidence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);

            pOdeTrigger->SetMinConfidence(minConfidence);

            LOG_INFO("Trigger '" << name << "' set minimum confidence = " 
                << minConfidence << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting minimum confidence");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerConfidenceMaxGet(const char* 
        name, float* maxConfidence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            *maxConfidence = pOdeTrigger->GetMaxConfidence();
            
            LOG_INFO("Trigger '" << name << "' returned maximum confidence = " 
                << *maxConfidence << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting maximum confidence");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerConfidenceMaxSet(const char* name, 
        float maxConfidence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);

            pOdeTrigger->SetMaxConfidence(maxConfidence);

            LOG_INFO("Trigger '" << name << "' set maximum confidence = " 
                << maxConfidence << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting minimum confidence");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerTrackerConfidenceMinGet(const char* 
        name, float* minConfidence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            *minConfidence = pOdeTrigger->GetMinTrackerConfidence();
            
            LOG_INFO("Trigger '" << name << "' returned minimum Tracker confidence = " 
                << *minConfidence << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting minimum Tracker confidence");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerTrackerConfidenceMinSet(const char* name, 
        float minConfidence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);

            pOdeTrigger->SetMinTrackerConfidence(minConfidence);

            LOG_INFO("Trigger '" << name << "' set minimum Tracker confidence = " 
                << minConfidence << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting minimum Tracker confidence");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerTrackerConfidenceMaxGet(const char* 
        name, float* maxConfidence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            *maxConfidence = pOdeTrigger->GetMaxTrackerConfidence();
            
            LOG_INFO("Trigger '" << name << "' returned maximum Tracker confidence = " 
                << *maxConfidence << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting maximum Tracker confidence");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerTrackerConfidenceMaxSet(const char* name, 
        float maxConfidence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);

            pOdeTrigger->SetMaxTrackerConfidence(maxConfidence);

            LOG_INFO("Trigger '" << name << "' set maximum Tracker confidence = " 
                << maxConfidence << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting minimum Tracker confidence");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerDimensionsMinGet(const char* name, 
        float* minWidth, float* minHeight)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->GetMinDimensions(minWidth, minHeight);
            
            LOG_INFO("Trigger '" << name << "' returned Minimum Width = " 
                << *minWidth << " and Minimum Height = " 
                    << *minHeight << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting minimum dimensions");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerDimensionsMinSet(const char* name, 
        float minWidth, float minHeight)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            // TODO: validate the min values for in-range
            pOdeTrigger->SetMinDimensions(minWidth, minHeight);

            LOG_INFO("Trigger '" << name << "' returned Minimum Width = " 
                << minWidth << " and Minimum Height = " << minHeight << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception setting minimum dimensions");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerDimensionsMaxGet(const char* name, 
        float* maxWidth, float* maxHeight)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->GetMaxDimensions(maxWidth, maxHeight);
            
            LOG_INFO("Trigger'" << name << "' returned Maximim Width = " 
                << *maxWidth << " and Minimum Height = " 
                    << *maxHeight << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting maximum dimensions");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerDimensionsMaxSet(const char* name, 
        float maxWidth, float maxHeight)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            // TODO: validate the max values for in-range
            pOdeTrigger->SetMaxDimensions(maxWidth, maxHeight);

            LOG_INFO("Trigger '" << name << "' set Maximim Width = " 
                << maxWidth << " and Minimum Height = " 
                    << maxHeight << " successfully");
            

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception setting maximum dimensions");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerFrameCountMinGet(const char* name, 
        uint* min_count_n, uint* min_count_d)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->GetMinFrameCount(min_count_n, min_count_d);

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting minimum frame count");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services:: OdeTriggerFrameCountMinSet(const char* name, 
        uint min_count_n, uint min_count_d)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            // TODO: validate the min values for in-range
            pOdeTrigger->SetMinFrameCount(min_count_n, min_count_d);

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting minimum frame count");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerInferDoneOnlyGet(const char* name, 
        boolean* inferDoneOnly)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            *inferDoneOnly = pOdeTrigger->GetInferDoneOnlySetting();
            
            LOG_INFO("Trigger '" << name << "' set Inference Done Only = " 
                << inferDoneOnly << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting Inference Done Only");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerInferDoneOnlySet(const char* name, 
        boolean inferDoneOnly)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);

            pOdeTrigger->SetInferDoneOnlySetting(inferDoneOnly);
            
            LOG_INFO("Trigger '" << name << "' set Inference Done Only = " 
                << inferDoneOnly << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting Inference Done Only");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerIntervalGet(const char* name, uint* interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            *interval = pOdeTrigger->GetInterval();
            
            LOG_INFO("Trigger '" << name << "' returned Interval = " 
                << *interval << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting Interval");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::OdeTriggerIntervalSet(const char* name, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->SetInterval(interval);

            LOG_INFO("Trigger '" << name << "' set Interval = " 
                << interval << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception setting Interval");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::OdeTriggerActionAdd(const char* name, const char* action)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, action);

            // Note: Actions can be added when in use, i.e. shared between
            // multiple ODE Triggers

            if (!m_odeTriggers[name]->AddAction(m_odeActions[action]))
            {
                LOG_ERROR("ODE Trigger '" << name
                    << "' failed to add ODE Action '" << action << "'");
                return DSL_RESULT_ODE_TRIGGER_ACTION_ADD_FAILED;
            }
            LOG_INFO("ODE Action '" << action
                << "' was added to ODE Trigger '" << name << "' successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name
                << "' threw exception adding ODE Action '" << action << "'");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeTriggerActionRemove(const char* name, const char* action)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, action);

            if (!m_odeActions[action]->IsParent(m_odeTriggers[name]))
            {
                LOG_ERROR("ODE Action'" << action << 
                    "' is not in use by ODE Trigger '" << name << "'");
                return DSL_RESULT_ODE_TRIGGER_ACTION_NOT_IN_USE;
            }

            if (!m_odeTriggers[name]->RemoveAction(m_odeActions[action]))
            {
                LOG_ERROR("ODE Trigger '" << name
                    << "' failed to remove ODE Action '" << action << "'");
                return DSL_RESULT_ODE_TRIGGER_ACTION_REMOVE_FAILED;
            }
            LOG_INFO("ODE Action '" << action
                << "' was removed from ODE Trigger '" << name << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name
                << "' threw exception remove ODE Action '" << action << "'");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeTriggerActionRemoveAll(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);

            m_odeTriggers[name]->RemoveAllActions();

            LOG_INFO("All Events Actions removed from ODE Trigger '" << name << "' successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw an exception removing All Events Actions");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerAreaAdd(const char* name, const char* area)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_ODE_AREA_NAME_NOT_FOUND(m_odeAreas, area);

            // Note: Areas can be added when in use, i.e. shared between
            // multiple ODE Triggers

            if (!m_odeTriggers[name]->AddArea(m_odeAreas[area]))
            {
                LOG_ERROR("ODE Trigger '" << name
                    << "' failed to add ODE Area '" << area << "'");
                return DSL_RESULT_ODE_TRIGGER_AREA_ADD_FAILED;
            }
            LOG_INFO("ODE Area '" << area
                << "' was added to ODE Trigger '" << name << "' successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name
                << "' threw exception adding ODE Area '" << area << "'");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeTriggerAreaRemove(const char* name, const char* area)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_ODE_AREA_NAME_NOT_FOUND(m_odeAreas, area);

            if (!m_odeAreas[area]->IsParent(m_odeTriggers[name]))
            {
                LOG_ERROR("ODE Area'" << area << 
                    "' is not in use by ODE Trigger '" << name << "'");
                return DSL_RESULT_ODE_TRIGGER_AREA_NOT_IN_USE;
            }

            if (!m_odeTriggers[name]->RemoveArea(m_odeAreas[area]))
            {
                LOG_ERROR("ODE Trigger '" << name
                    << "' failed to remove ODE Area '" << area << "'");
                return DSL_RESULT_ODE_TRIGGER_AREA_REMOVE_FAILED;
            }
            LOG_INFO("ODE Area '" << area
                << "' was removed from ODE Trigger '" << name << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name
                << "' threw exception remove ODE Area '" << area << "'");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeTriggerAreaRemoveAll(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);

            m_odeTriggers[name]->RemoveAllAreas();

            LOG_INFO("All Events Areas removed from ODE Trigger '" << name << "' successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw an exception removing All ODE Areas");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeTriggerAccumulatorAdd(const char* name, 
        const char* accumulator)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_ODE_ACCUMULATOR_NAME_NOT_FOUND(m_odeAccumulators, 
                accumulator);

            // check for in-use

            if (!m_odeTriggers[name]->AddAccumulator(m_odeAccumulators[accumulator]))
            {
                LOG_ERROR("ODE Trigger '" << name
                    << "' failed to add ODE Accumulator '" << accumulator << "'");
                return DSL_RESULT_ODE_TRIGGER_ACCUMULATOR_ADD_FAILED;
            }
            LOG_INFO("ODE Accumulator '" << accumulator
                << "' was added to ODE Trigger '" << name << "' successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name
                << "' threw exception adding ODE Accumulator '" << accumulator << "'");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerAccumulatorRemove(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);

            if (!m_odeTriggers[name]->RemoveAccumulator())
            {
                LOG_ERROR("ODE Trigger '" << name
                    << "' failed to remove ODE Accumulator");
                return DSL_RESULT_ODE_TRIGGER_ACCUMULATOR_REMOVE_FAILED;
            }
            LOG_INFO("ODE Accumulator was removed from ODE Trigger '" 
                << name << "' successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name
                << "' threw exception removing ODE Accumulator");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerHeatMapperAdd(const char* name, 
        const char* heatMapper)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_ODE_HEAT_MAPPER_NAME_NOT_FOUND(m_odeHeatMappers, 
                heatMapper);

            // check for in-use

            if (!m_odeTriggers[name]->AddHeatMapper(m_odeHeatMappers[heatMapper]))
            {
                LOG_ERROR("ODE Trigger '" << name
                    << "' failed to add ODE Heat-Mapper '" << heatMapper << "'");
                return DSL_RESULT_ODE_TRIGGER_HEAT_MAPPER_ADD_FAILED;
            }
            LOG_INFO("ODE Heat-Mapper '" << heatMapper
                << "' was added to ODE Trigger '" << name << "' successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name
                << "' threw exception adding ODE Heat-Mapper '" << heatMapper << "'");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerHeatMapperRemove(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);

            if (!m_odeTriggers[name]->RemoveHeatMapper())
            {
                LOG_ERROR("ODE Trigger '" << name
                    << "' failed to remove ODE Heat-Mapper");
                return DSL_RESULT_ODE_TRIGGER_HEAT_MAPPER_REMOVE_FAILED;
            }
            LOG_INFO("ODE Heat-Mapper was removed from ODE Trigger '" 
                << name << "' successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name
                << "' threw exception removing ODE Heat-Mapper");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerDelete(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            if (m_odeTriggers[name]->IsInUse())
            {
                LOG_INFO("ODE Trigger '" << name << "' is in use");
                return DSL_RESULT_ODE_TRIGGER_IN_USE;
            }
            m_odeTriggers.erase(name);

            LOG_INFO("ODE Trigger '" << name << "' deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw an exception on deletion");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            if (m_odeTriggers.empty())
            {
                return DSL_RESULT_SUCCESS;
            }
            for (auto const& imap: m_odeTriggers)
            {
                // In the case of Delete all
                if (imap.second->IsInUse())
                {
                    LOG_ERROR("ODE Trigger '" << imap.second->GetName() << "' is currently in use");
                    return DSL_RESULT_ODE_TRIGGER_IN_USE;
                }
            }
            m_odeTriggers.clear();

            LOG_INFO("All ODE Triggers deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger threw an exception on delete all");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }

    uint Services::OdeTriggerListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_odeTriggers.size();
    }

}