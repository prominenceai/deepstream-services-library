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

    DslReturnType Services::OdeAccumulatorNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure Accumulator name uniqueness 
            if (m_odeAccumulators.find(name) != m_odeAccumulators.end())
            {   
                LOG_ERROR("ODE Accumulator name '" << name 
                    << "' is not unique");
                return DSL_RESULT_ODE_ACCUMULATOR_NAME_NOT_UNIQUE;
            }
            m_odeAccumulators[name] = DSL_ODE_ACCUMULATOR_NEW(name);
            
            LOG_INFO("New ODE Accumulator '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Accumulator '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_ACCUMULATOR_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeAccumulatorActionAdd(const char* name, 
        const char* action)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_ACCUMULATOR_NAME_NOT_FOUND(m_odeAccumulators, name);
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, action);

            // Note: Actions can be added when in use, i.e. shared between
            // multiple ODE Triggers and ODE Accumulators

            if (!m_odeAccumulators[name]->AddAction(m_odeActions[action]))
            {
                LOG_ERROR("ODE Accumulator '" << name
                    << "' failed to add ODE Action '" << action << "'");
                return DSL_RESULT_ODE_ACCUMULATOR_ACTION_ADD_FAILED;
            }
            LOG_INFO("ODE Action '" << action
                << "' was added to ODE Accumulator '" << name << "' successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Accumulator '" << name
                << "' threw exception adding ODE Action '" << action << "'");
            return DSL_RESULT_ODE_ACCUMULATOR_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeAccumulatorActionRemove(const char* name, 
        const char* action)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_ACCUMULATOR_NAME_NOT_FOUND(m_odeAccumulators, name);
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, action);

            if (!m_odeActions[action]->IsParent(m_odeAccumulators[name]))
            {
                LOG_ERROR("ODE Action'" << action << 
                    "' is not in use by ODE Accumulator '" << name << "'");
                return DSL_RESULT_ODE_ACCUMULATOR_ACTION_NOT_IN_USE;
            }

            if (!m_odeAccumulators[name]->RemoveAction(m_odeActions[action]))
            {
                LOG_ERROR("ODE Accumulator '" << name
                    << "' failed to remove ODE Action '" << action << "'");
                return DSL_RESULT_ODE_ACCUMULATOR_ACTION_REMOVE_FAILED;
            }
            LOG_INFO("ODE Action '" << action
                << "' was removed from ODE Accumulator '" 
                    << name << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Accumulator '" << name
                << "' threw exception remove ODE Action '" << action << "'");
            return DSL_RESULT_ODE_ACCUMULATOR_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeAccumulatorActionRemoveAll(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_ACCUMULATOR_NAME_NOT_FOUND(m_odeAccumulators, name);

            m_odeAccumulators[name]->RemoveAllActions();

            LOG_INFO("All Events Actions removed from ODE Accumulator '" 
                << name << "' successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Accumulator '" << name 
                << "' threw an exception removing All Events Actions");
            return DSL_RESULT_ODE_ACCUMULATOR_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeAccumulatorDelete(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_ACCUMULATOR_NAME_NOT_FOUND(m_odeAccumulators, name);
            
            if (m_odeAccumulators[name]->IsInUse())
            {
                LOG_INFO("ODE Accumulator '" << name << "' is in use");
                return DSL_RESULT_ODE_ACCUMULATOR_IN_USE;
            }
            m_odeAccumulators.erase(name);

            LOG_INFO("ODE Accumulator '" << name << "' deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Accumulator '" << name 
                << "' threw an exception on deletion");
            return DSL_RESULT_ODE_ACCUMULATOR_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeAccumulatorDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            if (m_odeAccumulators.empty())
            {
                return DSL_RESULT_SUCCESS;
            }
            for (auto const& imap: m_odeAccumulators)
            {
                // In the case of Delete all
                if (imap.second->IsInUse())
                {
                    LOG_ERROR("ODE Accumulator '" << imap.second->GetName() 
                        << "' is currently in use");
                    return DSL_RESULT_ODE_ACCUMULATOR_IN_USE;
                }
            }
            m_odeAccumulators.clear();

            LOG_INFO("All ODE Accumulators deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Accumulator API threw an exception on delete all");
            return DSL_RESULT_ODE_ACCUMULATOR_THREW_EXCEPTION;
        }
    }

    uint Services::OdeAccumulatorListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_odeAccumulators.size();
    }
    
}