/*
The MIT License

Copyright (c) 2019-Present, ROBERT HOWELL

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
#include "DslOdeType.h"
#include "DslOdeAction.h"

namespace DSL
{
    OdeAction::OdeAction(const char* name)
        : Base(name)
    {
        LOG_FUNC();

    }

    OdeAction::~OdeAction()
    {
        LOG_FUNC();

    }

    // ********************************************************************

    CallbackOdeAction::CallbackOdeAction(const char* name, 
        dsl_ode_occurrence_handler_cb clientHandler, void* clientData)
        : OdeAction(name)
        , m_clientHandler(clientHandler)
        , m_clientData(clientData)
    {
        LOG_FUNC();

    }

    CallbackOdeAction::~CallbackOdeAction()
    {
        LOG_FUNC();

    }
    
    void CallbackOdeAction::HandleOccurrence(DSL_BASE_PTR pBaseType,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        DSL_ODE_TYPE_PTR pOdeType = std::dynamic_pointer_cast<OdeType>(pBaseType);

        m_clientHandler(pOdeType->s_eventCount, pOdeType->m_wName.c_str(),
            pFrameMeta, pObjectMeta, m_clientData);
    }

    // ********************************************************************

    DisplayOdeAction::DisplayOdeAction(const char* name)
        : OdeAction(name)
    {
        LOG_FUNC();

    }

    DisplayOdeAction::~DisplayOdeAction()
    {
        LOG_FUNC();

    }

    void DisplayOdeAction::HandleOccurrence(DSL_BASE_PTR pBaseType,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {

    }

    // ********************************************************************

    LogOdeAction::LogOdeAction(const char* name)
        : OdeAction(name)
    {
        LOG_FUNC();

    }

    LogOdeAction::~LogOdeAction()
    {
        LOG_FUNC();

    }

    void LogOdeAction::HandleOccurrence(DSL_BASE_PTR pBaseType, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        DSL_ODE_TYPE_PTR pOdeType = std::dynamic_pointer_cast<OdeType>(pBaseType);
        
        LOG_INFO("Event Name      : " << pOdeType->GetName());
        LOG_INFO("  Unique Id     : " << pOdeType->s_eventCount);
        LOG_INFO("  NTP Timestamp : " << pFrameMeta->ntp_timestamp);
        LOG_INFO("  Source Data   : ------------------------");
        LOG_INFO("    Id          : " << pFrameMeta->source_id);
        LOG_INFO("    Frame       : " << pFrameMeta->frame_num);
        LOG_INFO("    Width       : " << pFrameMeta->source_frame_width);
        LOG_INFO("    Heigh       : " << pFrameMeta->source_frame_height );
        LOG_INFO("  Object Data   : ------------------------");
        LOG_INFO("    Class Id    : " << pObjectMeta->class_id );
        LOG_INFO("    Tracking Id : " << pObjectMeta->object_id);
        LOG_INFO("    Label       : " << pObjectMeta->obj_label);
        LOG_INFO("    Confidence  : " << pObjectMeta->confidence);
        LOG_INFO("    Left        : " << pObjectMeta->rect_params.left);
        LOG_INFO("    Top         : " << pObjectMeta->rect_params.top);
        LOG_INFO("    Width       : " << pObjectMeta->rect_params.width);
        LOG_INFO("    Height      : " << pObjectMeta->rect_params.height);
        LOG_INFO("  Min Criteria  : ------------------------");
        LOG_INFO("    Confidence  : " << pOdeType->m_minConfidence);
        LOG_INFO("    Frame Count : " << pOdeType->m_minFrameCountN
            << " out of " << pOdeType->m_minFrameCountD);
        LOG_INFO("    Width       : " << pOdeType->m_minWidth);
        LOG_INFO("    Height      : " << pOdeType->m_minHeight);
    }

    // ********************************************************************

    PrintOdeAction::PrintOdeAction(const char* name)
        : OdeAction(name)
    {
        LOG_FUNC();

    }

    PrintOdeAction::~PrintOdeAction()
    {
        LOG_FUNC();

    }

    void PrintOdeAction::HandleOccurrence(DSL_BASE_PTR pBaseType, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        DSL_ODE_TYPE_PTR pOdeType = std::dynamic_pointer_cast<OdeType>(pBaseType);
        
        std::cout << "Event Name      : " << pOdeType->GetName() << "\n";
        std::cout << "  Unique Id     : " << pOdeType->s_eventCount << "\n";
        std::cout << "  NTP Timestamp : " << pFrameMeta->ntp_timestamp << "\n";
        std::cout << "  Source Data   : ------------------------" << "\n";
        std::cout << "    Id          : " << pFrameMeta->source_id << "\n";
        std::cout << "    Frame       : " << pFrameMeta->frame_num << "\n";
        std::cout << "    Width       : " << pFrameMeta->source_frame_width << "\n";
        std::cout << "    Heigh       : " << pFrameMeta->source_frame_height << "\n";
        std::cout << "  Object Data   : ------------------------" << "\n";
        std::cout << "    Class Id    : " << pObjectMeta->class_id << "\n";
        std::cout << "    Tracking Id : " << pObjectMeta->object_id << "\n";
        std::cout << "    Label       : " << pObjectMeta->obj_label << "\n";
        std::cout << "    Confidence  : " << pObjectMeta->confidence << "\n";
        std::cout << "    Left        : " << pObjectMeta->rect_params.left << "\n";
        std::cout << "    Top         : " << pObjectMeta->rect_params.top << "\n";
        std::cout << "    Width       : " << pObjectMeta->rect_params.width << "\n";
        std::cout << "    Height      : " << pObjectMeta->rect_params.height << "\n";
        std::cout << "  Min Criteria  : ------------------------" << "\n";
        std::cout << "    Confidence  : " << pOdeType->m_minConfidence << "\n";
        std::cout << "    Frame Count : " << pOdeType->m_minFrameCountN
            << " out of " << pOdeType->m_minFrameCountD << "\n";
        std::cout << "    Width       : " << pOdeType->m_minWidth << "\n";
        std::cout << "    Height      : " << pOdeType->m_minHeight << "\n";
    }

    // ********************************************************************

    RedactOdeAction::RedactOdeAction(const char* name, double red, double green, double blue, double alpha)
        : OdeAction(name)
    {
        LOG_FUNC();

    }

    RedactOdeAction::~RedactOdeAction()
    {
        LOG_FUNC();

    }

    void RedactOdeAction::HandleOccurrence(DSL_BASE_PTR pBaseType,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // hide the OSD display text
        if (pObjectMeta->text_params.display_text)
        {
            pObjectMeta->text_params.set_bg_clr = 0;
            pObjectMeta->text_params.font_params.font_size = 0;
        }
        // shade in the background
        pObjectMeta->rect_params.border_width = 0;
        pObjectMeta->rect_params.has_bg_color = 1;
        pObjectMeta->rect_params.bg_color.red = m_backgroundColor.red;
        pObjectMeta->rect_params.bg_color.green = m_backgroundColor.green;
        pObjectMeta->rect_params.bg_color.blue = m_backgroundColor.blue;
        pObjectMeta->rect_params.bg_color.alpha = m_backgroundColor.alpha;
    }

    // ********************************************************************
    
    QueueOdeAction::QueueOdeAction(const char* name, uint maxSize)
        : OdeAction(name)
        , m_maxSize(maxSize)
    {
        LOG_FUNC();

    }

    QueueOdeAction::~QueueOdeAction()
    {
        LOG_FUNC();

    }

    void QueueOdeAction::HandleOccurrence(DSL_BASE_PTR pBaseType, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
//        if (m_odeQueue.size() == m_maxSize)
//        {
//            m_odeQueue.pop();
//        }
//        m_odeQueue.push(pOdeOccurrence);
    }
    
//    DSL_ODE_OCCURRENCE_PTR QueueOdeAction::Dequeue()
//    {
//        if (!m_odeQueue.size())
//        {
//            return nullptr;
//        }
//        DSL_ODE_OCCURRENCE_PTR pOde = m_odeQueue.front();
//        m_odeQueue.pop();
//        return pOde;
//    }
    
    uint QueueOdeAction::GetMaxSize()
    {
        LOG_FUNC();

        return 0;
//        return m_maxSize;
    }
    
    uint QueueOdeAction::GetCurrentSize()
    {
        LOG_FUNC();
        
//        return m_odeQueue.size();
        return 0;
    }
    
}    
