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

#include "Dss.h"
#include "DssAppCtx.h"
#include "DssPipeline.h"

namespace DSS
{
 
    AppContext::AppContext(Config& config)
        : m_config(config)
        , m_pPipeline(NULL)
    {
        LOG_FUNC();
        
    }

    AppContext::~AppContext()
    {
        LOG_FUNC();
        
        if (m_pPipeline)
        {
            delete m_pPipeline;
        }
        
    }

//    bool AppContext::Configure(const std::string& cfgFilePathSpec)
//    {
//        LOG_FUNC();
//        
//        return m_pConfig->LoadFile(cfgFilePathSpec);
//    }

    bool AppContext::Update(Display *display)
    {
        LOG_FUNC();
                
        if (!m_config.IsTiledDisplayEnabled())
        {
            LOG_ERROR("Application has no tiled display");
            return false;
        }
                
        try 
        {
            m_pPipeline = new Pipeline();
            
        }
        catch(...)
        {
            return false;
        }
        
        m_config.ConfigureNewXWindows();
        
        return true;
    }
}