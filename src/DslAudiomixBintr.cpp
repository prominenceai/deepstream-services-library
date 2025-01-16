/*
The MIT License

Copyright (c) 2025, Prominence AI, Inc.

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
#include "DslAudiomixBintr.h"
#include "DslServices.h"

namespace DSL
{
    AudiomixBintr::AudiomixBintr(const char* name, 
        GstObject* parentBin, const char* ghostPadName)
        : Bintr(name, parentBin)
        , m_ghostPadName(ghostPadName)
    {
        LOG_FUNC();

        // Single Audiomix element for all Sources 
        m_pAudiomix = DSL_ELEMENT_NEW("audiomixer", name);

        AddChild(m_pAudiomix);

        // Float the Audiomixer as a src Ghost Pad for this AudiomixBintr
        m_pAudiomix->AddGhostPadToParent("src", ghostPadName);

    }

    AudiomixBintr::~AudiomixBintr()
    {
        LOG_FUNC();

        m_pAudiomix->RemoveGhostPadFromParent(m_ghostPadName.c_str());
    }

   bool AudiomixBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("AudiomixBintr '" << GetName() 
                << "' is already linked");
            return false;
        }
        // single element, nothing to link
        m_isLinked = true;
        return true;
    }        

    void AudiomixBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("AudiomixBintr '" << GetName() << "' is not linked");
            return;
        }
        // single element, nothing to link
        m_isLinked = false;
    }

    
}


