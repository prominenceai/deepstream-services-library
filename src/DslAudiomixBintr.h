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

#ifndef _DSL_AUDIOMIX_BINTR_H
#define _DSL_AUDIOMIX_BINTR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslSourceBintr.h"

namespace DSL
{

    #define DSL_AUDIOMIX_PTR std::shared_ptr<AudiomixBintr>
    #define DSL_AUDIOMIX_NEW(name, parentBin, srcPadName) \
        std::shared_ptr<AudiomixBintr> \
           (new AudiomixBintr(name, parentBin, srcPadName))

    /**
     * @class AudiomixBintr
     * @brief Implements a Pipeline Audiomix class for audio and video.
     */
    class AudiomixBintr : public Bintr
    {
    public: 
    
        /**
         * @brief ctor for the AudiomixBintr class
         * @param[in] name unique name for the AudiomixBintr to creat.
         * @param[in] parentBin gst bin pointer for the parent PipelineSourcesBintr.
         * @param[in] ghostPadName name to assign the src ghost pad for the audiomix.
         */
        AudiomixBintr(const char* name, 
            GstObject* parentBin, const char* ghostPadName);

        /**
         * @brief dtor for the AudiomixBintr class.
         */
        ~AudiomixBintr();

        /**
         * @brief & operator for the AudiomixBintr class.
         * @return returns shared pointer to the Audiomix elementr.
         */
        DSL_ELEMENT_PTR Get()
        {
            return m_pAudiomix;
        }

        /**
         * @brief required by all Bintrs - single element, nothing to link.
         * @returns true on successful link, false otherwise.
         */
        bool LinkAll();
        
        /**
         * @brief required by all Bintrs - single element, nothing to link.
         */
        void UnlinkAll();

    private:
    
        /**
         * @brief Name for the src ghost pad, assigned by the parent.
         */
        std::string m_ghostPadName;

        /**
         * @brief NVIDIA Audiomix element for this AudiomixBintr.
         */
        DSL_ELEMENT_PTR m_pAudiomix;

    };

}
    
#endif // _DSL_AUDIOMIX_BINTR_H    
