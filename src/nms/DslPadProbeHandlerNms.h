/*
The MIT License

Copyright (c) 2019-2022, Prominence AI, Inc.

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

#ifndef _DSL_NMS_PAD_PROBE_HANDLER_H
#define _DSL_NMS_PAD_PROBE_HANDLER_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslElementr.h"
#include "DslPadProbeHandler.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_PPH_NMS_PTR std::shared_ptr<NmsPadProbeHandler>
    #define DSL_PPH_NMS_NEW(name, labelFile, matchMethod, matchThreshold) \
        std::shared_ptr<NmsPadProbeHandler>(new NmsPadProbeHandler(name, \
            labelFile, matchMethod, matchThreshold))

    //----------------------------------------------------------------------------------------------

    /**
     * @class NmsPadProbeHandler
     * @brief Pad Probe Handler to Perform Non-Maximum-Suppression on all bbox
     * perditions for a specified class(s) using one of two methods. IoU, IoS
     */
    class NmsPadProbeHandler : public PadProbeHandler
    {
    public: 
    
        /**
         * @brief ctor for the NMS Pad Probe Handler
         * @param[in] name unique name for the PPH
         * @param[in] 
         */
        NmsPadProbeHandler(const char* name, const char* labelFile, 
            uint matchMethod, float matchThreshold);

        /**
         * @brief dtor for the NMS Pad Probe Handler
         */
        ~NmsPadProbeHandler();
        
        /**
         * @brief Gets the current model label file in use by the Pad Probe Handler.
         * @return path to the model label file. NULL indicates class agnostic NMS. 
         */
        const char* GetLabelFile();
        
        /**
         * @brief Sets the label file for the Pad Probe Handler to use
         * @param[in] label_file absolute or relative path to the inference model 
         * label file to use. Set "label_file" to NULL to perform class agnostic NMS.
         */
        bool SetLabelFile(const char* labelFile);
        
        /**
         * @brief Gets the current number of labels found in the label file.
         * @return number of class labels in the provided model label file.
         */
        uint GetNumLabels();

        /**
         * @brief Gets the current object match determination settings in use
         * by the Pad Probe Handler.
         * @param[out] match_method current method of object match determination, 
         * either DSL_NMS_MATCH_METHOD_IOU or DSL_NMS_MATCH_METHOD_IOS.
         * @param[out] match_threshold current threshold for object match determination
         * currently in use.
         */
        void GetMatchSettings(uint* matchMethod, float* matchThreshold);
        
        /**
         * @brief Sets the object match determination settings for the Pad Probe
         * Handler to use
         * @param[in] match_method new method for object match determination, either 
         * DSL_NMS_MATCH_METHOD_IOU or DSL_NMS_MATCH_METHOD_IOS.
         * @param[in] match_threshold new threshold for object match determination.
         */
        void SetMatchSettings(uint matchMethod, float matchThreshold);

        /**
         * @brief ODE Pad Probe Handler
         * @param[in] pBuffer Pad buffer
         * @return GstPadProbeReturn see GST reference, one of [GST_PAD_PROBE_DROP, 
         * GST_PAD_PROBE_OK, GST_PAD_PROBE_REMOVE, GST_PAD_PROBE_PASS, 
         * GST_PAD_PROBE_HANDLED]
         */
        GstPadProbeReturn HandlePadData(GstPadProbeInfo* pInfo);
        
    private:
    
        /**
         * @brief absolute or relative path to the inference model label file to use.
         * If Empty string, NMS will be class agnostic NMS
         */
        std::string m_labelFile;
        
        /**
         * @brief container of class labels parsed from the client provided label file.
         */
        std::vector<std::string> m_classLabels;
        
        /**
         * @brief nuber of object labels in the user provided label-file.
         * value will be set to 1 if no label file is provided (i.e. agnostic nms). 
         */
        uint m_numLabels;
        
        /**
         * @brief method for object match determination, either DSL_NMS_MATCH_METHOD_IOU 
         * or DSL_NMS_MATCH_METHOD_IOS.
         */
        uint m_matchMethod;
        
        /**
         * @brief threshold for object match determination.
         */
        float m_matchThreshold;
    };
        
}

#endif // _DSL_NMS_PAD_PROBE_HANDLER_H
        
