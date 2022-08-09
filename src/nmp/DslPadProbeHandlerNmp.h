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
     * @brief Callback typedef used to abstract NVIDIA's Object Meta API
     * nvds_remove_obj_meta_from_frame(). See @a _processNonMaximumObjectMeta
     * @param[in] frame_meta A pointer to frame meta from which @a obj_meta
     * is to be removed.
     * @param[in] obj_meta A pointer to the object meta to be removed from
     * @a frame_meta
     */
    typedef void (*remove_obj_meta_from_frame_cb)(NvDsFrameMeta * frame_meta,
        NvDsObjectMeta *obj_meta);
    
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_PPH_NMP_PTR std::shared_ptr<NmpPadProbeHandler>
    #define DSL_PPH_NMP_NEW(name, labelFile, processMethod, matchMethod, \
        matchThreshold) \
        std::shared_ptr<NmpPadProbeHandler>(new NmpPadProbeHandler(name, \
            labelFile, processMethod, matchMethod, matchThreshold))

    //----------------------------------------------------------------------------------------------

    /**
     * @class NmpPadProbeHandler
     * @brief Pad Probe Handler to Perform Non-Maximum-Processing -- either '
     * suppression or merging on all bbox predictions for a specified class(s) 
     * using one of two methods. IoU, IoS.
     */
    class NmpPadProbeHandler : public PadProbeHandler
    {
    public: 
    
        /**
         * @brief ctor for the NMS Pad Probe Handler
         * @param[in] name unique name for the new Pad Probe Handler.
         * @param[in] label_file absolute or relative path to inference model 
         * label file. Set "label_file" to NULL to perform class agnostic NMP.
         * @param[in] process_method method of processing non-maximum predictions. 
         * One of DSL_NMP_PROCESS_METHOD_SUPRESS or DSL_NMP_PROCESS_METHOD_MERGE. 
         * @param[in] match_method method for object match determination, either 
         * DSL_NMP_MATCH_METHOD_IOU or DSL_NMP_MATCH_METHOD_IOS.
         * @param[in] match_threshold threshold for object match determination.
         */
        NmpPadProbeHandler(const char* name, const char* labelFile, 
            uint processMethod, uint matchMethod, float matchThreshold);

        /**
         * @brief dtor for the NMS Pad Probe Handler
         */
        ~NmpPadProbeHandler();
        
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
         * @brief Gets the current non-maximum processing method.
         * @return Either DSL_NMP_PROCESS_METHOD_SUPRESS or DSL_NMP_PROCESS_METHOD_MERGE.
         */
        uint GetProcessMethod();
        
        /**
         * @brief Sets the the non-maximum process method to use
         * @param[in] process_method method of processing non-maximum predictions. 
         * One of DSL_NMP_PROCESS_METHOD_SUPRESS or DSL_NMP_PROCESS_METHOD_MERGE. 
         */
        void SetProcessMethod(uint processMode);

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
        
        /**
         * @brief inline function to add (store) the object meta to the 
         * m_objectMetaArray and m_predictionsArray containers.
         * @param[in] pObjectMeta pointer to object meta structure to store.
         */
        void _storeObjectMetaAndPrediction(NvDsObjectMeta* pObjectMeta);
        
        /**
         * @brief inline function to process all non-maximum predictions
         * according to the current settings for m_processMethod, m_matchMethod,
         * and m_matchThreshold.
         * @param[in] removeObj callback funtion to be called on to remove
         * each non-maximum occurrence. Using a callback allows the unit test code
         * to provide a test stub removing the dependency on the DeepStream function.
         * @param[in] pFrameMeta frame-meta the contains all of the object-meta
         * structions to suppressed or merged.
         */
        void _processNonMaximumObjectMeta(remove_obj_meta_from_frame_cb removeObj,
            NvDsFrameMeta* pFrameMeta);
        
        /**
         * @brief inline function to clear the m_objectMetaArray and m_predictionsArray containers.
         */
        void _clearObjectMetaAndPredictions();
        
        /**
         * @brief inline function to calculate the union of two bounding boxes.
         * @param[in] box1 first box coordinates to use for union calculation.
         * @param[in] box2 second box coordinates to use for union calculation.
         * @return bbox coordinates for the union of the two input boxes.
         */
        std::vector<float> _calculateBoxUnion(
            const std::vector<float> &box1, const std::vector<float> &box2);
            
        /**
         * @brief inline "test" function to get the current number of labels 
         * found in the label file. 1 if class-id agnostic.
         * @return number of class labels in the provided model label file.
         */
        uint _getNumLabels()
        {
            return m_numLabels;
        }
        
        /**
         * @brief inline "test" function to retrieve the object metata array
         * for test verification puposes only.
         */
        std::vector<std::vector<NvDsObjectMeta*>> _getObjectMetaArray()
        {
            return m_objectMetaArray;
        }

        /**
         * @brief inline "test" function to retrieve the predictions array
         * for test verification puposes only.
         */
        std::vector<std::vector<std::vector<float>>> _getPredictionsArray()
        {
            return m_predictionsArray;
        }

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
         * @brief method of processing non-maximum predictions. 
         * One of DSL_NMP_PROCESS_METHOD_SUPRESS or DSL_NMP_PROCESS_METHOD_MERGE. 
         */
        uint m_processMethod;
        
        /**
         * @brief method for object match determination, either DSL_NMS_MATCH_METHOD_IOU 
         * or DSL_NMS_MATCH_METHOD_IOS.
         */
        uint m_matchMethod;
        
        /**
         * @brief threshold for object match determination.
         */
        float m_matchThreshold;
        
        /**
         * @brief stores the object-meta parsed from a single frame.
         */
        std::vector<std::vector<NvDsObjectMeta*>> m_objectMetaArray;
        
        /**
         * @brief stores the object meta coordinates as predictions 
         * in a 2-dimensional array, 1 array for each class id. 
         */
        std::vector<std::vector<std::vector<float>>> m_predictionsArray;
        
    };
        
}

#endif // _DSL_NMS_PAD_PROBE_HANDLER_H
        
