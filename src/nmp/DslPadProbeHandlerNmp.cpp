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

#include <NumCpp.hpp>
//#include <numeric>
//#include <algorithm>

#include "Dsl.h"
#include "DslPadProbeHandlerNmp.h"
#include "DslBase.h"
#include "DslBintr.h"

namespace DSL
{
    #define VECTOR_RESERVE_SIZE 1000
    
    template<typename T>
    std::vector<uint32_t> argsort(const std::vector<T> &array)
    {
        std::vector<uint32_t> indices(array.size());
        
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
            [&array](int left, int right) -> bool
            {
                // sort indices according to corresponding array element
                return array[left] < array[right];
            });

        return indices;
    } 
   
    NmpPadProbeHandler::NmpPadProbeHandler(const char* name,
        const char* labelFile, uint processMethod, uint matchMethod, 
        float matchThreshold)
        : PadProbeHandler(name)
        , m_processMethod(processMethod)
        , m_matchMethod(matchMethod)
        , m_matchThreshold(matchThreshold)
    {
        LOG_FUNC();
        
        if (!SetLabelFile(labelFile))
        {
            throw;
        }
        // Enable now
        if (!SetEnabled(true))
        {
            throw;
        }
    }

    NmpPadProbeHandler::~NmpPadProbeHandler()
    {
        LOG_FUNC();
    }
    
    const char* NmpPadProbeHandler::GetLabelFile()
    {
        LOG_FUNC();

        return m_labelFile.c_str();
    }
    
    bool NmpPadProbeHandler::SetLabelFile(const char* labelFile)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_padHandlerMutex);
        
        // The labels can be listed in one of two ways
        // 1. multiple labels per line delimeted by ';'
        // 2. one label per line
        
        m_labelFile = labelFile;
        m_classLabels.clear();
        
        if (!m_labelFile.size())
        {
            LOG_INFO("Setting NmpPadProbeHandler '" << GetName() 
                << "' to perform class agnostic NMS");
            m_numLabels = 1;
        }
        else
        {
            
            std::ifstream labelStream(m_labelFile);
            std::string delim{';'};

            if (!labelStream.is_open())
            {
                LOG_ERROR("NmpPadProbeHandler '" << GetName() 
                    << "' failed to open label file '" << m_labelFile << "'");
                return false;
            }
            if (!labelStream.good())
            {
                LOG_ERROR("NmpPadProbeHandler '" << GetName() 
                    << "' opened label file '" << m_labelFile 
                    << "' in a bad state '" << labelStream.rdstate() << "'");
                return false;
            } 
            
            while (!labelStream.eof())
            {
                std::string line;

                std::getline(labelStream, line, '\n');
                if (line.empty())
                {
                    continue;
                }
                // if it's a single lable for this line without deliminator
                if (line.find(delim, 0) == std::string::npos)
                {
                    m_classLabels.push_back(line);
                    continue;
                }

                size_t pos(0), oldpos(0);
                
                while ((pos = line.find(delim, oldpos)) != std::string::npos)
                {
                    m_classLabels.push_back(line.substr(oldpos, pos - oldpos));
                    oldpos = pos + delim.length();
                }
                m_classLabels.push_back(line.substr(oldpos));
            }
            // Set the number of labels 
            m_numLabels = m_classLabels.size();
        }
        
        // Size the Predictions and Object Meta Arrays according to the number 
        // of class labels - 1 for class agnostic NMS, size of m_classLabels
        m_predictionsArray.resize(m_numLabels);
        m_objectMetaArray.resize(m_numLabels);

        for (int lb=0; lb<m_numLabels; lb++) {
            m_predictionsArray[lb].reserve(VECTOR_RESERVE_SIZE);
            m_objectMetaArray[lb].reserve(VECTOR_RESERVE_SIZE);
        }
        
        LOG_INFO("NmpPadProbeHandler '" << GetName() << "' found " << m_numLabels  
            << " labels in label-file '" << m_labelFile << "'");
        return true;
    }

    uint NmpPadProbeHandler::GetProcessMethod()
    {
        LOG_FUNC();

        return m_processMethod;
    }
    
    void NmpPadProbeHandler::SetProcessMethod(uint processMode)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_padHandlerMutex);

        m_processMethod = processMode;
    }
    
    void NmpPadProbeHandler::GetMatchSettings(uint* matchMethod, 
        float* matchThreshold)
    {
        LOG_FUNC();

        *matchMethod = m_matchMethod;
        *matchThreshold = m_matchThreshold;
    }
    
    void NmpPadProbeHandler::SetMatchSettings(uint matchMethod, 
        float matchThreshold)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_padHandlerMutex);

        m_matchMethod = matchMethod;
        m_matchThreshold = matchThreshold;
    }
    
    GstPadProbeReturn NmpPadProbeHandler::HandlePadData(GstPadProbeInfo* pInfo)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_padHandlerMutex);

        if (!m_isEnabled)
        {
            return GST_PAD_PROBE_OK;
        }
        GstBuffer* pGstBuffer = (GstBuffer*)pInfo->data;
    
        NvDsBatchMeta* pBatchMeta = gst_buffer_get_nvds_batch_meta(pGstBuffer);
        
        // For each frame in the batched meta data
        for (NvDsMetaList* pFrameMetaList = pBatchMeta->frame_meta_list; 
            pFrameMetaList; pFrameMetaList = pFrameMetaList->next)
        {
            // Check for valid frame data
            NvDsFrameMeta* pFrameMeta = (NvDsFrameMeta*)(pFrameMetaList->data);
            if (pFrameMeta != NULL)
            {
                // For each detected object in the frame.
                for (NvDsMetaList* pObjectMetaList = pFrameMeta->obj_meta_list; 
                    pObjectMetaList; pObjectMetaList = pObjectMetaList->next)
                {
                    // Store the object metadata and it bbox coordinates as 
                    // a unique prediction. 
                    _storeObjectMetaAndPrediction((NvDsObjectMeta*)
                        (pObjectMetaList->data));
                }
                // Note: we pass in the nvidia DS remove object function here. The 
                // unit test code will test/call the _processNonMaximumObjectMeta 
                // function using a test stub. This removes the dependecy on the 
                // nvidia function when called under test (calling the nvida function
                // with test object meta will result in a SIGSEGV
                _processNonMaximumObjectMeta(nvds_remove_obj_meta_from_frame,
                    pFrameMeta);
                    
                // Clear the 
                _clearObjectMetaAndPredictions();
            }
        }
        return GST_PAD_PROBE_OK;
    }
    
    inline void NmpPadProbeHandler::_storeObjectMetaAndPrediction(NvDsObjectMeta* pObjectMeta)
    {
        if (pObjectMeta == NULL)
        {
            LOG_ERROR("Pad Probe Handler '" << GetName() 
                << "' received invalid Object Metadata");
            return;
        }
        // if class agnostic
        if (m_numLabels == 1) {
            m_objectMetaArray[0].emplace_back(pObjectMeta);
            m_predictionsArray[0].emplace_back(std::vector<float>
            {
                pObjectMeta->rect_params.left,
                pObjectMeta->rect_params.top,
                pObjectMeta->rect_params.left + pObjectMeta->rect_params.width,
                pObjectMeta->rect_params.top + pObjectMeta->rect_params.height, 
                pObjectMeta->confidence,
            });
        }
        else {
            m_objectMetaArray[pObjectMeta->class_id].emplace_back(pObjectMeta);
            m_predictionsArray[pObjectMeta->class_id].emplace_back(std::vector<float>
            {
                pObjectMeta->rect_params.left,
                pObjectMeta->rect_params.top,
                pObjectMeta->rect_params.left + pObjectMeta->rect_params.width,
                pObjectMeta->rect_params.top + pObjectMeta->rect_params.height, 
                pObjectMeta->confidence,
            });
        }
    }     

    inline void NmpPadProbeHandler::_processNonMaximumObjectMeta(
        remove_obj_meta_from_frame_cb removeObj, NvDsFrameMeta* pFrameMeta)
    {
        for (int lb=0; lb<m_numLabels; lb++)
        {
            if (m_predictionsArray[lb].size() == 0)
                continue;

            nc::NdArray<float> nd_predictions{m_predictionsArray[lb]};

            std::unordered_map<int, std::vector<int>> keep_to_merge_list;
            
            // we extract coordinates for every
            // prediction box present in P
            
            auto x1 = nd_predictions(nd_predictions.rSlice(), 0);
            auto y1 = nd_predictions(nd_predictions.rSlice(), 1);
            auto x2 = nd_predictions(nd_predictions.rSlice(), 2);
            auto y2 = nd_predictions(nd_predictions.rSlice(), 3);
            
            auto scores = nd_predictions(nd_predictions.rSlice(), 4);

            auto areas = (x2 - x1) * (y2 - y1);

            // # sort the prediction boxes in P
            // # according to their confidence scores
            // order = scores.argsort()
            std::vector<uint32_t> order = argsort(scores.toStlVector());
            
            nc::NdArray<nc::uint32> nd_order{order};
            
            // # initialise an empty list for
            // # filtered prediction boxes

            std::vector<unsigned int> keep, remove;
            
            //while (order.size() > 0) {
            while (nc::shape(nd_order).size() > 0) {

                auto idx = nd_order[-1];
                
                // order.pop_back();
                nd_order = nd_order(0, nc::Slice(0,-1));
                if (nc::shape(nd_order).size() == 0) {
                    keep_to_merge_list[idx].emplace_back(idx);        
                    break;      
                }   
                
                // select coordinates of BBoxes according to
                // the indices in order
                
                nc::NdArray<nc::uint32> index_order = nd_order;
                nc::NdArray<nc::uint32> index{idx};
                
                auto xx1 = x1[index_order];
                auto xx2 = x2[index_order];
                auto yy1 = y1[index_order];
                auto yy2 = y2[index_order];

                // find the coordinates of the intersection boxes

                for(auto it = xx1.begin(); it != xx1.end(); ++it) 
                    if (*it < x1[index].item()) 
                        *it = x1[index].item();

                for(auto it = yy1.begin(); it != yy1.end(); ++it) 
                    if (*it < y1[index].item()) 
                        *it = y1[index].item();
                                
                for(auto it = xx2.begin(); it != xx2.end(); ++it) 
                    if (*it > x2[index].item()) 
                        *it = x2[index].item();

                for(auto it = yy2.begin(); it != yy2.end(); ++it) 
                    if (*it > y2[index].item()) 
                        *it = y2[index].item();

                // find height and width of the intersection boxes
                auto w = xx2 - xx1;
                auto h = yy2 - yy1;
                
                // take max with 0.0 to avoid negative w and h
                // due to non-overlapping boxes
                
                w = nc::clip(w, 0.0f, float(1e9));
                h = nc::clip(h, 0.0f, float(1e9));
                
                // find the intersection area - used by both methods
                auto intersection = w * h;
                
                // find the areas of BBoxes according the indices in order
                auto rem_areas = areas[index_order];
                
                nc::NdArray<bool> mask(false);
                
                if (m_matchMethod == DSL_NMP_MATCH_METHOD_IOU)
                {
                    // find the union of every prediction T in P
                    // with the prediction S
                    // Note that areas[idx] represents area of S
                    // find the IoU of every prediction in P with S
                    
                    auto _union = (rem_areas - intersection) + areas[index].item();
                    
                    mask = (intersection / _union) < m_matchThreshold;                    
                }
                else // (m_matchMethod == DSL_NMP_MATCH_METHOD_IOS)
                {
                    // find the smaller area of every prediction T in P
                    // with the prediction S
                    // Note that areas[idx] represents area of S
                    
                    auto smaller = rem_areas;
                    for(auto it = smaller.begin(); it != smaller.end(); ++it)
                        if (*it > areas[index].item())
                            *it = areas[index].item();           
 
                    mask = (intersection / smaller) < m_matchThreshold;                    
                    
                }

                auto rm_idx = 0;
		keep_to_merge_list[idx].emplace_back(idx);		
                for(auto it = mask.begin(); it != mask.end(); ++it, ++rm_idx)
                {
                    if (*it == 0)
                    {
                        if (m_processMethod == DSL_NMP_PROCESS_METHOD_MERGE)
                        {
                            keep_to_merge_list[idx].emplace_back(index_order[rm_idx]);
                        }
                        remove.emplace_back(index_order[rm_idx]);
                    }
                }

                nd_order = nd_order[mask];
            }

            if (m_processMethod == DSL_NMP_PROCESS_METHOD_MERGE)
            {
                for (auto it = keep_to_merge_list.begin(); 
                    it != keep_to_merge_list.end(); ++it)
                {
                    for (auto &merge_ind : it->second)
                        m_predictionsArray[lb][it->first] = 
                            _calculateBoxUnion(m_predictionsArray[lb][it->first], 
                                m_predictionsArray[lb][merge_ind]);

                    m_objectMetaArray[lb][it->first]->rect_params.left = 
                        m_predictionsArray[lb][it->first][0];
                    m_objectMetaArray[lb][it->first]->rect_params.top = 
                        m_predictionsArray[lb][it->first][1];
                    m_objectMetaArray[lb][it->first]->rect_params.width = 
                        m_predictionsArray[lb][it->first][2] - 
                        m_predictionsArray[lb][it->first][0];
                    m_objectMetaArray[lb][it->first]->rect_params.height = 
                        m_predictionsArray[lb][it->first][3] - 
                        m_predictionsArray[lb][it->first][1]; 
                }
            }
            for (auto &x: remove) 
            {
                removeObj(pFrameMeta, m_objectMetaArray[lb][x]);
            }
        }
    }    
    
    inline void NmpPadProbeHandler::_clearObjectMetaAndPredictions()
    {
        for (int lb=0; lb<m_numLabels; lb++) {
            m_predictionsArray[lb].clear();
            m_objectMetaArray[lb].clear();
        }
    }
    
    inline std::vector<float> NmpPadProbeHandler::_calculateBoxUnion(
        const std::vector<float> &box1, const std::vector<float> &box2)
    {
        float x1 = std::min(box1[0], box2[0]);
        float y1 = std::min(box1[1], box2[1]);
        float x2 = std::max(box1[2], box2[2]);
        float y2 = std::max(box1[3], box2[3]);
        return std::vector<float>{x1, y1, x2, y2, box1[4]};
    }    
}
