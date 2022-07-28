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
#include "DslPadProbeHandlerNms.h"
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
   
    NmsPadProbeHandler::NmsPadProbeHandler(const char* name,
        const char* labelFile, uint matchMethod, float matchThreshold)
        : PadProbeHandler(name)
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

    NmsPadProbeHandler::~NmsPadProbeHandler()
    {
        LOG_FUNC();
    }
    
    const char* NmsPadProbeHandler::GetLabelFile()
    {
        LOG_FUNC();

        return m_labelFile.c_str();
    }
    
    bool NmsPadProbeHandler::SetLabelFile(const char* labelFile)
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
            LOG_INFO("Setting NmsPadProbeHandler '" << GetName() 
                << "' to perform class agnostic NMS");
            m_numLabels = 1;
            return true;
        }
        
        std::ifstream labelStream(m_labelFile);
        std::string delim{';'};

        if (!labelStream.is_open())
        {
            LOG_ERROR("NmsPadProbeHandler '" << GetName() 
                << "' failed to open label file '" << m_labelFile << "'");
            return false;
        }
        if (!labelStream.good())
        {
            LOG_ERROR("NmsPadProbeHandler '" << GetName() 
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
        m_numLabels = m_classLabels.size();
        LOG_INFO("NmsPadProbeHandler '" << GetName() << "' found " << m_numLabels 
            << " labels in label-file '" << m_labelFile << "'");
        return true;
    }

    uint NmsPadProbeHandler::GetNumLabels()
    {
        LOG_FUNC();

        return m_numLabels;
    }
    
    void NmsPadProbeHandler::GetMatchSettings(uint* matchMethod, 
        float* matchThreshold)
    {
        LOG_FUNC();

        *matchMethod = m_matchMethod;
        *matchThreshold = m_matchThreshold;
    }
    
    void NmsPadProbeHandler::SetMatchSettings(uint matchMethod, 
        float matchThreshold)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_padHandlerMutex);

        m_matchMethod = matchMethod;
        m_matchThreshold = matchThreshold;
    }
    
    GstPadProbeReturn NmsPadProbeHandler::HandlePadData(GstPadProbeInfo* pInfo)
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
                NvDsMetaList* pObjectMetaList = pFrameMeta->obj_meta_list;
                std::vector<std::vector<NvDsObjectMeta*>> obj_array;
                std::vector<std::vector<std::vector<float>>> predictions;

                // [fix] parse the label file            
                // rjh num_labels = CLASS_AGNOSTIC ? 1 : 4;
                predictions.resize(m_numLabels);
                obj_array.resize(m_numLabels);

                for (int lb=0; lb<m_numLabels; lb++) {
                    predictions[lb].reserve(VECTOR_RESERVE_SIZE);
                    obj_array[lb].reserve(VECTOR_RESERVE_SIZE);
                }

                // For each detected object in the frame.
                while (pObjectMetaList)
                {
                    // Check for valid object data
                    NvDsObjectMeta* pObjectMeta = (NvDsObjectMeta*)(pObjectMetaList->data);

                    // Important - we need to advance the list pointer before removing.
                    pObjectMetaList = pObjectMetaList->next;

                    if (pObjectMeta != NULL)
                    {
                        // https://github.com/obss/sahi/blob/91a0becb0d86f0943c57f966a86a845a70c0eb77/sahi/postprocess/combine.py#L17-L40
                        // if class agnostic
                        if (m_numLabels = 1) {
                            obj_array[0].emplace_back(pObjectMeta);
                            predictions[0].emplace_back(std::vector<float>{
                                        pObjectMeta->rect_params.left,
                                        pObjectMeta->rect_params.top,
                                        pObjectMeta->rect_params.left + pObjectMeta->rect_params.width,
                                        pObjectMeta->rect_params.top + pObjectMeta->rect_params.height, 
                                        pObjectMeta->confidence,
                                        });
                        }
                        else {
                            obj_array[pObjectMeta->class_id].emplace_back(pObjectMeta);
                            predictions[pObjectMeta->class_id].emplace_back(std::vector<float>{
                                         pObjectMeta->rect_params.left,
                                         pObjectMeta->rect_params.top,
                                         pObjectMeta->rect_params.left + pObjectMeta->rect_params.width,
                                         pObjectMeta->rect_params.top + pObjectMeta->rect_params.height, 
                                         pObjectMeta->confidence,
                                         });

                        }
                        // nvds_remove_obj_meta_from_frame(pFrameMeta, pObjectMeta);
                    }
                }
                
                for (int lb=0; lb<m_numLabels; lb++) {
                    // print_2dVector(predictions);                            
                    // https://dpilger26.github.io/NumCpp/doxygen/html/classnc_1_1_nd_array.html#a9d7045ecdff86bac3306a8bfd9a787eb
                    if (predictions[lb].size() == 0)
                        continue;

                    nc::NdArray<float> nd_predictions{predictions[lb]};
        //            # we extract coordinates for every
        //            # prediction box present in P
        //            x1 = predictions[:, 0]
        //            y1 = predictions[:, 1]
        //            x2 = predictions[:, 2]
        //            y2 = predictions[:, 3]
                    
                    auto x1 = nd_predictions(nd_predictions.rSlice(), 0);
                    auto y1 = nd_predictions(nd_predictions.rSlice(), 1);
                    auto x2 = nd_predictions(nd_predictions.rSlice(), 2);
                    auto y2 = nd_predictions(nd_predictions.rSlice(), 3);
                    
                    // scores = predictions[:, 4]
                    auto scores = nd_predictions(nd_predictions.rSlice(), 4);

                    // areas = (x2 - x1) * (y2 - y1)
                    auto areas = (x2 - x1) * (y2 - y1);

                    // std::vector<size_t> order = argsort(obj_array);
                    // https://dpilger26.github.io/NumCpp/doxygen/html/classnc_1_1_nd_array.html#a1fb3a21ab9c10a2684098df919b5b440
                    
                    // # sort the prediction boxes in P
                    // # according to their confidence scores
                    // order = scores.argsort()
                    std::vector<uint32_t> order = argsort(scores.toStlVector());
                    
                    nc::NdArray<nc::uint32> nd_order{order};
                    // std::sort(obj_array.begin(), obj_array.end(), confidence_compare);
                    
                    // # initialise an empty list for
                    // # filtered prediction boxes
                    // keep = []

                    std::vector<unsigned int> keep, remove;
                    
                    // return DSL_PAD_PROBE_OK;

                    //while (order.size() > 0) {
                    while (nc::shape(nd_order).size() > 0) {
                        // auto idx = order.back();

                        auto idx = nd_order[-1];
                        
                        // order.pop_back();
                        nd_order = nd_order(0, nc::Slice(0,-1));
                        if (nc::shape(nd_order).size() == 0) 
                            break;    
                        
        //                # select coordinates of BBoxes according to
        //                # the indices in order
        //                xx1 = torch.index_select(x1, dim=0, index=order)
        //                xx2 = torch.index_select(x2, dim=0, index=order)
        //                yy1 = torch.index_select(y1, dim=0, index=order)
        //                yy2 = torch.index_select(y2, dim=0, index=order)
                        
                        nc::NdArray<nc::uint32> index_other = nd_order;
                        nc::NdArray<nc::uint32> index{idx};
                        
                        auto xx1 = x1[index_other];
                        auto xx2 = x2[index_other];
                        auto yy1 = y1[index_other];
                        auto yy2 = y2[index_other];

        //                # find the coordinates of the intersection boxes
        //                xx1 = torch.max(xx1, x1[idx])
        //                yy1 = torch.max(yy1, y1[idx])
        //                xx2 = torch.min(xx2, x2[idx])
        //                yy2 = torch.min(yy2, y2[idx])

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

        //                # find height and width of the intersection boxes
        //                w = xx2 - xx1
        //                h = yy2 - yy1
                        
                        auto w = xx2 - xx1;
                        auto h = yy2 - yy1;
                        
        //                # take max with 0.0 to avoid negative w and h
        //                # due to non-overlapping boxes
        //                w = torch.clamp(w, min=0.0)
        //                h = torch.clamp(h, min=0.0)
                        
                        w = nc::clip(w, 0.0f, float(1e9));
                        h = nc::clip(h, 0.0f, float(1e9));
                        
        //                # find the intersection area
        //                inter = w * h
                        
                        auto inter = w * h;
                        
        //                # find the areas of BBoxes according the indices in order
        //                rem_areas = torch.index_select(areas, dim=0, index=order)i
                        
                        auto rem_areas = areas[index_other];
                        
                        if (m_matchMethod == DSL_NMS_MATCH_METHOD_IOU) {
        //                    if match_metric == "IOU":
        //                    # find the union of every prediction T in P
        //                    # with the prediction S
        //                    # Note that areas[idx] represents area of S
        //                    union = (rem_areas - inter) + areas[idx]
        //                    # find the IoU of every prediction in P with S
        //                    match_metric_value = inter / union
                            
                            auto _union = (rem_areas - inter) + areas[index].item();
                            auto match_metric_value = inter / _union;

                            // mask = match_metric_value < match_threshold
                            auto mask = match_metric_value < m_matchThreshold;                    
                            
                            auto rm_idx = 0;
                            for(auto it = mask.begin(); it != mask.end(); ++it, ++rm_idx) {
                                if (*it == 0)
                                    remove.emplace_back(index_other[rm_idx]);
                            }

                            nd_order = nd_order[mask];
                        }
                        else if (m_matchMethod == DSL_NMS_MATCH_METHOD_IOS) {
        //                    # find the smaller area of every prediction T in P
        //                    # with the prediction S
        //                    # Note that areas[idx] represents area of S
        //                    smaller = torch.min(rem_areas, areas[idx])
        //                    # find the IoU of every prediction in P with S
        //                    match_metric_value = inter / smaller
                            
                            auto smaller = rem_areas;
                            for(auto it = smaller.begin(); it != smaller.end(); ++it)
                                if (*it > areas[index].item())
                                    *it = areas[index].item();                    
                            auto match_metric_value = inter / smaller;

                            auto mask = match_metric_value < m_matchThreshold;                    
                            
                            auto rm_idx = 0;
                            for(auto it = mask.begin(); it != mask.end(); ++it, ++rm_idx) {
                                if (*it == 0)
                                    remove.emplace_back(index_other[rm_idx]);
                            }

                            nd_order = nd_order[mask];
                        }
                        else 
                        {
                            LOG_ERROR("Invalid NMS Match Method for Pad Probe Handler '"
                                << GetName());
                            return GST_PAD_PROBE_REMOVE; 
                        }            
                    }

                    for (auto &x: remove) {
                        nvds_remove_obj_meta_from_frame(pFrameMeta, obj_array[lb][x]);
                    }
                }
            
            }
        }
        return GST_PAD_PROBE_OK;
    }
     
}