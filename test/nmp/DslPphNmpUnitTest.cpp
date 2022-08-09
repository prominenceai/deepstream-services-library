/*
The MIT License

Copyright (c) 2022, Prominence AI, Inc.

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

#include "catch.hpp"
#include "DslServices.h"
#include "DslPadProbeHandlerNmp.h"

static std::string name("nms-pph");

static std::string labelFile1(
    "/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/labels.txt");

static std::string labelFile2(
    "/opt/nvidia/deepstream/deepstream/samples/models/Secondary_CarColor/labels.txt");

static std::string noLabelFile;

using namespace DSL;

SCENARIO( "A new Non Maximum Processor PPH is created correctly", 
    "[NmsPph]" )
{
    GIVEN( "Attributes for a new NMP PPH" )
    {
        uint processMethod(DSL_NMP_PROCESS_METHOD_SUPRESS);
        uint matchMethod(DSL_NMP_MATCH_METHOD_IOS);
        float matchThreshold(0.5);

        WHEN( "When the NMP PPH is created with a lableFile - 1 label per line" )
        {
            DSL_PPH_NMP_PTR pNmsPph = DSL_PPH_NMP_NEW(name.c_str(), 
                labelFile1.c_str(), processMethod, matchMethod, matchThreshold);

            THEN( "All members are setup correctly" )
            {
                std::string retLabelFile(pNmsPph->GetLabelFile());
                REQUIRE( retLabelFile == labelFile1 );
                
                // 4 labels in "Primary_Detector/labels.txt"
                REQUIRE( pNmsPph->_getNumLabels() == 4 );
                
                REQUIRE( pNmsPph->GetProcessMethod() == processMethod );
                
                uint retMatchMethod(9);
                float retMatchThreshold(0);
                pNmsPph->GetMatchSettings(&retMatchMethod, &retMatchThreshold);
                REQUIRE( retMatchMethod == matchMethod );
                REQUIRE( retMatchThreshold == matchThreshold );
            }
        }
        WHEN( "When the NMP PPH is created with a lableFile - all labels on 1 line" )
        {
            DSL_PPH_NMP_PTR pNmsPph = DSL_PPH_NMP_NEW(name.c_str(), 
                labelFile2.c_str(), processMethod, matchMethod, matchThreshold);

            THEN( "All members are setup correctly" )
            {
                std::string retLabelFile(pNmsPph->GetLabelFile());
                REQUIRE( retLabelFile == labelFile2 );
                
                // 12 labels in "Secondary_CarColor/labels.txt"
                REQUIRE( pNmsPph->_getNumLabels() == 12 );
            }
        }
        WHEN( "When the NMP PPH is created without a lableFile" )
        {
            DSL_PPH_NMP_PTR pNmsPph = DSL_PPH_NMP_NEW(name.c_str(), 
                noLabelFile.c_str(), processMethod, matchMethod, matchThreshold);

            THEN( "All members are setup correctly" )
            {
                std::string retLabelFile(pNmsPph->GetLabelFile());
                REQUIRE( retLabelFile == noLabelFile );
                
                // one label for class agnostic nms"
                REQUIRE( pNmsPph->_getNumLabels() == 1 );
            }
        }
    }
}

SCENARIO( "A new Non Maximum Processor PPH can store/clear object metadata and predictions correctly", 
    "[NmsPph]" )
{
    GIVEN( "A new Non Maximum Processor PPH and 3 instances of object metadata" )
    {
        uint processMethod(DSL_NMP_PROCESS_METHOD_SUPRESS);
        uint matchMethod(DSL_NMP_MATCH_METHOD_IOS);
        float matchThreshold(0.5);

        NvDsObjectMeta objectMeta1 = {0};
        objectMeta1.class_id = 0; 
        objectMeta1.rect_params.left = 10;
        objectMeta1.rect_params.top = 10;
        objectMeta1.rect_params.width = 100;
        objectMeta1.rect_params.height = 100;

        NvDsObjectMeta objectMeta2 = {0};
        objectMeta2.class_id = 1; 
        objectMeta2.rect_params.left = 11;
        objectMeta2.rect_params.top = 11;
        objectMeta2.rect_params.width = 100;
        objectMeta2.rect_params.height = 100;

        NvDsObjectMeta objectMeta3 = {0};
        objectMeta3.class_id = 2; 
        objectMeta3.rect_params.left = 12;
        objectMeta3.rect_params.top = 12;
        objectMeta3.rect_params.width = 100;
        objectMeta3.rect_params.height = 100;
            
        WHEN( "The NMS PPH is created with a valid label file" )
        {
            DSL_PPH_NMP_PTR pNmsPph = DSL_PPH_NMP_NEW(name.c_str(), 
                labelFile1.c_str(), processMethod, matchMethod, matchThreshold);
                
            THEN( "The object metadata and predictions can be stored and cleared correctly" )
            {
                pNmsPph->_storeObjectMetaAndPrediction(&objectMeta1);
                pNmsPph->_storeObjectMetaAndPrediction(&objectMeta2);
                pNmsPph->_storeObjectMetaAndPrediction(&objectMeta3);

                auto objectMetaArray = pNmsPph->_getObjectMetaArray();
                
                // size must be equal to num labels
                REQUIRE( objectMetaArray.size() == pNmsPph->_getNumLabels() );
                REQUIRE( objectMetaArray[0].size() == 1 );
                REQUIRE( objectMetaArray[1].size() == 1 );
                REQUIRE( objectMetaArray[2].size() == 1 );
                REQUIRE( objectMetaArray[3].size() == 0 );
                
                auto predicationsArray = pNmsPph->_getPredictionsArray();
                
                REQUIRE( predicationsArray.size() == pNmsPph->_getNumLabels() );
                REQUIRE( predicationsArray[0].size() == 1 );
                REQUIRE( predicationsArray[1].size() == 1 );
                REQUIRE( predicationsArray[2].size() == 1 );
                REQUIRE( predicationsArray[3].size() == 0 );
                
                pNmsPph->_clearObjectMetaAndPredictions();

                objectMetaArray = pNmsPph->_getObjectMetaArray();
                
                REQUIRE( objectMetaArray.size() == pNmsPph->_getNumLabels() );
                REQUIRE( objectMetaArray[0].size() == 0 );
                REQUIRE( objectMetaArray[1].size() == 0 );
                REQUIRE( objectMetaArray[2].size() == 0 );
                REQUIRE( objectMetaArray[3].size() == 0 );

                predicationsArray = pNmsPph->_getPredictionsArray();
                
                REQUIRE( predicationsArray.size() == pNmsPph->_getNumLabels() );
                REQUIRE( predicationsArray[0].size() == 0 );
                REQUIRE( predicationsArray[1].size() == 0 );
                REQUIRE( predicationsArray[2].size() == 0 );
                REQUIRE( predicationsArray[3].size() == 0 );
            }
        }
        WHEN( "The NMS PPH is created without a label file - class agnostic" )
        {
            DSL_PPH_NMP_PTR pNmsPph = DSL_PPH_NMP_NEW(name.c_str(), 
                noLabelFile.c_str(), processMethod, matchMethod, matchThreshold);
                
            THEN( "The object metadata and predictions can be stored and cleared correctly" )
            {
                pNmsPph->_storeObjectMetaAndPrediction(&objectMeta1);
                pNmsPph->_storeObjectMetaAndPrediction(&objectMeta2);
                pNmsPph->_storeObjectMetaAndPrediction(&objectMeta3);

                auto objectMetaArray = pNmsPph->_getObjectMetaArray();
                
                // size must be equal to num labels
                REQUIRE( objectMetaArray.size() == pNmsPph->_getNumLabels() );
                REQUIRE( objectMetaArray[0].size() == 3 );

                pNmsPph->_clearObjectMetaAndPredictions();
                objectMetaArray = pNmsPph->_getObjectMetaArray();
                
                REQUIRE( objectMetaArray.size() == pNmsPph->_getNumLabels() );
                REQUIRE( objectMetaArray[0].size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Non Maximum Processor PPH can calculate the union of two bounding boxes", 
    "[NmsPph]" )
{
    GIVEN( "A new Non Maximum PPH" )
    {
        uint processMethod(DSL_NMP_PROCESS_METHOD_SUPRESS);
        uint matchMethod(DSL_NMP_MATCH_METHOD_IOS);
        float matchThreshold(0.5);

        DSL_PPH_NMP_PTR pNmsPph = DSL_PPH_NMP_NEW(name.c_str(), 
            labelFile1.c_str(), processMethod, matchMethod, matchThreshold);
            
        WHEN( "Two bboxes with the same coordinates are checked" )
        {
            // v = {x1, y1, x2, y2, confidence}
            const std::vector<float> bbox1 = {100, 100, 200, 200, 0.99};
            const std::vector<float> bbox2 = {100, 100, 200, 200, 0.77};

            THEN( "The resulting bbox union is equal to bbox1" )
            {
                auto bboxUnion = pNmsPph->_calculateBoxUnion(
                    bbox1, bbox2);
                    
                REQUIRE( bboxUnion == bbox1 );
            }
        }
        WHEN( "Two bboxes with overlaping coordinates are checked" )
        {
            const std::vector<float> bbox1 = {100, 100, 200, 200, 0.99};
            const std::vector<float> bbox2 = {200, 100, 300, 200, 0.77};

            THEN( "The resulting bbox union has the correct coordinates" )
            {
                const std::vector<float> expectedResult = {100, 100, 300, 200, 0.99};
                auto bboxUnion = pNmsPph->_calculateBoxUnion(
                    bbox1, bbox2);
                    
                REQUIRE( bboxUnion == expectedResult );
            }
        }
    }
}

static void remove_obj_meta_from_frame(NvDsFrameMeta * frame_meta,
        NvDsObjectMeta *obj_meta)
{
    std::cout << "remove_obj_meta_from_frame called" << std::endl;
    std::cout << "class id   :  " << obj_meta->class_id << std::endl;
    std::cout << "left       : " << obj_meta->rect_params.left << std::endl;
    std::cout << "top        : " << obj_meta->rect_params.top << std::endl;
    std::cout << "width      : " << obj_meta->rect_params.width << std::endl;
    std::cout << "height     : " << obj_meta->rect_params.height << std::endl;
    std::cout << "confidence : " << std::to_string(obj_meta->confidence) << std::endl;
    std::cout << std::endl;
}

SCENARIO( "A new Non Maximum Processor PPH can supress non maximum object meta", 
    "[NmsPph]" )
{
    GIVEN( "A new Non Maximum Processor PPH and 3 instances of object metadata" )
    {
        uint processMethod(DSL_NMP_PROCESS_METHOD_SUPRESS);
        float matchThreshold(0.5);

        NvDsFrameMeta frameMeta =  {0};

        NvDsObjectMeta objectMeta1 = {0};
        objectMeta1.class_id = 0; 
        objectMeta1.rect_params.left = 10;
        objectMeta1.rect_params.top = 10;
        objectMeta1.rect_params.width = 100;
        objectMeta1.rect_params.height = 100;
        objectMeta1.confidence = 0.9;

        NvDsObjectMeta objectMeta2 = {0};
        objectMeta2.class_id = 0; 
        objectMeta2.rect_params.left = 11;
        objectMeta2.rect_params.top = 11;
        objectMeta2.rect_params.width = 100;
        objectMeta2.rect_params.height = 100;
        objectMeta2.confidence = 0.8;

        NvDsObjectMeta objectMeta3 = {0};
        objectMeta3.class_id = 0; 
        objectMeta3.rect_params.left = 12;
        objectMeta3.rect_params.top = 12;
        objectMeta3.rect_params.width = 100;
        objectMeta3.rect_params.height = 100;
        objectMeta3.confidence = 0.7;
            
        WHEN( "The NMS PPH is created with a valid label file and match method = IOU" )
        {
            DSL_PPH_NMP_PTR pNmsPph = DSL_PPH_NMP_NEW(name.c_str(), 
                labelFile1.c_str(), processMethod, DSL_NMP_MATCH_METHOD_IOU, 
                matchThreshold);
                
            THEN( "The object metadata and predictions can be stored and cleared correctly" )
            {
                pNmsPph->_storeObjectMetaAndPrediction(&objectMeta1);
                pNmsPph->_storeObjectMetaAndPrediction(&objectMeta2);
                pNmsPph->_storeObjectMetaAndPrediction(&objectMeta3);
                
                pNmsPph->_processNonMaximumObjectMeta(remove_obj_meta_from_frame,
                    &frameMeta);
            }
        }
        WHEN( "The NMS PPH is created without a label file and match method = IOU" )
        {
            DSL_PPH_NMP_PTR pNmsPph = DSL_PPH_NMP_NEW(name.c_str(), 
                noLabelFile.c_str(), processMethod, DSL_NMP_MATCH_METHOD_IOU, 
                matchThreshold);
                
            THEN( "The object metadata and predictions can be stored and cleared correctly" )
            {
                pNmsPph->_storeObjectMetaAndPrediction(&objectMeta1);
                pNmsPph->_storeObjectMetaAndPrediction(&objectMeta2);
                pNmsPph->_storeObjectMetaAndPrediction(&objectMeta3);

                pNmsPph->_processNonMaximumObjectMeta(remove_obj_meta_from_frame,
                    &frameMeta);
            }
        }
        WHEN( "The NMS PPH is created with a valid label file and match method = IOS" )
        {
            DSL_PPH_NMP_PTR pNmsPph = DSL_PPH_NMP_NEW(name.c_str(), 
                labelFile1.c_str(), processMethod, DSL_NMP_MATCH_METHOD_IOS, 
                matchThreshold);
                
            THEN( "The object metadata and predictions can be stored and cleared correctly" )
            {
                pNmsPph->_storeObjectMetaAndPrediction(&objectMeta1);
                pNmsPph->_storeObjectMetaAndPrediction(&objectMeta2);
                pNmsPph->_storeObjectMetaAndPrediction(&objectMeta3);
                
                pNmsPph->_processNonMaximumObjectMeta(remove_obj_meta_from_frame,
                    &frameMeta);
            }
        }
        WHEN( "The NMS PPH is created without a label file and match method = IOS" )
        {
            DSL_PPH_NMP_PTR pNmsPph = DSL_PPH_NMP_NEW(name.c_str(), 
                noLabelFile.c_str(), processMethod, DSL_NMP_MATCH_METHOD_IOS, 
                matchThreshold);
                
            THEN( "The object metadata and predictions can be stored and cleared correctly" )
            {
                pNmsPph->_storeObjectMetaAndPrediction(&objectMeta1);
                pNmsPph->_storeObjectMetaAndPrediction(&objectMeta2);
                pNmsPph->_storeObjectMetaAndPrediction(&objectMeta3);

                pNmsPph->_processNonMaximumObjectMeta(remove_obj_meta_from_frame,
                    &frameMeta);
            }
        }
    }
}
