/*
The MIT License

Copyright (c) 2021-2022, Prominence AI, Inc.

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
#include "DslOdeTrackedObject.h"

using namespace DSL;

SCENARIO( "A TrackedObject is created correctly", "[TrackedObject]" )
{
    GIVEN( "Attributes for a new TrackedObject" ) 
    {
        uint frame_num(4321);
        // Object Meta test data
        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = 0;
        objectMeta.object_id = 1234; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;
        
        uint maxHistory(10);
        
        WHEN( "A TrackedObject is created" )
        {
            std::shared_ptr<TrackedObject> pTrackedObject = std::shared_ptr<TrackedObject>
                (new TrackedObject(objectMeta.object_id, frame_num));
                
            THEN( "All attributes are setup correctly" )
            {
                REQUIRE( pTrackedObject->m_trackingId == objectMeta.object_id );
                REQUIRE( pTrackedObject->m_frameNumber == frame_num );
                REQUIRE( pTrackedObject->m_maxHistory == 0 );
                REQUIRE( pTrackedObject->m_bboxTrace.size() == 0 );
            }
        }
        WHEN( "A TrackedObject is created" )
        {
            std::shared_ptr<TrackedObject> pTrackedObject = std::shared_ptr<TrackedObject>
                (new TrackedObject(objectMeta.object_id, frame_num, 
                    &objectMeta.rect_params, maxHistory));
                
            THEN( "All attributes are setup correctly" )
            {
                REQUIRE( pTrackedObject->m_trackingId == objectMeta.object_id );
                REQUIRE( pTrackedObject->m_frameNumber == frame_num );
                REQUIRE( pTrackedObject->m_maxHistory ==  maxHistory );
                REQUIRE( pTrackedObject->m_bboxTrace.size() == 1 );
                
                std::shared_ptr<std::vector<dsl_coordinate>> pTrace = 
                    pTrackedObject->GetTrace(DSL_BBOX_POINT_NORTH_WEST);
                    
                REQUIRE( pTrace->at(0).x == 10 );
                REQUIRE( pTrace->at(0).y == 10 );
            }
        }
    }
}

SCENARIO( "A TrackedObject generates the correct trace", "[TrackedObject]" )
{
    GIVEN( "A new TrackedObject" ) 
    {
        uint frame_num(4321);
        // Object Meta test data
        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = 0;
        objectMeta.object_id = 1234; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;

        uint maxHistory(10);
        
        std::shared_ptr<TrackedObject> pTrackedObject = std::shared_ptr<TrackedObject>
            (new TrackedObject(objectMeta.object_id, frame_num, 
                &objectMeta.rect_params, maxHistory));
        
        WHEN( "A TrackedObject is created" )
        {
            objectMeta.rect_params.left = 20;
            objectMeta.rect_params.top = 20;
            objectMeta.rect_params.width = 210;
            objectMeta.rect_params.height = 110;
            
            pTrackedObject->PushBbox(&objectMeta.rect_params);

            objectMeta.rect_params.left = 30;
            objectMeta.rect_params.top = 30;
            objectMeta.rect_params.width = 220;
            objectMeta.rect_params.height = 120;
            
            pTrackedObject->PushBbox(&objectMeta.rect_params);

            objectMeta.rect_params.left = 40;
            objectMeta.rect_params.top = 40;
            objectMeta.rect_params.width = 230;
            objectMeta.rect_params.height = 130;
            
            pTrackedObject->PushBbox(&objectMeta.rect_params);

            objectMeta.rect_params.left = 50;
            objectMeta.rect_params.top = 50;
            objectMeta.rect_params.width = 240;
            objectMeta.rect_params.height = 140;
            
            pTrackedObject->PushBbox(&objectMeta.rect_params);

            THEN( "All attributes are setup correctly" )
            {
                REQUIRE( pTrackedObject->m_trackingId == objectMeta.object_id );
                REQUIRE( pTrackedObject->m_frameNumber == frame_num );

                REQUIRE( pTrackedObject->m_bboxTrace.size() == 5 );
                
                std::shared_ptr<std::vector<dsl_coordinate>> pTrace = 
                    pTrackedObject->GetTrace(DSL_BBOX_POINT_NORTH_WEST);
                    
                std::vector<dsl_coordinate> expectedTrace = 
                    {{10,10},{20,20},{30,30},{40,40},{50,50}};

                for (auto i = 0; i < pTrace->size(); i++)
                {
                    REQUIRE( pTrace->at(i).x == expectedTrace.at(i).x );
                    REQUIRE( pTrace->at(i).y == expectedTrace.at(i).y );
                }
            }
        }
    }
}