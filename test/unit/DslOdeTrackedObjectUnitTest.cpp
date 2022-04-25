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

        std::string colorName  = "my-custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), 
            red, green, blue, alpha);
        
        uint maxHistory(10);
        
        WHEN( "A TrackedObject is created" )
        {
            std::shared_ptr<TrackedObject> pTrackedObject = std::shared_ptr<TrackedObject>
                (new TrackedObject(objectMeta.object_id, frame_num, 
                    (NvBbox_Coords*)&objectMeta.rect_params, pColor, 
                    DSL_DEFAULT_TRACKING_TRIGGER_MAX_TRACE_POINTS));
                
            THEN( "All attributes are setup correctly" )
            {
                REQUIRE( pTrackedObject->trackingId == objectMeta.object_id );
                REQUIRE( pTrackedObject->frameNumber == frame_num);

                DSL_RGBA_MULTI_LINE_PTR pTrace = 
                    pTrackedObject->GetTrace(DSL_BBOX_POINT_NORTH_WEST,
                        DSL_OBJECT_TRACE_TEST_METHOD_ALL_POINTS);
                    
                REQUIRE( pTrace->coordinates[0].x == 10 );
                REQUIRE( pTrace->coordinates[0].y == 10 );
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

        std::string colorName  = "my-custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), 
            red, green, blue, alpha);

        uint maxHistory(10);
        
        std::shared_ptr<TrackedObject> pTrackedObject = std::shared_ptr<TrackedObject>
            (new TrackedObject(objectMeta.object_id, frame_num, 
                (NvBbox_Coords*)&objectMeta.rect_params, pColor, maxHistory));
        
        WHEN( "A TrackedObject is updated" )
        {
            objectMeta.rect_params.left = 20;
            objectMeta.rect_params.top = 20;
            objectMeta.rect_params.width = 210;
            objectMeta.rect_params.height = 110;
            
            pTrackedObject->Update(1, (NvBbox_Coords*)&objectMeta.rect_params);

            objectMeta.rect_params.left = 30;
            objectMeta.rect_params.top = 30;
            objectMeta.rect_params.width = 220;
            objectMeta.rect_params.height = 120;
            
            pTrackedObject->Update(2, (NvBbox_Coords*)&objectMeta.rect_params);

            objectMeta.rect_params.left = 40;
            objectMeta.rect_params.top = 40;
            objectMeta.rect_params.width = 230;
            objectMeta.rect_params.height = 130;
            
            pTrackedObject->Update(3, (NvBbox_Coords*)&objectMeta.rect_params);

            objectMeta.rect_params.left = 50;
            objectMeta.rect_params.top = 50;
            objectMeta.rect_params.width = 240;
            objectMeta.rect_params.height = 140;
            
            pTrackedObject->Update(4, (NvBbox_Coords*)&objectMeta.rect_params);

            THEN( "All attributes are updated correctly" )
            {
                REQUIRE( pTrackedObject->trackingId == objectMeta.object_id );
                REQUIRE( pTrackedObject->frameNumber == 4);

                DSL_RGBA_MULTI_LINE_PTR pTrace = 
                    pTrackedObject->GetTrace(DSL_BBOX_POINT_NORTH_WEST,
                        DSL_OBJECT_TRACE_TEST_METHOD_ALL_POINTS);
                    
                std::vector<dsl_coordinate> expectedTrace = 
                    {{10,10},{20,20},{30,30},{40,40},{50,50}};

                for (auto i = 0; i < pTrace->num_coordinates; i++)
                {
                    REQUIRE( pTrace->coordinates[i].x == expectedTrace.at(i).x );
                    REQUIRE( pTrace->coordinates[i].y == expectedTrace.at(i).y );
                }
            }
        }
    }
}

SCENARIO( "A TrackedObjects Container is created correctly", "[TrackedObject]" )
{
    GIVEN( "Attributes for a new TrackedObjects container" ) 
    {
        uint maxTracePoints(10);

        WHEN( "A TrackedObjects container is creted" )
        {
            std::shared_ptr<TrackedObjects>pTrackedObjectsPerSource = 
                std::shared_ptr<TrackedObjects>(new TrackedObjects(
                    maxTracePoints));
        
            THEN( "All attributes are setup correctly" )
            {
                // Empty container should fail to find or update a source
                REQUIRE( pTrackedObjectsPerSource->GetObject(0,0) == nullptr );
                REQUIRE( pTrackedObjectsPerSource->IsTracked(0,0) == false );
            }
        }
    }
}

SCENARIO( "A TrackedObjects Container adds a Tracked Object correctly", "[TrackedObject]" )
{
    GIVEN( "A new TrackedObjects container" ) 
    {
        NvDsFrameMeta frameMeta =  {0};
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.frame_num = 1;
        frameMeta.source_id = 987;

        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = 432;
        objectMeta.object_id = 123;
        objectMeta.rect_params.left = 20;
        objectMeta.rect_params.top = 20;
        objectMeta.rect_params.width = 210;
        objectMeta.rect_params.height = 110;

        std::string colorName  = "my-custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), 
            red, green, blue, alpha);

        uint maxTracePoints(10);

        std::shared_ptr<TrackedObjects>pTrackedObjectsPerSource = 
            std::shared_ptr<TrackedObjects>(new TrackedObjects(
                maxTracePoints));

        WHEN( "An Object is added to be container" )
        {
            REQUIRE( pTrackedObjectsPerSource->Track(&frameMeta, 
                &objectMeta, pColor) != nullptr );

            // Second call for the same object must fail
            REQUIRE( pTrackedObjectsPerSource->Track(&frameMeta, 
                &objectMeta, pColor) == nullptr );
            
            THEN( "It's correctly returned on GetObject" )
            {
                REQUIRE( pTrackedObjectsPerSource->IsTracked(frameMeta.source_id,
                        objectMeta.object_id) == true );
                        
                std::shared_ptr<TrackedObject> pTrackedObject = 
                    pTrackedObjectsPerSource->GetObject(frameMeta.source_id,
                        objectMeta.object_id);
                        
                REQUIRE( pTrackedObject->trackingId == objectMeta.object_id );
                REQUIRE( pTrackedObject->frameNumber == frameMeta.frame_num );
            }
        }
    }
}

SCENARIO( "A TrackedObjects Container manages multiple Tracked Object correctly", "[TrackedObject]" )
{
    GIVEN( "A new TrackedObjects container" ) 
    {
        NvDsFrameMeta frameMeta =  {0};
        frameMeta.ntp_timestamp = INT64_MAX;
        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = 1;
        objectMeta.rect_params.left = 20;
        objectMeta.rect_params.top = 20;
        objectMeta.rect_params.width = 210;
        objectMeta.rect_params.height = 110;
        
        std::string colorName  = "my-custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), 
            red, green, blue, alpha);

        uint maxTracePoints(10);

        std::shared_ptr<TrackedObjects>pTrackedObjectsPerSource = 
            std::shared_ptr<TrackedObjects>(new TrackedObjects(
                maxTracePoints));

        WHEN( "Several Objects are added to be container" )
        {

            frameMeta.frame_num = 1;
            
            frameMeta.source_id = 1;
            objectMeta.object_id = 1;

            REQUIRE( pTrackedObjectsPerSource->Track(&frameMeta, 
                &objectMeta, pColor) != nullptr );
            REQUIRE( pTrackedObjectsPerSource->IsTracked(frameMeta.source_id,
                objectMeta.object_id) == true );

            frameMeta.source_id = 2;
            objectMeta.object_id = 2;

            REQUIRE( pTrackedObjectsPerSource->Track(&frameMeta, 
                &objectMeta, pColor) != nullptr );
            REQUIRE( pTrackedObjectsPerSource->IsTracked(frameMeta.source_id,
                objectMeta.object_id) == true );

            frameMeta.source_id = 3;
            objectMeta.object_id = 3;

            REQUIRE( pTrackedObjectsPerSource->Track(&frameMeta, 
                &objectMeta, pColor) != nullptr );
            REQUIRE( pTrackedObjectsPerSource->IsTracked(frameMeta.source_id,
                objectMeta.object_id) == true );

            
            THEN( "It's objects can be updated correctly" )
            {
                        
                uint newFrameNumber = 2;

                frameMeta.source_id = 1;
                objectMeta.object_id = 1;
                
                std::shared_ptr<TrackedObject> pTrackedObject = 
                    pTrackedObjectsPerSource->GetObject(frameMeta.source_id,
                        objectMeta.object_id);
                        
                pTrackedObject->Update(newFrameNumber, 
                    (NvBbox_Coords*)&objectMeta.rect_params); 

                frameMeta.source_id = 2;
                objectMeta.object_id = 2;

                pTrackedObject = pTrackedObjectsPerSource->GetObject(frameMeta.source_id,
                        objectMeta.object_id);
                        
                pTrackedObject->Update(newFrameNumber, 
                    (NvBbox_Coords*)&objectMeta.rect_params);
                    
                frameMeta.source_id = 3;
                objectMeta.object_id = 3;

                pTrackedObject = pTrackedObjectsPerSource->GetObject(frameMeta.source_id,
                    objectMeta.object_id);
                        
                pTrackedObject->Update(newFrameNumber, 
                    (NvBbox_Coords*)&objectMeta.rect_params);

                // All should still be tracked after purging with the current frame number
                
                pTrackedObjectsPerSource->Purge(newFrameNumber);
                
                frameMeta.source_id = 1;
                objectMeta.object_id = 1;
                REQUIRE( pTrackedObjectsPerSource->IsTracked(frameMeta.source_id,
                        objectMeta.object_id) == true );
                
                frameMeta.source_id = 2;
                objectMeta.object_id = 2;
                REQUIRE( pTrackedObjectsPerSource->IsTracked(frameMeta.source_id,
                        objectMeta.object_id) == true );
                
                frameMeta.source_id = 3;
                objectMeta.object_id = 3;
                REQUIRE( pTrackedObjectsPerSource->IsTracked(frameMeta.source_id,
                        objectMeta.object_id) == true );

                // All should be purged on the next frame if not uptdate
                
                pTrackedObjectsPerSource->Purge(newFrameNumber+1);
                
                frameMeta.source_id = 1;
                objectMeta.object_id = 1;
                REQUIRE( pTrackedObjectsPerSource->IsTracked(frameMeta.source_id,
                        objectMeta.object_id) == false );
                
                frameMeta.source_id = 2;
                objectMeta.object_id = 2;
                REQUIRE( pTrackedObjectsPerSource->IsTracked(frameMeta.source_id,
                        objectMeta.object_id) == false );
                
                frameMeta.source_id = 3;
                objectMeta.object_id = 3;
                REQUIRE( pTrackedObjectsPerSource->IsTracked(frameMeta.source_id,
                        objectMeta.object_id) == false );
            }
        }
    }
}

