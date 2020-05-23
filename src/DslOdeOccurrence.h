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

#ifndef _DSL_ODE_OCCURRENCE_H
#define _DSL_ODE_OCCURRENCE_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslBase.h"

namespace DSL
{
    #define MAX_NAME_SIZE 32
    
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_ODE_OCCURRENCE_PTR std::shared_ptr<OdeOccurrence>
    #define DSL_ODE_OCCURRENCE_NEW() \
        std::shared_ptr<OdeOccurrence>(new OdeOccurrence())

    struct BoundingBox
    {
        /**
        * @brief width of the box in pixels.
        */
        uint width;   

        /**
        * @brief height of the box in pixels.
        */
        uint height;   

        /**
        * @brief left coordinate of the box in pixels.
        */
        uint left;   

        /**
        * @brief top coordinate of the box in pixels.
        */
        uint top;   
    };
    
    struct OdeOccurrence
    {
        /**
         * @brief Ode type Identifer
         */
        uint event_type; 
        
         /** 
         * @brief an array to store the name of the Object Detection Event
         */
        char event_name[MAX_NAME_SIZE];

        /** 
         * @brief Unique Event Occurrence Identifier
         */
        uint64_t event_id;

        /** 
         * @brief Network Time Protocol (NTP) timestamp
         */
        uint64_t ntp_timestamp;
        
        /** 
         * @brief source_id of the frame in the batch e.g. camera_id.
         */
        uint source_id;

        /**
         * @brief current frame number of the source 
         */
        uint frame_num;

        /** 
         * @brief width of the frame at the input of stream muxer
         */
        uint source_frame_width;

        /** 
         * @brief height of the frame at the input of stream muxer
         */
        uint source_frame_height;

        /** 
         * @brief Index of the object class infered by the primary detector/classifier 
         */
        uint class_id;

        /** 
         * @brief an array to store the string describing the class of the detected
         * object 
         */
        char object_label[MAX_LABEL_SIZE];

        /** 
         * @brief Unique ID for tracking the object. @ref UNTRACKED_OBJECT_ID indicates the
         * object has not been tracked 
         */
        uint64_t object_id;

        /** 
         * @brief confidence value of the object, set by inference component 
         */
        float confidence;
        
        /** 
         * @brief min confidence event criteria 
         */
        float min_confidence;
        
        /**
         * @breif objects bounding box coordinates and dimensions
         */ 
        BoundingBox box;
        
        /**
         * @breif objects bounding box event criteria
         */ 
        BoundingBox box_criteria;
        
        /**
         * @breif min number 'n' of 'd' frames event criteria
         */ 
        uint min_frame_count_n;

        /**
         * @breif min number 'n' of 'd' frames event criteria
         */ 
        uint min_frame_count_d;
    };
}

#endif // _DSL_ODE_OCCURRENCE_H
    