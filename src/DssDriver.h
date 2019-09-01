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

#ifndef _DSS_DRIVER_H
#define _DSS_DRIVER_H

#include <gst/gst.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>


#include "deepstream_common.h"
#include "deepstream_config.h"
#include "deepstream_osd.h"
#include "deepstream_perf.h"
#include "deepstream_primary_gie.h"
#include "deepstream_sinks.h"
#include "deepstream_sources.h"
#include "deepstream_streammux.h"
#include "deepstream_tiled_display.h"
#include "deepstream_dsexample.h"
#include "deepstream_tracker.h"
#include "deepstream_secondary_gie.h"


namespace DSS {

    /**
     * @class Driver
     * @file  DssDriver.h
     * @brief Implements a singlton instance 
     */
    class Driver
    {
    public:
    
        /** 
         * @brief Returns a pointer to this singleton
         * 
         * @return instance pointer to Driver
         */
        static Driver* GetDriver();

        bool IsDisplayActive(){ return m_pDisplay; };
        
        /** 
         * @brief Handles all pending events
         * 
         * @return true if all events were handled succesfully
         */
        bool HandleXWindowEvents(); 

        /**
         * @brief handle to the single main loop
        */
        GMainLoop* m_pMainLoop;
                
            
    private:
        /**
         * @brief private ctor for this singleton class
         */
        Driver();

        /**
         * @brief private dtor for this singleton class
         */
        ~Driver();
        
        /**
         * @brief instance pointer for this singleton class
         */
        static Driver* m_pInstatnce;
        
        /**
         * @brief mutex to prevent driver reentry
        */
        GMutex m_driverMutex;

        /**
         * @brief a single display for the driver
        */
        Display* m_pDisplay;
        
        /**
         * @brief mutex for all display critical code
        */
        GMutex m_displayMutex;
        
        /**
         * @brief handle to the X Window event thread, 
         * active for the life of the driver
        */
        GThread* m_pXWindowEventThread;

        /**
         * @brief Only one application context at this time
        */
        AppContext m_appContext;

    };  

    static gboolean EventThread(gpointer arg);
    
    static gpointer XWindowEventThread(gpointer arg);

}


#endif // _DSS_DRIVER_H