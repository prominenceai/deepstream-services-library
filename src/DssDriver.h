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

#include "DssConfig.h"
#include "DssPipeline.h"
#include "DsdApi.h"

#undef DSD_SOURCE_NEW
#define DSD_SOURCE_NEW(source, type, width, height, fps-n, fps-d) \
    Driver::GetDriver()->SourceNew(source, type, width, height, fps-n, fps-d)

#undef DSD_SOURCE_DELETE
#define DSD_SOURCE_DELETE(source) \
    Driver::GetDriver()->SourceDelete(source)

#undef DSD_STREAMMUX_NEW
#define DSD_STREAMMUX_NEW(streammux, live, batchSize, batchTimeout, width, height) \
    Driver::GetDriver()->StreamMuxNew(streammux, live, batchSize, batchTimeout, width, height)

#undef DSD_STREAMMUX_DELETE
#define DSD_STREAMMUX_DELETE(streammux) \
    Driver::GetDriver()->StreamMuxDelete(streammux)

#undef DSD_DISPLAY_NEW
#define DSD_DISPLAY_NEW(display, rows, columns, width, height) \
    Driver::GetDriver()->DisplayNew(display, rows, columns, width, height)

#undef DSD_DISPLAY_DELETE
#define DSD_DISPLAY_DELETE(display) \
    Driver::GetDriver()->DisplayDelete(display)

#undef DSD_GIE_NEW
#define DSD_GIE_NEW(gie, model, infer, batchSize, bc1, bc2, bc3, bc4) \
    Driver::GetDriver()->GieNew(gie, model, infer, batchSize, bc1, bc2, bc3, bc4)

#undef DSD_GIE_DELETE
#define DSD_GIE_DELETE(gie) \
    Driver::GetDriver()->GieDelete(gie)

#undef DSD_CONFIG_DELETE
#define DSD_CONFIG_DELETE(config) \
    Driver::GetDriver()->ConfigNew(config)

#undef DSD_CONFIG_NEW
#define DSD_CONFIG_NEW(config) \
    Driver::GetDriver()->ConfigNew(config)

#undef DSD_CONFIG_DELETE
#define DSD_CONFIG_DELETE(config) \
    Driver::GetDriver()->ConfigDelete(config)
        
#undef DSD_CONFIG_FILE_SAVE        
#define DSD_CONFIG_FILE_SAVE(config, file) \
    Driver::GetDriver()->ConfigFileSave(config, file)
    
#undef DSD_CONFIG_FILE_LOAD
#define DSD_CONFIG_FILE_LOAD(config, file) \
    Driver::GetDriver()->ConfigFileLoad(config, file)

#undef DSD_CONFIG_FILE_OVERWRITE
#define DSD_CONFIG_FILE_OVERWRITE(config, file) \
    Driver::GetDriver()->ConfigFileOverWrite(config, file)

#undef DSD_CONFIG_SOURCE_ADD
#define DSD_CONFIG_SOURCE_ADD(config, source) \
    Driver::GetDriver()->ConfigSourceAdd(config, source)

#undef DSD_CONFIG_SOURCE_REMOVE
#define DSD_CONFIG_SOURCE_REMOVE(config, source) \
    Driver::GetDriver()->ConfigSourceRemove(config, source)

#undef DSD_CONFIG_OSD_ADD
#define DSD_CONFIG_OSD_ADD(config, osd) \
    Driver::GetDriver()->ConfigOsdAdd(config, osd)

#undef DSD_CONFIG_OSD_REMOVE
#define DSD_CONFIG_OSD_REMOVE(config, osd) \
    Driver::GetDriver()->ConfigOsdRemove(config, osd)

#undef DSD_CONFIG_GIE_ADD
#define DSD_CONFIG_GIE_ADD(config, gie) \
    Driver::GetDriver()->ConfigGieAdd(config, gie)

#undef DSD_CONFIG_GIE_REMOVE
#define DSD_CONFIG_GIE_REMOVE(config, gie) \
    Driver::GetDriver()->ConfigGieRemove(config, gie)

#undef DSD_PIPELINE_NEW
#define DSD_PIPELINE_NEW(pipeline, config) \
    Driver::GetDriver()->PipelineNew(pipeline, config)
    
#undef DSD_PIPELINE_DELETE
#define DSD_PIPELINE_DELETE(pipeline) \
    Driver::GetDriver()->PipelineDelete(pipeline)
    
#undef DSD_PIPELINE_PAUSE
#define DSD_PIPELINE_PAUSE(pipeline) \
    Driver::GetDriver()->PipelinePause(pipeline) 

#undef DSD_PIPELINE_PLAY
#define DSD_PIPELINE_PLAY(pipeline) \
    Driver::GetDriver()->PipelinePlay(pipeline)

#undef DSD_PIPELINE_GET_STATE
#define DSD_PIPELINE_GET_STATE(pipeline) \
    Driver::GetDriver()->PipelineGetState(pipeline)


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
        
        #define DSD_SOURCE_NEW(source, type, width, height, fps-n, fps-d)
        #define DSD_SOURCE_DELETE(source)
        #define DSD_STREAMMUX_NEW(streammux, live, batchSize, batchTimeout, width, height)
        #define DSD_STREAMMUX_DELETE(streammux)
        #define DSD_DISPLAY_NEW(display, rows, columns, width, height)
        #define DSD_DISPLAY_DELETE(display)
        #define DSD_GIE_NEW(gie, model, infer, batchSize, bc1, bc2, bc3, bc4)
        #define DSD_GIE_DELETE(gie)
        #define DSD_CONFIG_NEW(config)
        #define DSD_CONFIG_DELETE(config)
        #define DSD_CONFIG_NEW(config)
        #define DSD_CONFIG_DELETE(config)
        #define DSD_CONFIG_FILE_LOAD(config, file)
        #define DSD_CONFIG_FILE_SAVE(config, file)
        #define DSD_CONFIG_FILE_OVERWRITE(config, file)
        #define DSD_CONFIG_SOURCE_ADD(config, source)
        #define DSD_CONFIG_SOURCE_REMOVE(config, source)
        #define DSD_CONFIG_OSD_ADD(config, osd)
        #define DSD_CONFIG_OSD_REMOVE(config, source)
        #define DSD_CONFIG_GIE_ADD(config, osd)
        #define DSD_CONFIG_GIE_REMOVE(config, source)
        #define DSD_PIPELINE_NEW(pipeline, config)
        #define DSD_PIPELINE_DELETE(pipeline)
        #define DSD_PIPELINE_PAUSE(pipeline)
        #define DSD_PIPELINE_PLAY(pipeline)
        #define DSD_PIPELINE_GET_STATE(pipeline)
                        
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
         * @brief pointer to a Application Context object
         * 
         */
        AppContext* m_pAppContext;

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
        AppContext* m_appContext;

    };  

    /**
     * @brief 
     * @param arg
     * @return 
     */
    static gboolean EventThread(gpointer arg);
    
    static gpointer XWindowEventThread(gpointer arg);

}


#endif // _DSS_DRIVER_H