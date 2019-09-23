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

#ifndef _DSD_DRIVER_H
#define _DSD_DRIVER_H

#include "DsdPipelineBintr.h"
#include "DsdSourceBintr.h"
#include "DsdStreamMuxBintr.h"
#include "DsdSinkBintr.h"
#include "DsdOsdBintr.h"
#include "DsdGieBintr.h"
#include "DsdDisplayBintr.h"

typedef int DsdReturnType;

namespace DSD {
    
    enum compTypes
    {
        SOURCE_CAMERA_CSI = 0,
        OSD_WITH_CLOCK,
        SINK_OVERLAY,
        STREAM_MUTEX,
        GIE_CLASIFIER,
        TILED_DISPLAY
    };

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
        
        DsdReturnType SourceNew(const std::string& source, guint type, 
            gboolean live, guint width, guint height, guint fps_n, guint fps_d);
        
        DsdReturnType SourceDelete(const std::string& source);
        
        DsdReturnType SinkNew(const std::string& sink, guint displayId, guint overlayId,
            guint offsetX, guint offsetY, guint width, guint height);
        
        DsdReturnType SinkDelete(const std::string& sink);
        
        DsdReturnType StreamMuxNew(const std::string& streammux, gboolean live, 
            guint batchSize, guint batchTimeout, guint width, guint height);
        
        DsdReturnType StreamMuxDelete(const std::string& streammux);
        
        DsdReturnType DisplayNew(const std::string& display, 
            guint rows, guint columns, guint width, guint height);
        
        DsdReturnType DisplayDelete(const std::string& display);
        
        DsdReturnType GieNew(const std::string& gie, const std::string& configFilePath, 
            guint batchSize, guint interval, guint uniqueId, guint gpuId, const 
            std::string& modelEngineFile, const std::string& rawOutputDir);
        
        DsdReturnType GieDelete(const std::string& gie);
        
        DsdReturnType PipelineSourceAdd(const std::string& pipeline, const std::string& source);
        
        DsdReturnType PipelineSourceRemove(const std::string& pipeline, const std::string& source);
        
        DsdReturnType PipelineSinkAdd(const std::string& pipeline, const std::string& sink);
        
        DsdReturnType PipelineSinkRemove(const std::string& pipeline, const std::string& sink);
        
        DsdReturnType PipelineStreamMuxAdd(const std::string& pipeline, const std::string& streammux);
        
        DsdReturnType PipelineStreamMuxRemove(const std::string& pipeline, const std::string& streammux);

        DsdReturnType PipelineOsdAdd(const std::string& pipeline, const std::string& osd);
        
        DsdReturnType PipelineOsdRemove(const std::string& pipeline, const std::string& osd);
        
        DsdReturnType PipelineGieAdd(const std::string& pipeline, const std::string& gie);
        
        DsdReturnType PipelineGieRemove(const std::string& pipeline, const std::string& gie);
        
        DsdReturnType PipelineDisplayAdd(const std::string& pipeline, const std::string& display);
        
        DsdReturnType PipelineDisplayRemove(const std::string& pipeline, const std::string& display);
        
        DsdReturnType PipelineNew(const std::string& pipeline);
        
        DsdReturnType PipelineDelete(const std::string& pipeline);
        
        DsdReturnType PipelinePause(const std::string& pipeline);
        
        DsdReturnType PipelinePlay(const std::string& pipeline);
        
        DsdReturnType PipelineGetState(const std::string& pipeline);
                        
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
         * @brief mutex for all display critical code
        */
        GMutex m_displayMutex;
        
        /**
         * @brief handle to the X Window event thread, 
         * active for the life of the driver
        */

        /**
         * @brief a single display for the driver
        */
        Display* m_pXDisplay;
        
        GThread* m_pXWindowEventThread;
        

        std::map <std::string, Bintr*> m_allComps;
        
    };  

    static gboolean MainLoopThread(gpointer arg);

    /**
     * @brief 
     * @param arg
     * @return 
     */
    static gboolean EventThread(gpointer arg);
    
    static gpointer XWindowEventThread(gpointer arg);

}


#endif // _DSD_DRIVER_H