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

#ifndef _DSL_DRIVER_H
#define _DSL_DRIVER_H

#include "DslPipelineBintr.h"
#include "DslSourceBintr.h"
#include "DslSinkBintr.h"
#include "DslOsdBintr.h"
#include "DslGieBintr.h"
#include "DslDisplayBintr.h"

typedef int DslReturnType;

namespace DSL {
    
    /**
     * @class Services
     * @brief Implements a singlton instance 
     */
    class Services
    {
    public:
    
        /** 
         * @brief Returns a pointer to this singleton
         * 
         * @return instance pointer to Services
         */
        static Services* GetServices();
        
        DslReturnType SourceCsiNew(const char* source, 
            guint width, guint height, guint fps_n, guint fps_d);
        
        DslReturnType SourceUriNew(const char* source, 
            const char* uri, guint cudadecMemType, guint intraDecode);
        
        DslReturnType StreamMuxNew(const char* streammux, gboolean live, 
            guint batchSize, guint batchTimeout, guint width, guint height);
        
        DslReturnType SinkNew(const char* sink, guint displayId, guint overlayId,
            guint offsetX, guint offsetY, guint width, guint height);
        
        DslReturnType OsdNew(const char* osd, gboolean isClockEnabled);
        
        DslReturnType GieNew(const char* gie, const char* configFilePath, 
            guint batchSize, guint interval, guint uniqueId, guint gpuId, 
            const char* modelEngineFile, const char* rawOutputDir);
        
        DslReturnType DisplayNew(const char* display, 
            guint rows, guint columns, guint width, guint height);
        
        DslReturnType ComponentDelete(const char* component);
        
        DslReturnType PipelineNew(const char* pipeline);
        
        DslReturnType PipelineDelete(const char* pipeline);
        
        DslReturnType PipelineComponentsAdd(const char* pipeline, const char** components);
        
        DslReturnType PipelineComponentsRemove(const char* pipeline, const char** components);
        
        DslReturnType PipelineStreamMuxPropertiesSet(const char* pipeline,
            gboolean areSourcesLive, guint batchSize, guint batchTimeout, guint width, guint height);

        DslReturnType PipelinePause(const char* pipeline);
        
        DslReturnType PipelinePlay(const char* pipeline);
        
        DslReturnType PipelineGetState(const char* pipeline);
                        
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
        Services();

        /**
         * @brief private dtor for this singleton class
         */
        ~Services();
        
        /**
         * @brief instance pointer for this singleton class
         */
        static Services* m_pInstatnce;
        
        /**
         * @brief mutex to prevent Services reentry
        */
        GMutex m_servicesMutex;

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
        

        std::map <std::string, std::shared_ptr<PipelineBintr>> m_pipelines;
        
        std::map <std::string, std::shared_ptr<Bintr>> m_components;
        
        static std::string m_configFileDir;
        
        static std::string m_modelFileDir;
        
        static std::string m_streamFileDir;
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


#endif // _DSL_DRIVER_H