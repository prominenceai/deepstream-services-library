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
        
        DslReturnType SetDumpDotDir(const char* dir);
        
        DslReturnType SourceCsiNew(const char* source, 
            uint width, uint height, uint fps_n, uint fps_d);
        
        DslReturnType SourceUriNew(const char* source, 
            const char* uri, uint cudadecMemType, uint intraDecode);
            
        boolean SourceIsLive(const char* source);
        
        uint GetNumSourceInUse();
        
        uint GetNumSourceInUseMax();
        
        void SetNumSourceInUseMax(uint max);
        
        DslReturnType StreamMuxNew(const char* streammux, boolean live, 
            uint batchSize, uint batchTimeout, uint width, uint height);
        
        DslReturnType SinkNew(const char* sink, uint displayId, uint overlayId,
            uint offsetX, uint offsetY, uint width, uint height);
        
        DslReturnType OsdNew(const char* osd, boolean isClockEnabled);
        
        DslReturnType GieNew(const char* gie, const char* configFilePath, 
            uint batchSize, uint interval, uint uniqueId, uint gpuId, 
            const char* modelEngineFile, const char* rawOutputDir);
        
        DslReturnType DisplayNew(const char* display, uint width, uint height);
        
        boolean ComponentIsInUse(const char* component);
        
        DslReturnType ComponentDelete(const char* component);

        DslReturnType ComponentDeleteMany(const char** components);

        DslReturnType ComponentDeleteAll();
        
        uint ComponentListSize();
        
        const char** ComponentListAll();
        
        DslReturnType PipelineNew(const char* pipeline);
        
        DslReturnType PipelineNewMany(const char** pipelines);
        
        DslReturnType PipelineDelete(const char* pipeline);
        
        DslReturnType PipelineDeleteMany(const char** pipelines);

        DslReturnType PipelineDeleteAll();

        uint PipelineListSize();
        
        const char** PipelineListAll();

        DslReturnType PipelineComponentAdd(const char* pipeline, const char* component);

        DslReturnType PipelineComponentAddMany(const char* pipeline, const char** components);
        
        DslReturnType PipelineComponentRemove(const char* pipeline, const char* component);

        DslReturnType PipelineComponentRemoveMany(const char* pipeline, const char** components);
        
        DslReturnType PipelineStreamMuxSetBatchProperties(const char* pipeline,
            uint batchSize, uint batchTimeout);

        DslReturnType PipelineStreamMuxSetOutputSize(const char* pipeline,
            uint width, uint height);

        DslReturnType PipelinePause(const char* pipeline);
        
        DslReturnType PipelinePlay(const char* pipeline);
        
        DslReturnType PipelineGetState(const char* pipeline);
        
        DslReturnType PipelineDumpToDot(const char* pipeline, char* filename);
        
        DslReturnType PipelineDumpToDotWithTs(const char* pipeline, char* filename);
        
        DslReturnType PipelineStateChangeListenerAdd(const char* pipeline, 
            dsl_state_change_listener_cb listener, void* userdata);
        
        DslReturnType PipelineStateChangeListenerRemove(const char* pipeline, 
            dsl_state_change_listener_cb listener);
                        
        DslReturnType PipelineDisplayEventHandlerAdd(const char* pipeline, 
            dsl_display_event_handler_cb handler, void* userdata);

        DslReturnType PipelineDisplayEventHandlerRemove(const char* pipeline, 
            dsl_display_event_handler_cb handler);

                        
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
         * @brief maximum number of sources that can be in use at one time
         * Set to the default in service contructor, the value can be read
         * and updated as the first call to DSL.
         */
        uint m_numSourceInUseMax;
        
        /**
         * @brief map of all pipelines creaated by the client, key=name
         */
        std::map <std::string, std::shared_ptr<PipelineBintr>> m_pipelines;
        
        /**
         * @brief used to return a list of all pipeline names to the client
         */
        std::vector<const char*> m_pipelineNames;
        
        /**
         * @brief map of all pipeline components creaated by the client, key=name
         */
        std::map <std::string, std::shared_ptr<Bintr>> m_components;
        
        /**
         * @brief used to return a list of all component names to the client
         */
        std::vector<const char*> m_componentNames;
        
        static std::string m_configFileDir;
        
        static std::string m_modelFileDir;
        
        static std::string m_streamFileDir;
    };  

    static gboolean MainLoopThread(gpointer arg);
}


#endif // _DSL_DRIVER_H