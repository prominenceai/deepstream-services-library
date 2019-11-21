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
        
        /***************************************************************
         **** all Services defined below are documented in DslApi.h ****
         ***************************************************************/
        DslReturnType SetDumpDotDir(const wchar_t* dir);
        
        DslReturnType SourceCsiNew(const wchar_t* name, 
            uint width, uint height, uint fps_n, uint fps_d);
        
        DslReturnType SourceUriNew(const wchar_t* name, 
            const wchar_t* uri, uint cudadecMemType, uint intraDecode, uint dropFrameInterval);
            
        boolean SourceIsLive(const wchar_t* name);
        
        uint GetNumSourceInUse();
        
        uint GetNumSourceInUseMax();
        
        void SetNumSourceInUseMax(uint max);
        
        DslReturnType OverlaySinkNew(const wchar_t* name, 
            uint offsetX, uint offsetY, uint width, uint height);
        
        DslReturnType OsdNew(const wchar_t* name, boolean isClockEnabled);
        
        DslReturnType PrimaryGieNew(const wchar_t* name, const wchar_t* inferConfigFile,
            const wchar_t* modelEngineFile, uint interval, uint uniqueId);
        
        DslReturnType DisplayNew(const wchar_t* name, uint width, uint height);
        
        boolean ComponentIsInUse(const wchar_t* component);
        
        DslReturnType ComponentDelete(const wchar_t* component);

        DslReturnType ComponentDeleteMany(const wchar_t** components);

        DslReturnType ComponentDeleteAll();
        
        uint ComponentListSize();
        
        const wchar_t** ComponentListAll();
        
        DslReturnType PipelineNew(const wchar_t* pipeline);
        
        DslReturnType PipelineNewMany(const wchar_t** pipelines);
        
        DslReturnType PipelineDelete(const wchar_t* pipeline);
        
        DslReturnType PipelineDeleteMany(const wchar_t** pipelines);

        DslReturnType PipelineDeleteAll();

        uint PipelineListSize();
        
        const wchar_t** PipelineListAll();

        DslReturnType PipelineComponentAdd(const wchar_t* pipeline, const wchar_t* component);

        DslReturnType PipelineComponentAddMany(const wchar_t* pipeline, const wchar_t** components);
        
        DslReturnType PipelineComponentRemove(const wchar_t* pipeline, const wchar_t* component);

        DslReturnType PipelineComponentRemoveMany(const wchar_t* pipeline, const wchar_t** components);
        
        DslReturnType PipelineStreamMuxSetBatchProperties(const wchar_t* pipeline,
            uint batchSize, uint batchTimeout);

        DslReturnType PipelineStreamMuxSetOutputSize(const wchar_t* pipeline,
            uint width, uint height);

        DslReturnType PipelinePause(const wchar_t* pipeline);
        
        DslReturnType PipelinePlay(const wchar_t* pipeline);
        
        DslReturnType PipelineStop(const wchar_t* pipeline);
        
        DslReturnType PipelineGetState(const wchar_t* pipeline);
        
        DslReturnType PipelineDumpToDot(const wchar_t* pipeline, wchar_t* filename);
        
        DslReturnType PipelineDumpToDotWithTs(const wchar_t* pipeline, wchar_t* filename);
        
        DslReturnType PipelineStateChangeListenerAdd(const wchar_t* pipeline, 
            dsl_state_change_listener_cb listener, void* userdata);
        
        DslReturnType PipelineStateChangeListenerRemove(const wchar_t* pipeline, 
            dsl_state_change_listener_cb listener);
                        
        DslReturnType PipelineDisplayEventHandlerAdd(const wchar_t* pipeline, 
            dsl_display_event_handler_cb handler, void* userdata);

        DslReturnType PipelineDisplayEventHandlerRemove(const wchar_t* pipeline, 
            dsl_display_event_handler_cb handler);
        
        GMainLoop* GetMainLoopHandle()
        {
            LOG_FUNC();
            LOG_INFO("Returning Handle to MainLoop");
            
            return m_pMainLoop;
        }
                        
        /** 
         * @brief Handles all pending events
         * 
         * @return true if all events were handled succesfully
         */
        bool HandleXWindowEvents(); 

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
         * @brief handle to the single main loop
        */
        GMainLoop* m_pMainLoop;
            
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
        static std::vector<const wchar_t*> m_pipelineNames;
        
        /**
         * @brief map of all pipeline components creaated by the client, key=name
         */
        std::map <std::string, std::shared_ptr<Bintr>> m_components;
        
        /**
         * @brief used to return a list of all component names to the client
         */
        static std::vector<const wchar_t*> m_componentNames;
    };  

    static gboolean MainLoopThread(gpointer arg);
}


#endif // _DSL_DRIVER_H