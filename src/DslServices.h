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
        DslReturnType SourceCsiNew(const char* name, 
            uint width, uint height, uint fps_n, uint fps_d);
        
        DslReturnType SourceUriNew(const char* name, 
            const char* uri, uint cudadecMemType, uint intraDecode, uint dropFrameInterval);
            
        DslReturnType SourcePause(const char* name);

        DslReturnType SourceResume(const char* name);

        boolean SourceIsLive(const char* name);
        
        uint GetNumSourceInUse();
        
        uint GetNumSourceInUseMax();
        
        void SetNumSourceInUseMax(uint max);
        
        DslReturnType PrimaryGieNew(const char* name, const char* inferConfigFile,
            const char* modelEngineFile, uint interval);
        
        DslReturnType SecondaryGieNew(const char* name, const char* inferConfigFile,
            const char* modelEngineFile, const char* inferOnGieName);
        
        DslReturnType DisplayNew(const char* name, uint width, uint height);
        
        DslReturnType DisplayDimensionsGet(const char* name, uint* width, uint* height);

        DslReturnType DisplayDimensionsSet(const char* name, uint width, uint height);

        DslReturnType DisplayTilesGet(const char* name, uint* cols, uint* rows);

        DslReturnType DisplayTilesSet(const char* name, uint cols, uint rows);
        
        DslReturnType OverlaySinkNew(const char* name, 
            uint offsetX, uint offsetY, uint width, uint height);
        
        DslReturnType OsdNew(const char* name, boolean isClockEnabled);

// TODO        
//        boolean ComponentIsInUse(const char* component);
        
        DslReturnType ComponentDelete(const char* component);

        DslReturnType ComponentDeleteAll();
        
        uint ComponentListSize();
        
        DslReturnType PipelineNew(const char* pipeline);
        
        DslReturnType PipelineDelete(const char* pipeline);
        
        DslReturnType PipelineDeleteAll();

        uint PipelineListSize();
        
        DslReturnType PipelineComponentAdd(const char* pipeline, const char* component);

        DslReturnType PipelineComponentRemove(const char* pipeline, const char* component);

        DslReturnType PipelineStreamMuxBatchPropertiesGet(const char* pipeline,
            uint* batchSize, uint* batchTimeout);

        DslReturnType PipelineStreamMuxBatchPropertiesSet(const char* pipeline,
            uint batchSize, uint batchTimeout);

        DslReturnType PipelineStreamMuxDimensionsGet(const char* pipeline,
            uint* width, uint* height);

        DslReturnType PipelineStreamMuxDimensionsSet(const char* pipeline,
            uint width, uint height);
            
        DslReturnType PipelineStreamMuxPaddingGet(const char* pipeline, boolean* enabled);

        DslReturnType PipelineStreamMuxPaddingSet(const char* pipeline, boolean enabled);

        DslReturnType PipelinePause(const char* pipeline);
        
        DslReturnType PipelinePlay(const char* pipeline);
        
        DslReturnType PipelineStop(const char* pipeline);
        
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
         * @brief map of all pipeline components creaated by the client, key=name
         */
        std::map <std::string, std::shared_ptr<Bintr>> m_components;
    };  

    static gboolean MainLoopThread(gpointer arg);
}


#endif // _DSL_DRIVER_H