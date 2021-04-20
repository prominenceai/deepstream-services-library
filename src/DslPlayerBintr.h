/*
The MIT License

Copyright (c) 2019-2021, Prominence AI, Inc.

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

#ifndef _DSL_PLAYER_BINTR_H
#define _DSL_PLAYER_BINTR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslBintr.h"
#include "DslPipelineStateMgr.h"
#include "DslPipelineXWinMgr.h"
#include "DslSourceBintr.h"
#include "DslSinkBintr.h"

namespace DSL
{

    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_PLAYER_BINTR_PTR std::shared_ptr<PlayerBintr>
    #define DSL_PLAYER_BINTR_NEW(name, pSource, pSink) \
        std::shared_ptr<PlayerBintr>(new PlayerBintr(name, pSource, pSink))    
    
    class PlayerBintr : public Bintr, public PipelineStateMgr,
        public PipelineXWinMgr
    {
    public: 
    
        PlayerBintr(const char* name, 
            DSL_SOURCE_PTR pSource, DSL_SINK_PTR pSink);

        ~PlayerBintr();

        /**
         * @brief Links all Child Bintrs owned by this Player Bintr
         * @return True success, false otherwise
         */
        bool LinkAll();

        /**
         * @brief Unlinks all Child Bintrs owned by this Player Bintr
         * Calling UnlinkAll when in an unlinked state has no effect.
         */
        void UnlinkAll();
        
        /**
         * @brief Attempts to link all and play the PlayerBintr
         * @return true if able to play, false otherwise
         */
        bool Play();

        /**
         * @brief Schedules a Timer Callback to call HandlePause in the mainloop context
         * @return true if HandlePause schedule correctly, false otherwise 
         */
        bool Pause();

        /**
         * @brief Pauses the Player by setting its state to GST_STATE_PAUSED
         * Import: must be called in the mainloop's context, i.e. timer callback
         */
        void HandlePause();

        /**
         * @brief Schedules a Timer Callback to call HandleStop in the mainloop context
         * @return true if HandleStop schedule correctly, false otherwise 
         */
        bool Stop();
        
        /**
         * @brief Stops the Player by setting its state to GST_STATE_NULL
         * Import: must be called in the mainloop's context, i.e. timer callback
         */
        void HandleStop();
        
        /**
         * @brief Terminates the player on event of EOS or XWindow Delete
         * Import: must be called by the BusWatch or Event Handler context.
         */
        void HandleTermination();

        /**
         * @brief adds a callback to be notified on Player Termination event
         * @param[in] listener pointer to the client's function to call on Termination event
         * @param[in] clientData opaque pointer to client data passed into the listener function.
         * @return true on successful add, false otherwise
         */
        bool AddTerminationEventListener(dsl_player_termination_event_listener_cb listener, void* clientData);

        /**
         * @brief removes a previously added callback
         * @param[in] listener pointer to the client's function to remove
         * @return true on successful remove, false otherwise
         */
        bool RemoveTerminationEventListener(dsl_player_termination_event_listener_cb listener);
        
    private:

        /**
         * @brief Mutex to protect the async GCond used to synchronize
         * the Application thread with the mainloop context on
         * asynchronous change of pipeline state.
         */
        GMutex m_asyncCommMutex;
        
        /**
         * @brief Condition used to block the application context while waiting
         * for a Pipeline change of state to be completed in the mainloop context
         */
        GCond m_asyncCondition;

        /**
         * @brief map of all currently registered Termination event listeners
         * callback functions mapped with the user provided data
         */
        std::map<dsl_player_termination_event_listener_cb, void*>m_terminationEventListeners;

        /**
         * @brief Queue element for Video converter.
         */
        DSL_ELEMENT_PTR m_pQueue;
        
        /**
         * @brief Video converter to convert from RAW memory to NVMM.
         */
        DSL_ELEMENT_PTR m_pConverter;
        
        /**
         * @brief Caps filter for the video converter.
         */
        DSL_ELEMENT_PTR m_pCapsFilter;
        
        /**
         * @brief shared pointer to the Player's child URI Source
         */
        DSL_SOURCE_PTR m_pSource;
        
        /**
         * @brief shared pointer to the Player's child Overlay Sink
         */
        DSL_SINK_PTR m_pSink;

    };

    /**
     * @brief Timer callback function to Pause a Player in the mainloop context.  
     * @param pPlayer shared pointer to the Player that started the timer to 
     * schedule the pause
     * @return false always to self destroy the on-shot timer.
     */
    static int PlayerPause(gpointer pPipeline);
    
    /**
     * @brief Timer callback function to Stop a Player in the mainloop context.  
     * @param pPlayer shared pointer to the Player that started the timer to 
     * schedule the stop
     * @return false always to self destroy the on-shot timer.
     */
    static int PlayerStop(gpointer pPipeline);

    /**
     * @brief EOS Listener Callback to add to the State Manager's EOS Listeners,
     * and the XWindow Manager's Delete Event Handlers
     * @param pPlayer pointer to the Player object that received the Event message
     */
    static void PlayerTerminate(void* pPlayer);
    
}

#endif //  DSL_PLAYER_BINTR_H
