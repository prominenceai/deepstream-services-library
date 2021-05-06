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

namespace DSL
{

    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_PLAYER_BINTR_PTR std::shared_ptr<PlayerBintr>
    #define DSL_PLAYER_BINTR_NEW(name, pSource, pSink) \
        std::shared_ptr<PlayerBintr>(new PlayerBintr(name, pSource, pSink))    

    #define DSL_PLAYER_RENDER_BINTR_PTR std::shared_ptr<RenderPlayerBintr>
    
    #define DSL_PLAYER_RENDER_VIDEO_BINTR_PTR std::shared_ptr<VideoRenderPlayerBintr>
    #define DSL_PLAYER_RENDER_VIDEO_BINTR_NEW(name, \
        filePath, renderType, offsetX, offsetY, zoom, repeatEnabled) \
        std::shared_ptr<VideoRenderPlayerBintr>(new VideoRenderPlayerBintr(name, \
            filePath, renderType, offsetX, offsetY, zoom, repeatEnabled))    

    #define DSL_PLAYER_RENDER_IMAGE_BINTR_PTR std::shared_ptr<ImageRenderPlayerBintr>
    #define DSL_PLAYER_RENDER_IMAGE_BINTR_NEW(name, \
        filePath, renderType, offsetX, offsetY, zoom, timeout) \
        std::shared_ptr<ImageRenderPlayerBintr>(new ImageRenderPlayerBintr(name, \
            filePath, renderType, offsetX, offsetY, zoom, timeout))    
    
    class PlayerBintr : public Bintr, public PipelineStateMgr,
        public PipelineXWinMgr
    {
    public: 
    
        /**
         * @brief ctor for the PlayerBintr class
         * @return 
         */
        PlayerBintr(const char* name, 
            DSL_BINTR_PTR pSource, DSL_BINTR_PTR pSink);

        /**
         * @brief ctor2 allows for derived classes to mange their Source/Sink
         */
        PlayerBintr(const char* name);

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
         * @brief Attempts to link all and Play the Pipeline
         * @return true if able to Play, false otherwise
         */
        bool Play();

        /**
         * @brief Completes the process of transitioning to a state of Playing
         */
        bool HandlePlay();

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
         * @brief Terminates the player on event of XWindow Delete
         * Important: must be called by the BusWatch or Event Handler context.
         */
        virtual void HandleTermination();
        
        /**
         * @brief Terminates the player on event of EOS
         * Important: must be called by the BusWatch or Event Handler context.
         */
        virtual void HandleEos();

        /**
         * @brief adds a callback to be notified on Player Termination event
         * @param[in] listener pointer to the client's function to call on Termination event
         * @param[in] clientData opaque pointer to client data passed into the listener function.
         * @return true on successful add, false otherwise
         */
        bool AddTerminationEventListener(dsl_player_termination_event_listener_cb listener, 
            void* clientData);

        /**
         * @brief removes a previously added callback
         * @param[in] listener pointer to the client's function to remove
         * @return true on successful remove, false otherwise
         */
        bool RemoveTerminationEventListener(dsl_player_termination_event_listener_cb listener);

    protected:

        /**
         * @brief Function to initiate the stop process by removing the TerminationEventListener,
         * disabling the source's EOS handler, and then sending an EOS event.
         * @return 
         */
        bool InitiateStop();
        
        /**
         * @brief Asynchronous comm flag for the Termination handler to communication
         * with the async Stop handler. If set, to notify clients of a termination event.
         * i.e the specific reason for stoppin.
         */
        bool m_inTermination;

        /**
         * @brief Queue to connect Source to Converter.
         */
        DSL_ELEMENT_PTR m_pQueue;
    
        /**
         * @brief Video converter to convert from RAW memory to NVMM.
         */
        DSL_ELEMENT_PTR m_pConverter;
        
        /**
         * @brief Caps filter for the video converter.
         */
        DSL_ELEMENT_PTR m_pConverterCapsFilter;

        /**
         * @brief shared pointer to the Player's child URI Source
         */
        DSL_BINTR_PTR m_pSource;
        
        /**
         * @brief shared pointer to the Player's child Overlay Sink
         */
        DSL_BINTR_PTR m_pSink;
    
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

    };

    /**
     * @class RenderPlayerBintr
     * @file DslPlayerBintr.h
     * @brief Implements a PlayerBintr with a FileSourceBintr or ImageSourceBintr
     * and OverlaySink or WindowSinkBintr
     */
    class RenderPlayerBintr : public PlayerBintr
    {
    public: 
    
        RenderPlayerBintr(const char* name, uint renderType, 
            uint offsetX, uint offsetY, uint zoom);

        ~RenderPlayerBintr();

        /**
         * @brief returns the current filePath for this RenderPlayerBintr
         * @return const string to the file path
         */
        const char* GetFilePath();
        
        /**
         * @brief sets the current filePath for this RenderPlayerBintr
         * The service will fail if the RenderPlayerBintr is linked when called
         * @return const string to the filePath to use
         */
        bool SetFilePath(const char* filePath);

        /**
         * @brief adds a filePath to the RenderPlayerBintr's queue
         * @return const string to the filePath to use
         */
        bool QueueFilePath(const char* filePath);

        /**
         * @brief Gets the current X and Y offsets for the RenderPlayerBintr
         * @param[out] offsetX current offset in the x direction in pixels
         * @param[out] offsetY current offset in the Y direction in pixels
         */
        void GetOffsets(uint* offsetX, uint* offsetY);
        
        /**
         * @brief Sets the current X and Y offsets for the RenderPlayerBintr
         * @param[in] offsetX new offset in the Y direction in pixels
         * @param[in] offsetY new offset in the Y direction in pixels
         * @return true on successful update, false otherwise
         */
        bool SetOffsets(uint offsetX, uint offsetY);

        /**
         * @brief returns the current zooom setting for this RenderPlayerBintr
         * @return zoom setting in units of %
         */
        uint GetZoom();
        
        /**
         * @brief sets the current filePath for this RenderPlayerBintr
         * The service will fail if the RenderPlayerBintr is linked when called
         * @return true on successful update, false otherwise
         */
        bool SetZoom(uint zoom);
        
        /**
         * @brief Sends an EOS event to the Player causing the player to 
         * play the next queued file path, or terminate if none.
         * @return 
         */
        bool Next();
        
        /**
         * @brief Handles the Stop and Play next queued file in a timer callback
         */
        void HandleStopAndPlay();
        
        /**
         * @brief on event of EOS, schedules HandleStopAndPlay if a file is queued,
         * or HandleStop to terminate the Player if not. 
         * Important: must be called by the BusWatch or Event Handler context.
         */
        void HandleEos();

        static const uint m_displayId;
        static const uint m_depth;
        
    protected:

        /**
         * @breif creates the sink once the Source has been created and the 
         * dimensions have been determined and saved to the member variables.
         * @return true on successfull creation, false otherwise
         */
        bool CreateRenderSink();
        
        /**
         * @brief updates the Render Sink with new Dimensions
         * @return true on successful update, false otherwise.
         */
        bool SetDimensions();

        /**
         * @brief Sink Type, either DSL_RENDER_TYPE_OVERLAY or DSL_RENDER_TYPE_WINDOW
         */
        uint m_renderType;
        
        /**
         * @brief zoom factor in units of %
         */
        uint m_zoom;
        
        /**
         * @brief offset of the Image or Stream in the X direction
         */
        uint m_offsetX;

        /**
         * @brief offset of the Image or Stream in the Y direction
         */
        uint m_offsetY;
    
        /**
         * @brief width of the Image or Stream
         */
        uint m_width;

        /**
         * @brief height of the Image or Stream
         */
        uint m_height;
        
        /**
         * @brief queue of file paths to play,
         */
        std::queue<std::string> m_filePathQueue;

        /**
         * @brief mutual exclusion over the file path queue.
         */
        GMutex m_filePathQueueMutex;
        
    };

    /**
     * @class VideoRenderPlayerBintr
     * @file DslPlayerBintr.h
     * @brief Implements a PlayerBintr with a FileSourceBintr
     * and OverlaySink or WindowSinkBintr
     */
    class VideoRenderPlayerBintr : public RenderPlayerBintr
    {
    public: 
    
        VideoRenderPlayerBintr(const char* name, const char* filePath, 
            uint renderType, uint offsetX, uint offsetY, uint zoom, bool repeatEnabled);

        ~VideoRenderPlayerBintr();

        /**
         * @brief returns the current repeat-enabled setting for this VideoRenderPlayerBintr
         * @return true if repeat on EOS is enabled, false otherwise
         */
        bool GetRepeatEnabled();
        
        /**
         * @brief sets the current repeat-enabled for this RenderPlayerBintr
         * The service will fail if the VideoRenderPlayerBintr is linked when called
         * @param[in] repeatEnabled set to true to enable, false otherwise
         * @return true on successful update, false otherwise
         */
        bool SetRepeatEnabled(bool repeatEnabled);

    private:
        
        bool m_repeatEnabled;
    };

    /**
     * @class ImageRenderPlayerBintr
     * @file DslPlayerBintr.h
     * @brief Implements a PlayerBintr with an ImageSourceBintr
     * and OverlaySink or WindowSinkBintr
     */
    class ImageRenderPlayerBintr : public RenderPlayerBintr
    {
    public: 
    
        ImageRenderPlayerBintr(const char* name, const char* filePath, 
            uint renderType, uint offsetX, uint offsetY, uint zoom, uint timeout);

        ~ImageRenderPlayerBintr();

        /**
         * @brief returns the current timeout setting for this ImageRenderPlayerBintr
         * @return true if send EOS on timeout is enabled, false otherwise
         */
        uint GetTimeout();
        
        /**
         * @brief sets the timeout for this RenderPlayerBintr.
         * The service will fail if the VideoRenderPlayerBintr is linked when called
         * @param[in] timeout timeout to display the image before sending an EOS event
         * in units of seconds.  0 = no timeout.
         * @return true on successful update, false otherwise
         */
        bool SetTimeout(uint timeout);

    private:
    
        uint m_timeout;
    };
    
    /**
     * @brief Timer callback function to Pause a Player in the mainloop context.  
     * @param pPlayer shared pointer to the Player that started the timer to 
     * schedule the pause
     * @return false always to self destroy the on-shot timer.
     */
    static int PlayerPause(gpointer pPlayer);
    
    /**
     * @brief Timer callback function to Stop a Player in the mainloop context.  
     * @param pPlayer shared pointer to the Player that started the timer to 
     * schedule the stop
     * @return false always to self destroy the on-shot timer.
     */
    static int PlayerStop(gpointer pPlayer);

    /**
     * @brief Timer callback function to Stop and Play a Player in the mainloop context.  
     * @param pPlayer shared pointer to the Player that started the timer to 
     * schedule the stop amd play
     * @return false always to self destroy the on-shot timer.
     */
    static int PlayerStopAndPlay(gpointer pPlayer);

    /**
     * @brief XWindow delete Callback to add to XWindow Manager's Delete Event Handlers
     * @param pPlayer pointer to the Player object that received the Event message
     */
    static void PlayerTerminate(void* pPlayer);
    
    /**
     * @brief EOS Listener Callback to add to the State Manager's EOS Listeners
     * @param pPlayer pointer to the Player object that received the Event message
     */
    static void PlayerHandleEos(void* pPlayer);
    
}

#endif //  DSL_PLAYER_BINTR_H
