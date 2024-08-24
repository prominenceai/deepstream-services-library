/*
The MIT License

Copyright (c) 2019-2024, Prominence AI, Inc.

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

#ifndef _DSL_RECORD_MGR_H
#define _DSL_RECORD_MGR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslBintr.h"
#include "DslMailer.h"

#include <gst-nvdssr.h>

namespace DSL
{
    #define DSL_RECORD_MGR_PTR std::shared_ptr<RecordMgr>

    class RecordMgr
    {
    public: 
    
        RecordMgr(const char* name, const char* outdir, uint gpuId, uint container, 
            dsl_record_client_listener_cb clientListener);

        ~RecordMgr();
        
        /**
         * @brief Creates the required context to start a NvDsSSRecord session
         * @return true of the context could be created, false otherwise.
         */
        bool CreateContext();
        
        /**
         * @brief Destroy a previously created client context
         */
        void DestroyContext();
        
        /**
         * @brief Gets the current outdir in use by this Bintr
         * @return relative or absolute pathspec as provided on construction or set call.
         */
        const char* GetOutdir();

        /**
         * @brief Sets the outdir to use by this Bintr
         * @param[in] relative or absolute pathspec to the existing directory to use
         * @return true on successfull set, false otherwise
         */
        bool SetOutdir(const char* outdir);

        /**
         * @brief Gets the container type used by this RecordMgr
         * @return container type in use 
         */
        uint GetContainer();
        
        /**
         * @brief Sets the current container type to be used by this RecordMgr
         * @param[in] container type to set
         * @return true if container type was set successfully, false others
         */
        bool SetContainer(uint container);
        
        /**
         * @brief Gets the current max size used by this RecordMgr.
         * @return max size of the video recording in seconds. 
         * default = DSL_DEFAULT_VIDEO_RECORD_MAX_SIZE_IN_SEC.
         */
        uint GetMaxSize();
        
        /**
         * @brief Sets the current cache size used by this RecordMgr.
         * @param[in] maxSize size of the video recording seconds.
         * default = DSL_DEFAULT_VIDEO_RECORD_MAX_SIZE_IN_SEC.
         */
        bool SetMaxSize(uint maxSize);
        
        /**
         * @brief Gets the current cache size used by this RecordMgr.
         * @return size of the video cache in seconds .
         * default = DSL_DEFAULT_VIDEO_RECORD_CACHE_SIZE_IN_SEC.
         */
        uint GetCacheSize();
        
        /**
         * @brief Sets the current cache size used by this RecordMgr.
         * @param[in] cacheSize size of cache in seconds.
         * default = DSL_DEFAULT_VIDEO_RECORD_CACHE_SIZE_IN_SEC.
         */
        bool SetCacheSize(uint cacheSize);
        
        /**
         * @brief Gets the current width and height settings for this RecordSinkBintr
         * Zero values indicates no transcode
         * @param[out] width the current width setting in pixels
         * @param[out] height the current height setting in pixels
         */ 
        void GetDimensions(uint* width, uint* height);
        
        /**
         * @brief Sets the current width and height settings for this RecordSinkBintr
         * Zero values indicates no transcode
         * The caller is required to provide valid width and height values
         * @param[in] width the width value to set in pixels
         * @param[in] height the height value to set in pixels
         * @return false if the RecordSink is currently linked. True otherwise
         */ 
        bool SetDimensions(uint width, uint hieght);
        
        /**
         * @brief Start recording to file
         * @param[out] session unique Id for the new recording session, 
         * @param[in] start seconds before the current time. Should be less 
         * than video cache size.
         * @param[in] duration of recording in seconds from start
         * @param[in] clientData returned on call to client callback
         * @return true on succesful start, false otherwise
         */
        bool StartSession(uint start, uint duration, void* clientData);
        
        /**
         * @brief implements a timer thread to notify the client listener 
         * in the main loop context.
         * @return false always to self remove timer once the client has 
         * been notified. Timer/tread will be restarted on next call to StartSession()
         */
        int NotifyClientListener();
        
        /**
         * @brief Stop recording to file
         * @param[in] sync if true the function will block until the asynchronous
         * stop has completed or DSL_RECORDING_STOP_WAIT_TIMEOUT. 
         * @return true on succesful stop, false otherwise
         */
        bool StopSession(bool sync);

        /**
         * @brief Queries the Record Bin context to check the Key Frame
         * @return true if the Bin has the Key Frame ???
         */
        bool GotKeyFrame();
        
        /**
         * @brief Queires the Record Bin context to check if the recording is on
         * @return true if recording is currently on
         */
        bool IsOn();
        
        /**
         * @brief Queries the Record Bin context to check if reset has been done.
         * @return true if reset has been done.
         */
        bool ResetDone();

        /**
         * @brief adds a Video Player, Render or RTSP type, to this RecordMgr
         * @param pPlayer shared pointer to an Video Player to add
         * @return true on successfull add, false otherwise
         */
        bool AddVideoPlayer(DSL_BINTR_PTR pPlayer);
        
        /**
         * @brief adds a Video Player, Render or RTSP type, to this RecordMgr
         * @param pPlayer shared pointer to an Video Player to add
         * @return true on successfull add, false otherwise
         */
        bool RemoveVideoPlayer(DSL_BINTR_PTR pPlayer);
        
        /**
         * @brief adds a SMTP Mailer to this RecordMgr
         * @param[in] pMailer shared pointer to a Mailer to add
         * @param[in] subject subject line to use for all email
         * @return true on successfull add, false otherwise
         */
        bool AddMailer(DSL_MAILER_PTR pMailer, const char* subject);
        
        /**
         * @brief removes a SMTP Mailer from this RecordMgr
         * @param[in] pMailer shared pointer to an Mailer to remove
         * @return true on successfull remove, false otherwise
         */
        bool RemoveMailer(DSL_MAILER_PTR pMailer);
        

        /**
         * @brief Record complete handler function to conver the gchar* strings in
         * NvDsSRRecordingInfo to 
         * @return true if reset has been done.
         */
        void* HandleRecordComplete(NvDsSRRecordingInfo* pNvDsInfo);

protected:
        
        /**
         * @brief unique name for the RecordMgr
         */
        std::string m_name;

        /**
         * @brief absolute or relative path 
         */
        std::string m_outdir;

        /**
         * @brief GPU_ID provided on construction
         */
        uint m_parentGpuId;

        /**
         * @brief SR context, once created, must be passed to 
         */
        NvDsSRContext* m_pContext;
        
        /**
         * @brief SR context initialization parameters, provided by client
         */
        NvDsSRInitParams m_initParams;
        
        /**
         * @brief current session id assigned on record start.
         */
        NvDsSRSessionId m_currentSessionId;
        
        /**
         * @brief Mutex to protect Start and Stop Session services that can
         * be called by multiple threads; Actions, Users, Callbacks, etc.
         */
        DslMutex m_recordMgrMutex;
        
        /**
         * @brief gnome timer Id for the asynchronous "start-session" client notification.
         */
        uint m_listenerNotifierTimerId;
        
        /**
         * @brief boolean flag to specify whether an async stop recording session 
         * is in progress or not. 
         */
        bool m_stopSessionInProgress;

        /**
         * @brief client listener function to be called on session complete
         */
        dsl_record_client_listener_cb m_clientListener;
        
        /**
         * @brief map of all Video Players to play recordings.
         */
        std::map<std::string, DSL_BINTR_PTR> m_videoPlayers;

        /**
         * @brief map of all Mailers to send email.
         */
        std::map<std::string, std::shared_ptr<MailerSpecs>> m_mailers;
        
        
        void* m_clientData;
        
    };

    //******************************************************************************************
    
    static int RecordMgrListenerNotificationHandler(gpointer pRecordMgr);

    static void* RecordCompleteCallback(NvDsSRRecordingInfo* pNvDsInfo, void* pRecordSinkBintr);
}


#endif //  _DSL_RECORD_MGR_H

