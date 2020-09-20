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

#ifndef _DSL_RECORD_MGR_H
#define _DSL_RECORD_MGR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslBintr.h"

#include <gst-nvdssr.h>

namespace DSL
{
    class RecordMgr
    {
    public: 
    
        RecordMgr(const char* name, const char* outdir, uint container, 
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
         * @brief Gets the current cache size used by this RecordSinkBint
         * @return size of the video cache in seconds 
         * default = DSL_DEFAULT_VIDEO_RECORD_CACHE_IN_SEC
         */
        uint GetCacheSize();
        
        /**
         * @brief Sets the current cache size used by this RecordSinkBint
         * @param[in] videoCacheSize size of video cache in seconds 
         * default = DSL_DEFAULT_VIDEO_RECORD_CACHE_IN_SEC
         */
        bool SetCacheSize(uint videoCacheSize);
        
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
         * @param[in] start seconds before the current time. Should be less than video cache size.
         * @param[in] duration of recording in seconds from start
         * @param[in] clientData returned on call to client callback
         * @return true on succesful start, false otherwise
         */
        bool StartSession(uint start, uint duration, void* clientData);
        
        /**
         * @brief Stop recording to file
         * @param[in] session unique sission Id of the recording session to stop
         * @return true on succesful start, false otherwise
         */
        bool StopSession();

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
         * @brief client listener function to be called on session complete
         */
        dsl_record_client_listener_cb m_clientListener;
        
        void* m_clientData;
        
    };

    //******************************************************************************************

    static void* RecordCompleteCallback(NvDsSRRecordingInfo* pNvDsInfo, void* pRecordSinkBintr);

}


#endif //  _DSL_RECORD_MGR_H

