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

#ifndef _DSL_SINK_BINTR_H
#define _DSL_SINK_BINTR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslBintr.h"
#include "DslElementr.h"

#include <gst-nvdssr.h>

namespace DSL
{
    #define DSL_TAP_PTR std::shared_ptr<TapBintr>

    #define DSL_RECORD_TAP_PTR std::shared_ptr<RecordTapBintr>
    #define DSL_RECORD_TAP_NEW(name, outdir, container, clientListener) std::shared_ptr<RecordTapBintr>( \
        new RecordTapBintr(name, outdir, container, clientListener))

    class TapBintr : public Bintr
    {
    public: 
    
        TapBintr(const char* name);

        ~TapBintr();
  
        bool LinkToSource(DSL_NODETR_PTR pTee);

        bool UnlinkFromSource();
        
    protected:

        /**
         * @brief Queue element as sink for all Tap Bintrs.
         */
        DSL_ELEMENT_PTR m_pQueue;
    };

    //-------------------------------------------------------------------------

    class RecordTapBintr : public TapBintr
    {
    public: 
    
        RecordTapBintr(const char* name, const char* outdir, uint container, NvDsSRCallbackFunc clientListener);

        ~RecordTapBintr();
  
        /**
         * @brief Links all Child Elementrs owned by this Bintr
         * @return true if all links were succesful, false otherwise
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elemntrs owned by this Bintr
         * Calling UnlinkAll when in an unlinked state has no effect.
         */
        void UnlinkAll();

        /**
         * @brief Gets the current outdir in use by this Bintr
         * @return relative or absolute pathspec as provided on construction or set call.
         */
        const char* GetOutdir();

        /**
         * @brief Sets the outdir to use by this RecordTapBintr
         * @param[in] relative or absolute pathspec to the existing directory to use
         * @return true on successfull set, false otherwise
         */
        bool SetOutdir(const char* outdir);

        /**
         * @brief Gets the current cache size setting used by this RecordTapBintr
         * @return size of the video cache in seconds 
         * default = DSL_DEFAULT_SINK_VIDEO_CACHE_IN_SEC
         */
        uint GetCacheSize();
        
        /**
         * @brief Sets the current cache size used by this RecordTapBint
         * @param[in] videoCacheSize size of video cache in seconds 
         * default = DSL_DEFAULT_SINK_VIDEO_CACHE_IN_SEC
         */
        bool SetCacheSize(uint videoCacheSize);
        
        /**
         * @brief Gets the current width and height settings for this RecordTapBintr
         * Zero values indicates no transcode
         * @param[out] width the current width setting in pixels
         * @param[out] height the current height setting in pixels
         */ 
        void GetDimensions(uint* width, uint* height);
        
        /**
         * @brief Sets the current width and height settings for this RecordTapBintr
         * Zero values indicates no transcode
         * The caller is required to provide valid width and height values
         * @param[in] width the width value to set in pixels
         * @param[in] height the height value to set in pixels
         * @return false if the RecordTapBintr is currently linked. True otherwise
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
        bool StartSession(uint* session, uint start, uint duration, void* clientData);
        
        /**
         * @brief Stop recording to file
         * @param[in] session unique sission Id of the recording session to stop
         * @return true on succesful start, false otherwise
         */
        bool StopSession(uint session);

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
         * @brief Queries the Record Bin context to check if reset has been
         * @return true if reset has been done.
         */
        bool ResetDone();

    private:

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
         * @brief Node to wrap NVIDIA's Record Bin
         */
        DSL_NODETR_PTR m_pRecordBin;
    };

}
#endif // _DSL_SINK_BINTR_H
