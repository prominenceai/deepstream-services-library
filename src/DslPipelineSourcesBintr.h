
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

#ifndef _DSL_PIPELINE_SOURCES_BINTR_H
#define _DSL_PIPELINE_SOURCES_BINTR_H

#include "Dsl.h"
#include "DslSourceBintr.h"

namespace DSL
{
    class PipelineSourcesBintr : public Bintr
    {
    public: 
    
        PipelineSourcesBintr(const char* name);

        ~PipelineSourcesBintr();
        
        void AddChild(std::shared_ptr<Bintr> pChildBintr);
        
        void RemoveChild(std::shared_ptr<Bintr> pChildBintr);
        
        void RemoveAllChildren();
        
        void AddSourceGhostPad();
        
        /**
         * @brief interates through the list of child source bintrs setting 
         * their Sensor Id's and linking to the StreamMux
         */
        void LinkAll();
        
        /**
         * @brief interates through the list of child source bintrs unlinking
         * them from the StreamMux and reseting their Sensor Id's
         */
        void UnlinkAll();
        
        void SetStreamMuxProperties(gboolean areSourcesLive, guint batchSize, guint batchTimeout, 
            guint width, guint height);

    private:

        GstElement* m_pStreamMux;
        
        /**
         @brief
         */
        gboolean m_areSourcesLive;

        /**
         @brief
         */
        gint m_batchSize;

        /**
         @brief
         */
        gint m_batchTimeout;
        /**
         @brief
         */
        gint m_streamMuxWidth;

        /**
         @brief
         */
        gint m_streamMuxHeight;

        /**
         @brief
         */
        gboolean m_enablePadding;
    };

    /**
     * @brief 
     * @param[in] pBin
     * @param[in] pPad
     * @param[in] pSource (callback user data) pointer to the unique URI source opject
     */
    static void OnPadAddedCB(GstElement* pBin, GstPad* pPad, gpointer pSource);

    /**
     * @brief 
     * @param[in] pChildProxy
     * @param[in] pObject
     * @param[in] name
     * @param[in] pSource (callback user data) pointer to the unique URI source opject
     */
    static void OnChildAddedCB(GstChildProxy* pChildProxy, GObject* pObject,
        gchar* name, gpointer pSource);

    /**
     * @brief 
     * @param[in] pObject
     * @param[in] arg0
     * @param[in] pSource
     */
    static void OnSourceSetupCB(GstElement* pObject, GstElement* arg0, gpointer pSource);

    /**
     * Probe function to drop certain events to support custom
     * logic of looping of each source stream.
     */

    /**
     * @brief Probe function to drop certain events to support
     * custom logic of looping of each URI source (file) stream.
     * @param pPad
     * @param pInfo
     * @param pSource
     * @return 
     */
    static GstPadProbeReturn StreamBufferRestartProbCB(GstPad* pPad, 
        GstPadProbeInfo* pInfo, gpointer pSource);

    /**
     * @brief 
     * @param pSource
     * @return 
     */
    static gboolean StreamBufferSeekCB(gpointer pSource);
    
}

#endif // _DSL_PIPELINE_SOURCES_BINTR_H
