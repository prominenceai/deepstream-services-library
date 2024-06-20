/*
The MIT License

Copyright (c) 2019-2023, Prominence AI, Inc.

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

#ifndef _DSL_DEWARPER_BINTR_H
#define _DSL_DEWARPER_BINTR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslElementr.h"
#include "DslQBintr.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_DEWARPER_PTR std::shared_ptr<DewarperBintr>
    #define DSL_DEWARPER_NEW(name, configFile, cameraId) \
        std::shared_ptr<DewarperBintr>(new DewarperBintr(name, configFile, cameraId))
        
    class DewarperBintr : public QBintr
    {
    public: 
    
        /**
         * @brief Ctor for the DewarperBintr class
         * @param[in] name unique name to give to the Dewarper
         * @param[in] absolute or relative path to the Dewarper config text file
         * @param[in] cameraId refers to the first column of the CSV files (i.e. 
         * csv_files/nvaisle_2M.csv & csv_files/nvspot_2M.csv). The dewarping 
         * parameters for the given camera are read from CSV files and used to 
         * generate dewarp surfaces (i.e. multiple aisle and spot surface) from 
         * 360d input video stream from the CSV file.
         */
        DewarperBintr(const char* name, 
            const char* configFile, uint cameraId);

        /**
         * @brief dtor for the DewarperBintr class
         */
        ~DewarperBintr();

        /**
         * @brief Required by all Bintrs. but not used by Dewarper
         * @param[in] pParentBintr Parent Pipeline 
         */
        bool AddToParent(DSL_BASE_PTR pParentBintr);

        /**
         * @brief Links all Child Elementrs owned by this Bintr
         * @return true if all links were succesful, false otherwise
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elementrs owned by this Bintr
         * Calling UnlinkAll when in an unlinked state has no effect.
         */
        void UnlinkAll();

        /**
         * @brief Gets the path to the Dewarper Config File in use by 
         * this DewarperBintr
         * @return fully qualified patspec in use by this Bintr.
         */
        const char* GetConfigFile();
        
        /**
         * @brief Sets the path to the Config File for thr DewarperBintr to use.
         * @param[in] configFile fully qualified patspec to a Dewarper config-file.
         */
        bool SetConfigFile(const char* configFile);

        /**
         * @brief Gets the current camera-id for this DewarperBintr.
         * @return Current camera-id - refers to the first column of the CSV files
         * (i.e. csv_files/nvaisle_2M.csv & csv_files/nvspot_2M.csv). The dewarping
         * parameters for the given camera are read from CSV files and used to 
         * generate dewarp surfaces (i.e. multiple aisle and spot surface) from 
         * 360d input video stream from the CSV file.

         */
        uint GetCameraId();
        
        /**
         * @brief Sets the camera-id for this DewarperBintr
         * @param[in] cameraId refers to the first column of the CSV files (i.e. 
         * csv_files/nvaisle_2M.csv & csv_files/nvspot_2M.csv). The dewarping
         * parameters for the given camera are read from CSV files and used to 
         * generate dewarp surfaces (i.e. multiple aisle and spot surface) from 
         * 360d input video stream from the CSV file.
         */
        bool SetCameraId(uint cameraId);
        
        /**
         * @brief Set the GPU ID for all Elementrs
         * @return true if successfully set, false otherwise.
         */
        bool SetGpuId(uint gpuId);
        
        /**
         * @brief Gets the the number of dewarped output surfaces per frame buffer.
         * @return the number of dewarped output surfaces per frame buffer.
         */
        uint GetNumBatchBuffers();
        
        /**
         * @brief Sets the number of dewarped output surfaces per frame buffer.
         * @param num number of dewarped output surfaces per frame buffer.
         * @return true if successfull set, false otherwise.
         */
        bool SetNumBatchBuffers(uint num);

        /**
         * @brief Sets the NVIDIA buffer memory type.
         * @brief nvbufMemType new memory type to use, one of the 
         * DSL_NVBUF_MEM_TYPE constant values.
         * @return true if successfully set, false otherwise.
         */
        bool SetNvbufMemType(uint nvbufMemType);

    private:

        /** 
         * @brief refers to the first column of the CSV files (i.e. 
         * csv_files/nvaisle_2M.csv & csv_files/nvspot_2M.csv). The dewarping
         * parameters for the given camera are read from CSV files and used to 
         * generate dewarp surfaces (i.e. multiple aisle and spot surface) from 
         * 360d input video stream from the CSV file.
         */
        uint m_cameraId;
        
        /**
         * @brief pathspec to the config file used by this DewarperBintr
         */
        std::string m_configFile;
        
        /**
         * @brief Number of Surfaces per output Buffer
         */
        uint m_numBatchBuffers;

        /**
         * @brief Vidconverter plugin for the DewarperBintr.
         */
        DSL_ELEMENT_PTR m_pVidConv;

        /**
         * @brief Dewarper Element for the DewarperBintr
         */
        DSL_ELEMENT_PTR  m_pDewarper;
    };
    
} // DSL

#endif // _DSL_DEWARPER_BINTR_H
