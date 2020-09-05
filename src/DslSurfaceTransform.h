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

#ifndef _DSL_SURFACE_TRANSFORM_H
#define _DSL_SURFACE_TRANSFORM_H

#include "Dsl.h"
#include <nvbufsurftransform.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/highgui/highgui.hpp"

namespace DSL
{
    /**
     * @class DslCudaStream
     * @brief Wrapper class for a new Cuda Stream
     */
    class DslCudaStream
    {
    public:
    
        /**
         * @brief ctor for the DslCudaStream class
         * @param gpuId GPU ID valid for multi GPU systems
         */
        DslCudaStream(uint32_t gpuId)
            : stream(NULL)
        {
            LOG_FUNC();

            cudaError_t cudaError = cudaSetDevice(gpuId);
            if (cudaError != cudaSuccess)
            {
                LOG_ERROR("cudaSetDevice failed with error '" << cudaError << "'");
                throw;
            }
            cudaError = cudaStreamCreate(&stream);
            if (cudaError != cudaSuccess)
            {    
                LOG_ERROR("cudaStreamCreate failed with error '" << cudaError << "'");
                throw;
            }
        }

        /**
         * @brief dtor for the DslCudaStream class
         */
        ~DslCudaStream()
        {
            LOG_FUNC();

            if (stream)
            {
                cudaStreamDestroy(stream);
            }
        }

    public:
    
        // cuda stream if created successfully, NULL otherwise.
        cudaStream_t stream;
    };

    // ---------------------------------------------------------------------------------------------------------------
    /**
     * @struct DslMapInfo
     * @brief Info map to access the buffer of batched surfaces
     */
    struct DslMapInfo : public GstMapInfo
    {
    public:
    
        /**
         * @brief ctor for the DslMapInfo structure
         * @param pBuffer buffer to map
         */
        DslMapInfo(GstBuffer* pBuffer)
            : GstMapInfo GST_MAP_INFO_INIT
            , m_pBuffer(NULL)
        {
            LOG_FUNC();
            
            if (!gst_buffer_map(pBuffer, this, GST_MAP_READ))
            {
                LOG_ERROR("Failed to map gst buffer");
                throw;
            }
            // set only when successful
            m_pBuffer = pBuffer;
        }

        /**
         * @brief dtor for the DslMapInfo structure
         */
        ~DslMapInfo()
        {
            LOG_FUNC();
            
            if (m_pBuffer)
            {
                gst_buffer_unmap(m_pBuffer, this);
            }
        }
        
    private:

        /**
         * @brief mapped buffer
         */
        GstBuffer* m_pBuffer;
    };

    // ---------------------------------------------------------------------------------------------------------------
    
    /**
     * @struct DslMonoSurface
     * @brief New mono surface buffer copied from a single surface in a batched surface buffer
     */
    struct DslMonoSurface : public NvBufSurface
    {
    public: 
    
        /**
         * @brief ctor for the DslMonoSurface structure
         * @param mapInfo mapped buffer info as source surface buffer
         * @param index index into the batched surface list
         */
        DslMonoSurface(DslMapInfo& mapInfo, int index)
            : NvBufSurface{0}
            , width(0)
            , height(0)
        {   
            LOG_FUNC();
            
            // copy the shared surface properties
            (NvBufSurface)*this = *(NvBufSurface*)mapInfo.data;
            
            // set the buffer surface as a mono surface
            numFilled = 1;
            batchSize = 1;
            
            // copy the single indexed surface to the new surfaceList of one
            surfaceList = &(((NvBufSurface*)mapInfo.data)->surfaceList[index]);
            
            // new width and height properties for the mono surface, since there's only one.
            width = surfaceList[0].width;
            height = surfaceList[0].height;
        }   

        /**
         * @brief dtor for the DslMonoSurface structure
         */
        ~DslMonoSurface()
        {
            LOG_FUNC();
        }
        
    public:

        /**
         * @brief width width of the mono surface buffer in pixels
         */
        int width;
        
        /**
         * @brief height height of the mono surface buffer in pixels
         */
        int height;

    private:
        
    };

    // ---------------------------------------------------------------------------------------------------------------
    
    /**
     * @struct DslTransformParams
     * @brief Surface transform params with coordinates and dimensions for both source and
     * destination surfaces. Scaling of source to destination is not (currently) supported.
     */
    struct DslTransformParams : public NvBufSurfTransformParams
    {
    public:

        /**
         * @brief ctor for the DslTransformParams structure
         * @param left x-positional coordinate for upper left corner
         * @param top y-positional coordinate for upper left corner
         * @param width width of the source and destination rectangles for the transform
         * @param height height of the source and destination rectangles for the transform
         */
        DslTransformParams(uint32_t left, uint32_t top, uint32_t width, uint32_t height)
            : NvBufSurfTransformParams{0}
            , m_srcRect{top, left, width, height} // this is the correct order for Transform (top first)
            , m_dstRect{0, 0, width, height} // this may be too limiting, only supports cropping
        {
            LOG_FUNC();

            // src and dst rectangles are set up by pointer
            src_rect = &m_srcRect;
            dst_rect = &m_dstRect;
            transform_flag = NVBUFSURF_TRANSFORM_CROP_SRC | NVBUFSURF_TRANSFORM_CROP_DST;
            transform_filter = NvBufSurfTransformInter_Default;    
        }   

        /**
         * @brief dtor for the DslTransformParams structure
         */
        ~DslTransformParams()
        {
            LOG_FUNC();
        }

    private:

        /**
         * @brief m_srcRect coordinates and dimensions of the rectangle to transform within the source surface.
         */
        NvBufSurfTransformRect m_srcRect;

        /**
         * @brief m_srcRect coordinates and dimensions of the rectangle to transform for the destination surface.
         */
        NvBufSurfTransformRect m_dstRect;
    };
    
    // ---------------------------------------------------------------------------------------------------------------

    /**
     * @struct DslSurfaceCreateParams
     * @brief Surface create params with dimensions and memory allocation size.
     */
    struct DslSurfaceCreateParams : public NvBufSurfaceCreateParams
    {
    public:
    
        /**
         * @brief ctor for the DslSurfaceCreateParams structure
         * @param gpuId GPU ID valid for multi GPU systems
         * @param width width of the new surface if only 1
         * @param height height of the new surface if only 1`
         * @param size of memory to create, width and height are ignored if set
         * set size to 0 for no addition memory allocated when batch size = 1
         */
        DslSurfaceCreateParams(uint32_t gpuId, uint32_t width, uint32_t height, uint32_t size)
            : NvBufSurfaceCreateParams{gpuId, width, height, size, false, 
                NVBUF_COLOR_FORMAT_RGBA, NVBUF_LAYOUT_PITCH, NVBUF_MEM_DEFAULT}
        {
            LOG_FUNC();
        }

        /**
         * @brief dtor for the DslSurfaceCreateParams structure
         */
        ~DslSurfaceCreateParams()
        {
            LOG_FUNC();
        }
    };
    
    // ---------------------------------------------------------------------------------------------------------------


    /**
     * @struct DslSurfaceTransformSessionParams
     * @brief Structure of "Transform Session" config params, with a set session params function
     */
    struct DslSurfaceTransformSessionParams : public NvBufSurfTransformConfigParams
    {
    public:
    
        /**
         * @brief ctor for the DslCudaStream class
         * @param gpuId GPU ID valid for multi GPU systems
         */
        DslSurfaceTransformSessionParams(int32_t gpuId, DslCudaStream& cudaStream)
            : NvBufSurfTransformConfigParams{NvBufSurfTransformCompute_Default, gpuId, cudaStream.stream}
        {
            LOG_FUNC();
        }
        
        /**
         * @brief dtor for the DslSurfaceTransformSessionParams structure
         */
        ~DslSurfaceTransformSessionParams()
        {
            LOG_FUNC();
        }
        
        /**
         * @brief function to set the Transform Session params
         */
        bool Set()
        {
            LOG_FUNC();

            NvBufSurfTransform_Error error = NvBufSurfTransformSetSessionParams(this);
            if (error != NvBufSurfTransformError_Success)
            {
                LOG_ERROR("NvBufSurfTransformSetSessionParams failed with error '" << error << "'");
                return false;
            }    

            return true;
        }
    };
    
    // ---------------------------------------------------------------------------------------------------------------
    
        /**
     * @class DslBufferSurface
     * @brief Wrapper class for a new batched surface buffer
     */
    class DslBufferSurface
    {
    public:
    
        /**
         * @brief ctor for the DslBufferSurface class
         * @param gpuId GPU ID valid for multi GPU systems
         */
        DslBufferSurface(uint32_t batchSize, DslSurfaceCreateParams& surfaceCreateParams)
            : m_pBufSurface(NULL)
            , m_isMapped(false)
        {
            LOG_FUNC();

            if (NvBufSurfaceCreate(&m_pBufSurface, batchSize, &surfaceCreateParams) != NvBufSurfTransformError_Success)
            {
                LOG_ERROR("NvBufSurfaceCreate failed");
                throw;
            }
            if (NvBufSurfaceMemSet(m_pBufSurface, -1, -1, 0) != NvBufSurfTransformError_Success)
            {
                LOG_ERROR("NvBufSurfaceMemSet failed");
                throw;
            }
        }
        
        /**
         * @brief dtor for the DslBufferSurface class
         */
        ~DslBufferSurface()
        {
            LOG_FUNC();

            if (m_isMapped)
            {
                NvBufSurfaceUnMap(m_pBufSurface, -1, -1);
            }
            if (m_pBufSurface)
            {
                NvBufSurfaceDestroy(m_pBufSurface);
            }
        }

        /**
         * @brief address-of operator
         * @return returns a pointer to the actual gst batched surface buffer
         */
        NvBufSurface* operator&(){return m_pBufSurface;};
        
        /**
         * @brief function to transform a mono source surface to a surface indexed within the batch.
         * @return true on successful transform, false otherwise
         */
        bool TransformMonoSurface(DslMonoSurface& srcSurface, uint32_t index, DslTransformParams& transformParams)
        {
            LOG_FUNC();

            NvBufSurfTransform_Error error = NvBufSurfTransform(&srcSurface, &m_pBufSurface[index], &transformParams);
            if (error != NvBufSurfTransformError_Success)
            {
                LOG_ERROR("NvBufSurfTransform failed with error '" << error << "'");
                return false;
            }
            return true;
        }
        
        /**
         * @brief function to map the newly transformed batched surface buffer.
         * @return true on successful mapping, false otherwise
         */
        bool Map()
        {
            if (NvBufSurfaceMap(m_pBufSurface, -1, -1, NVBUF_MAP_READ) != NvBufSurfTransformError_Success)
            {
                LOG_ERROR("NvBufSurfaceMap failed");
                return false;
            }
            return true;
        }
        
        /**
         * @brief function to synchronize a mapped, and transformed batched surface buffer, modified by hardware, for CPU access
         * @return true on successful mapping, false otherwise
         */
        bool SyncForCpu()
        {
            if (NvBufSurfaceSyncForCpu(m_pBufSurface, -1, -1) != NvBufSurfTransformError_Success)
            {
                LOG_ERROR("NvBufSurfaceSyncForCpu failed");
                return false;
            }
            return true;
        }
        
        
    private:    

        /**
         * @brief pointer to a batched surface buffer
         */
        NvBufSurface* m_pBufSurface;
        
        /**
         * @brief set to true once mapped so that the buffer can be unmapped before destruction
         */
        bool m_isMapped;
        
    };

}
#endif // _DSL_SURFACE_TRANSFORM_H
