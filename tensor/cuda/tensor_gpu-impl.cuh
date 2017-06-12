#ifndef LZC_TENSOR_GPU_IMPL_H
#define LZC_TENSOR_GPU_IMPL_H

#include "../tensor.h"

namespace lzc {
    namespace cuda {
        // define some CONST
        // 1 MEM_UNIT_BITS 一个内存单位的位数 
        // 2 MAX_THREADS_PER_BLOCK  
        // 3 MEM_UNIT 一个内存单元的大小
        // 4 MEM_UNIT_MASK 内存单元最大的数
        // 5 ALIGN_BITS  实际连续位数，也就是每个内存单元内保持内存连续，内存单元之间可以不连续？
        // 6 ALIGN_WIDTH  连续实际宽度
        // 7 BASE_THREAD_BITS  线程数位数
        // 8 BASE_THREAD_NUM  线程实际数
        // 9 BASE_GRID_NUM  
        // 10 MAX_GRID_NUM
        #if __CUDA_ARCH__ >= 200
        const int MEM_UNIT_BITS = 5;
        const int MAX_THREADS_PER_BLOCK = 1024; // hardware limit 
        #else
        const int MEM_UNIT_BITS = 4;
        const int MAX_THREADS_PER_BLOCK = 512; // hardware limit 
        #endif 
        const int MEM_UNIT = 1 << MEM_UNIT_BITS;
        const int MEM_UNIT_MASK = MEM_UNIT - 1;

        const int ALIGN_BITS = MEM_UNIT_BITS; // >= 200 对应32位连续
        const int ALIGN_WIDTH = 1 << MEM_UNIT_BITS; 

        const int BASE_THREAD_BITS = 8; // hard-code cuda-by-example
        const int BASE_THREAD_NUM = 1 << BASE_THREAD_BITS; // 256
        
        const int BASE_GRID_NUM = 32; // hard-code cuda-by-example
        const int MAX_GRID_NUM = 65535; // hardware limit
    }; // cuda

}; // lzc

namespace lzc {
    // cudaMallocPitch: linear memory alloc for 2D,3D
    // cudaError_t: cuda erro type
    template <int dim>
    inline void alloc_space(Tensor<gpu, dim> &t) {
        size_t pitch;
        cudaError_t ret = cudaMallocPitch((void **)&t._dptr, &pitch, t._shape[dim - 1] * sizeof(real_t), t.flat_to_2d(true)._shape[1]);
        utils::Assert(ret == cudaSuccess, cudaGetErrorString(ret));
        // pitch is the num of "width" bytes. that is t._shape[dim - 1] * sizeof(float)
        t._shape._stride = pitch / sizeof(real_t);
    }

    template <int dim>
    inline void free_space(Tensor<gpu, dim> &t)  {
        cudaFree((void *)t._dptr);
        t._dptr = NULL;
    }
    
    template <class A, class B, int dim>
    inline void copy(Tensor<A, dim> dst, Tensor<B, dim> &src, cudaMemcpyKind kind) {
        utils::Assert(dst._shape == src._shape, "copy::shape mismatch");
        Tensor<A, 2> dst2 = dst.flat_to_2d(true);
        Tensor<B, 2> src2 = src.flat_to_2d(true);
        cudaError_t ret = cudaMemcpy2D(dst2._dptr, dst2._shape._stride * sizeof(real_t), src2._dptr, src2._shape._stride * sizeof(real_t), src2._shape[0] * sizeof(real_t), src2._shape[1], kind); 
        utils::Assert(ret == cudaSuccess, cudaGetErrorString(ret));
    } 

    template <int dim>
    inline void copy(Tensor<gpu, dim> dst, Tensor<cpu, dim> &src) {
        copy<gpu, cpu, dim>(dst, src, cudaMemcpyHostToDevice);
    }

    template <int dim>
    inline void copy(Tensor<cpu, dim> dst, Tensor<gpu, dim> &src) {
        copy<cpu, gpu, dim>(dst, src, cudaMemcpyDeviceToHost);
    }

    template <int dim>
    inline void copy(Tensor<gpu, dim> dst, Tensor<gpu, dim> &src) {
        copy<gpu, gpu, dim>(dst, src, cudaMemcpyDeviceToDevice);
    }
}; // lzc
#endif
