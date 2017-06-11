#ifndef LZC_TENSOR_GPU_IMPL_H
#define LZC_TENSOR_GPU_IMPL_H

#include "../tensor.h"
#include "tensor_gpu_op.h"

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
    _XINLINE_ void alloc_space(Tensor<gpu, dim> &t) {
        size_t pitch;
        cudaError_t ret = cudaMallocPitch((void **)&t._dptr, &pitch, t._shape[dim - 1] * sizeof(real_t), t.flat_to_2d(true)._shape[1]);
        utils::Assert(ret == cudaSuccess, cudaGetErrorString(ret));
        // pitch is the num of "width" bytes. that is t._shape[dim - 1] * sizeof(float)
        t._shape._stride = pitch / sizeof(real_t);
    }

    template <class SV, int dim>
    _XINLINE_ void store(Tensor<gpu, dim> t, real_t v) {
        GTensor2D gt = t.flat_to_2d();
        for (int i = 0; i < gt._shape[0]; ++i) {
            for (int j = 0; j < gt._shape[1]; ++j) {
                sv::GSaver<SV>::save(gt[i][j], v);
            }
        } 
    }

    template <int dim>
    _XINLINE_ void free_space(Tensor<gpu, dim> &t)  {
        cudaFree((void *)t._dptr);
        t._dptr = nullptr;
    }

    template <int dim>
    _XINLINE_ Tensor<gpu, dim> new_gtensor(const Shape<dim> &shape, real_t init_v) {
        Tensor<gpu, dim> ret(shape);
        alloc_space(ret);
        store<sv::saveto, dim>(ret, init_v);
        return ret;
    }
}; // lzc
#endif
