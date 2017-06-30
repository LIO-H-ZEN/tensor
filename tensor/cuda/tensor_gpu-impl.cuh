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
        cudaError_t ret = cudaMallocPitch((void **)&t._dptr, &pitch, t._shape[dim - 1] * sizeof(real_t), t.flat_to_2d()._shape[0]);
        utils::Assert(ret == cudaSuccess, cudaGetErrorString(ret));
        // pitch is the num of "width" bytes. that is t._shape[dim - 1] * sizeof(float)
        // 所以stride就是gpu分配内存不是严格的个数*sizeof(real_t)，而是会补齐到512之类的
        t._shape._stride = pitch / sizeof(real_t);
    }

    template <int dim>
    inline void free_space(Tensor<gpu, dim> &t)  {
        cudaFree((void *)t._dptr);
        t._dptr = NULL;
    }
    
    template <class A, class B, int dim>
    inline void copy(Tensor<A, dim> dst, const Tensor<B, dim> &src, cudaMemcpyKind kind) {
        utils::Assert(dst._shape == src._shape, "copy::shape mismatch");
        Tensor<A, 2> dst_2d = dst.flat_to_2d();
        Tensor<B, 2> src_2d = src.flat_to_2d();
        cudaError_t ret = cudaMemcpy2D(dst_2d._dptr, dst_2d._shape._stride * sizeof(real_t), src_2d._dptr, src_2d._shape._stride * sizeof(real_t), src_2d._shape[1] * sizeof(real_t), src_2d._shape[0], kind); 
        utils::Assert(ret == cudaSuccess, cudaGetErrorString(ret));
    } 

    template <int dim>
    inline void copy(Tensor<gpu, dim> dst, const Tensor<cpu, dim> &src) {
        copy<gpu, cpu, dim>(dst, src, cudaMemcpyHostToDevice);
    }

    template <int dim>
    inline void copy(Tensor<cpu, dim> dst, const Tensor<gpu, dim> &src) {
        copy<cpu, gpu, dim>(dst, src, cudaMemcpyDeviceToHost);
    }

    template <int dim>
    inline void copy(Tensor<gpu, dim> dst, const Tensor<gpu, dim> &src) {
        copy<gpu, gpu, dim>(dst, src, cudaMemcpyDeviceToDevice);
    }

}; // lzc

namespace lzc {
    // 先实现个能work的
    // 也就是block * thread 能cover的
    // 这里一开始用的引用，用了之后爆内存非法访问。这个还需要理解下。
    // 确定kernel函数的参数确实不能这么写。
    template <class SV, class OP, int threads_per_block>
    __global__ void map_kernel(GTensor2D dst, const GTensor2D lst, const GTensor2D rst) {
        // blockIdx  threadIdx
        const index_t tid = blockIdx.x * threads_per_block + threadIdx.x; 
        const index_t stride = dst._shape._stride;
        const index_t x = tid / stride; 
        const index_t y = tid % stride; 
        if (x < dst._shape[0] && y < dst._shape[1]) {
            SV::save(dst[x][y], OP::map(lst[x][y], rst[x][y]));
        }
    }    

    template <class SV, class OP, int dim>
    inline void map(Tensor<gpu, dim> dst, const Tensor<gpu, dim> &lst, const Tensor<gpu, dim> &rst) {
        dim3 thread_dim3(cuda::BASE_THREAD_NUM, 1, 1); // 3D in a block 
        const index_t num_block = (dst._shape.mem_size() + cuda::BASE_THREAD_NUM - 1) / cuda::BASE_THREAD_NUM; 
        if (num_block < cuda::MAX_GRID_NUM) {
            dim3 grid_dim3(num_block, 1, 1); // 3D in a grid
            GTensor2D dst_2d = dst.flat_to_2d();
            GTensor2D lst_2d = lst.flat_to_2d();
            GTensor2D rst_2d = rst.flat_to_2d();
            map_kernel<SV, OP, cuda::BASE_THREAD_NUM><<<grid_dim3, thread_dim3>>>(dst_2d, lst_2d, rst_2d);
        } else {
            utils::Error("not implement"); 
        }
    }

    template <class SV, int threads_per_block>
    __global__ void store_kernel(GTensor2D dst, real_t v) {
        const index_t tid = blockIdx.x * threads_per_block + threadIdx.x; 
        const index_t stride = dst._shape._stride;
        const index_t x = tid / stride; 
        const index_t y = tid % stride; 
        if (x < dst._shape[0] && y < dst._shape[1]) {
            SV::save(dst[x][y], v);
        }
    }

    template <class SV, int dim>
    inline void store(Tensor<gpu, dim> dst, real_t v) {
        GTensor2D dst2 = dst.flat_to_2d();
        dim3 thread_dim3(cuda::BASE_THREAD_NUM, 1, 1); // 3D in a block 
        const index_t num_block = (dst2._shape.mem_size() + cuda::BASE_THREAD_NUM - 1) / cuda::BASE_THREAD_NUM; 
        if (num_block < cuda::MAX_GRID_NUM) {
            dim3 grid_dim3(num_block, 1, 1); // 3D in a grid
            store_kernel<SV, cuda::BASE_THREAD_NUM><<<grid_dim3, thread_dim3>>>(dst2, v);
        } else {
            utils::Error("not implement"); 
        }
    }

    template <int dim>
    inline Tensor<gpu, dim> new_gtensor(const Shape<dim> &shape, real_t init_v) {
        Tensor<gpu, dim> ret(shape);
        alloc_space(ret);
        store<sv::saveto, dim>(ret, init_v);
        return ret;
    }
}; // lzc
#endif
