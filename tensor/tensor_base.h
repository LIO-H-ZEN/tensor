#ifndef LZC_TENSOR_BASE_H
#define LZC_TENSOR_BASE_H

#include <iostream>
#include <cstring>

// CUDACC用来区分是否是nvcc编译的
// always_inline
#ifdef _XINLINE_
    #error "_XINLINE_ must be undefined"
#endif
#ifdef __CUDACC__
    #define _XINLINE_ inline __attribute__((always_inline)) __device__ __host__
#else
    #define _XINLINE_ inline __attribute__((always_inline))
#endif

// float or double
#define IS_SINGLE_PRECISION 1 

namespace lzc {
#if IS_SINGLE_PRECISION
    typedef float real_t;
#else
    typedef double real_t;
#endif
    typedef unsigned index_t; // 0 ~ 65535 limit of #(gpu thread block) 
}; // lzc

namespace lzc {
    namespace op {
        struct plus {
            _XINLINE_ static real_t map(real_t l, real_t r) {
                return l + r;
            }
        }; // plus 
        struct minus {
            _XINLINE_ static real_t map(real_t l, real_t r) {
                return l - r;
            }
        }; // minus 
        struct mul {
            _XINLINE_ static real_t map(real_t l, real_t r) {
                return l * r;
            }
        }; // mul 
        struct div {
            _XINLINE_ static real_t map(real_t l, real_t r) {
                return l / r;
            }
        }; // div 
    }; // op
    
    namespace sv {
        struct saveto {
            _XINLINE_ static void save(real_t& l, real_t r) {
                l = r;
            }
        }; // saveto 
        struct add_to {
            _XINLINE_ static void save(real_t& l, real_t r) {
                l += r;
            }
        }; // add_to 
        struct minus_to {
            _XINLINE_ static void save(real_t& l, real_t r) {
                l -= r;
            }
        }; // minus_to 
        struct mul_to {
            _XINLINE_ static void save(real_t& l, real_t r) {
                l -= r;
            }
        }; // mul_to 
        struct div_to {
            _XINLINE_ static void save(real_t& l, real_t r) {
                l /= r;
            }
        }; // div_to 
    }; // sv 
};

#endif
