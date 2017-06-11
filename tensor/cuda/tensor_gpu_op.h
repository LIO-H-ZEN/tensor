#ifndef LZC_OP_TENSOR_GPU_OP_H 
#define LZC_OP_TENSOR_GPU_OP_H 

#ifndef _DINLINE_
#define _DINLINE_ inline __device__
#else
#error "_DINLINE_ must not be defined"
#endif

#include "../tensor.h"
#include "../tensor_op.h"

namespace lzc {
    namespace op {
        template <class OPTYPE>
        struct GBinaryMapper {
            _DINLINE_ real_t map(real_t l, real_t r) {}
        }; // gbm
        
        template <>
        struct GBinaryMapper<plus> {
            _DINLINE_ real_t map(real_t l, real_t r) {
                return l + r;
            }
        }; // gbm
    }; // op

    namespace sv {
        template <class OPTYPE>
        struct GSaver {
            _DINLINE_ void save(real_t& l, real_t r) {}
        }; // saver
        
        template <>
        struct GSaver<saveto> {
            _DINLINE_ void save(real_t& l, real_t r) {
                l = r;
            }
        }; // saver
    }; // op
}; // lzc
#endif
