#ifndef LZC_OP_TENSOR_OP_H
#define LZC_OP_TENSOR_OP_H

#include "tensor.h"

namespace lzc {

    namespace op {
        struct plus {};
    }; // op
    
    namespace sv {
        struct saveto {};
    }; // op
    
    namespace op {
    
        template <class OPTYPE>
        struct BinaryMapper {
            inline static real_t map(real_t l, real_t r) {}
        }; // bm
        
        template <>
        struct BinaryMapper<plus> {
            inline static real_t map(real_t l, real_t r) {
                return l + r;
            }
        }; // bm
    };
    
    namespace sv {
        template <class OPTYPE>
        struct Saver {
            inline static void save(real_t l, real_t r) {}
        }; // sv 
    
        template <>
        struct Saver<saveto> {
            inline static void save(real_t& l, real_t r) {
                l = r;
            }
        }; // sv 
    }; // op
}; // lzc
#endif
