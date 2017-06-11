#ifndef LZC_OP_TENSOR_OP_H
#define LZC_OP_TENSOR_OP_H

#include "tensor.h"

namespace lzc {
    namespace op {
        struct plus {
            _XINLINE_ static real_t map(real_t l, real_t r) {
                return l + r;
            }
        }; // plus 
    }; // op
    
    namespace sv {
        struct saveto {
            _XINLINE_ static void save(real_t& l, real_t r) {
                l = r;
            }
        }; // saveto 
    }; // sv 
}; // lzc
#endif
