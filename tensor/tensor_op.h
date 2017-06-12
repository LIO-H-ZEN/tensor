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
}; // lzc
#endif
