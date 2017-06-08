#ifndef LZC_TENSOR_CPU_IMPL_H
#define LZC_TENSOR_CPU_IMPL_H

#include "tensor_op.h"

namespace lzc {
    // map implementation
    template <class SV, class OP>
    inline void map(CTensor2D dst, const CTensor2D &lst, const CTensor2D &rst) {
        for ( index_t x = 0; x < dst._shape[0]; ++x) {
            for (index_t y = 0; y < dst._shape[1]; ++y) {
                op::Saver<SV>::map(dst[x][y], op::BinaryMapper<OP>::map(lst[x][y], rst[x][y]));
            }
        } 
    }
}; // lzc
#endif
