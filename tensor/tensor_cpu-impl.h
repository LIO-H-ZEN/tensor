#ifndef LZC_TENSOR_CPU_IMPL_H
#define LZC_TENSOR_CPU_IMPL_H

#include "tensor_op.h"

namespace lzc {
    // map implementation
    template <class SV, class OP>
    inline void map(CTensor2D dst, const CTensor2D &lst, const CTensor2D &rst) {
        for ( index_t x = 0; x < dst._shape[0]; ++x) {
            for (index_t y = 0; y < dst._shape[1]; ++y) {
                sv::Saver<SV>::save(dst[x][y], op::BinaryMapper<OP>::map(lst[x][y], rst[x][y]));
            }
        } 
    }
    
    template <int dimension>
    inline void alloc_space(Tensor<cpu, dimension> &t) {
        t._shape._stride = t._shape[dimension - 1];
        t._dptr = new real_t[t._shape.mem_size()];        
    }

    template <int dimension>
    inline void free_space(Tensor<cpu, dimension> &t) {
        delete [] t._dptr;        
    }

    template <class SV>
    inline void store(CTensor2D t, real_t v) {
        for (int i = 0; i < t._shape[0]; ++i) {
            for (int j = 0; j < t._shape[1]; ++j) {
                sv::Saver<SV>::save(t[i][j], v); 
            }
        }
    }
     
    template <int dimension>
    inline Tensor<cpu, dimension> new_ctensor(const Shape<dimension> &shape, real_t init_v) {
        Tensor<cpu, dimension> ret(shape);
        alloc_space(ret);
        store<sv::saveto>(ret.flat_to_2d(), init_v);
        return ret;
    }
}; // lzc
#endif
