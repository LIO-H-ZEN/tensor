#ifndef LZC_TENSOR_CPU_IMPL_H
#define LZC_TENSOR_CPU_IMPL_H

#include "tensor_op.h"
#include "../util/utils.h"

namespace lzc {
    // map implementation
    template <class SV, class OP>
    inline void map(CTensor2D dst, const CTensor2D &lst, const CTensor2D &rst) {
        utils::Assert(lst._shape == rst._shape, "left right tensor must in same shape");
        utils::Assert(dst._shape == rst._shape, "des right tensor must in same shape");
        for ( index_t x = 0; x < dst._shape[0]; ++x) {
            for (index_t y = 0; y < dst._shape[1]; ++y) {
                SV::save(dst[x][y], OP::map(lst[x][y], rst[x][y]));
            }
        } 
    }
    
    template <int dim>
    inline void alloc_space(Tensor<cpu, dim> &t) {
        t._shape._stride = t._shape[dim - 1];
        t._dptr = new real_t[t._shape.mem_size()];        
    }

    template <int dim>
    inline void free_space(Tensor<cpu, dim> &t) {
        delete [] t._dptr;        
    }

    template <class SV, int dim>
    inline void store(Tensor<cpu, dim> t, real_t v) {
        CTensor2D ct2d = t.flat_to_2d();
        for (index_t i = 0; i < ct2d._shape[0]; ++i) {
            for (index_t j = 0; j < ct2d._shape[1]; ++j) {
                SV::save(ct2d[i][j], v); 
            }
        }
    }
     
    template <int dim>
    inline Tensor<cpu, dim> new_ctensor(const Shape<dim> &shape, real_t init_v) {
        Tensor<cpu, dim> ret(shape);
        alloc_space(ret);
        store<sv::saveto, dim>(ret, init_v);
        return ret;
    }
}; // lzc
#endif
