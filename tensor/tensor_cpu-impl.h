#ifndef LZC_TENSOR_CPU_IMPL_H
#define LZC_TENSOR_CPU_IMPL_H

#include "tensor_base.h"
#include "../util/utils.h"

namespace lzc {
    // map implementation
    template <class SV, class OP, int dim>
    inline void map(Tensor<cpu, dim> dst, const Tensor<cpu, dim> &lst, const Tensor<cpu, dim> &rst) {
        utils::Assert(lst._shape == rst._shape, "left right tensor must in same shape");
        utils::Assert(dst._shape == rst._shape, "des right tensor must in same shape");
        CTensor2D dst_2d = dst.flat_to_2d();
        CTensor2D lst_2d = lst.flat_to_2d();
        CTensor2D rst_2d = rst.flat_to_2d();
        for ( index_t x = 0; x < dst_2d._shape[0]; ++x) {
            for (index_t y = 0; y < dst_2d._shape[1]; ++y) {
                SV::save(dst_2d[x][y], OP::map(lst_2d[x][y], rst_2d[x][y]));
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

    template <int dim>
    inline void copy(Tensor<cpu, dim> dst, const Tensor<cpu, dim> &src) {
        utils::Assert(dst._shape == src._shape, "dst and src must be in same shape");    
        CTensor2D dst2 = dst.flat_to_2d();
        CTensor2D src2 = src.flat_to_2d();
        for (int i = 0; i < dst2._shape[0]; i++) {
            memcpy(dst2[i]._dptr, src2[i]._dptr, dst2._shape[1] * sizeof(real_t));
        }
    }
}; // lzc
#endif
