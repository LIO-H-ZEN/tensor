#ifndef LZC_TENSOR_H
#define LZC_TENSOR_H

#include "tensor_base.h"

namespace lzc {
template <int dimension>
class Shape {
public:
    const static int DIM = dimension;
    const static int SUBDIM = dimension - 1;
public:
    _XINLINE_ Shape( void ) {}
    _XINLINE_ index_t& operator[](size_t idx) {
        return _array[idx];
    }
    _XINLINE_ const index_t& stride( void ) const {
        return _stride;
    }
    _XINLINE_ bool operator==(const Shape<DIM> &other) const {
        #pragma unroll 
        for (int i = 0; i < DIM; ++i) {
            if (this->_array[i] != other._array[i]) {
                return false;
            }
        }
        return true;
    }

    _XINLINE_ Shape<2> flat_to_2d() const {
        Shape<2> s;
        s._stride = this->_stride;
        s._array[1] = this->_array[DIM - 1];
        index_t left = 1;
        #pragma unroll
        for (int i = 0; i < DIM - 1; ++i) {
            left *= this->_array[i];
        } 
        s._array[0] = left;
        return s;
    }
    _XINLINE_ size_t size( void ) const {
        size_t ret = this->_array[0];
        #pragma unroll
        for (int i = 1; i < DIM; ++i) {
            ret *= this->_array[i];
        }
        return ret;
    } 

    // hard-code just to make it work
    _XINLINE_ size_t mem_size( void ) const {
        // stride reserve the basic dim
        size_t ret = this->_stride;
        #pragma unroll
        for (int i = 0; i < DIM - 1; ++i) {
            ret *= this->_array[i];
        }
        return ret;
    }

    _XINLINE_ Shape<SUBDIM> sub_shape( void ) const {
        Shape<SUBDIM> s;
        s._stride = this->_stride;
        #pragma unroll
        for (int i = 1; i < DIM; ++i) {
            s._array[i - 1] = this->_array[i];
        }
        return s;
    }
public:
    // from left to right: 1st dim, 2st dim, ....
    index_t _array[DIM]; 
    index_t _stride;     
}; // class Shape

_XINLINE_ Shape<1> shape1(index_t a1) {
    Shape<1> s;
    s[0] = a1;
    return s;
}

_XINLINE_ Shape<2> shape2(index_t a1, index_t a2) {
    Shape<2> s;
    s[0] = a1;
    s[1] = a2;
    return s;
}

_XINLINE_ Shape<3> shape3(index_t a1, index_t a2, index_t a3) {
    Shape<3> s;
    s[0] = a1;
    s[1] = a2;
    s[2] = a3;
    return s;
}

_XINLINE_ Shape<4> shape4(index_t a1, index_t a2, index_t a3, index_t a4) {
    Shape<4> s;
    s[0] = a1;
    s[1] = a2;
    s[2] = a3;
    s[3] = a4;
    return s;
}
}; // lzc

namespace lzc {
struct cpu {
    const static bool isCPU = true;
};
struct gpu {
    const static bool isCPU = false;
};

template <class device, int dim>
class Tensor {
public:
    const static bool DEVICE_TYPE = device::isCPU;
    const static index_t DIM = dim; 
    const static index_t SUBDIM = dim - 1; 
    real_t *_dptr;
    Shape<DIM> _shape;
public:
    _XINLINE_ Tensor( void ) {}
    _XINLINE_ Tensor(Shape<DIM> shape) : _shape(shape) {}
    _XINLINE_ Tensor(real_t *dptr, Shape<DIM> shape): _dptr(dptr),_shape(shape) {} 

    _XINLINE_ Tensor<device, 2> flat_to_2d() const{
        return Tensor<device, 2>((real_t*)_dptr, _shape.flat_to_2d());
    }  
    
    _XINLINE_ Tensor<device, SUBDIM> operator[](index_t idx) const {
        Shape<SUBDIM> s = this->_shape.sub_shape();
        return Tensor<device, SUBDIM>((real_t*)_dptr + idx * s.mem_size(), s);
    }

    _XINLINE_ Tensor<device, DIM> slice(index_t se, index_t ed) const {
        Shape<DIM> s = this->_shape;
        s[0] = ed - se;
        return Tensor<device, DIM>((real_t*)_dptr + se * s.sub_shape().mem_size(), s);
    }
}; // class Tensor
template <class device>
class Tensor<device, 1> {
public:
    real_t *_dptr;
    Shape<1> _shape;
    _XINLINE_ Tensor( void ) {}
    _XINLINE_ Tensor(real_t *dptr, Shape<1> shape) : _dptr(dptr),_shape(shape) {}
    
    _XINLINE_ real_t& operator[](index_t idx) const {
        return _dptr[idx];
    } 

    _XINLINE_ Tensor<device, 1> slice(int se, int ed) const {
        Shape<1> s;
        s._array[0] = ed - se;
        return Tensor<device, 1>((real_t*)_dptr + se, s);
    }

}; // class tensor
}; // lzc

namespace lzc {
    typedef Tensor<cpu, 1> CTensor1D;
    typedef Tensor<cpu, 2> CTensor2D;
    typedef Tensor<cpu, 3> CTensor3D;
    typedef Tensor<cpu, 4> CTensor4D;
    typedef Tensor<gpu, 1> GTensor1D;
    typedef Tensor<gpu, 2> GTensor2D;
    typedef Tensor<gpu, 3> GTensor3D;
    typedef Tensor<gpu, 4> GTensor4D;
}; // lzc

namespace lzc {
    template <class SV, class OP, int dim>
    inline void map(Tensor<cpu, dim> dst, const Tensor<cpu, dim> &lst, const Tensor<cpu, dim> &rst);
    template <class SV, class OP, int dim>
    inline void map(Tensor<gpu, dim> dst, const Tensor<gpu, dim> &lst, const Tensor<gpu, dim> &rst);

    // alloc memory for tensor according to its shape
    // and set its stride
    // 在host上分配空间，所以inline就行了
    // 下面几个函数情况一样，都是在host上调用cuda api
    template <int dim>
    inline void alloc_space(Tensor<cpu, dim> &t);
    template <int dim>
    inline void alloc_space(Tensor<gpu, dim> &t);

    // free_space
    template <int dimension>
    inline void free_space(Tensor<cpu, dimension> &t); 
    template <int dimension>
    inline void free_space(Tensor<gpu, dimension> &t); 

    // store (cpu implemented, gpu hold)
    template <class SV, int dim>
    inline void store(Tensor<cpu, dim> t, real_t v);
    template <class SV, int dim>
    inline void store(Tensor<gpu, dim> t, real_t v);

    // copy
    template <int dim>
    inline void copy(Tensor<cpu, dim> dst, const Tensor<cpu, dim> &src);
    template <int dim>
    inline void copy(Tensor<gpu, dim> dst, const Tensor<cpu, dim> &src);
    template <int dim>
    inline void copy(Tensor<cpu, dim> dst, const Tensor<gpu, dim> &src);
    template <int dim>
    inline void copy(Tensor<gpu, dim> dst, const Tensor<gpu, dim> &src);
}; // lzc

namespace lzc {
    template <int dim>
    inline Tensor<cpu, dim> new_ctensor(const Shape<dim> &shape, real_t init_v);

    template <int dim>
    inline Tensor<gpu, dim> new_gtensor(const Shape<dim> &shape, real_t init_v);
}; // lzc

#include "tensor_cpu-impl.h"
#ifdef __CUDACC__
#include "cuda/tensor_gpu-impl.cuh"
#endif

#endif
