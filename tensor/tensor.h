#ifndef LZC_TENSOR_H
#define LZC_TENSOR_H

#include <iostream>

namespace lzc {
    typedef float real_t;
    typedef unsigned index_t; // 0 ~ 65535 limit of #(gpu thread block) 
}; // namepsace lzc

namespace lzc {
template <int dimension>
class Shape {
public:
    const static int DIM = dimension;
    const static int SUBDIM = dimension - 1;
public:
    Shape( void ) {}
    inline index_t& operator[](size_t idx) {
        return _array[idx];
    }
    inline const index_t& stride( void ) const {
        return _stride;
    }
    inline bool operator==(const Shape<DIM> &other) const {
        #pragma unroll 
        for (int i = 0; i < DIM; ++i) {
            if (this->_array[i] != other._array[i]) {
                return false;
            }
        }
        return true;
    }

    inline Shape<2> flat_to_2d( void ) const {
        Shape<2> s;
        s._stride = this->_stride;
        s._array[0] = this->_array[0];
        index_t left = 1;
        #pragma unroll
        for (int i = 1; i < DIM; ++i) {
            left *= this->_array[i];
        } 
        s._array[1] = left;
        return s;
    }
    inline size_t size( void ) const {
        size_t ret = this->_array[0];
        #pragma unroll
        for (int i = 1; i < DIM; ++i) {
            ret *= this->_array[i];
        }
        return ret;
    } 

    // hard-code just to make it work
    inline size_t mem_size( void ) const {
        // stride reserve the basic dim
        size_t ret = this->_stride;
        #pragma unroll
        for (int i = 0; i < DIM - 1; ++i) {
            ret *= this->_array[i];
        }
        return ret;
    }

    inline Shape<SUBDIM> sub_shape( void ) const {
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

inline Shape<1> shape1(index_t a1) {
    Shape<1> s;
    s[0] = a1;
    return s;
}

inline Shape<2> shape2(index_t a1, index_t a2) {
    Shape<2> s;
    s[0] = a1;
    s[1] = a2;
    return s;
}

inline Shape<3> shape3(index_t a1, index_t a2, index_t a3) {
    Shape<3> s;
    s[0] = a1;
    s[1] = a2;
    s[2] = a3;
    return s;
}

inline Shape<4> shape4(index_t a1, index_t a2, index_t a3, index_t a4) {
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

template <class device, int dimension>
class Tensor {
public:
    const static bool DEVICE_TYPE = device::isCPU;
    const static index_t DIM = dimension; 
    const static index_t SUBDIM = dimension - 1; 
    real_t *_dptr;
    Shape<DIM> _shape;
public:
    Tensor( void ) {}
    Tensor(Shape<DIM> shape) : _shape(shape) {}
    Tensor(real_t *dptr, Shape<DIM> shape): _dptr(dptr),_shape(shape) {} 

    inline Tensor<device, 2> flat_to_2d() const{
        return Tensor<device, 2>((real_t*)_dptr, _shape.flat_to_2d());
    }  
    
    inline Tensor<device, SUBDIM> operator[](index_t idx) const {
        Shape<SUBDIM> s = this->_shape.sub_shape();
        return Tensor<device, SUBDIM>((real_t*)_dptr + idx * s.mem_size(), s);
    }

    inline Tensor<device, DIM> slice(index_t se, index_t ed) const {
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
    Tensor( void ) {}
    Tensor(real_t *dptr, Shape<1> shape) : _dptr(dptr),_shape(shape) {}
    
    inline real_t& operator[](index_t idx) const {
        return _dptr[idx];
    } 

    inline Tensor<device, 1> slice(int se, int ed) const {
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
    template <class SV, class OP>
    inline void map(CTensor2D dst, const CTensor2D &lst, const CTensor2D &rst);
    inline void map(GTensor2D dst, const GTensor2D &lst, const GTensor2D &rst);

    // alloc memory for tensor according to its shape
    // and set its stride
    template <int dimension>
    inline void alloc_space(Tensor<cpu, dimension> &t);

    template <int dimension>
    inline void alloc_space(Tensor<gpu, dimension> &t);

    template <int dimension>
    inline void free_space(Tensor<cpu, dimension> &t); 

    template <int dimension>
    inline void free_space(Tensor<gpu, dimension> &t); 

    template <class SV>
    inline void store(CTensor2D t, real_t v);

    template <class SV>
    inline void store(GTensor2D t, real_t v);
}; // lzc

namespace lzc {
    template <int dimension>
    inline Tensor<cpu, dimension> new_ctensor(const Shape<dimension> &shape, real_t init_v);

}; // lzc

#include "tensor_cpu-impl.h"
#ifdef __CUDA_ARCH__
// #include "cuda/tensor_gpu-impl.h"
#endif

#endif
