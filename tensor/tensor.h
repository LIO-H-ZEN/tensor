#ifndef LZC_TENSOR_H
#define LZC_TENSOR_H

#pragma GCC push_options
#pragma GCC optimize ("unroll-loops")

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
        return _shape[idx];
    }
    inline const index_t& stride( void ) const {
        return _stride;
    }
    inline bool operator==(const Shape<DIM> &other) const {
        #pragma unroll 
        for (int i = 0; i < DIM; ++i) {
            if (this->_shape[i] != other._shape[i]) {
                return false;
            }
        }
        return true;
    }
    inline Shape<2> flat_to_2d( void ) const {
        Shape<2> s;
        s._stride = this->_stride;
        s._shape[0] = this->_shape[0];
        index_t left = 1;
        #pragma unroll
        for (int i = 1; i < DIM; ++i) {
            left *= this->_shape[i];
        } 
        s._shape[1] = left;
        return s;
    }
    inline size_t size( void ) const {
        size_t ret = this->_shape[0];
        #pragma unroll
        for (int i = 1; i < DIM; ++i) {
            ret *= this->_shape[i];
        }
        return ret;
    } 
    virtual inline size_t mem_size( void ) const {}

    inline Shape<SUBDIM> sub_shape( void ) const {
        Shape<SUBDIM> s;
        s._stride = this->_stride;
        #pragma unroll
        for (int i = 1; i < DIM; ++i) {
            s._shape[i - 1] = this->_shape[i];
        }
        return s;
    }
public:
    // from left to right: 1st dim, 2st dim, ....
    index_t _shape[]; 
    index_t _stride;     
}; // class Shape
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
    real_t *_dptr;
    Shape<dimension> _shape;
    const static bool DEVICE_TYPE = device::isCPU;
    const static index_t DIM = dimension; 
    const static index_t SUBDIM = dimension - 1; 
public:
    Tensor( void ) {}
    Tensor(real_t *dptr, Shape<DIM> shape): _dptr(dptr),_shape(shape) {} 

    inline Tensor<device, 2> flat_to_2d() const{
        return Tensor((real_t*)_dptr, _shape.flat_to_2d());
    }  
    
    inline Tensor<device, SUBDIM> operator[](index_t idx) const {
        Shape<SUBDIM> = _shape.sub_shape();
        return Tensor((real_t*)_dptr + idx * s.mem_size(), s);
    }

    inline Tensor<device, DIM> slice(index_t se, index_t ed) const {
        Shape<DIM> s = this->_shape;
        s[0] = ed - se;
        return Tensor((real_t*)_dptr + se * s.sub_shape().mem_size(), s);
    }
}; // class Tensor
template <class device>
class Tensor<device, 1> {
public:
    real_t *_dptr;
    Shape<1> _shape;
    Tensor( void ) {}
    Tensor(real_t *dptr, Shape<1> shape) : _dptr(dptr),_shape(shape) {}
    
    inline real_t operator[](index_t idx) const {
        return _dptr[idx];
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

#pragma GCC pop_options
#endif