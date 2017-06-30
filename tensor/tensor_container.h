#ifndef LZC_TENSOR_CONTAINER
#define LZC_TENSOR_CONTAINER

#include "tensor.h"

namespace lzc {
template <class device, int dim>
class TensorContainer{
public:
    TensorContainer( void ) {
        _data._shape._stride = 0;
        _data._dptr = NULL;
        _view._dptr = NULL; 
    }
    TensorContainer(const Shape<dim> &shape) {
        _data._dptr = NULL;
        this->container_alloc_by_shape(shape); 
    }
    ~TensorContainer(void) {
        this->container_free_space();
    }

    inline void resize(const Shape<dim> &shape) {
        Shape<2> shape_2d = shape.flat_to_2d(true);
        if (_data._shape[0] > shape_2d[0] || _data._shape[1] > shape_2d[1]) {
            container_alloc_by_shape(shape);
        } else {
            _view._shape = shape;
            _view._shape._stride = _data._shape._stride;
        }
    }
    
    // 隐式类型转换 
    inline operator Tensor<device, dim> (void) const{
        return _view;
    }
    // 重载
    inline Tensor<device, dim> operator()(void) const{
        return _view;
    }
private:
    inline void container_free_space(void) {
        if (_data._dptr != NULL) {
            free_space(_data);
            _data._dptr = _view._dptr = NULL; 
        }
    }
    inline void container_alloc_by_shape(const Shape<dim> &shape) {
        if (_data._dptr != NULL) {
            this->container_free_space();
        } 
        _data._shape = shape.flat_to_2d();
        alloc_space(_data);
        _view._dptr = _data._dptr;
        _view._shape = shape;
        _view._shape._stride = _data._shape._stride;
    }
    // 实际上都是2d tensor，尤其是gpu，申请的内存都是2D的 
    Tensor<device, 2> _data;    
    // 使用方看到的tensor,和_data共享一个_dptr内存 
    Tensor<device, dim> _view;
}; // tc 
}; // lzc
#endif
