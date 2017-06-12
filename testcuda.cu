#include "tensor/tensor.h"

using namespace lzc;
using namespace std;

void test_gpu_tensor( CTensor2D &t) {
    GTensor2D t2(shape2(10, 10));
    alloc_space(t2);
    copy(t2, t); 
    cout << "GPU-t2 alloc done. copy done." << endl;
    // 切记gtensor要想在host访问，要copy回来。
    // 另外发现实际分配的width不是10，而是128
    /*
    for (index_t i = 0; i < t2._shape[0]; ++i) {
        for (index_t j = 0; j < t2._shape[1]; ++j) {
            cout << t2[i][j] << " ";
        }
        cout << endl;
    }
    */
    CTensor2D res(shape2(10, 10));
    alloc_space(res);
    copy(res, t2);
    cout << "copy from GPU-t2 to res and print res:" << endl; 
    for (index_t i = 0; i < res._shape[0]; ++i) {
        for (index_t j = 0; j < res._shape[1]; ++j) {
            cout << res[i][j] << " ";
        }
        cout << endl;
    }
    free_space(t2);
    free_space(res);
}
