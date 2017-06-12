#include "tensor/tensor.h"

using namespace lzc;
using namespace std;

void test_gpu_tensor( CTensor2D &t) {
    GTensor2D t2 = new_gtensor(shape2(10, 10), -2);
    copy(t2, t); 
    cout << "GPU-t2 alloc done. copy done." << endl;

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

    free_space(res);

    GTensor2D t3 = new_gtensor(shape2(10, 10), -1);
    GTensor2D t4 = new_gtensor(shape2(10, 10), -99);
    map<sv::saveto, op::plus>(t4, t3, t2);    
    
    cout << "gpu-plus done. res:" << endl;  
    CTensor2D t5(shape2(10, 10));
    alloc_space(t5);
    copy(t5, t4);
    for (index_t i = 0; i < t5._shape[0]; ++i) {
        for (index_t j = 0; j < t5._shape[1]; ++j) {
            cout << t5[i][j] << " ";
        }
        cout << endl;
    }

    free_space(t2);
    free_space(t3);
    free_space(t4);
    free_space(t5);
}
