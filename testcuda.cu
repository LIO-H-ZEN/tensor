#include "tensor/tensor.h"

using namespace lzc;
using namespace std;

void test_gpu_tensor( CTensor2D &t) {
    GTensor2D t2(shape2(10, 10));
    /*copy(t2, t); 
    cout << "t2:" << endl;
    for (index_t i = 0; i < t2._shape[0]; ++i) {
        for (index_t j = 0; j < t2._shape[1]; ++j) {
            cout << t2[i][j] << " ";
        }
        cout << endl;
    }*/

}
