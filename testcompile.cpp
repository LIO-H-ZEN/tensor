#include "tensor/tensor.h"

using namespace lzc;
using namespace std;

void test_cpu_tensor( CTensor2D &t ) {
    CTensor2D t2(shape2(10, 10));
    cout << "alloc space for t2 .... " << endl;
    alloc_space(t2);
    copy(t2, t); 
    cout << "CPU-t2:" << endl;
    for (index_t i = 0; i < t2._shape[0]; ++i) {
        for (index_t j = 0; j < t2._shape[1]; ++j) {
            cout << t2[i][j] << " ";
        }
        cout << endl;
    }
    free_space(t2);
}

void test_gpu_tensor(CTensor2D &t);

int main( void ) {
    cout << "t1:" << endl;
    CTensor2D t = new_ctensor(shape2(10, 10), -1);
    for (index_t i = 0; i < t._shape[0]; ++i) {
        for (index_t j = 0; j < t._shape[1]; ++j) {
            cout << t[i][j] << " ";
        }
        cout << endl;
    }
    test_cpu_tensor(t);
    test_gpu_tensor(t);
    free_space(t);
}
