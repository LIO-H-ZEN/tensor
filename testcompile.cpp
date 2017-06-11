#include "tensor/tensor.h"

using namespace lzc;
using namespace std;

void test_cpu_tensor( void ) {
    cout << "t1:" << endl;
    CTensor2D t = new_ctensor(shape2(10, 10), -1);
    for (index_t i = 0; i < t._shape[0]; ++i) {
        for (index_t j = 0; j < t._shape[1]; ++j) {
            cout << t[i][j] << " ";
        }
        cout << endl;
    }
    CTensor2D t2(shape2(10, 10));
    cout << "alloc space for t2 .... " << endl;
    alloc_space(t2);
    copy(t2, t); 
    cout << "t2:" << endl;
    for (index_t i = 0; i < t2._shape[0]; ++i) {
        for (index_t j = 0; j < t2._shape[1]; ++j) {
            cout << t2[i][j] << " ";
        }
        cout << endl;
    }
}

int main( void ) {
    test_cpu_tensor();
}
