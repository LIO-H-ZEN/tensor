#include "tensor/tensor.h"
#include "tensor/tensor_container.h"

using namespace lzc;
using namespace std;

extern void test_gpu_tensor_2(CTensor3D &t);

void test_gpu_tensor_2(CTensor3D &t) {
    Shape<3> s = t._shape; 
    TensorContainer<gpu, 3> gmat1(s), gmat2(s), gmat3(s);
    copy(gmat1(), t);
    copy(gmat2(), t);
    map<sv::saveto, op::plus>(gmat3(), gmat2(), gmat1()); 
    copy(t, gmat3());
    for (index_t i = 0; i < s[0]; ++i) {
        for (index_t j = 0; j < s[1]; ++j) {
            for (index_t k = 0; k < s[2]; ++k) {
                cout << t[i][j][k] << " ";
            }
        }
        cout << endl;
    }

    cout << "Tensor = TensorContainer test: " << endl;
    GTensor3D gmt1 = gmat1;
    copy(t, gmt1);
    for (index_t i = 0; i < s[0]; ++i) {
        for (index_t j = 0; j < s[1]; ++j) {
            for (index_t k = 0; k < s[2]; ++k) {
                cout << t[i][j][k] << " ";
            }
        }
        cout << endl;
    }
}
