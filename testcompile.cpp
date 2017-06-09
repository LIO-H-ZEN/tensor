#include "tensor/tensor.h"

using namespace lzc;
using namespace std;

int main( void ) {
    /*
    float a[100];
    Shape<2> s;
    s._stride = 1;
    s[0] = 10;
    s[1] = 10;
    for (int i =0; i < 100; ++i) {
        a[i] = i;
    }
    CTensor2D mat(&a[0], s);

    // test [] slice
    cout << mat[0][0] << endl;    
    cout << mat[1][0] << endl;    
    cout << mat[0][1] << endl;    
    cout << mat[9][0] << endl;    
    cout << mat[9][9] << endl;    
    CTensor2D m = mat.slice(9, 10);
    CTensor1D m1 = mat[9].slice(7, 9);
    cout << m[0][7] << endl; 
    cout << m1[0] << endl;

    // test op
    float b[100];
    CTensor2D pmat(&b[0], s); 
    map<sv::saveto, op::plus>(pmat, mat, mat); 
    cout << pmat[0][0] << endl;
    cout << pmat[0][1] << endl;
    cout << pmat[1][0] << endl;
    */
    CTensor2D t = new_ctensor(shape2(10, 10), -1);
    cout << t[1][0] << endl;
}
