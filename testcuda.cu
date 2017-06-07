#include "tensor/tensor.h"

using namespace lzc;
using namespace std;

int main( void ) {
    float a[100];
    Shape<2> s;
    s._stride = 1;
    s[0] = 10;
    s[1] = 10;
    for (int i =0; i < 100; ++i) {
        a[i] = i;
    }
    CTensor2D mat(&a[0], s);
    cout << mat[0][0] << endl;    
    cout << mat[1][0] << endl;    
    cout << mat[0][1] << endl;    
    cout << mat[9][0] << endl;    
    cout << mat[9][9] << endl;    
    CTensor2D m = mat.slice(9, 10);
    cout << m[0][7] << endl; 
}
