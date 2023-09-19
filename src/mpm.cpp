#include "array/Array2D.cuh"

int main()
{
    Array2D<float, 3> a(10);
    a.resize(20);
    a.syncToDevice();
}