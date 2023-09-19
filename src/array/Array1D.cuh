#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

#include <vector>
#include <assert.h>

/*
One-dimensional array class for representing device data
All arrays are allocated on host and device
*/

template <typename dataT>
class Array1D
{
public:
    Array1D() = default;
    Array1D(size_t size);
    ~Array1D();
    Array1D(const Array1D &other) = delete;
    Array1D &operator=(const Array1D &other) = delete;
    void resize(size_t size);
    dataT &operator()(size_t i);
    const dataT &operator()(size_t i) const;
    void syncToDevice();
    void syncToHost();

private:
    std::vector<dataT> h_data;
    dataT *d_data;
};

template <typename dataT>
Array1D<dataT>::Array1D(size_t size)
{
    checkCudaErrors(cudaMalloc((void **)&d_data, sizeof(dataT) * size));
    h_data.resize(size);
}

template <typename dataT>
Array1D<dataT>::~Array1D()
{
    if (h_data.size() > 0)
        checkCudaErrors(cudaFree(d_data));
}

template <typename dataT>
dataT &Array1D<dataT>::operator()(size_t i)
{
    assert(i < h_data.size());
    return h_data[i];
}

template <typename dataT>
const dataT &Array1D<dataT>::operator()(size_t i) const
{
    assert(i < h_data.size());
    return h_data[i];
}

template <typename dataT>
void Array1D<dataT>::resize(size_t size)
{
    if (h_data.size() > 0)
        checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaMalloc((void **)&d_data, sizeof(dataT) * size));

    h_data.resize(size);
}

template <typename dataT>
void Array1D<dataT>::syncToDevice()
{
    checkCudaErrors(cudaMemcpy(d_data, h_data.data(), sizeof(dataT) * h_data.size(), cudaMemcpyHostToDevice));
}

template <typename dataT>
void Array1D<dataT>::syncToHost()
{
    checkCudaErrors(cudaMemcpy(h_data.data(), d_data, sizeof(dataT) * h_data.size(), cudaMemcpyDeviceToHost));
}