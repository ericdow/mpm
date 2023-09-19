#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

#include <array>
#include <vector>
#include <assert.h>
#include <iostream>

/*
Two-dimensional array class for representing device data
The first dimension is static, the second dimension is dynamic
All arrays are allocated on host and device
*/

using dimType = uint32_t;

template <typename dataT, dimType dim>
class Array2D
{
public:
    Array2D();
    Array2D(size_t size);
    ~Array2D();
    Array2D(const Array2D &other) = delete;
    Array2D &operator=(const Array2D &other) = delete;
    void resize(size_t size);
    dataT &operator()(dimType i, size_t j);
    const dataT &operator()(dimType i, size_t j) const;
    void syncToDevice();
    void syncToHost();

private:
    std::array<std::vector<dataT>, dim> h_data;
    std::array<dataT *, dim> d_data_ptrs; // device pointers on host
    dataT **d_data;                       // device pointers on device
};

template <typename dataT, dimType dim>
Array2D<dataT, dim>::Array2D()
{
    static_assert(dim > 1, "Use Array1D");
    checkCudaErrors(cudaMalloc((void **)&d_data, sizeof(dataT *) * dim));
}

template <typename dataT, dimType dim>
Array2D<dataT, dim>::Array2D(size_t size)
    : Array2D()
{
    for (dimType d = 0; d < dim; ++d)
    {
        checkCudaErrors(cudaMalloc((void **)&d_data_ptrs[d], sizeof(dataT) * size));
        h_data[d].resize(size);
    }
    checkCudaErrors(cudaMemcpy(d_data, d_data_ptrs.data(), sizeof(dataT *) * dim, cudaMemcpyHostToDevice));
}

template <typename dataT, dimType dim>
Array2D<dataT, dim>::~Array2D()
{
    if (h_data[0].size() > 0)
    {
        for (dimType d = 0; d < dim; ++d)
            checkCudaErrors(cudaFree(d_data_ptrs[d]));
        checkCudaErrors(cudaFree(d_data));
    }
}

template <typename dataT, dimType dim>
dataT &Array2D<dataT, dim>::operator()(dimType i, size_t j)
{
    assert(i < dim);
    assert(j < h_data[0].size());
    return h_data[i][j];
}

template <typename dataT, dimType dim>
const dataT &Array2D<dataT, dim>::operator()(dimType i, size_t j) const
{
    assert(i < dim);
    assert(j < h_data[0].size());
    return h_data[i][j];
}

template <typename dataT, dimType dim>
void Array2D<dataT, dim>::resize(size_t size)
{
    for (dimType d = 0; d < dim; ++d)
    {
        if (h_data[d].size() > 0)
            checkCudaErrors(cudaFree(d_data_ptrs[d]));
        checkCudaErrors(cudaMalloc((void **)&d_data_ptrs[d], sizeof(dataT) * size));
        h_data[d].resize(size);
    }
    checkCudaErrors(cudaMemcpy(d_data, d_data_ptrs.data(), sizeof(dataT *) * dim, cudaMemcpyHostToDevice));
}

template <typename dataT, dimType dim>
void Array2D<dataT, dim>::syncToDevice()
{
    for (dimType d = 0; d < dim; ++d)
        checkCudaErrors(cudaMemcpy(d_data_ptrs[d], h_data[d].data(), sizeof(dataT) * h_data[d].size(), cudaMemcpyHostToDevice));
}

template <typename dataT, dimType dim>
void Array2D<dataT, dim>::syncToHost()
{
    for (dimType d = 0; d < dim; ++d)
        checkCudaErrors(cudaMemcpy(h_data[d].data(), d_data_ptrs[d], sizeof(dataT) * h_data[d].size(), cudaMemcpyDeviceToHost));
}