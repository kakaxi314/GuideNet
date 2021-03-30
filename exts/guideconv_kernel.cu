//
// Created by jie on 09/02/19.
//

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

namespace {

    template<typename scalar_t>
    __global__ void
    conv2d_kernel_lf(scalar_t *__restrict__ x, scalar_t *__restrict__ y, scalar_t *__restrict__ z, size_t N1,
                        size_t N2, size_t Ci, size_t Co, size_t B,
                        size_t K) {
        int col_index = threadIdx.x + blockIdx.x * blockDim.x;
        int row_index = threadIdx.y + blockIdx.y * blockDim.y;
        int cha_index = threadIdx.z + blockIdx.z * blockDim.z;
        if ((row_index < N1) && (col_index < N2) && (cha_index < Co)) {
            for (int b = 0; b < B; b++) {
                scalar_t result = 0;
                for (int i = -int((K - 1) / 2.); i < (K + 1) / 2.; i++) {
                    for (int j = -int((K - 1) / 2.); j < (K + 1) / 2.; j++) {

                        if ((row_index + i < 0) || (row_index + i >= N1) || (col_index + j < 0) ||
                            (col_index + j >= N2)) {
                            continue;
                        }

                        result += x[b * N1 * N2 * Ci + cha_index * N1 * N2 + (row_index + i) * N2 + col_index + j] *
                                  y[b * N1 * N2 * Ci * K * K + cha_index * N1 * N2 * K * K +
                                    (i + (K - 1) / 2) * K * N1 * N2 +
                                    (j + (K - 1) / 2) * N1 * N2 + row_index * N2 + col_index];
                    }
                }
                z[b * N1 * N2 * Co + cha_index * N1 * N2 + row_index * N2 + col_index] = result;
            }
        }
    }


    template<typename scalar_t>
    __global__ void conv2d_kernel_lb(scalar_t *__restrict__ x, scalar_t *__restrict__ y, scalar_t *__restrict__ gx,
                                         scalar_t *__restrict__ gy, scalar_t *__restrict__ gz, size_t N1, size_t N2,
                                         size_t Ci, size_t Co, size_t B,
                                         size_t K) {
        int col_index = threadIdx.x + blockIdx.x * blockDim.x;
        int row_index = threadIdx.y + blockIdx.y * blockDim.y;
        int cha_index = threadIdx.z + blockIdx.z * blockDim.z;
        if ((row_index < N1) && (col_index < N2) && (cha_index < Co)) {
            for (int b = 0; b < B; b++) {
                scalar_t result = 0;
                for (int i = -int((K - 1) / 2.); i < (K + 1) / 2.; i++) {
                    for (int j = -int((K - 1) / 2.); j < (K + 1) / 2.; j++) {

                        if ((row_index - i < 0) || (row_index - i >= N1) || (col_index - j < 0) ||
                            (col_index - j >= N2)) {
                            continue;
                        }
                        result += gz[b * N1 * N2 * Ci + cha_index * N1 * N2 + (row_index - i) * N2 + col_index - j
                                  ] *
                                  y[b * N1 * N2 * Ci * K * K + cha_index * N1 * N2 * K * K +
                                    (i + (K - 1) / 2) * K * N1 * N2 +
                                    (j + (K - 1) / 2) * N1 * N2 + (row_index - i) * N2 + col_index - j];
                        gy[b * N1 * N2 * Ci * K * K + cha_index * N1 * N2 * K * K + (i + (K - 1) / 2) * K * N1 * N2 +
                           (j + (K - 1) / 2) * N1 * N2 + (row_index - i) * N2 + col_index - j] =
                                gz[b * N1 * N2 * Ci + cha_index * N1 * N2 + (row_index - i) * N2 + col_index - j
                                ] * x[b * N1 * N2 * Ci + cha_index * N1 * N2 + row_index * N2 + col_index];

                    }
                }
                gx[b * N1 * N2 * Co + cha_index * N1 * N2 + row_index * N2 + col_index] = result;
            }
        }
    }
}


void Conv2d_LF_Cuda(at::Tensor x, at::Tensor y, at::Tensor z, size_t N1, size_t N2, size_t Ci, size_t Co, size_t B,
                    size_t K) {
    dim3 blockSize(32, 32, 1);
    dim3 gridSize((N2 + blockSize.x - 1) / blockSize.x, (N1 + blockSize.y - 1) / blockSize.y,
                  (Co + blockSize.z - 1) / blockSize.z);
    AT_DISPATCH_FLOATING_TYPES(x.type(), "Conv2d_LF", ([&] {
        conv2d_kernel_lf<scalar_t> << < gridSize, blockSize >> > (
                x.data<scalar_t>(), y.data<scalar_t>(), z.data<scalar_t>(),
                        N1, N2, Ci, Co, B, K);
    }));
}


void
Conv2d_LB_Cuda(at::Tensor x, at::Tensor y, at::Tensor gx, at::Tensor gy, at::Tensor gz, size_t N1, size_t N2, size_t Ci,
               size_t Co, size_t B, size_t K) {
    dim3 blockSize(32, 32, 1);
    dim3 gridSize((N2 + blockSize.x - 1) / blockSize.x, (N1 + blockSize.y - 1) / blockSize.y,
                  (Co + blockSize.z - 1) / blockSize.z);
    AT_DISPATCH_FLOATING_TYPES(x.type(), "Conv2d_LB", ([&] {
        conv2d_kernel_lb<scalar_t> << < gridSize, blockSize >> > (
                x.data<scalar_t>(), y.data<scalar_t>(),
                        gx.data<scalar_t>(), gy.data<scalar_t>(), gz.data<scalar_t>(),
                        N1, N2, Ci, Co, B, K);
    }));
}
