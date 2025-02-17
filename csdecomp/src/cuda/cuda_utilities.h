#pragma once
#include <cuda_runtime.h>

#include <cassert>
#include <cfloat>
#include <iostream>

namespace csdecomp {
// In cuda_utilites all handy device functions are gathered.

#define CUDACHECKERROR(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

/**
 * @brief A structure for managing GPU memory.
 *
 * This struct provides a simple interface for allocating and managing memory
 * that can be shared between host and device in CUDA applications.
 */
template <typename T>
struct CudaPtr {
  T *host;    ///< Pointer to host memory
  T *device;  ///< Pointer to device (GPU) memory
  unsigned long size;

  CudaPtr() : host(nullptr), device(nullptr), size(0) {}

  CudaPtr(T *data, unsigned long size) : host(data), size(size) {
    if (cudaMalloc(&device, size * sizeof(T)) != ::cudaSuccess) {
      throw std::runtime_error("Failed to cudaMalloc!");
    }
  }

  CudaPtr(const CudaPtr &) = delete;
  CudaPtr &operator=(const CudaPtr &) = delete;

  ~CudaPtr() { CUDACHECKERROR(cudaFree((void *)device)); }

  void copyHostToDevice() {
    if (host == nullptr) {
      throw std::runtime_error("Tried to copy from a dangling host pointer!");
    }

    cudaMemcpy((void *)device, host, size * sizeof(T), cudaMemcpyHostToDevice);
  }

  void copyDeviceToHost() {
    if (host == nullptr) {
      throw std::runtime_error("Tried to copy to dangling host pointer!");
    }

    cudaMemcpy((void *)host, device, size * sizeof(T), cudaMemcpyDeviceToHost);
  }
};

#ifdef __CUDACC__

// Sets 4x4 block to identity of a 4xN matrix starting from the start_ptr. This
// assumes column major storage of a 4xN matrix, and sets the following 16
// floats to identity.
__device__ inline void set4x4BlockIdentityOf4xNMatrix(float *start_ptr) {
  start_ptr[0] = 1;
  start_ptr[1] = 0;
  start_ptr[2] = 0;
  start_ptr[3] = 0;

  start_ptr[4] = 0;
  start_ptr[5] = 1;
  start_ptr[6] = 0;
  start_ptr[7] = 0;

  start_ptr[8] = 0;
  start_ptr[9] = 0;
  start_ptr[10] = 1;
  start_ptr[11] = 0;

  start_ptr[12] = 0;
  start_ptr[13] = 0;
  start_ptr[14] = 0;
  start_ptr[15] = 1;
}

// printing of colmajor 4x4 matrices
__device__ inline void printMatrix4x4(const float *matrix) {
  printf("4x4 Matrix (column-major):\n");
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      // Index calculation for column-major order
      int index = col * 4 + row;
      printf("%8.3f ", matrix[index]);
    }
    printf("\n");
  }
}

__device__ inline void printMatrixNxM(const float *matrix, const int N,
                                      const int M) {
  printf("%dx%d Matrix (column-major):\n", N, M);
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < M; ++col) {
      // Index calculation for column-major order
      int index = col * N + row;
      printf("%8.3f ", matrix[index]);
    }
    printf("\n");
  }
}

__device__ inline void printMatrixNxMRowMajor(const float *matrix, const int N,
                                              const int M) {
  printf("%dx%d Matrix (row-major):\n", N, M);
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < M; ++col) {
      // Index calculation for column-major order
      int index = col + row * M;
      printf("%8.3f ", matrix[index]);
    }
    printf("\n");
  }
}

// printing of float arrays, requries array length N.
__device__ inline void printArrayN(const float *arr, const int N) {
  printf("%dx1 Array float :\n", N);
  for (int idx = 0; idx < N; ++idx) {
    printf("%8.3f \n", arr[idx]);
  }
  printf("\n");
}

__device__ inline void printArrayN(const int *arr, const int N) {
  printf("%dx1 Array int:\n", N);
  for (int idx = 0; idx < N; ++idx) {
    printf("%d \n", arr[idx]);
  }
  printf("\n");
}

__device__ inline void printArrayN(const uint8_t *arr, const int N) {
  printf("%dx1 Array int:\n", N);
  for (int idx = 0; idx < N; ++idx) {
    printf("%d \n", arr[idx]);
  }
  printf("\n");
}

/**
 * @brief Performs matrix multiplication of A and B, storing the result in
 * 'result'.
 *
 * This device function multiplies matrix A (NxM) with matrix B (MxK) and stores
 * the result in the 'result' matrix (NxK). All matrices are assumed to be in
 * column-major order.
 *
 * @param result Pointer to the output matrix (NxK) to store the multiplication
 * result.
 * @param A Pointer to the first input matrix (NxM).
 * @param B Pointer to the second input matrix (MxK).
 * @param N Number of rows in matrix A and in the result matrix.
 * @param M Number of columns in matrix A and number of rows in matrix B.
 * @param K Number of columns in matrix B and in the result matrix.
 *
 * @note This function is designed to be called from device code (CUDA kernels).
 * @note All matrices are assumed to be in column-major order for memory access
 * efficiency.
 * @note No bounds checking is performed; ensure input dimensions are correct xD
 */
__device__ inline void matmulF(float *result, const float *A, const float *B,
                               const int N, const int M, const int K) {
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      float sum = 0;
      for (int m = 0; m < M; ++m) {
        // a_nm = A[n + m*N]
        // b_mk = B[m + k*M]
        sum += A[n + m * N] * B[m + k * M];
      }
      // result_nk = result[n + k*N]
      result[n + k * N] = sum;
    }
  }
}

/**
 * @brief Performs matrix multiplication of A and B, storing the result in
 * 'result'.
 *
 * This device function multiplies matrix A (NxM) with matrix B (MxK) and stores
 * the result in the 'result' matrix (NxK). All matrices are assumed to be in
 * column-major order.
 *
 * @param result Pointer to the output matrix (NxK) to store the multiplication
 * result.
 * @param A Pointer to the first input matrix (NxM).
 * @param B Pointer to the second input matrix (MxK).
 * @param N Number of rows in matrix A and in the result matrix.
 * @param M Number of columns in matrix A and number of rows in matrix B.
 * @param K Number of columns in matrix B and in the result matrix.
 *
 * @note This function is designed to be called from device code (CUDA kernels).
 * @note All matrices are assumed to be in column-major order for memory access
 * efficiency.
 * @note No bounds checking is performed; ensure input dimensions are correct xD
 */
__device__ inline void matmulFRowMajor(float *result, const float *A,
                                       const float *B, const int N, const int M,
                                       const int K) {
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      float sum = 0;
      for (int m = 0; m < M; ++m) {
        // a_nm = A[n*M + m]
        // b_mk = B[m*K + k]
        sum += A[n * M + m] * B[m * K + k];
      }
      // result_nk = result[n*K + k]
      result[n * K + k] = sum;
    }
  }
}

/**
 * @brief Finds the maximum difference between corresponding elements of vectors
 * a and b.
 *
 * This device function compares each pair of elements from vectors a and b,
 * calculates their difference (a[i] - b[i]), and returns the maximum difference
 * found.
 *
 * @param a Pointer to the first input vector.
 * @param b Pointer to the second input vector.
 * @param N Number of elements in each vector.
 *
 * @return The maximum value of (a[i] - b[i]) across all i from 0 to N-1.
 *
 * @note This function is designed to be called from device code (CUDA kernels).
 * @note The function assumes both vectors have at least N elements.
 * @note No bounds checking is performed; ensure input dimensions are correct.
 * @note If all differences are negative or both vectors are empty, it returns
 * -FLT_MAX.
 */
__device__ inline float findMaxAMinusB(const float *a, const float *b,
                                       const int N) {
  float result = -FLT_MAX;
  for (int n = 0; n < N; ++n) {
    if (a[n] - b[n] > result) {
      result = a[n] - b[n];
    }
  }
  return result;
}

/**
 * @brief Computes the element-wise difference of two vectors (a - b) on the
 * GPU.
 *
 * This device function subtracts each element of vector b from the
 * corresponding element of vector a and stores the result in the result vector.
 *
 * @param result Pointer to the output vector where the result will be stored
 * @param a Pointer to the first input vector
 * @param b Pointer to the second input vector
 * @param N Number of elements in each vector
 */
__device__ inline void vectorAMinusB(float *result, const float *a,
                                     const float *b, const int N) {
  for (int n = 0; n < N; ++n) {
    result[n] = a[n] - b[n];
  }
}

/**
 * @brief Calculates the appropriate grid and block sizes for CUDA kernel
 * launches.
 *
 * This function determines the optimal grid and block sizes for a 2D CUDA
 * kernel launch based on the desired number of threads and the maximum threads
 * per block.
 *
 * @param num_threads_for_x Number of threads needed in the x-dimension
 * @param num_threads_for_y Number of threads needed in the y-dimension
 * @param grid_size Pointer to store the calculated grid size
 * @param block_size Pointer to store the calculated block size
 * @param threads_per_block_in_x Number of threads per block in x-dimension
 * (default: 32)
 * @param threads_per_block_in_y Number of threads per block in y-dimension
 * (default: 32)
 *
 * @note The function asserts that the grid_size and block_size pointers are not
 * null, and that the total threads per block does not exceed 1024.
 */
inline void GetGridAndBlockSize(int num_threads_for_x, int num_threads_for_y,
                                dim3 *grid_size, dim3 *block_size,
                                int threads_per_block_in_x = 32,
                                int threads_per_block_in_y = 32) {
  assert(grid_size != nullptr && block_size != nullptr);
  constexpr int kMaxThreadsPerBlock = 1024;
  assert(threads_per_block_in_x * threads_per_block_in_y <=
         kMaxThreadsPerBlock);
  assert(threads_per_block_in_x > 0 && threads_per_block_in_y > 0);
  const dim3 multi_block_size(threads_per_block_in_x, threads_per_block_in_y);
  const int bx =
      (num_threads_for_x + multi_block_size.x - 1) / multi_block_size.x;
  const int by =
      (num_threads_for_y + multi_block_size.y - 1) / multi_block_size.y;
  const dim3 multi_grid_size(bx, by);
  *grid_size = dim3(bx, by);
  *block_size = dim3(threads_per_block_in_x, threads_per_block_in_y);
}
#endif

}  // namespace csdecomp