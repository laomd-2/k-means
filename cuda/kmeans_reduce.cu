#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <random>
#include <vector>
#include <fstream>
#include <iostream>

__device__ float
squared_l2_distance(float x_1, float y_1, float x_2, float y_2) {
  return (x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2);
}

__global__ void fine_reduce(const float* __restrict__ data_x,
                            const float* __restrict__ data_y,
                            int data_size,
                            const float* __restrict__ means_x,
                            const float* __restrict__ means_y,
                            int * __restrict__ label,
                            float* __restrict__ new_sums_x,
                            float* __restrict__ new_sums_y,
                            int k,
                            int* __restrict__ counts) {
  extern __shared__ float shared_data[];

  const int local_index = threadIdx.x;
  const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_index >= data_size) return;

  // Load the mean values into shared memory.
  if (local_index < k) {
    shared_data[local_index] = means_x[local_index];
    shared_data[k + local_index] = means_y[local_index];
  }

  __syncthreads();

  // Load once here.
  const float x_value = data_x[global_index];
  const float y_value = data_y[global_index];

  float best_distance = FLT_MAX;
  int best_cluster = -1;
  for (int cluster = 0; cluster < k; ++cluster) {
    const float distance = squared_l2_distance(x_value,
                                               y_value,
                                               shared_data[cluster],
                                               shared_data[k + cluster]);
    if (distance < best_distance) {
      best_distance = distance;
      best_cluster = cluster;
    }
  }
  label[global_index] = best_cluster;

  __syncthreads();

  // reduction

  const int x = local_index;
  const int y = local_index + blockDim.x;
  const int count = local_index + blockDim.x + blockDim.x;

  for (int cluster = 0; cluster < k; ++cluster) {
    shared_data[x] = (best_cluster == cluster) ? x_value : 0;
    shared_data[y] = (best_cluster == cluster) ? y_value : 0;
    shared_data[count] = (best_cluster == cluster) ? 1 : 0;
    __syncthreads();

    // Reduction for this cluster.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (local_index < stride) {
        shared_data[x] += shared_data[x + stride];
        shared_data[y] += shared_data[y + stride];
        shared_data[count] += shared_data[count + stride];
      }
      __syncthreads();
    }

    // Now shared_data[0] holds the sum for x.

    if (local_index == 0) {
      const int cluster_index = blockIdx.x * k + cluster;
      new_sums_x[cluster_index] = shared_data[x];
      new_sums_y[cluster_index] = shared_data[y];
      counts[cluster_index] = shared_data[count];
    }
    __syncthreads();
  }
}

__global__ void coarse_reduce(float* __restrict__ means_x,
                              float* __restrict__ means_y,
                              float* __restrict__ new_sum_x,
                              float* __restrict__ new_sum_y,
                              int k,
                              int* __restrict__ counts) {
  extern __shared__ float shared_data[];

  const int index = threadIdx.x;
  const int y_offset = blockDim.x;

  shared_data[index] = new_sum_x[index];
  shared_data[y_offset + index] = new_sum_y[index];
  __syncthreads();

  for (int stride = blockDim.x / 2; stride >= k; stride /= 2) {
    if (index < stride) {
      shared_data[index] += shared_data[index + stride];
      shared_data[y_offset + index] += shared_data[y_offset + index + stride];
    }
    __syncthreads();
  }

  if (index < k) {
    const int count = max(1, counts[index]);
    means_x[index] = new_sum_x[index] / count;
    means_y[index] = new_sum_y[index] / count;
    new_sum_y[index] = 0;
    new_sum_x[index] = 0;
    counts[index] = 0;
  }
}

int main(int argc, const char *argv[])
{
    std::ifstream fx(argv[1]), fy(argv[2]);
    thrust::host_vector<float> h_x;
    thrust::host_vector<float> h_y;

    // Load x and y into host vectors ... (omitted)
    float x;
    int n, k;
    fx >> n >> k;
    fy >> n >> k;
    for (int i = 0; i < n; i++)
    {
        fx >> x;
        h_x.push_back(x);
        fy >> x;
        h_y.push_back(x);
    }

    const size_t number_of_elements = h_x.size();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    thrust::device_vector<float> d_x = h_x;
    thrust::device_vector<float> d_y = h_y;

    std::mt19937 rng(std::random_device{}());
    std::shuffle(h_x.begin(), h_x.end(), rng);
    std::shuffle(h_y.begin(), h_y.end(), rng);
    thrust::device_vector<float> d_mean_x(h_x.begin(), h_x.begin() + k);
    thrust::device_vector<float> d_mean_y(h_y.begin(), h_y.begin() + k);

    thrust::device_vector<float> d_sums_x(k);
    thrust::device_vector<float> d_sums_y(k);
    thrust::device_vector<int> d_counts(k, 0), d_label(number_of_elements, 0);

    const int threads = 1024;
    const int blocks = (number_of_elements + threads - 1) / threads;

    // * 3 for x, y and counts.
    const int fine_shared_memory = 3 * threads * sizeof(float);
    // * 2 for x and y. Will have k * blocks threads for the coarse reduction.
    const int coarse_shared_memory = 2 * k * blocks * sizeof(float);

    int number_of_iterations = 100;
    for (size_t iteration = 0; iteration < number_of_iterations; ++iteration)
    {
        fine_reduce<<<blocks, threads, fine_shared_memory>>>(thrust::raw_pointer_cast(d_x.data()),
                                                             thrust::raw_pointer_cast(d_y.data()),
                                                             number_of_iterations,
                                                             thrust::raw_pointer_cast(d_mean_x.data()),
                                                             thrust::raw_pointer_cast(d_mean_y.data()),
                                                             thrust::raw_pointer_cast(d_label.data()),
                                                             thrust::raw_pointer_cast(d_sums_x.data()),
                                                             thrust::raw_pointer_cast(d_sums_y.data()),
                                                             k,
                                                             thrust::raw_pointer_cast(d_counts.data()));
        cudaDeviceSynchronize();

        coarse_reduce<<<1, k * blocks, coarse_shared_memory>>>(thrust::raw_pointer_cast(d_mean_x.data()),
                                                               thrust::raw_pointer_cast(d_mean_y.data()),
                                                               thrust::raw_pointer_cast(d_sums_x.data()),
                                                               thrust::raw_pointer_cast(d_sums_y.data()),
                                                               k,
                                                               thrust::raw_pointer_cast(d_counts.data()));

        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << milliseconds / 1000.0 << "s" << std::endl;

    thrust::host_vector<float> h_mean_x(d_mean_x), h_mean_y(d_mean_y);
    thrust::host_vector<int> h_label(d_label);
    std::ofstream fout(argv[3]);
    fout << h_label.size() << std::endl;
    for (int x : h_label)
        fout << x << ' ';
    fout << std::endl;

    fout << k << std::endl;
    for (int i = 0; i < k; i++)
    {
        fout << h_mean_x[i] << ' ' << h_mean_y[i] << ' ';
    }
    fout << std::endl;
}