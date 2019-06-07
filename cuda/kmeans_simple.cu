#if !defined(KMEANS_CUDA)
#define KMEANS_CUDA

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <random>
#include "kmeans.cuh"

__device__ float squared_l2_distance(float x_1, float y_1, float x_2, float y_2) {
  return (x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2);
}

// In the assignment step, each point (thread) computes its distance to each
// cluster centroid and adds its x and y values to the sum of its closest
// centroid, as well as incrementing that centroid's count of assigned points.
__global__ void assign_clusters(const thrust::device_ptr<float> data_x,
                     const thrust::device_ptr<float> data_y,
                     int data_size,
                     const thrust::device_ptr<float> means_x,
                     const thrust::device_ptr<float> means_y,
                     thrust::device_ptr<int> label,
                     thrust::device_ptr<float> new_sums_x,
                     thrust::device_ptr<float> new_sums_y,
                     int k, int p,
                     thrust::device_ptr<int> counts)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = blockDim.x * gridDim.x;
    for (int i = 0; i < p; i++) {
        int index = i * width + id;
        if (index >= data_size)
            return;

        // Make global loads once.
        const float x = data_x[index];
        const float y = data_y[index];

        float best_distance = FLT_MAX;
        int best_cluster = 0;
        for (int cluster = 0; cluster < k; ++cluster)
        {
            const float distance =
                squared_l2_distance(x, y, means_x[cluster], means_y[cluster]);
            if (distance < best_distance)
            {
                best_distance = distance;
                best_cluster = cluster;
            }
        }

        label[index] = best_cluster;
        atomicAdd(thrust::raw_pointer_cast(new_sums_x + best_cluster), x);
        atomicAdd(thrust::raw_pointer_cast(new_sums_y + best_cluster), y);
        atomicAdd(thrust::raw_pointer_cast(counts + best_cluster), 1);
    }
}

// Each thread is one cluster, which just recomputes its coordinates as the mean
// of all points assigned to it.
__global__ void compute_new_means(thrust::device_ptr<float> means_x,
                       thrust::device_ptr<float> means_y,
                       const thrust::device_ptr<float> new_sum_x,
                       const thrust::device_ptr<float> new_sum_y,
                       const thrust::device_ptr<int> counts)
{
    const int cluster = threadIdx.x;
    const int count = max(1, counts[cluster]);
    means_x[cluster] = new_sum_x[cluster] / count;
    means_y[cluster] = new_sum_y[cluster] / count;
}

#endif // KMEANS_CUDA
