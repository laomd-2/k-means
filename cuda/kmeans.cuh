#if !defined(KMEANS_CUDA)
#define KMEANS_CUDA

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <random>

__device__ float squared_l2_distance(float x_1, float y_1, float x_2, float y_2);

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
                                thrust::device_ptr<int> counts);

// Each thread is one cluster, which just recomputes its coordinates as the mean
// of all points assigned to it.
__global__ void compute_new_means(thrust::device_ptr<float> means_x,
                                  thrust::device_ptr<float> means_y,
                                  const thrust::device_ptr<float> new_sum_x,
                                  const thrust::device_ptr<float> new_sum_y,
                                  const thrust::device_ptr<int> counts);

#endif // KMEANS_CUDA
