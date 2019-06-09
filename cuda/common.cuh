#ifndef __LAOMD__UTILS
#define __LAOMD__UTILS

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstdlib>
#include <ctime>

__device__ __host__ float squared_l2_distance(float x_1, float y_1, float x_2, float y_2) {
    return (x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2);
}

__device__ int find_nearest_cluster(float x, float y, const float* means_x, const float* means_y, int k) {
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
    return best_cluster;
}

__global__ void get_distance(const float* data_x, const float* data_y, float* distance, int data_size,
                             const float* d_mean_x, const float* d_mean_y, int num_choices) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = blockDim.x * gridDim.x;
    
    for (int index = id; index < data_size; index+=width) {
        float dd = FLT_MAX;
        for (int j = 0; j < num_choices; j++) {   // 已选的中心
            float d = squared_l2_distance(data_x[index], data_y[index], d_mean_x[j], d_mean_y[j]);
            if (d < dd)     dd = d;
        }
        distance[index + 1] = dd;
    }
}

__global__ void choice_cluster(const float* data_x, const float* data_y, const float* distance, int data_size,
                             float* d_mean_x, float* d_mean_y, int num_choices, float seed) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = blockDim.x * gridDim.x;
    
    float val = seed * distance[data_size];
    for (int index = id; index < data_size; index+=width) {
        if (distance[index] <= val && val < distance[index + 1]) {
            d_mean_x[num_choices] = data_x[index];
            d_mean_y[num_choices] = data_y[index];
            break;
        }
    }
}

#endif