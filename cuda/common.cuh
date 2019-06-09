#ifndef __LAOMD__UTILS
#define __LAOMD__UTILS

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstdlib>
#include <ctime>

/**
 * @description: 欧拉距离
 */
__device__ __host__ float squared_l2_distance(float x_1, float y_1, float x_2, float y_2) {
    return (x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2);
}

/**
 * @description: 计算点（x, y）最近的中心点
 * @param
    means_*：中心点数组
    k: 中心点个数
 * @return: 最近中心点的类下标
 */
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

/**
 * @description: kmeans++1.1步，计算每个聚类点到已选中心点的最近距离
 * @param
    data_*：聚类点
    distance：存储每个点的最近距离
    data_size：聚类点个数
    d_mean_*：已选中心点
    num_choices：已选中心点个数
 * @return: 
 */
__global__ void get_distance(const float* data_x, const float* data_y, float* distance, int data_size,
                             const float* d_mean_x, const float* d_mean_y, int num_choices) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = blockDim.x * gridDim.x;
    
    for (int index = id; index < data_size; index+=width) {
        float dd = FLT_MAX;
        for (int j = 0; j < num_choices; j++) {   // 已选的中心
            float d = squared_l2_distance(data_x[index], data_y[index], d_mean_x[j], d_mean_y[j]);
            if (d < dd)     dd = d; // 选出最近的距离
        }
        distance[index + 1] = dd;
    }
}

/**
 * @description: kmeans++1.3步，根据生成的随机数选出一个中心点。（偏向于选择离已选中心点远的点）
 * @param
    data_*：聚类点
    distance：距离前缀和
    data_size：聚类点个数
    d_mean_*：已选中心点
    num_choices：已选中心点个数
    seed: 0-1的随机数
 * @return: 
 */
__global__ void choice_cluster(const float* data_x, const float* data_y, const float* distance, int data_size,
                             float* d_mean_x, float* d_mean_y, int num_choices, float seed) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = blockDim.x * gridDim.x;
    
    float val = seed * distance[data_size]; // val在0到前缀和最大值
    for (int index = id; index < data_size; index+=width) {
        // val在该聚类点所属的距离区间（这个地方体现，距离越大概率越大）
        if (distance[index] <= val && val < distance[index + 1]) {
            d_mean_x[num_choices] = data_x[index];
            d_mean_y[num_choices] = data_y[index];
            break;
        }
    }
}

#endif