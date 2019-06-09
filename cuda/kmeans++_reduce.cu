/*
 * @Description: 全都使用归约，不使用原子指令的kmeans++
 * @Author: 劳马东
 * @Date: 2019-06-07 18:36:45
 * @LastEditTime: 2019-06-09 23:17:41
 */
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <cfloat>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include "reduce.cuh"
#include "common.cuh"
#include "io.cuh"

/**
 * @description: 非原子指令实现归约1.1步，将block全局值写到global memory
 * @param
    local：每个线程的局部数组，1*k
    global：k*num_blocks数组，即每行存储一个中心点的数据，每个block的数据存储在一列
    k：聚类中心点数
 * @return: 
 */
__device__ inline void sum_in_block(float* local, float* global, int k)
{
    for (int i = 0; i < k; i++) // 先计算每个block的归约值
        local[i] = blockReduce(local[i], thrust::plus<float>(), 0.0f);
    if (threadIdx.x == 0) {
        for (int i = 0; i < k; i++) {
            // k * num_blocks
            // 这里虽然不能合并访存，但在compute_new_means归并时可以合并访存
            // 另一个考虑是k通常较小而且不是warp的倍数，而num_blocks通常是
            global[i * gridDim.x + blockIdx.x] = local[i];
        }
    }
}

/**
 * @description: 给每个聚类点分配一个类
 * @param
    new_sum_*：每个新类所有点某一维度的和
    counts：每个新类拥有的聚类点
 * @return: 
 */
__global__ void assign_clusters(const float* data_x, const float* data_y, int* label, int data_size,
                                const float* means_x, const float* means_y, 
                                float* new_sums_x, float* new_sums_y, float* counts, int k)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = blockDim.x * gridDim.x;

    // 假设k不大于16
    float local_sum_x[16] = {0.0f};
    float local_sum_y[16] = {0.0f};
    float local_count[16] = {0.0f};

    for (int index = id; index < data_size; index += width) {
        const float x = data_x[index];
        const float y = data_y[index];
        int best_cluster = find_nearest_cluster(x, y, means_x, means_y, k);
        label[index] = best_cluster;
        local_sum_x[best_cluster] += x;
        local_sum_y[best_cluster] += y;
        local_count[best_cluster]++;
    }

    // 把线程的局部值归约到block，然后写到global memory
    sum_in_block(local_sum_x, new_sums_x, k);
    sum_in_block(local_sum_y, new_sums_y, k);
    sum_in_block(local_count, counts, k);
}

/**
 * @description: 对new_sums_*和counts归约
 * @param
    in：k*num_blocks数组
    width：num_blocks
 * @return: 
 */
__device__ inline float sum_in_device1block(const float* in, int width) {
    float s = 0.0f;
    int wid = threadIdx.x / warpSize;
    int laneid = threadIdx.x & (warpSize - 1);
    // 一个warp算一行
    int offset = wid * width;
    for (int i = laneid; i < width; i += warpSize) {
        s += in[offset + i];
    }
    // 将warp的值归约，即得到一个类的和
    s = warpReduce(s, thrust::plus<float>());
    return s;
}

/**
 * @description: 计算新的中心点
 * @param
    max_diff：新旧中心点的最大偏差，用于判断是否收敛
 * @return: 
 */
__global__ void compute_new_means(float* new_means_x, float* new_means_y,
                       const float* new_sum_x, const float* new_sum_y, const float* counts, 
                       int blocks, float* max_diff)
{
    int cluster = threadIdx.x / warpSize;
    int laneid = threadIdx.x & (warpSize - 1);

    float sum_x = sum_in_device1block(new_sum_x, blocks);
    float sum_y = sum_in_device1block(new_sum_y, blocks);
    float count = max(1.0f, sum_in_device1block(counts, blocks));

    // max_val存储每个类新旧中心点的偏差
    __shared__ float max_val[32];
    if (laneid == 0) {
        // 旧中心点的坐标
        float mean_x = new_means_x[cluster], mean_y= new_means_y[cluster];
        // 计算均值向量，新中心点
        float new_mean_x = sum_x / count, new_mean_y = sum_y / count;
        new_means_x[cluster] = new_mean_x;
        new_means_y[cluster] = new_mean_y;
        // 新旧中心点的欧拉距离
        max_val[cluster] = squared_l2_distance(mean_x, mean_y, new_mean_x, new_mean_y);
    }
    if (cluster == 0) {
        // 归约最大值，最大偏差
        float val = warpReduce(max_val[laneid], thrust::maximum<float>());
        if (laneid == 0)    *max_diff = val;
    }
}

int main(int argc, const char *argv[])
{
    thrust::host_vector<float> h_x;
    thrust::host_vector<float> h_y;
    int k;
    k = read_data(argv[1], h_x);
    k = read_data(argv[2], h_y);
    int number_of_elements = h_x.size();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    thrust::device_vector<float> d_x(h_x);
    thrust::device_vector<float> d_y(h_y);
    thrust::device_vector<float> d_mean_x(k);
    thrust::device_vector<float> d_mean_y(k);
    // 第一个位置存储0
    thrust::device_vector<float> distance(1 + number_of_elements);

    srand(time(NULL));
    int index = rand() % number_of_elements;
    // kmeans++ 1.0步，随机选择第一个中心点
    d_mean_x[0] = d_x[index];
    d_mean_y[0] = d_y[index];

    // 选取其余的k-1个中心点
    const int threads = 1024;
    int blocks = std::min(64, (number_of_elements + threads - 1) / threads);
    for (int i = 1; i < k; i++) {
        get_distance<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_x.data()),
            thrust::raw_pointer_cast(d_y.data()),
            thrust::raw_pointer_cast(distance.data()),
            number_of_elements, 
            thrust::raw_pointer_cast(d_mean_x.data()),
            thrust::raw_pointer_cast(d_mean_y.data()),
            i
        );
        cudaDeviceSynchronize();
        // 计算前缀和
        thrust::inclusive_scan(distance.begin(), distance.end(), distance.begin());
        // 生成选择中心点的随机数（0-1）
        float seed = (rand() % number_of_elements) / (float)number_of_elements;
        choice_cluster<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_x.data()),
            thrust::raw_pointer_cast(d_y.data()),
            thrust::raw_pointer_cast(distance.data()),
            number_of_elements, 
            thrust::raw_pointer_cast(d_mean_x.data()),
            thrust::raw_pointer_cast(d_mean_y.data()),
            i, seed
        );
        cudaDeviceSynchronize();
    }
    
    thrust::device_vector<float> d_sums_x(k * blocks, 0.0f), d_sums_y(k * blocks, 0.0f), 
                                 d_counts(k * blocks, 0.0f);
    thrust::device_vector<int> d_label(number_of_elements, 0);
    float *d_s;
    cudaMalloc(&d_s, sizeof(float));

    float tol = 1e-4f, s = tol + 1.0f;
    int number_of_iterations = 300, iteration;
    // 收敛或最大迭代次数时，算法结束
    for (iteration = 0; s > tol && iteration < number_of_iterations; ++iteration)
    {
        // 第二步，分配每个点所属的类
        assign_clusters<<<blocks, threads, k * sizeof(float)>>>(
            thrust::raw_pointer_cast(d_x.data()),
            thrust::raw_pointer_cast(d_y.data()),
            thrust::raw_pointer_cast(d_label.data()),
            number_of_elements,
            thrust::raw_pointer_cast(d_mean_x.data()),
            thrust::raw_pointer_cast(d_mean_y.data()),
            thrust::raw_pointer_cast(d_sums_x.data()),
            thrust::raw_pointer_cast(d_sums_y.data()),
            thrust::raw_pointer_cast(d_counts.data()),
            k
        );
        cudaDeviceSynchronize();

        // 第三步，更新聚类中心
        // 这里使用的是k*warpSize的block，每个warp归约计算一个类
        compute_new_means<<<1, k * 32>>>(
            thrust::raw_pointer_cast(d_mean_x.data()),
            thrust::raw_pointer_cast(d_mean_y.data()),
            thrust::raw_pointer_cast(d_sums_x.data()),
            thrust::raw_pointer_cast(d_sums_y.data()),
            thrust::raw_pointer_cast(d_counts.data()),
            blocks, d_s
        );
        cudaDeviceSynchronize();
        cudaMemcpy(&s, d_s, sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_s);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << milliseconds / 1000.0 << "s with " << iteration << " rounds" << std::endl;

    output(argv[3], d_label, d_mean_x, d_mean_y);
}