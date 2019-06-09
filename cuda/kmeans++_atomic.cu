/*
 * @Description: 使用原子指令实现device级别的归约的kmeans++
 * @Author: 劳马东
 * @Date: 2019-06-07 18:36:45
 * @LastEditTime: 2019-06-09 23:17:11
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
 * @description: 使用原子指令归约的核函数
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

    // 将线程的局部和归约到block，然后用原子指令增加到global memory
    deviceReduceBlockAtomic(local_sum_x, new_sums_x, k);
    deviceReduceBlockAtomic(local_sum_y, new_sums_y, k);
    deviceReduceBlockAtomic(local_count, counts, k);
}

// new_sum_*和counts（1*k）都归约好了，因此这里不用归约
__global__ void compute_new_means(float* new_means_x, float* new_means_y,
                       float* new_sum_x, float* new_sum_y, float* counts, 
                       int blocks, float* max_diff)
{
    const int cluster = threadIdx.x;
    const int count = max(1.0f, counts[cluster]);

    float mean_x = new_means_x[cluster], mean_y = new_means_y[cluster];
    float new_mean_x = new_sum_x[cluster] / count, new_mean_y = new_sum_y[cluster] / count;
    new_means_x[cluster] = new_mean_x;
    new_means_y[cluster] = new_mean_y;

    new_sum_x[cluster] = 0.0f;
    new_sum_y[cluster] = 0.0f;
    counts[cluster] = 0.0f;

    // 归约最大值
    float diff = warpReduce(squared_l2_distance(mean_x, mean_y, new_mean_x, new_mean_y), thrust::maximum<float>());
    if (cluster == 0)   *max_diff = diff;
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
    thrust::device_vector<float> distance(1 + number_of_elements);

    srand(time(NULL));
    int index = rand() % number_of_elements;
    d_mean_x[0] = d_x[index];
    d_mean_y[0] = d_y[index];

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
        thrust::inclusive_scan(distance.begin(), distance.end(), distance.begin());
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
    for (iteration = 0; s > tol && iteration < number_of_iterations; ++iteration)
    {
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

        // 这里使用的是1*k的block，每个线程计算一个类
        compute_new_means<<<1, k>>>(
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