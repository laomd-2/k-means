#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>

__device__ float squared_l2_distance(float x_1, float y_1, float x_2, float y_2) {
  return (x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2);
}

// In the assignment step, each point (thread) computes its distance to each
// cluster centroid and adds its x and y values to the sum of its closest
// centroid, as well as incrementing that centroid's count of assigned points.
__constant__ float means_x[32], means_y[32];

__global__ void assign_clusters(const float* data_x,
                     const float* data_y,
                     int data_size,
                     const float* means_x,
                     const float* means_y,
                     int* label,
                     float* new_sums_x,
                     float* new_sums_y,
                     int k, int p,
                     int* counts)
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
        atomicAdd(new_sums_x + best_cluster, x);
        atomicAdd(new_sums_y + best_cluster, y);
        atomicAdd(counts + best_cluster, 1);
    }
}

// Each thread is one cluster, which just recomputes its coordinates as the mean
// of all points assigned to it.
__global__ void compute_new_means(float* new_means_x,
                       float* new_means_y,
                       float* new_sum_x,
                       float* new_sum_y,
                       int* counts)
{
    const int cluster = threadIdx.x;
    const int count = max(1, counts[cluster]);
    counts[cluster] = 0;
    new_means_x[cluster] = new_sum_x[cluster] / count;
    new_means_y[cluster] = new_sum_y[cluster] / count;
    new_sum_x[cluster] = 0.0f;
    new_sum_y[cluster] = 0.0f;
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

    thrust::device_vector<float> d_x(h_x);
    thrust::device_vector<float> d_y(h_y);

    srand(time(NULL));
    thrust::device_vector<float> d_mean_x(k);
    thrust::device_vector<float> d_mean_y(k);
    for (int i = 0; i < k; i++) {
        int index = rand() % number_of_elements;
        d_mean_x[i] = h_x[i];
        d_mean_y[i] = h_y[i];
    }

    thrust::device_vector<float> d_sums_x(k);
    thrust::device_vector<float> d_sums_y(k);
    thrust::device_vector<int> d_counts(k, 0), d_label(number_of_elements, 0);

    const int threads = 1024;
    int blocks = (number_of_elements + threads - 1) / threads, p = 1;
    if (blocks > 32)
    {
        blocks = 32;
        p = (number_of_elements + 32767) / 32768;
    }

    int number_of_iterations = 1000;
    for (size_t iteration = 0; iteration < number_of_iterations; ++iteration)
    {
        assign_clusters<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_x.data()),
            thrust::raw_pointer_cast(d_y.data()),
            number_of_elements,
            thrust::raw_pointer_cast(d_mean_x.data()),
            thrust::raw_pointer_cast(d_mean_y.data()),
            thrust::raw_pointer_cast(d_label.data()),
            thrust::raw_pointer_cast(d_sums_x.data()),
            thrust::raw_pointer_cast(d_sums_y.data()),
            k, p,
            thrust::raw_pointer_cast(d_counts.data())
        );
        cudaDeviceSynchronize();

        compute_new_means<<<1, k>>>(
            thrust::raw_pointer_cast(d_mean_x.data()),
            thrust::raw_pointer_cast(d_mean_y.data()),
            thrust::raw_pointer_cast(d_sums_x.data()),
            thrust::raw_pointer_cast(d_sums_y.data()),
            thrust::raw_pointer_cast(d_counts.data())
        );
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << milliseconds / 1000.0 << "s" << std::endl;

    std::ofstream fout(argv[3]);
    fout << d_label.size() << std::endl;
    for (int x : d_label)
        fout << x << ' ';
    fout << std::endl;

    fout << k << std::endl;
    for (int i = 0; i < k; i++)
    {
        fout << d_mean_x[i] << ' ' << d_mean_y[i] << ' ';
    }
    fout << std::endl;
}