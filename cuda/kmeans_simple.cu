#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <algorithm>
#include <random>
#include <cfloat>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>

#define WARP_SIZE 32

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

__device__ void reduce_in_block(float* local_sum, int* local_count, 
                                float* new_sums_x, float* new_sums_y, int* counts, int k)
{
    // reduction
    extern __shared__ float shared_data[];
    for (int i = 0; i < k; i++) {
        for (int stride = WARP_SIZE/2; stride>0; stride>>=1) {
            local_sum[2 * i] += __shfl_down(local_sum[2 * i], stride);
            local_sum[2 * i + 1] += __shfl_down(local_sum[2 * i + 1], stride);
            local_count[i] += __shfl_down(local_count[i], stride);
        }
        // warp之间隐式同步，无需原子操作
        if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
            shared_data[3 * i] += local_sum[2 * i];
            shared_data[3 * i + 1] += local_sum[2 * i + 1];
            shared_data[3 * i + 2] += local_count[i];
        }
    }
    if (threadIdx.x == 0) {
        for (int i = 0; i < k; i++) {
            int offset = i * gridDim.x;
            new_sums_x[offset + blockIdx.x] = shared_data[3 * i];
            new_sums_y[offset+ blockIdx.x] = shared_data[3 * i + 1];
            counts[offset + blockIdx.x] = int(shared_data[3 * i + 2] + 0.5f);
        }
    }
}

__global__ void assign_clusters(const float* data_x, const float* data_y, int* label, int data_size,
                                const float* means_x, const float* means_y, 
                                float* new_sums_x, float* new_sums_y, int* counts, int k, int p)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = blockDim.x * gridDim.x;

    float local_sum[32] = {0.0f};
    int local_count[16] = {0};

    for (int i = 0; i < p; i++) {
        int index = i * width + id;
        if (index >= data_size) return;

        const float x = data_x[index];
        const float y = data_y[index];
        int best_cluster = find_nearest_cluster(x, y, means_x, means_y, k);
        label[index] = best_cluster;
        local_sum[2 * best_cluster] = x;
        local_sum[2 * best_cluster + 1] = y;
        local_count[best_cluster]++;
    }

    reduce_in_block(local_sum, local_count, new_sums_x, new_sums_y, counts, k);
}

// Each thread is one cluster, which just recomputes its coordinates as the mean
// of all points assigned to it.
__global__ void compute_new_means(float* new_means_x, float* new_means_y,
                       float* new_sum_x, float* new_sum_y, int* counts, 
                       int blocks, float* max_diff)
{
    const int cluster = threadIdx.x;
    float sum_x = 0.0f, sum_y = 0.0f;
    int count = 0, offset = cluster * blocks;
    for (int i = 0; i < blocks; i++) {
        sum_x += new_sum_x[offset + i];
        new_sum_x[offset + i] = 0.0f;
        sum_y += new_sum_y[offset + i];
        new_sum_y[offset + i] = 0.0f;
        count += counts[offset + i];
        counts[offset + i] = 0;
    }
    if (count == 0) count = 1;

    float mean_x = new_means_x[cluster], mean_y = new_means_y[cluster];
    float new_mean_x = sum_x / count, new_mean_y = sum_y / count;
    new_means_x[cluster] = new_mean_x;
    new_means_y[cluster] = new_mean_y;

    // 归约最大值
    float diff = squared_l2_distance(mean_x, mean_y, new_mean_x, new_mean_y);
    for (int stride = WARP_SIZE/2; stride>0; stride>>=1) {
        float d = __shfl_down(diff, stride);
        if (d > diff)   diff = d;
    }
    if (cluster == 0)   *max_diff = diff;
}

void init_clusters(const thrust::host_vector<float>& h_x, const thrust::host_vector<float>& h_y,
                   thrust::device_vector<float>& d_mean_x,  thrust::device_vector<float>& d_mean_y, int k) {
    int n = h_x.size();
    int index = rand() % n;
    d_mean_x[0] = h_x[index];
    d_mean_y[0] = h_y[index];
    
    thrust::host_vector<float> distance(n);
    std::default_random_engine e;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int i = 1; i < k; i++) {   // 待选中心
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            float dd = FLT_MAX;
            for (int j = 0; j < i; j++) {   // 已选的中心
                float d = squared_l2_distance(h_x[k], h_y[k], d_mean_x[j], d_mean_y[j]);
                if (d < dd)     dd = d;
            }
            sum += dd;
            distance[k] = dd;
        }
        int num = u(e) * sum;
        int index = 0;
        while (sum > 0) {
            sum -= distance[index];
            index++;
        }
        d_mean_x[i] = h_x[index];
        d_mean_y[i] = h_y[index];
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

    thrust::device_vector<float> d_x(h_x);
    thrust::device_vector<float> d_y(h_y);

    srand(time(NULL));
    thrust::device_vector<float> d_mean_x(k);
    thrust::device_vector<float> d_mean_y(k);
    init_clusters(h_x, h_y, d_mean_x, d_mean_y, k);

    const int threads = 1024;
    int blocks = (number_of_elements + threads - 1) / threads, p = 1;
    if (blocks > 64)
    {
        blocks = 64;
        p = (number_of_elements + 65535) / 65536;
    }

    thrust::device_vector<float> d_sums_x(k * blocks);
    thrust::device_vector<float> d_sums_y(k * blocks);
    thrust::device_vector<int> d_counts(k * blocks, 0), d_label(number_of_elements, 0);

    int number_of_iterations = 300;
    float tol = 1e-4f, s = tol + 1.0f, *d_s;
    cudaMalloc(&d_s, sizeof(float));
    size_t iteration;
    for (iteration = 0; s > tol && iteration < number_of_iterations; ++iteration)
    {
        assign_clusters<<<blocks, threads, 3 * k * sizeof(float)>>>(
            thrust::raw_pointer_cast(d_x.data()),
            thrust::raw_pointer_cast(d_y.data()),
            thrust::raw_pointer_cast(d_label.data()),
            number_of_elements,
            thrust::raw_pointer_cast(d_mean_x.data()),
            thrust::raw_pointer_cast(d_mean_y.data()),
            thrust::raw_pointer_cast(d_sums_x.data()),
            thrust::raw_pointer_cast(d_sums_y.data()),
            thrust::raw_pointer_cast(d_counts.data()),
            k, p
        );
        cudaDeviceSynchronize();

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