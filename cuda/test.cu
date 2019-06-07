#include <vector>
#include <iostream>
#include <fstream>
#include "kmeans.cuh"


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
    for (int i = 0; i < n; i++) {
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

    std::mt19937 rng(std::random_device{}());
    std::shuffle(h_x.begin(), h_x.end(), rng);
    std::shuffle(h_y.begin(), h_y.end(), rng);
    thrust::device_vector<float> d_mean_x(h_x.begin(), h_x.begin() + k);
    thrust::device_vector<float> d_mean_y(h_y.begin(), h_y.begin() + k);

    thrust::device_vector<float> d_sums_x(k);
    thrust::device_vector<float> d_sums_y(k);
    thrust::device_vector<int> d_counts(k, 0), d_label(number_of_elements, 0);

    const int threads = 1024;
    int blocks = (number_of_elements + threads - 1) / threads, p = 1;
    if (blocks > 64) {
        blocks = 64;
        p = (number_of_elements + 65535) / 65536;
    }

    int number_of_iterations = 100;
    for (size_t iteration = 0; iteration < number_of_iterations; ++iteration)
    {
        thrust::fill(d_sums_x.begin(), d_sums_x.end(), 0);
        thrust::fill(d_sums_y.begin(), d_sums_y.end(), 0);
        thrust::fill(d_counts.begin(), d_counts.end(), 0);

        assign_clusters<<<blocks, threads>>>(d_x.data(),
                                             d_y.data(),
                                             number_of_elements,
                                             d_mean_x.data(),
                                             d_mean_y.data(),
                                             d_label.data(),
                                             d_sums_x.data(),
                                             d_sums_y.data(),
                                             k, p,
                                             d_counts.data());
        cudaDeviceSynchronize();

        compute_new_means<<<1, k>>>(d_mean_x.data(),
                                    d_mean_y.data(),
                                    d_sums_x.data(),
                                    d_sums_y.data(),
                                    d_counts.data());
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << milliseconds / 1000.0 << "s" << std::endl;

    std::ofstream fout(argv[3]);
    fout << d_label.size() << std::endl;
    for (int x: d_label)  fout << x << ' ';
    fout << std::endl;

    fout << k << std::endl;
    for (int i = 0; i < k; i++) {
        fout << d_mean_x[i] << ' ' << d_mean_y[i] << ' ';
    }
    fout << std::endl;
}