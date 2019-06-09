#if !defined(LAOMD_IO)
#define LAOMD_IO

#include <fstream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

/**
 * @description: 输入聚类的各个坐标点
 * @param
    h_x: 存储坐标点一个维度的坐标
 * @return: 
 */
int read_data(const char* file, thrust::host_vector<float>& h_x) {
    std::ifstream fx(file);
    float x;
    int n, k;
    fx >> n >> k;
    for (int i = 0; i < n; i++)
    {
        fx >> x;
        h_x.push_back(x);
    }
    return k;
}

/**
 * @description: 输出每个点所属的类、每个类的中心点
 * @param
    d_label: 每个点所属的类
    d_mean_*: 每个中心点一个维度的坐标值
 * @return: 
 */
void output(const char* file, const thrust::device_vector<int>& d_label,
            const thrust::device_vector<float>& d_mean_x, const thrust::device_vector<float>& d_mean_y) {
    std::ofstream fout(file);
    fout << d_label.size() << std::endl;
    for (int x : d_label)
        fout << x << ' ';
    fout << std::endl;

    int k = d_mean_x.size();
    fout << k << std::endl;
    for (int i = 0; i < k; i++)
    {
        fout << d_mean_x[i] << ' ' << d_mean_y[i] << ' ';
    }
    fout << std::endl;
}
#endif // LAOMD_IO
