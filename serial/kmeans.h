#include <algorithm>
#include <cstdlib>
#include <limits>
#include <vector>

typedef std::vector<double> Point;

#define square(x) (x) * (x)

double l2_distance(const Point &a, const Point &b)
{
    double sum = 0;
    int n = a.size();
    for (int i = 0; i < n; i++)
        sum += square(a[i] - b[i]);
    return sum;
}

std::pair<std::vector<Point>, std::vector<int>> k_means(const std::vector<Point> &data,
                                                        int k, double tol, int max_iter, int seed)
{
    srand(seed);

    std::vector<Point> means(k);
    int n = data.size(), dim = data[0].size();
    // 随机选取k个质心
    for (auto &cluster : means)
        cluster = data[rand() % n];

    std::vector<int> label(data.size());
    while (max_iter > 0)
    {
        // Find label.
        for (int i = 0; i < data.size(); ++i)
        {
            double min_dis = std::numeric_limits<double>::max();
            int nearest_cluster = -1;
            for (int cluster = 0; cluster < k; ++cluster)
            {
                double distance = l2_distance(data[i], means[cluster]);
                if (distance < min_dis)
                {
                    min_dis = distance;
                    nearest_cluster = cluster;
                }
            }
            label[i] = nearest_cluster;
        }

        // Sum up and count points for each cluster.
        std::vector<Point> new_means(k, Point(dim, 0));
        std::vector<int> counts(k, 0);
        for (int i = 0; i < data.size(); ++i)
        {
            int cluster = label[i];
            counts[cluster]++;
            for (int j = 0; j < dim; j++)
                new_means[cluster][j] += data[i][j];
        }

        double pre = -1.0;
        for (int cluster = 0; cluster < k; ++cluster)
        {
            // Turn 0/0 into 0/1 to avoid zero division.
            const auto count = std::max<size_t>(1, counts[cluster]);
            for (int j = 0; j < dim; j++)
                new_means[cluster][j] /= count;
            pre = std::max(pre, l2_distance(new_means[cluster], means[cluster]));
        }
        means = new_means;
        if (pre < tol)
            break;
    }

    return std::make_pair(means, label);
}