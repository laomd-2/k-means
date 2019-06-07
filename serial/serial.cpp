#include <iostream>
#include <fstream>
#include <vector>
#include "kmeans.h"
using namespace std;


int main(int argc, char const *argv[])
{
    ifstream fin("../data.txt");
    cin.rdbuf(fin.rdbuf());

    int n, dim, k;
    cin >> n >> dim >> k;
    vector<Point> X(n, Point(dim));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < dim; j++)
            cin >> X[i][j];
    
    auto [pivots, label] = k_means(X, k, 1e-4, 100, 0);
    ofstream fout("../result/result.txt");
    fout << label.size() << endl;
    for (int x: label)  fout << x << ' ';
    fout << endl;

    fout << pivots.size() << endl;
    for (auto& pivot: pivots)
        for (auto x: pivot)
            fout << x << ' ';
    fout << endl;
    return 0;
}
