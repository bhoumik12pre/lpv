#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
using namespace std;

void merge(vector<int>& a, int l, int m, int r) {
    vector<int> tmp;

    int i = l;
    int j = m + 1;

    while (i <= m && j <= r)
        tmp.push_back(a[i] <= a[j] ? a[i++] : a[j++]);

    while (i <= m)
        tmp.push_back(a[i++]);

    while (j <= r)
        tmp.push_back(a[j++]);

    for (int k = l; k <= r; k++)
        a[k] = tmp[k - l];
}

void sequential_merge_sort(vector<int>& a, int l, int r) {
    if (l >= r)
        return;

    int m = (l + r) / 2;

    sequential_merge_sort(a, l, m);
    sequential_merge_sort(a, m + 1, r);

    merge(a, l, m, r);
}

void parallel_merge_sort(vector<int>& a, int l, int r, int depth = 0) {
    if (l >= r)
        return;

    int m = (l + r) / 2;

    if (depth <= 4) {
        #pragma omp parallel sections
        {
            #pragma omp section
            parallel_merge_sort(a, l, m, depth + 1);

            #pragma omp section
            parallel_merge_sort(a, m + 1, r, depth + 1);
        }
    }
    else {
        sequential_merge_sort(a, l, m);
        sequential_merge_sort(a, m + 1, r);
    }

    merge(a, l, m, r);
}

int main() {
    int n = 100000;

    vector<int> a(n), b(n);

    srand(42);

    for (int i = 0; i < n; i++)
        a[i] = b[i] = rand() % 1000000;

    double t1 = omp_get_wtime();

    sequential_merge_sort(a, 0, n - 1);

    double t2 = omp_get_wtime();

    parallel_merge_sort(b, 0, n - 1);

    double t3 = omp_get_wtime();

    cout << "Sequential Merge Sort: " << (t2 - t1) << " s" << endl;
    cout << "Parallel Merge Sort: " << (t3 - t2) << " s" << endl;

    return 0;
}