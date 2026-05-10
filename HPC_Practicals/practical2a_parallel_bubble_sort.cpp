#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
using namespace std;

void sequential_bubble(vector<int>& a) {
    int n = a.size();

    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (a[j] > a[j + 1])
                swap(a[j], a[j + 1]);
        }
    }
}

void parallel_bubble(vector<int>& a) {
    int n = a.size();

    for (int phase = 0; phase < n; phase++) {
        int start = phase % 2;

        #pragma omp parallel for shared(a)
        for (int i = start; i < n - 1; i += 2) {
            if (a[i] > a[i + 1])
                swap(a[i], a[i + 1]);
        }
    }
}

int main() {
    int n = 10000;

    vector<int> a(n), b(n);

    srand(42);

    for (int i = 0; i < n; i++) {
        a[i] = b[i] = rand() % 100000;
    }

    double t1 = omp_get_wtime();

    sequential_bubble(a);

    double t2 = omp_get_wtime();

    parallel_bubble(b);

    double t3 = omp_get_wtime();

    cout << "Sequential Bubble Sort: " << (t2 - t1) << " s" << endl;
    cout << "Parallel Bubble Sort: " << (t3 - t2) << " s" << endl;
    cout << "Speedup: " << (t2 - t1) / (t3 - t2) << "x" << endl;

    return 0;
}