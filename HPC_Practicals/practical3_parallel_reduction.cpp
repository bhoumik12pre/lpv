#include <iostream>
#include <vector>
#include <climits>
#include <omp.h>
using namespace std;

int main() {
    int n = 1000000;

    vector<int> a(n);

    srand(42);

    for (int i = 0; i < n; i++)
        a[i] = rand() % 100000;

    int mn = INT_MAX;
    int mx = INT_MIN;
    long long sum = 0;

    #pragma omp parallel for reduction(min:mn) reduction(max:mx) reduction(+:sum)
    for (int i = 0; i < n; i++) {
        if (a[i] < mn)
            mn = a[i];

        if (a[i] > mx)
            mx = a[i];

        sum += a[i];
    }

    double avg = (double)sum / n;

    cout << "Minimum: " << mn << endl;
    cout << "Maximum: " << mx << endl;
    cout << "Sum: " << sum << endl;
    cout << "Average: " << avg << endl;

    return 0;
}