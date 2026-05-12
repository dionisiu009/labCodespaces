#include <iomanip>
#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;

double calc_sequential(const vector<double>& a) {
    double sum = 0.0;
    double p = 1.0;
    for (size_t i = 0; i < a.size(); i++) {
        p *= a[i];
        sum += p;
    }
    return sum;
}

double calc_parallel(const vector<double>& a, int threads) {
    double total_sum = 0.0;
    int n = static_cast<int>(a.size());

    vector<double> local_S(threads, 0.0);
    vector<double> local_P(threads, 1.0);

#pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        int chunk = n / nthreads;
        int start = tid * chunk;
        int end = (tid == nthreads - 1) ? n : (tid + 1) * chunk;

        double S = 0.0;
        double P = 1.0;

        for (int i = start; i < end; i++) {
            P *= a[i];
            S += P;
        }

        local_S[tid] = S;
        local_P[tid] = P;
    }

    double current_P = 1.0;
    for (int t = 0; t < threads; t++) {
        total_sum += current_P * local_S[t];
        current_P *= local_P[t];
    }

    return total_sum;
}

int main() {
    vector<int> dimensions = {1000000, 5000000, 10000000, 50000000};
    vector<int> threads_count = {2, 4, 6, 8, 10, 20};

    cout << "Initialization..." << endl;

    for (int n : dimensions) {
        vector<double> a(n, 1.0000001);

        cout << "------------------------------------------------" << endl;
        cout << "Dimension N = " << n << endl;

        double start_seq = omp_get_wtime();
        double res_seq = calc_sequential(a);
        double end_seq = omp_get_wtime();
        double time_seq = end_seq - start_seq;

        cout << "Sequential Mode | Result: " << scientific << res_seq
             << " | Time: " << fixed << setprecision(5) << time_seq << " sec" << endl;

        for (int m : threads_count) {
            double start_par = omp_get_wtime();
            double res_par = calc_parallel(a, m);
            double end_par = omp_get_wtime();
            double time_par = end_par - start_par;

            double speedup = time_seq / time_par;

            cout << "Threads: " << setw(2) << m
                 << " | Result: " << scientific << res_par
                 << " | Time: " << fixed << setprecision(5) << time_par << " sec"
                 << " | Speedup S_m(n): " << fixed << setprecision(2) << speedup << endl;
        }
    }

    return 0;
}
