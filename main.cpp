#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <vector>

double calc_sequential(const std::vector<double>& a) {
    double sum = 0.0;
    double p = 1.0;
    for (std::size_t i = 0; i < a.size(); i++) {
        p *= a[i];
        sum += p;
    }
    return sum;
}

double calc_parallel(const std::vector<double>& a, int threads) {
    if (threads <= 0) {
        return 0.0;
    }

    double total_sum = 0.0;
    int n = static_cast<int>(a.size());

    std::vector<double> local_sums(threads, 0.0);
    std::vector<double> local_products(threads, 1.0);

#pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        int chunk = n / nthreads;
        int start = tid * chunk;
        int end = (tid == nthreads - 1) ? n : (tid + 1) * chunk;

        double partial_sum = 0.0;
        double partial_product = 1.0;

        for (int i = start; i < end; i++) {
            partial_product *= a[i];
            partial_sum += partial_product;
        }

        local_sums[tid] = partial_sum;
        local_products[tid] = partial_product;
    }

    double current_P = 1.0;
    for (int t = 0; t < threads; t++) {
        total_sum += current_P * local_sums[t];
        current_P *= local_products[t];
    }

    return total_sum;
}

int main() {
    std::vector<int> dimensions = {1000000, 5000000, 10000000, 50000000};
    std::vector<int> threads_count = {2, 4, 6, 8, 10, 20};

    std::cout << "Initialization..." << std::endl;

    for (int n : dimensions) {
        std::vector<double> a(n, 1.0000001);

        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Dimension N = " << n << std::endl;

        double start_seq = omp_get_wtime();
        double res_seq = calc_sequential(a);
        double end_seq = omp_get_wtime();
        double time_seq = end_seq - start_seq;

        std::cout << "Sequential Mode | Result: " << std::scientific << res_seq
                  << " | Time: " << std::fixed << std::setprecision(5) << time_seq << " sec" << std::endl;

        for (int m : threads_count) {
            double start_par = omp_get_wtime();
            double res_par = calc_parallel(a, m);
            double end_par = omp_get_wtime();
            double time_par = end_par - start_par;

            constexpr double kTimeEps = 1e-12;
            double speedup = 0.0;
            if (time_par <= kTimeEps) {
                speedup = std::numeric_limits<double>::infinity();
            } else if (time_seq <= kTimeEps) {
                speedup = 0.0;
            } else {
                speedup = time_seq / time_par;
            }

            std::cout << "Threads: " << std::setw(2) << m
                      << " | Result: " << std::scientific << res_par
                      << " | Time: " << std::fixed << std::setprecision(5) << time_par << " sec"
                      << " | Speedup S_m(n): " << std::fixed << std::setprecision(2) << speedup << std::endl;
        }
    }

    return 0;
}
