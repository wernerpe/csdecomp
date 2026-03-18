#include <omp.h>

#include <iostream>
#include <thread>
#include <vector>

int main() {
  // Print hardware info
  std::cout << "Hardware threads available: "
            << std::thread::hardware_concurrency() << std::endl;
  std::cout << "OpenMP max threads: " << omp_get_max_threads() << std::endl;

  // Set number of threads to 20
  omp_set_num_threads(omp_get_max_threads());
  std::cout << "Set OpenMP threads to:" << omp_get_max_threads() << std::endl;

  // Test parallel region
  std::vector<int> thread_ids(20, -1);

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int num_threads = omp_get_num_threads();

#pragma omp single
    {
      std::cout << "Number of threads in parallel region: " << num_threads
                << std::endl;
    }

    // Each thread records its ID
    if (tid < 20) {
      thread_ids[tid] = tid;
    }

#pragma omp critical
    {
      std::cout << "Hello from thread " << tid << " of " << num_threads
                << std::endl;
    }
  }

  // Verify all threads were used
  std::cout << "\nThread verification:" << std::endl;
  int active_threads = 0;
  for (int i = 0; i < 20; ++i) {
    if (thread_ids[i] != -1) {
      active_threads++;
    }
  }
  std::cout << "Active threads detected: " << active_threads << std::endl;

  return 0;
}