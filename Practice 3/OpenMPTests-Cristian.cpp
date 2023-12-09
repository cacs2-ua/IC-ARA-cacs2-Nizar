#include <omp.h>
#include <iostream>
#include <limits>
#include <vector>
#include <chrono>
#include <thread>

//Parallelizing a simple for loop that fills an array with values.
int parallellismLoop() {
    const int size = 100;
    int array[size];

    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        array[i] = i * i;
    }

    // Print the first 10 elements for demonstration
    for (int i = 0; i < 10; ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

//Parallel processing of an array (e.g., finding the maximum element).

int parallellismArray() {
    const int size = 100;
    int array[size];

    // Initialize the array with some values
    for (int i = 0; i < size; ++i) {
        array[i] = i;
    }

    int maxVal = std::numeric_limits<int>::min();

    #pragma omp parallel for reduction(max:maxVal)
    for (int i = 0; i < size; ++i) {
        if (array[i] > maxVal) {
            maxVal = array[i];
        }
    }

    std::cout << "Maximum value: " << maxVal << std::endl;

    return 0;
}


//Parallel quicksort algorithm (simplified version)
void quicksort(std::vector<int>& arr, int left, int right) {
    int i = left, j = right;
    int tmp;
    int pivot = arr[(left + right) / 2];

    // Partition
    while (i <= j) {
        while (arr[i] < pivot)
            i++;
        while (arr[j] > pivot)
            j--;
        if (i <= j) {
            tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
            i++;
            j--;
        }
    }

    // Recursion
    if (left < j) {
        #pragma omp task
        quicksort(arr, left, j);
    }
    if (i < right) {
        #pragma omp task
        quicksort(arr, i, right);
    }
}

int parallellismQuicksort() {
    std::vector<int> arr = {24, 12, 35, 47, 23, 47, 21, 13};

    #pragma omp parallel
    {
        #pragma omp single
        quicksort(arr, 0, arr.size() - 1);
    }

    for (auto val : arr) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}


//Handling multiple tasks in parallel
void process(int id) {
    // Simulate some work
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "Processed task " << id << std::endl;
}

int parallellismTasks() {
    const int numTasks = 5;

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < numTasks; ++i) {
                #pragma omp task
                process(i);
            }
        }
    }

    return 0;
}


//Parallel processing of a large dataset (e.g., summing elements)
int parallellismLargeSumming() {
    const int size = 1000000;
    std::vector<int> data(size, 1);  // Large dataset initialized with 1s

    long long sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; ++i) {
        sum += data[i];
    }

    std::cout << "Sum: " << sum << std::endl;

    return 0;
}


//Implementing a simple three-stage pipeline
void stage1(int& data) {
    // Process stage 1
    data += 5;
}

void stage2(int& data) {
    // Process stage 2
    data *= 2;
}

void stage3(int& data) {
    // Process stage 3
    data -= 3;
}

int parallellismPipeline() {
    const int numItems = 10;
    std::vector<int> data(numItems, 0);

    #pragma omp parallel for
    for (int i = 0; i < numItems; ++i) {
        stage1(data[i]);
        stage2(data[i]);
        stage3(data[i]);
    }

    for (auto val : data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}


//Parallel processing using object methods.
class Processor {
public:
    void process(int id) {
        // Simulate some work
        std::cout << "Processing item " << id << std::endl;
    }
};

int parallellismObjectMethods() {
    const int numItems = 10;
    std::vector<Processor> processors(numItems);

    #pragma omp parallel for
    for (int i = 0; i < numItems; ++i) {
        processors[i].process(i);
    }

    return 0;
}

//Parallelizing a recursive function (e.g., Fibonacci sequence)
int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    int a, b;

    #pragma omp task shared(a)
    a = fibonacci(n - 1);

    #pragma omp task shared(b)
    b = fibonacci(n - 2);

    #pragma omp taskwait
    return a + b;
}

int parallellismRecursiveFunction() {
    int result;

    #pragma omp parallel
    {
        #pragma omp single
        result = fibonacci(10);
    }

    std::cout << "Fibonacci(10) = " << result << std::endl;

    return 0;
}


int main() {
    // Here you can test all the modules

    return 0;
}