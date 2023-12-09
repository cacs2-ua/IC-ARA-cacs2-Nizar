#include <iostream>
#include <vector>
#include <omp.h>

int main() {
    const int size = 1000000;
    std::vector<int> numbers(size, 1);  // Vector de 1 millón de elementos, inicializados en 1

    int sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; ++i) {
        sum += numbers[i];
    }

    std::cout << "Suma total: " << sum << std::endl;

    return 0;
}