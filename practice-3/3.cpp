#include <vector>
#include <iostream>
#include <chrono>
#include <thread>
int main() {
    // default initialization and add elements with push_back
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> v1;
    for (int i = 0; i < 10000; i++)
    v1.push_back(i);
    std::this_thread::sleep_for(std::chrono::seconds(3));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-
    start);
    std::cout<<"Default initialization:"<<elapsed.count()<<"ms"<<std::endl;
    // initialized with required size and add elements with direct access
    start = std::chrono::high_resolution_clock::now();
    std::vector<float> v2(10000);
    for (int i = 0; i < 10000; i++)
    v2[i] = i;
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    std::cout<<"Initialization with size:"<<elapsed.count()<<"ms"<<std::endl;
}