#include <iostream>
#include <future>
#include <chrono>
#include <thread>
int task(int id, int millis) {
std::this_thread::sleep_for(std::chrono::milliseconds(millis));
std::cout<<"Task"<<id<<" completed"<<std::endl;
return id;
}
int main() {
auto start = std::chrono::high_resolution_clock::now();
std::future<int> task1 = std::async(std::launch::async, task, 1, 2000);
std::future<int> task2 = std::async(std::launch::async, task, 2, 3000);
task1.wait();
int taskId = task2.get();
auto end = std::chrono::high_resolution_clock::now();
auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
std::cout<<"Completed in:"<<elapsed.count()<<"ms"<<std::endl;
}