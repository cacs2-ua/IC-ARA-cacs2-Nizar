#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;


// Parallelizing looping through a vector
int vectorLoop(){
    const int size = 1000000;
    std::vector<int> numbers(size, 1);

    int sum = 0;

    #pragma omp parallel for reduction(+:sum) // Used for parallelizing the loop and summing the results
    for (int i = 0; i < size; ++i) {
        sum += numbers[i];
    }

    return sum;
}


// Critical region parallelization
int criticalRegion(){

    int counter = 0;

    #pragma omp parallel // Used for parallelizing the loop
    {
        #pragma omp critical // Used for critical regions, where only one thread can access at a time
        {
            counter++;
        }
    }

    return counter;
}


// Parallelization of two different sections
int* parallelSections(){

    int resultA, resultB;

    #pragma omp parallel sections // Used for parallelizing the sections
    {
        #pragma omp section // Used to create a section
        {
            resultA = 2 + 3;
        }

        #pragma omp section // Used to create a section
        {
            resultB = 5 - 1;
        }
    }

    int* returnArray = new int[2];
    returnArray[0] = resultA;
    returnArray[1] = resultB;
    return returnArray;
}


// Parallelization of two different tasks
void taskA() {
    #pragma omp task // Used to create a task
    {
        int result = 2 + 3;
    }
}
void taskB() {
    #pragma omp task // Used to create a task
    {
        int result = 2 + 3;
    }
}
void taskParallelization(){

    #pragma omp parallel // Used for parallelizing the tasks
    {
        #pragma omp single // Used to create a single thread
        {
            taskA();
            taskB();
        }
        #pragma omp taskwait // Used to wait for all tasks to finish
    }
    cout << " [Task]  Completed two different parallel tasks succesfully!" << endl << endl;
    return;
}


// Parallelization of a loop with an atomic operation
int forAtomic(){
    const int size = 1000000;
    int sum = 0;

    #pragma omp parallel for // Used for parallelizing the loop
    for (int i = 0; i < size; ++i) {
        #pragma omp atomic // Used for atomic operations, without many threds accessing the same variable
        sum += i;
    }

    return sum;
}


// Parallelization of a loop with a barrier
int barrierThreads(){

    const int size = 5;
    int sharedArray[size];

    #pragma omp parallel // Used for parallelizing the loop
    {
        #pragma omp for // Used for declaring the loop
        for (int i = 0; i < size; ++i) {
            sharedArray[i] = i;
        }

        #pragma omp barrier // Used for waiting for all threads to reach the barrier

        #pragma omp for // Used for declaring the loop
        for (int i = 0; i < size; ++i) {
            sharedArray[i] *= 2;
        }
    }
    
    int sum = 0;
    cout << " [Barrier]  Array result: [" << flush;
    for (int i = 0; i < size; ++i) {
        sum += sharedArray[i];
        if (i == size - 1) {
            cout << sharedArray[i] << "]" << flush;
            break;
        }
        cout << sharedArray[i] << ", " << flush;
    }

    return sum;
}


// Parallelization of an ordered loop with a private variable
void threadPrivate(){

    int privCounter;

    #pragma omp parallel num_threads(6) private(privCounter) // Used for parallelizing the loop with 6 threads using a private variable
    {
        #pragma omp for ordered // Used for declaring the loop as ordered
        for(int i = 0; i < 6; ++i) {
            #pragma omp ordered // Used for assuring that the loop is executed in order
            {
                privCounter = omp_get_thread_num(); // Used for getting the thread number
                if(i == 0){
                    cout << " [Thread Private]  Threads: " << privCounter << flush;
                }
                else{
                    if(i == 5)
                        cout << ", " << privCounter << endl << endl;
                    else
                        cout << ", " << privCounter << flush;
                }
            }
        }
    }
}



// Main function showing the results
int main() {
    
    cout << "\n\n   TASK 1: Study of the OpenMP API\n\n" << endl;

    // Vector structure
    int sum = vectorLoop();
    cout << " [Vector]  Total sum: " << sum << endl << endl;

    // Critical region
    int counter = criticalRegion();
    cout << " [Critical Region]  Counter value: " << counter << endl << endl;

    // Parallel sections
    int* result = parallelSections();
    cout << " [Parallel Sections]  Result A: " << result[0] << "   Result B: " << result[1] << endl << endl;

    // Parallel tasks
    taskParallelization();

    // For loop and atomic
    int sumAtomic = forAtomic();
    cout << " [For/Atomic]  Total sum: " << sumAtomic << endl << endl;

    // Barrier
    int sumBarrier = barrierThreads();
    cout << "   Total sum: " << sumBarrier << endl << endl;

    // Ordered thread private
    threadPrivate();

    return 0;
}
