#include <omp.h>
#include <iostream>
#define N 1000
#define CHUNKSIZE 100

using namespace std;

int main(int argc, char *argv[]) {
    int i, chunk;
    float a[N], b[N], c[N];
    /* We initialize the vectors */
    for (i=0; i < N; i++)
        a[i] = b[i] = i * 1.0;
    chunk = CHUNKSIZE;
    #pragma omp parallel shared(a,b,c,chunk) private(i)
    {
        #pragma omp for schedule(dynamic,chunk) nowait
        for (i=0; i < N; i++)
        c[i] = a[i] + b[i];
} /* end of parallel region */
    cout << "hola";
    return 0;
}