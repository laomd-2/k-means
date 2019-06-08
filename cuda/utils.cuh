#ifndef __LAOMD__UTILS
#define __LAOMD__UTILS

#include <thrust/tuple.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
using namespace thrust;

template <class T>
struct bigger_tuple {
    __device__ __host__
    tuple<T,int> operator()(const tuple<T,int> &a, const tuple<T,int> &b) 
    {
        if (a > b) return a;
        else return b;
    } 

};

template <class T>
__device__ int argmax(T* first, int size) {
    // create implicit index sequence [0, 1, 2, ... )
    counting_iterator<int> begin(0); counting_iterator<int> end(size);
    tuple<T,int> init(*first,0); 
    tuple<T,int> smallest;
    smallest = reduce(device, make_zip_iterator(make_tuple(first, begin)), make_zip_iterator(make_tuple(first + size, end)),
                      init, bigger_tuple<T>());
    return get<1>(smallest);
}

#endif