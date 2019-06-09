#ifndef __LAOMD__UTILS
#define __LAOMD__UTILS

#include <thrust/tuple.h>
#include <thrust/reduce.h>
using namespace thrust;

template <class T>
struct bigger_tuple
{
    __device__ __host__
        tuple<T, int>
        operator()(const tuple<T, int> &a, const tuple<T, int> &b)
    {
        if (a > b)
            return a;
        else
            return b;
    }
};

template <class T>
int argmax(const device_vector<T> &vec)
{
    // create implicit index sequence [0, 1, 2, ... )
    counting_iterator<int> begin(0);
    counting_iterator<int> end(vec.size());
    tuple<T, int> init(vec[0], 0);
    tuple<T, int> smallest;
    smallest = reduce(make_zip_iterator(make_tuple(vec.begin(), begin)), make_zip_iterator(make_tuple(vec.end(), end)),
                      init, bigger_tuple<T>());
    return get<1>(smallest);
}

template <class BinaryFunc>
__inline__ __device__ float warpReduce(float val, BinaryFunc f)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val = f(__shfl_down(val, offset), val);
    return val;
}

template <class BinaryFunc>
__inline__ __device__ float warpAllReduce(float val, BinaryFunc f)
{
    for (int mask = warpSize / 2; mask > 0; mask >>= 1)
        val = f(__shfl_xor(val, mask), val);
    return val;
}

template <class BinaryFunc>
__inline__ __device__ float blockReduce(float val, BinaryFunc f, float default_val)
{
    static __shared__ float shared[32]; // Shared mem for 32 partial sums
    int lane = (threadIdx.x & (warpSize - 1));
    int wid = threadIdx.x / warpSize;

    val = warpReduce(val, f); // Each warp performs partial reduction
    if (lane == 0)
        shared[wid] = val; // Write reduced value to shared memory
    __syncthreads(); // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : default_val;
    if (wid == 0)
        val = warpReduce(val, f); //Final reduce within first warp
    return val;
}
#endif