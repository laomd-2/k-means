#if !defined(LAOMD_REDUCE)
#define LAOMD_REDUCE


template <class BinaryFunc>
__inline__ __device__ float warpReduce(float val, BinaryFunc f)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val = f(__shfl_down(val, offset), val);
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

__device__ inline void deviceReduceBlockAtomic(float* local, float* global, int k)
{
    for (int i = 0; i < k; i++) {
        local[i] = blockReduce(local[i], thrust::plus<float>(), 0.0f);
        if (threadIdx.x == 0) {
            atomicAdd(global + i, local[i]);
        }
    }
}

__device__ inline void deviceReduceWarpAtomic(float* local, float* global, int k)
{
    int laneId = threadIdx.x & (warpSize - 1);
    for (int i = 0; i < k; i++) {
        local[i] = warpReduce(local[i], thrust::plus<float>());
        if (laneId == 0) {
            atomicAdd(global + i, local[i]);
        }
    }
}

__device__ inline void deviceReduceThreadAtomic(float* local, float* global, int k) {
    for (int i = 0; i < k; i++) {
        atomicAdd(global + i, local[i]);
    }
}

#endif // LAOMD_REDUCE
