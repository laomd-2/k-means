/*
 * @Description: 一些归约函数，以warp、block、device为单位
 * @Author: 劳马东
 * @Date: 2019-06-09 21:20:26
 * @LastEditTime: 2019-06-09 22:38:59
 */
#if !defined(LAOMD_REDUCE)
#define LAOMD_REDUCE

/**
 * @description: 用洗牌指令实现warp为单位的归约
 * @param
    val: 归约的局部值
    f: 归约操作
 * @return: 归约的warp全局值
 */
template <class BinaryFunc>
__inline__ __device__ float warpReduce(float val, BinaryFunc f)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val = f(__shfl_down(val, offset), val);
    return val;
}

/**
 * @description: block为单位的归约
 * @param
    val: 归约的局部值
    f: 归约操作
 * @return: 归约的block全局值
 */
template <class BinaryFunc>
__inline__ __device__ float blockReduce(float val, BinaryFunc f, float default_val)
{
    static __shared__ float shared[32]; // 最多有32个warp
    int lane = (threadIdx.x & (warpSize - 1));
    int wid = threadIdx.x / warpSize;

    val = warpReduce(val, f); // 每个warp先归约一次，得到warp全局值
    if (lane == 0)
        shared[wid] = val; // 将warp全局值写到shared memory
    __syncthreads();

    // block中不一定够32个warp，用除法计算warp数
    val = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : default_val;
    if (wid == 0)       // 将所有的warp全局值在进行一次warp归约
        val = warpReduce(val, f);
    return val;
}

/**
 * @description: 不同层次的原子操作，实现device归约
 * @param 
    local: 每个线程的局部数组
    global: 结果数组
    k: 数组长度
 * @return: 
 */
__device__ inline void deviceReduceBlockAtomic(float* local, float* global, int k)
{
    // 先计算block的全局归约值，在原子加
    for (int i = 0; i < k; i++) {
        local[i] = blockReduce(local[i], thrust::plus<float>(), 0.0f);
        if (threadIdx.x == 0) {
            atomicAdd(global + i, local[i]);
        }
    }
}

__device__ inline void deviceReduceWarpAtomic(float* local, float* global, int k)
{
    // 先计算warp的全局归约值，在原子加
    int laneId = threadIdx.x & (warpSize - 1);
    for (int i = 0; i < k; i++) {
        local[i] = warpReduce(local[i], thrust::plus<float>());
        if (laneId == 0) {
            atomicAdd(global + i, local[i]);
        }
    }
}

__device__ inline void deviceReduceThreadAtomic(float* local, float* global, int k) {
    // 每个线程直接原子加
    for (int i = 0; i < k; i++) {
        atomicAdd(global + i, local[i]);
    }
}

#endif // LAOMD_REDUCE
