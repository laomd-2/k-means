#if !defined(LAOMD_SCAN)
#define LAOMD_SCAN

#include <algorithm>

__device__ float warp_inclusive_scan(float val1) {
    float val2;
    int laneid = threadIdx.x % 32;
    for(int stride = 1; stride < 32; stride<<=1){
        val2 = __shfl_up(val1, stride);
        if (laneid >= stride)
            val1 += val2;
    }
    return val1;
}

__global__ void block_inclusive_scan(float *src, int n) {
    __shared__ float shared[32];

    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    int wid = threadIdx.x / warpSize;
    int laneid = threadIdx.x % warpSize;
    float *dst = src;

    for (int index = tid; index < n; index += blockDim.x * gridDim.x) {
        float val1 = warp_inclusive_scan(src[index]);
        if (laneid == warpSize - 1)     shared[wid] = val1;
        __syncthreads();
        if (wid == 0) shared[laneid] = warp_inclusive_scan(shared[laneid]);
        __syncthreads();
        if (wid != 0) val1 += shared[wid - 1];
        dst[index] = val1;
    }
}

#endif // LAOMD_SCAN
