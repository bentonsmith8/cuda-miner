#include <cuda.h>

/* Random Kernel ----------------
*       Generates a random value.
*/
__device__
unsigned int random_kernel(unsigned int seed, unsigned int index) {

    curandState_t state;
    curand_init(
        seed,  // the seed can be the same for every thread and is set to be the time
        index, // the sequence number should be different for every thread
        0,     // an offset into the random number sequence at which to begin sampling
        &state // the random state object
    );

    // generate a random number
    return (unsigned int)(curand(&state));

} // End Random Kernel //

/* Nonce Kernel ----------------------------------
*       Generates an array of random nonce values.
*/
__global__
void nonce_kernel(unsigned int* nonce_array, unsigned int array_size, unsigned int mod, unsigned int seed) {

    // Calculate thread rank
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    // Generate random nonce values for every item in the array
    if (index < array_size) {
        unsigned int rand = random_kernel(seed, index);
        nonce_array[index] = rand % mod;
    }

} // End Nonce Kernel //

/* Reduction Kernel --------------------------------------
*       Reduces Hash Values to a local minimum
*/
__global__
void reduction_kernel(BYTE* hash_out, BYTE* nonce_out, BYTE* hash_in, BYTE* nonce_in, size_t trials) {

    // Calculate thread index
    uint t_idx = threadIdx.x;
    uint index = blockDim.x * blockIdx.x + t_idx;

    // shared reduction arrays for the block
    __shared__ BYTE hash_red[ BLOCK_SIZE * 32];
    __shared__ BYTE nonce_red[ BLOCK_SIZE * 12];

    // set up reduction arrays if thread is inside array
    // otherwise set value to max
    if (index < trials) {
        memcpy(&hash_red[t_idx*32], &hash_in[index*32], 32*sizeof(BYTE));
        memcpy(&nonce_red[t_idx*12], &nonce_in[index*12], 12*sizeof(BYTE));
    } else {
        memset(&hash_red[t_idx*32], 0xFF, 32*sizeof(BYTE));
        memset(&nonce_red[t_idx*12], 0xFF, 12*sizeof(BYTE));
    }

    for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        // if the hash at t_idx + stride is less than at t_idx
        // then move the one at t_idx + stride over to t_idx
        if (t_idx < stride && 
            hash_red[t_idx*32 + 32*stride] < hash_red[t_idx*32])
        {
            memcpy(&hash_red[t_idx*32], &hash_red[t_idx*32 + 32*stride], 32*sizeof(BYTE));
            memcpy(&nonce_red[t_idx*12], &nonce_red[t_idx*12 + 32*stride], 12*sizeof(BYTE));
        }
    }

    // store the block minimum in the device memory
    if (t_idx == 0) {
        memcpy(&hash_out[blockIdx.x*32], &hash_red[0], 32*sizeof(BYTE));
        memcpy(&nonce_out[blockIdx.x*12], &nonce_red[0], 12*sizeof(BYTE));
    }

} // End Reduction Kernel //
