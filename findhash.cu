#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

#include "sha256.cuh"
#include "findhash.cuh"

// kernel function to find hashes in parallel
__global__ void sha256(BYTE *str, size_t str_len, BYTE *nonces, BYTE *hashes, size_t trials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < trials) {
        size_t nonce_len = nonces[idx*12];
        size_t new_len = str_len + nonce_len+1;
        BYTE new_str[96];
        memcpy(new_str, str, str_len);
        new_str[str_len] = ' ';
        memcpy(new_str+str_len+1, nonces+idx*12+1, nonce_len);
        SHA256_CTX ctx;
        sha256_init(&ctx);
        sha256_update(&ctx, new_str, new_len);
        sha256_final(&ctx, &hashes[idx*32]);
    }

}

// host function to generate nonces for each thread
// this is inefficient, but it works
__host__ void generate_nonces(BYTE *nonces, size_t trials, long unsigned int *int_nonces)
{
    long int rand_int = rand();
    char c[12];
    size_t len;
    for (int i = 0; i < trials; i++)
    {
        int_nonces[i] = rand_int+i;
        sprintf(c, "%ld", rand_int+i);
        len = strlen(c);
        nonces[i*12] = (len&0xff);
        memcpy(nonces+i*12+1, c, len);
    }
}

// host function to copy constants to device
__host__ void pre_sha256() {
    cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice);
}

// host function to call kernel function
extern "C" {
    void find_hash(const char *str, int trials, BYTE *hash, long unsigned int *nonce, int *num_zeros) {
        time_t t;
        srand((unsigned) time(&t));

        size_t str_len = strlen(str);

        long unsigned int *int_nonces = (long unsigned int*)malloc(trials*sizeof(long unsigned int));
        BYTE* nonces = (BYTE*)malloc(trials*12*sizeof(BYTE));
        generate_nonces(nonces, trials, int_nonces);

        int num_blocks = ceil((float)trials/BLOCK_SIZE);

        dim3 dimGrid(num_blocks, 1, 1);
        dim3 dimBlock(BLOCK_SIZE, 1, 1);

        BYTE *d_str;
        cudaMalloc((void**)&d_str, str_len*sizeof(char));
        cudaMemcpy(d_str, str, str_len*sizeof(char), cudaMemcpyHostToDevice);

        BYTE *d_nonces;
        cudaMalloc((void**)&d_nonces, trials*12*sizeof(BYTE));
        cudaMemcpy(d_nonces, nonces, trials*12*sizeof(BYTE), cudaMemcpyHostToDevice);

        BYTE *d_hashes;
        cudaMalloc((void**)&d_hashes, 32*trials*sizeof(BYTE));

        pre_sha256();

        sha256<<<dimGrid, dimBlock>>>(d_str, str_len, d_nonces, d_hashes, trials);

        cudaDeviceSynchronize();

        BYTE *all_hash = (BYTE*)malloc(32*trials*sizeof(BYTE));
        cudaMemcpy(all_hash, d_hashes, 32*trials*sizeof(BYTE), cudaMemcpyDeviceToHost);

        int smallest = 0;
        unsigned long int l1, l2;
        for (int i = 0; i < trials; i++)
        {
            l1 = all_hash[i*32]<<24 | all_hash[i*32+1]<<16 | all_hash[i*32+2]<<8 | all_hash[i*32+3];
            l2 = all_hash[smallest*32]<<24 | all_hash[smallest*32+1]<<16 | all_hash[smallest*32+2]<<8 | all_hash[smallest*32+3];
            if (l1 < l2)
            {
                smallest = i;
            }
        }
        memcpy(hash, all_hash+smallest*32, 32*sizeof(BYTE));

        *nonce = int_nonces[smallest];

        *num_zeros = 0;
        for (int i = 0; i < 32; i++)
        {
            if (all_hash[smallest*32+i] == 0)
            {
                *num_zeros += 2;
            }
            else
            {
                if (all_hash[smallest*32+i] < 16)
                {
                    *num_zeros += 1;
                }
                break;
            }
        }

        cudaFree(d_str);
        cudaFree(d_nonces);
        cudaFree(d_hashes);

        free(all_hash);
        free(int_nonces);
        free(nonces);

    }
}
