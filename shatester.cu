#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "sha256.cuh"
#include "findhash.cuh"
#include "helpers.cuh"

int main(int argc, char const *argv[])
{
    if (argc != 3)
    {
        printf("Usage: %s <string> <trials>\n", argv[0]);
        return 1;
    }

    time_t t;
    srand((unsigned) time(&t));

    int target = 7;

    const char *str = argv[1];
    size_t str_len = strlen(str);
    int trials = atoi(argv[2]);

    // long unsigned int *int_nonces = (long unsigned int*)malloc(trials*sizeof(long unsigned int));
    // BYTE* nonces = (BYTE*)malloc(trials*12*sizeof(BYTE));
    // generate_nonces(nonces, trials, int_nonces);

    // for (int i = 0; i < trials; i++) {
    //     printf("%d: ", nonces[i*12]);
    //     for (int j = 0; j < nonces[i*12]; j++)
    //     {
    //         printf("%c", nonces[i*12+1+j]);
    //     }
    //     printf("\n");
    // }


    // int num_blocks = ceil((float)trials/BLOCK_SIZE);

    // dim3 dimGrid(num_blocks, 1, 1);
    // dim3 dimBlock(BLOCK_SIZE, 1, 1);

    // BYTE *d_str;
    // cudaMalloc((void**)&d_str, str_len*sizeof(char));
    // cudaMemcpy(d_str, str, str_len*sizeof(char), cudaMemcpyHostToDevice);


    // BYTE *d_nonces;
    // cudaMalloc((void**)&d_nonces, trials*12*sizeof(BYTE));
    // cudaMemcpy(d_nonces, nonces, trials*12*sizeof(BYTE), cudaMemcpyHostToDevice);

    // BYTE *d_hashes;
    // cudaMalloc((void**)&d_hashes, 32*trials*sizeof(BYTE));

    // pre_sha256();

    // sha256<<<dimGrid, dimBlock>>>(d_str, str_len, d_nonces, d_hashes, trials);

    // BYTE* d_red_hashes;
    // cudaMalloc((void**)&d_red_hashes, 32*num_blocks*sizeof(BYTE));

    // BYTE* d_red_nonces;
    // cudaMalloc((void**)&d_red_nonces, 12*num_blocks*sizeof(BYTE));

    // cudaDeviceSynchronize();

    // size_t shared_mem = (32+12)*BLOCK_SIZE*sizeof(BYTE);

    // reduction_kernel<<<dimGrid, dimBlock>>>(d_red_hashes, d_red_nonces, d_hashes, d_nonces, trials);

    // cudaDeviceSynchronize();

    // BYTE *all_hash = (BYTE*)malloc(32*trials*sizeof(BYTE));
    // cudaMemcpy(all_hash, d_hashes, 32*trials*sizeof(BYTE), cudaMemcpyDeviceToHost);
    
    // for (int i = 0; i < trials; i++) {
    //     printf("%ld: ", int_nonces[i]);
    //     for (int j = 0; j < 32; j++)
    //     {
    //         printf("%02x", hash[i*32+j]);
    //     }
    //     printf("\n");
    // }

    // BYTE *red_hashes = (BYTE*)malloc(32*num_blocks*sizeof(BYTE));
    // cudaMemcpy(red_hashes, d_red_hashes, 32*num_blocks*sizeof(BYTE), cudaMemcpyDeviceToHost);

    // BYTE *red_nonces = (BYTE*)malloc(12*num_blocks*sizeof(BYTE));
    // cudaMemcpy(red_nonces, d_red_nonces, 12*num_blocks*sizeof(BYTE), cudaMemcpyDeviceToHost);

    // cudaDeviceSynchronize();

    // printf("All hashes:\n");
    // for (size_t i = 0; i < trials; i++)
    // {
    //     for (size_t j = 0; j < 32; j++)
    //     {
    //         printf("%02x", all_hash[i*32+j]);
    //     }
    //     printf("\n");
    // }

    // printf("\n");

    BYTE *hash = (BYTE*)malloc(32*sizeof(BYTE));
    long unsigned int nonce;
    int num_zeros;

    find_hash(str, trials, hash, &nonce, &num_zeros);

    printf("Done hashing.\n");

    printf("Hash: ");
    for (int i = 0; i < 32; i++)
    {
        printf("%02x", hash[i]);
    }
    printf("\n");
    printf("Nonce: %ld\n", nonce);
    printf("Number of zeros: %d\n", num_zeros);

    // find smallest hash
    // int smallest = 0;
    // unsigned long int l1, l2;
    // for (int i = 1; i < trials; i++)
    // {
    //     l1 = all_hash[i*32]<<24 | all_hash[i*32+1]<<16 | all_hash[i*32+2]<<8 | all_hash[i*32+3];
    //     l2 = all_hash[smallest*32]<<24 | all_hash[smallest*32+1]<<16 | all_hash[smallest*32+2]<<8 | all_hash[smallest*32+3];
    //     if (l1 < l2)
    //     {
    //         smallest = i;
    //     }
    // }

    // printf("Smallest hash: ");
    // for (int i = 0; i < 32; i++)
    // {
    //     printf("%02x", all_hash[smallest*32+i]);
    // }
    // printf("\n");

    // printf("Nonce: %ld\n", int_nonces[smallest]);

    // // check to see if smallest hash has the target number of leading zeros in hex
    // int num_zeros = 0;
    // for (int i = 0; i < 32; i++)
    // {
    //     if (all_hash[smallest*32+i] == 0)
    //     {
    //         num_zeros += 2;
    //     }
    //     else
    //     {
    //         if (all_hash[smallest*32+i] < 16)
    //         {
    //             num_zeros++;
    //         }
    //         break;
    //     }
    // }

    // printf("Number of leading zeros: %d\n", num_zeros);


    // printf("Reduced hashes:\n");
    // for (size_t i = 0; i < num_blocks; i++)
    // {
    //     for (size_t j = 0; j < 32; j++)
    //     {
    //         printf("%02x", red_hashes[i*32+j]);
    //     }
    //     printf("\n");
    // }

    // for (size_t i = 0; i < num_blocks; i++)
    // {
    //     char nonce[11];
    //     memcpy(nonce, red_nonces+i*12+1, (unsigned char)red_nonces[i*12]);
    //     printf("%s: ", nonce);
    //     for (size_t j = 0; j < 32; j++)
    //     {
    //         printf("%02x", red_hashes[i*32+j]);
    //     }
    //     printf("\n");
    // }
    



    // cudaFree(d_str);
    // cudaFree(d_nonces);
    // cudaFree(d_hashes);
    // cudaFree(d_red_hashes);
    // cudaFree(d_red_nonces);

    // free(red_hashes);
    // free(red_nonces);
    // free(nonces);
    // free(int_nonces);

    return 0;
}
