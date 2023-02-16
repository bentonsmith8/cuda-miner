#ifndef _FINDHASH_H
#define _FINDHASH_H

#define BLOCK_SIZE 1024

extern "C" { void find_hash(const char *str, int trials, BYTE *hash, long unsigned int *nonce, int *num_zeros); }
void find_hash(const char *str, int trials, BYTE *hash, long unsigned int *nonce, int *num_zeros);
#endif