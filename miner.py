import ctypes
import time
import requests

# compile the C code into a shared library first
libFindHash = ctypes.cdll.LoadLibrary('./libfindhash.so')
# set the arg types of find_hash
libFindHash.find_hash.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_int)]

# how many trials to run on GPU before finding a block to submit
trials = 20_000_000
# how many attempt before rechecking the latest block
attempt_limit = 10
# how many leading zeros are required
difficulty = 7

BLOCK_URL = "https://maiti.info/anindya/services/mining/latestblock.php"
SUBMIT_URL = "https://maiti.info/anindya/services/mining/newblock.php?identity=%s&nonce=%s&newblock=%s"
identity = "wedoalittlemining"

# wrap the C function in a python function
def findhash(data, trials):
    data = ctypes.c_char_p(data.encode('utf-8'))
    hash_p = ctypes.create_string_buffer(32)
    nonce = ctypes.c_uint32()
    trials = ctypes.c_int(trials)
    num_zeros = ctypes.c_int(0)
    libFindHash.find_hash(data, trials, hash_p, ctypes.byref(nonce), ctypes.byref(num_zeros))
    return hash_p.raw, nonce.value, num_zeros.value

# convert bytes to a hex string
def bytes_to_hex(b):
    return ''.join(["%02x" % x for x in b])

# mine a block and submit it to the server
def mine(trials):
    zeros = 0
    attempts = 0
    last_hash = url_to_lines(BLOCK_URL)[0]
    print("Last hash: %s" % last_hash)
    while zeros < difficulty:
        start = time.perf_counter()
        t_hash, nonce, zeros = findhash(last_hash, trials)
        end = time.perf_counter()
        print("Hash: %s Nonce: %d Zeros: %d" % (bytes_to_hex(t_hash), nonce, zeros))
        hashrate = trials / (end - start) / 1e6
        print(f"Hash rate: {hashrate:3.2f} MH/sec")
        attempts += 1
        if attempts > attempt_limit and zeros < difficulty:
            print("Failed to mine a block")
            return

    print('success!')
    submit = SUBMIT_URL % (identity, nonce, bytes_to_hex(t_hash))
    response = url_to_lines(submit)[0]
    print(response)

def url_to_lines(url):
    contents = requests.get(url).content
    contents = contents.decode('utf-8')
    contents = contents.split('\n')
    return contents


def main():
    while True:
        mine(trials)

if __name__ == "__main__":
    main()