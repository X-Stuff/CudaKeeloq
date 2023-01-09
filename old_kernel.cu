#include "stdio.h"
#include "stdint.h"
#include "time.h"
#include "stdlib.h"

#define KeeLoq_NLF		0x3A5C742E
#define bit(x,n)		(((x)>>(n))&1)
#define g5(x,a,b,c,d,e)	(bit(x,a)+bit(x,b)*2+bit(x,c)*4+bit(x,d)*8+bit(x,e)*16)

FILE* fp_log;

uint32_t* dev_ctext = nullptr;
uint32_t* dev_p01 = nullptr;
uint32_t* dev_p02 = nullptr;
uint32_t* dev_p11 = nullptr;
uint32_t* dev_p12 = nullptr;
uint32_t* dev_p21 = nullptr;
uint32_t* dev_p22 = nullptr;
uint32_t* dev_p31 = nullptr;
uint32_t* dev_p32 = nullptr;
uint64_t* dev_key0 = nullptr;
uint64_t* dev_key1 = nullptr;
uint64_t* dev_key2 = nullptr;
uint64_t* dev_key3 = nullptr;
uint64_t* dev_skey0 = nullptr;
uint64_t* dev_skey1 = nullptr;
uint64_t* dev_skey2 = nullptr;
uint64_t* dev_skey3 = nullptr;
uint32_t* dev_p1fin = nullptr;
uint32_t* dev_p2fin = nullptr;
uint32_t* dev_p3fin = nullptr;
uint64_t* dev_keyfin = nullptr;

__device__ uint32_t decrypt(const uint32_t data, const uint64_t key)
{
    uint32_t x = data, r;

    for (r = 0; r < 528; r++)
    {
        x = (x << 1) ^ bit(x, 31) ^ bit(x, 15) ^ (uint32_t)bit(key, (15 - r) & 63) ^ bit(KeeLoq_NLF, g5(x, 0, 8, 19, 25, 30));
    }
    return x;
}


__device__ __host__ uint64_t xorshift64(uint64_t x64)
{
    x64 ^= x64 << 13;
    x64 ^= x64 >> 7;
    x64 ^= x64 << 17;
    return x64;
}

__global__ void rekey(uint64_t* key0, uint64_t* key1, uint64_t* key2, uint64_t* key3, int size) {
    uint64_t val;

    val = key0[0] = xorshift64(key3[size - 1]);
    val = key1[0] = xorshift64(val);
    val = key2[0] = xorshift64(val);
    val = key3[0] = xorshift64(val);


    for (int i = 1; i < size; i++) {
        val = key0[i] = xorshift64(val);
        val = key1[i] = xorshift64(val);
        val = key2[i] = xorshift64(val);
        val = key3[i] = xorshift64(val);
    }
}

__global__ void Kernel(uint32_t* ctext, uint64_t* key, uint32_t* p1, uint32_t* p2, int size, uint64_t* finkey, uint32_t* finp1, uint32_t* finp2, uint32_t* finp3) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        p1[i] = decrypt(ctext[0], key[i]);
        p2[i] = decrypt(ctext[1], key[i]);

        if (p2[i] == (p1[i] + 1)) {
            finkey[0] = key[i];
            finp1[0] = p1[i];
            finp2[0] = p2[i];
            finp3[0] = decrypt(ctext[2], key[i]);
        }


    }
}


// Helper function for using CUDA to add vectors in parallel.
void initkey(uint64_t* key0, uint64_t* key1, uint64_t* key2, uint64_t* key3, uint32_t* ctext, int size) {

    // Allocate GPU buffers for three vectors (two input, one output)
    cudaMalloc((void**)&dev_key0, size * sizeof(uint64_t));
    cudaMalloc((void**)&dev_key1, size * sizeof(uint64_t));
    cudaMalloc((void**)&dev_key2, size * sizeof(uint64_t));
    cudaMalloc((void**)&dev_key3, size * sizeof(uint64_t));
    //shaddowKeys
    cudaMalloc((void**)&dev_skey0, size * sizeof(uint64_t));
    cudaMalloc((void**)&dev_skey1, size * sizeof(uint64_t));
    cudaMalloc((void**)&dev_skey2, size * sizeof(uint64_t));
    cudaMalloc((void**)&dev_skey3, size * sizeof(uint64_t));

    cudaMalloc((void**)&dev_p01, size * sizeof(uint32_t));
    cudaMalloc((void**)&dev_p02, size * sizeof(uint32_t));
    cudaMalloc((void**)&dev_p11, size * sizeof(uint32_t));
    cudaMalloc((void**)&dev_p12, size * sizeof(uint32_t));
    cudaMalloc((void**)&dev_p21, size * sizeof(uint32_t));
    cudaMalloc((void**)&dev_p22, size * sizeof(uint32_t));
    cudaMalloc((void**)&dev_p31, size * sizeof(uint32_t));
    cudaMalloc((void**)&dev_p32, size * sizeof(uint32_t));

    cudaMalloc((void**)&dev_keyfin, 2 * sizeof(uint64_t));
    cudaMalloc((void**)&dev_p1fin, 2 * sizeof(uint32_t));
    cudaMalloc((void**)&dev_p2fin, 2 * sizeof(uint32_t));
    cudaMalloc((void**)&dev_p3fin, 2 * sizeof(uint32_t));
    cudaMalloc((void**)&dev_ctext, 3 * sizeof(uint32_t));


    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(dev_key0, key0, size * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_key1, key1, size * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_key2, key2, size * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_key3, key3, size * sizeof(uint64_t), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_skey0, dev_key0, size * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_skey1, dev_key1, size * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_skey2, dev_key2, size * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_skey3, dev_key3, size * sizeof(uint64_t), cudaMemcpyDeviceToDevice);

    cudaMemcpy(dev_p01, 0, size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_p02, 0, size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_p11, 0, size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_p12, 0, size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_p21, 0, size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_p22, 0, size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_p31, 0, size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_p32, 0, size * sizeof(uint32_t), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_p1fin, 0, 2 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_p2fin, 0, 2 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_p3fin, 0, 2 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_keyfin, 0, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ctext, ctext, 3 * sizeof(uint32_t), cudaMemcpyHostToDevice);



    // Launch a kernel on the GPU with one thread for each element.
    // 2 is number of computational blocks and (size + 1) / 2 is a number of threads in a block
    //addKernel << <1, (size + 1) >> > (dev_key, dev_p1, dev_p2, size);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    //cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
    //cudaMemcpy(p1, dev_p1, size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    //cudaMemcpy(p2, dev_p2, size * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    //cudaFree(dev_key);
    //cudaFree(dev_p1);
    //cudaFree(dev_p2);
}

int old_main(int argc, char** argv) {

    if (argc <= 3) {
        printf("Enter 3 Hopping-codes in Format 0x12345678 !\n");
        return - 1;

    }

    const int arraySize = 1024;
    //uint32_t p1[arraySize];
    //uint32_t p2[arraySize];
    uint64_t key0[arraySize];
    uint64_t key1[arraySize];
    uint64_t key2[arraySize];
    uint64_t key3[arraySize];
    uint64_t ctr = 0;
    uint64_t finalkey[2] = { 0 };
    uint32_t p1fin[2];
    uint32_t p2fin[2];
    uint32_t p3fin[2];

    uint32_t ctext[3] = {
        0xbe65ba43,
        0xac49547c,
        0x46a43de8
    };

    srand(time(NULL));


    sscanf(argv[1],"0x%08x", &ctext[0]);
    sscanf(argv[2], "0x%08x", &ctext[1]);
    sscanf(argv[3], "0x%08x", &ctext[2]);
    if (argc >= 5) {
        sscanf(argv[4], "0x%llx", &key0[0]);
    }

    if (key0[0] == 0) {
        key0[0] = xorshift64(xorshift64((time(NULL)))) * rand() + 0x1fffffffffffffffUl;
    }

    for (int i = 1; i < arraySize; i++) {
        key0[i] = xorshift64(key0[i - 1]);
    }
    key1[0] = xorshift64(key0[arraySize - 1]);

    for (int i = 1; i < arraySize; i++) {
        key1[i] = xorshift64(key1[i - 1]);
    }
    key2[0] = xorshift64(key1[arraySize - 1]);

    for (int i = 1; i < arraySize; i++) {
        key2[i] = xorshift64(key2[i - 1]);
    }
    key3[0] = xorshift64(key2[arraySize - 1]);

    for (int i = 1; i < arraySize; i++) {
        key3[i] = xorshift64(key3[i - 1]);
    }

    initkey(key0, key1, key2, key3, ctext, arraySize);

    ctext[0] = ctext[1] = ctext[2]= 0;

    cudaMemcpy(ctext, dev_ctext, 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("Cuda accelerated Keeloq Bruteforcer...\n\nHoppingcodes: 0x%08X 0x%08X 0x%08X\nStartkey: 0x%llx\nPress Enter to Start Brute Force...\n", ctext[0], ctext[1], ctext[2],key0[0]);
    getchar();

    cudaStream_t stream0,stream1,stream2,stream3,stream4;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);

    while (1 == 1) {

        Kernel << <4, (arraySize + 1) / 4, 0, stream0 >> > (dev_ctext, dev_key0, dev_p01, dev_p02, arraySize, dev_keyfin, dev_p1fin, dev_p2fin, dev_p3fin);
        Kernel << <4, (arraySize + 1) / 4, 0, stream1 >> > (dev_ctext, dev_key1, dev_p11, dev_p12, arraySize, dev_keyfin, dev_p1fin, dev_p2fin, dev_p3fin);
        Kernel << <4, (arraySize + 1) / 4, 0, stream2 >> > (dev_ctext, dev_key2, dev_p21, dev_p22, arraySize, dev_keyfin, dev_p1fin, dev_p2fin, dev_p3fin);
        Kernel << <4, (arraySize + 1) / 4, 0, stream3 >> > (dev_ctext, dev_key3, dev_p31, dev_p32, arraySize, dev_keyfin, dev_p1fin, dev_p2fin, dev_p3fin);
        rekey << <1, 1, 0, stream4 >> > (dev_skey0, dev_skey1, dev_skey2, dev_skey3, arraySize);
        //cudaDeviceSynchronize();

        cudaMemcpy(dev_key0, dev_skey0, arraySize * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_key1, dev_skey1, arraySize * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_key2, dev_skey2, arraySize * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_key3, dev_skey3, arraySize * sizeof(uint64_t), cudaMemcpyDeviceToDevice);

        cudaMemcpy(finalkey, dev_keyfin, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        if (finalkey[0] != 0) {
            cudaMemcpy(p1fin, dev_p1fin, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(p2fin, dev_p2fin, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(p3fin, dev_p3fin, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

            if (p3fin[0] == (p2fin[0] + 1)) {
                fp_log = fopen("logfile.log", "a");
                fprintf(fp_log, "\nPossible Key Found!!! Key: %llX %04X / %04X / %04X Counter: %llX\n\a\a\a\a", finalkey[0], p1fin[0], p2fin[0], p3fin[0], ctr * arraySize * 4);
                printf("\nPossible Key Found!!! Key: %llX %04X / %04X / %04X Counter: %llX\n\a\a\a\a", finalkey[0], p1fin[0], p2fin[0], p3fin[0], ctr * arraySize * 4);
                fclose(fp_log);
                getchar();
                return 0;
            }
            else {
                fp_log = fopen("logfile.log", "a");
                fprintf(fp_log, "Match! Key: %llX %04X / %04X / %04X Counter: %llX\n", finalkey[0], p1fin[0], p2fin[0], p3fin[0], ctr* arraySize);
                printf("\nMatch! Key: %llX %04X / %04X / %04X Counter: %llX\n\a", finalkey[0], p1fin[0], p2fin[0], p3fin[0], ctr * arraySize);
                finalkey[0] = finalkey[1] = 0;
                fclose(fp_log);
                cudaMemcpy(dev_keyfin, finalkey, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice);
            }
        }

        if (ctr % 0xFFFF == 0) {
            printf(">");
            //cudaMemcpy(key, dev_key, arraySize * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            //cudaMemcpy(p1, dev_p1, arraySize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            //cudaMemcpy(p2, dev_p2, arraySize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            //printf("Key 0: %I64X %04X / %04X Counter: %I64X\n", key[0], p1[0], p2[0], ctr * arraySize);
        }

        //rekey << <1, 1 >> > (dev_key0, dev_key1, dev_key2, dev_key3, arraySize);
        ctr++;

    }

    cudaDeviceReset();

    return 0;
}