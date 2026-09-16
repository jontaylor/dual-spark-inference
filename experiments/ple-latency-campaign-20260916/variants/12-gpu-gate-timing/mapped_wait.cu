// GB10 system-coherent CPU publication and graph-compatible GPU wait.
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" void gb10_release(uint64_t* pointer, uint64_t value) {
    __atomic_store_n(pointer, value, __ATOMIC_RELEASE);
}
extern "C" uint64_t gb10_acquire(const uint64_t* pointer) {
    return __atomic_load_n(pointer, __ATOMIC_ACQUIRE);
}
__device__ __forceinline__ uint64_t acquire(const uint64_t* pointer) {
    uint64_t value;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(pointer) : "memory");
    return value;
}
__device__ __forceinline__ uint64_t nanoseconds() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}
__global__ void wait_ready(uint64_t* flags, uint64_t timeout_ns, bool trap) {
    uint64_t expected = acquire(flags + 8);
    uint64_t start = nanoseconds();
    uint64_t polls = 0;
    while (acquire(flags) < expected) {
        ++polls;
        if (nanoseconds() - start >= timeout_ns) {
            uint64_t end = nanoseconds();
            flags[20] = end;
            flags[22] = acquire(flags + 19);
            flags[17] = end - start;
            flags[18] = polls;
            flags[16] = UINT64_MAX;
            flags[24] = expected;
            __threadfence_system();
            if (trap) asm volatile("trap;");
            return;
        }
        __nanosleep(100);
    }
    uint64_t end = nanoseconds();
    flags[20] = end;
    flags[22] = acquire(flags + 19);
    flags[17] = end - start;
    flags[18] = polls;
    flags[16] = 0;
    flags[24] = expected;
    __threadfence_system();
}
__global__ void early_work(uint64_t duration_ns) {
    uint64_t start = nanoseconds();
    while (nanoseconds() - start < duration_ns) __nanosleep(100);
}
extern "C" int gb10_wait(uint64_t* flags, uint64_t timeout_ns, int trap, void* stream) {
    wait_ready<<<1, 1, 0, (cudaStream_t)stream>>>(flags, timeout_ns, trap);
    return (int)cudaGetLastError();
}
extern "C" int gb10_early(uint64_t duration_ns, void* stream) {
    early_work<<<1, 1, 0, (cudaStream_t)stream>>>(duration_ns);
    return (int)cudaGetLastError();
}
