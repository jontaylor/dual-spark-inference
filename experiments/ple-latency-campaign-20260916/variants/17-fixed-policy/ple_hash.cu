// SPDX-License-Identifier: Apache-2.0
// Exact PLE row IDs directly from runner GPU inputs into mapped host storage.
#include <cuda_runtime.h>
#include <stdint.h>

__device__ int64_t context(const int32_t *tokens, const int32_t *history,
                          int64_t start, int64_t length, int64_t column,
                          int64_t ngram, int64_t eos) {
    if(column<0)return history[column+ngram-1];
    if(column>=length)return eos;
    int64_t token=tokens[start+column];
    return token<0?0:token;
}
__global__ void hash_ids(const int32_t *tokens,int64_t count,
                        const int32_t *query,int64_t requests,
                        const int32_t *history,int64_t history_stride,
                        int64_t ngram,int64_t heads,int64_t eos,
                        const int64_t *multipliers,const int64_t *sizes,
                        const int64_t *offsets,int64_t *out,int32_t *error,uint64_t *timing) {
    if(timing&&blockIdx.x==0&&threadIdx.x==0){
        uint64_t now;asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));timing[19]=now;
    }
    __shared__ int valid;
    __shared__ int64_t max_length;
    int64_t total_heads=(ngram-1)*heads;
    if(threadIdx.x==0){
        valid=query[0]==0&&query[requests]>0&&query[requests]<=count;
        max_length=1;
        for(int64_t r=0;r<requests;r++){
            int64_t length=(int64_t)query[r+1]-query[r];
            if(length<0)valid=0;
            if(length>max_length)max_length=length;
        }
        for(int64_t h=0;h<total_heads;h++)
            if(sizes[h]<=0||offsets[h]<0||offsets[h]>INT64_MAX-sizes[h])valid=0;
        if(!valid)atomicExch(error,-1);
    }
    __syncthreads();
    if(!valid)return;
    int64_t r=blockIdx.x,start=query[r],length=(int64_t)query[r+1]-start;
    int64_t end=r+1==requests?count:query[r+1];
    const int32_t *hist=history+r*history_stride;
    for(int64_t i=threadIdx.x;i<(end-start)*total_heads;i+=blockDim.x){
        int64_t pos=start+i/total_heads,index=i%total_heads;
        int64_t column=pos-start;if(column>=max_length)column=max_length-1;
        int64_t token=context(tokens,hist,start,length,column,ngram,eos);
        uint64_t mixed=(uint64_t)token*(uint64_t)multipliers[0];
        int reset=0;
        for(int64_t shift=1;shift<=index/heads+1;shift++){
            int64_t previous=context(tokens,hist,start,length,column-shift,ngram,eos);
            if(previous==eos)reset=1;
            if(reset)previous=eos;
            mixed^=(uint64_t)previous*(uint64_t)multipliers[shift];
        }
        int64_t rem=(int64_t)mixed%sizes[index];
        if(rem<0)rem+=sizes[index];
        out[pos*total_heads+index]=rem+offsets[index];
    }
}
static int launch_hash(const int32_t *tokens,int64_t count,
                               const int32_t *query,int64_t requests,
                               const int32_t *history,int64_t history_stride,
                               int64_t ngram,int64_t heads,int64_t eos,
                               const int64_t *multipliers,const int64_t *sizes,
                               const int64_t *offsets,int64_t *out,
                               int32_t *error,void *stream,uint64_t *timing) {
    if(!tokens||!query||!history||!multipliers||!sizes||!offsets||!out||!error||
       count<1||requests<1||requests>count||requests>65535||
       ngram<2||ngram>8||heads<1||heads>128||history_stride<ngram-1)return -1;
    hash_ids<<<(unsigned)requests,256,0,(cudaStream_t)stream>>>(
        tokens,count,query,requests,history,history_stride,ngram,heads,eos,
        multipliers,sizes,offsets,out,error,timing);
    return (int)cudaGetLastError();
}

extern "C" int gb10_ple_hash_gpu(const int32_t *tokens,int64_t count,
                               const int32_t *query,int64_t requests,
                               const int32_t *history,int64_t history_stride,
                               int64_t ngram,int64_t heads,int64_t eos,
                               const int64_t *multipliers,const int64_t *sizes,
                               const int64_t *offsets,int64_t *out,
                               int32_t *error,void *stream) {
    return launch_hash(tokens,count,query,requests,history,history_stride,ngram,heads,eos,multipliers,sizes,offsets,out,error,stream,nullptr);
}

extern "C" int gb10_ple_hash_gpu_timed(const int32_t *tokens,int64_t count,
                               const int32_t *query,int64_t requests,
                               const int32_t *history,int64_t history_stride,
                               int64_t ngram,int64_t heads,int64_t eos,
                               const int64_t *multipliers,const int64_t *sizes,
                               const int64_t *offsets,int64_t *out,
                               int32_t *error,void *stream,uint64_t *timing) {
    if(!timing)return -1;
    return launch_hash(tokens,count,query,requests,history,history_stride,ngram,heads,eos,multipliers,sizes,offsets,out,error,stream,timing);
}
