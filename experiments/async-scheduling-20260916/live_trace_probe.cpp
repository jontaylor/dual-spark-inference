// Disposable-process prototype: late CUPTI activity attachment after graph creation.
// Stop requires the caller to have synchronized CUDA; not yet safe for live injection.
#include <cupti.h>
#include <cupti_activity.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <cstdint>
static FILE* out = nullptr;
static std::mutex lock;
static size_t kernels=0, dropped=0;
static CUpti_SubscriberHandle subscriber;
static unsigned callbacks=0;
static void CUPTIAPI callback(void*, CUpti_CallbackDomain, CUpti_CallbackId, const void*) { callbacks++; }
static void quote(const char* name) {
  fputc('"',out);
  for (const unsigned char* p=(const unsigned char*)(name?name:""); *p; ++p) {
    if (*p=='"' || *p=='\\') fputc('\\',out);
    if (*p<32) fprintf(out,"\\u%04x",*p); else fputc(*p,out);
  }
  fputc('"',out);
}
static void CUPTIAPI requested(uint8_t** buffer, size_t* size, size_t* max_records) {
  *size=1024*1024; *buffer=(uint8_t*)malloc(*size); *max_records=0;
  if (!*buffer) *size=0;
}
static void CUPTIAPI completed(CUcontext ctx, uint32_t stream, uint8_t* buffer,
                               size_t size, size_t valid) {
  (void)size;
  std::lock_guard<std::mutex> guard(lock);
  CUpti_Activity* record=nullptr;
  while(cuptiActivityGetNextRecord(buffer,valid,&record)==CUPTI_SUCCESS) {
    if (record->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || record->kind==CUPTI_ACTIVITY_KIND_KERNEL) {
      auto* k=(CUpti_ActivityKernel10*)record;
      fprintf(out,"{\"kind\":\"kernel\",\"start\":%llu,\"end\":%llu,\"stream\":%u,\"correlation\":%u,\"name\":",
        (unsigned long long)k->start,(unsigned long long)k->end,k->streamId,k->correlationId);
      quote(k->name); fputs("}\n",out); kernels++;
    } else { fprintf(out,"{\"kind_id\":%d}\n",(int)record->kind); }
  }
  size_t lost=0;
  if(cuptiActivityGetNumDroppedRecords(ctx,stream,&lost)==CUPTI_SUCCESS) dropped+=lost;
  free(buffer);
}
extern "C" int trace_begin(const char* path) {
  if(out) return -1;
  out=fopen(path,"w"); if(!out) return -2;
  setvbuf(out,nullptr,_IOFBF,1024*1024);
  CUptiResult result=cuptiSubscribe(&subscriber,callback,nullptr);
  if(result==CUPTI_SUCCESS) result=cuptiEnableDomain(1,subscriber,CUPTI_CB_DOMAIN_DRIVER_API);
  if(result==CUPTI_SUCCESS) result=cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
  if(result==CUPTI_SUCCESS) result=cuptiActivityRegisterCallbacks(requested,completed);
  if(result==CUPTI_SUCCESS) result=cuptiActivityEnableAndDump(CUPTI_ACTIVITY_KIND_CONTEXT);
  if(result==CUPTI_SUCCESS) result=cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  uint64_t stamp=0;cuptiGetTimestamp(&stamp);
  fprintf(out,"{\"event\":\"begin\",\"cupti_ns\":%llu,\"result\":%d}\n",(unsigned long long)stamp,(int)result);
  fflush(out);return (int)result;
}
extern "C" int trace_stop_after_sync() {
  fprintf(out,"{\"event\":\"diagnostic\",\"callbacks\":%u,\"last_error\":%d}\n",callbacks,(int)cuptiGetLastError());
  auto flushed=cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
  auto disabled=cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  // This prototype caller explicitly synchronizes CUDA before stop.
  auto finalized=cuptiFinalize();
  std::lock_guard<std::mutex> guard(lock);
  fprintf(out,"{\"event\":\"end\",\"kernels\":%zu,\"dropped\":%zu,\"disable\":%d,\"flush\":%d,\"finalize\":%d}\n",
    kernels,dropped,(int)disabled,(int)flushed,(int)finalized);
  fclose(out);out=nullptr;return (int)finalized;
}
