#include <cupti.h>
#include <cupti_activity.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <cstdint>
#include <atomic>
#include <thread>
#include <chrono>
#include <string>
static FILE* out = nullptr;
static std::mutex lock;
static size_t kernels=0, dropped=0;
static CUpti_SubscriberHandle subscriber;
// 0 unused, 1 collecting, 4 collection disabled, -1 failed. CUPTI remains resident.
static std::atomic<int> state{0};
static void CUPTIAPI callback(void*, CUpti_CallbackDomain, CUpti_CallbackId, const void*) {}
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
    } else if (record->kind==CUPTI_ACTIVITY_KIND_RUNTIME || record->kind==CUPTI_ACTIVITY_KIND_DRIVER) {
      auto* api=(CUpti_ActivityAPI*)record;
      const char* name=nullptr;
      cuptiGetCallbackName(record->kind==CUPTI_ACTIVITY_KIND_RUNTIME?CUPTI_CB_DOMAIN_RUNTIME_API:CUPTI_CB_DOMAIN_DRIVER_API,api->cbid,&name);
      fprintf(out,"{\"kind\":\"api\",\"start\":%llu,\"end\":%llu,\"thread\":%u,\"correlation\":%u,\"name\":",(unsigned long long)api->start,(unsigned long long)api->end,api->threadId,api->correlationId);
      quote(name); fputs("}\n",out);
    } else if (record->kind==CUPTI_ACTIVITY_KIND_MEMCPY) {
      auto* c=(CUpti_ActivityMemcpy6*)record;
      fprintf(out,"{\"kind\":\"copy\",\"start\":%llu,\"end\":%llu,\"stream\":%u,\"bytes\":%llu,\"direction\":%u}\n",(unsigned long long)c->start,(unsigned long long)c->end,c->streamId,(unsigned long long)c->bytes,(unsigned)c->copyKind);
    }
  }
  size_t lost=0;
  if(cuptiActivityGetNumDroppedRecords(ctx,stream,&lost)==CUPTI_SUCCESS) dropped+=lost;
  free(buffer);
}
static int trace_begin(const char* path) {
  if(out) return -1;
  out=fopen(path,"w"); if(!out) return -2;
  setvbuf(out,nullptr,_IOFBF,1024*1024);
  CUptiResult result=cuptiSubscribe(&subscriber,callback,nullptr);
  if(result==CUPTI_SUCCESS) result=cuptiActivityRegisterCallbacks(requested,completed);
  if(result==CUPTI_SUCCESS) result=cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
  if(result==CUPTI_SUCCESS) result=cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER);
  if(result==CUPTI_SUCCESS) result=cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
  if(result==CUPTI_SUCCESS) result=cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  uint64_t stamp=0;cuptiGetTimestamp(&stamp);
  std::lock_guard<std::mutex> guard(lock);
  fprintf(out,"{\"event\":\"begin\",\"cupti_ns\":%llu,\"result\":%d}\n",(unsigned long long)stamp,(int)result);
  fflush(out);return (int)result;
}

extern "C" int trace_status() { return state.load(); }
extern "C" int trace_start_timed(const char* path, unsigned milliseconds) {
  if (!path || milliseconds<100 || milliseconds>30000) return -2;
  int expected=0;
  if (!state.compare_exchange_strong(expected,1)) return -3;
  std::string filename(path);
  std::thread([filename,milliseconds]() {
    // Allows debugger's dlopen/call to return and detach before CUPTI setup.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    if (trace_begin(filename.c_str()) != CUPTI_SUCCESS) { state.store(-1); return; }
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);
    auto result=cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    {
      std::lock_guard<std::mutex> guard(lock);
      uint64_t now=0;cuptiGetTimestamp(&now);
      fprintf(out,"{\"event\":\"stop_requested\",\"cupti_ns\":%llu,\"flush\":%d}\n",(unsigned long long)now,(int)result);
      fflush(out);
    }
    // cuptiFinalize at an API exit caused a post-detach launch failure in the
    // disposable concurrent-graph test. Do not tear down resources used by
    // in-flight work. Stop activity collection and remove our subscription.
    auto unsubscribed=cuptiUnsubscribe(subscriber);
    {
      std::lock_guard<std::mutex> guard(lock);
      fprintf(out,"{\"event\":\"collection_disabled\",\"result\":%d,\"kernels\":%zu,\"dropped\":%zu}\n",(int)unsubscribed,kernels,dropped);
      fflush(out);
    }
    state.store(unsubscribed==CUPTI_SUCCESS?4:-1);
  }).detach();
  return 0;
}
