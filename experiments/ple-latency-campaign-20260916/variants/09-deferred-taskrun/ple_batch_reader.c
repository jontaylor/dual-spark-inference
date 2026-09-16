// SPDX-License-Identifier: AGPL-3.0-or-later
// In-process, submit-before-wait exact-byte PLE reader. No mmap dereferences.
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/io_uring.h>
#include <linux/time_types.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define DEPTH 4096U
struct ring {
    int fd;
    void *sq, *cq;
    size_t sq_size, cq_size, entries_size;
    struct io_uring_sqe *entries;
    unsigned *head, *tail, *mask, *array;
    unsigned *cq_head, *cq_tail, *cq_mask;
    struct io_uring_cqe *cqes;
    unsigned pending;
};
struct item { int64_t id; uint32_t generation, index; };
struct reader {
    int fd, poisoned;
    int64_t rows, width, capacity;
    unsigned nrings;
    struct ring *rings;
    struct item *items;
    int64_t *unique_ids, *destinations, *miss_ids;
    uint8_t *scratch, *cache;
    int64_t *tags;
    size_t slots;
    uint64_t stats[5]; // calls, misses, submitted, first-wait-submitted, hits
    size_t item_slots;
    uint32_t generation;
    uint8_t **sources;
    int inflight, submit_error;
    int64_t active_count, active_misses;
    uint64_t active_submitted, deadline;
    uint8_t *active_out;
};
static unsigned acquire(unsigned *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
static void release(unsigned *p, unsigned v) { __atomic_store_n(p,v,__ATOMIC_RELEASE); }
static uint64_t now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (uint64_t)t.tv_sec*1000+t.tv_nsec/1000000; }
static void ring_close(struct ring *r) {
    if (r->entries && r->entries != MAP_FAILED) munmap(r->entries,r->entries_size);
    if (r->cq && r->cq != MAP_FAILED && r->cq != r->sq) munmap(r->cq,r->cq_size);
    if (r->sq && r->sq != MAP_FAILED) munmap(r->sq,r->sq_size);
    if (r->fd>=0) close(r->fd);
}
static int ring_open(struct ring *r) {
    struct io_uring_params p={.flags=IORING_SETUP_SINGLE_ISSUER|IORING_SETUP_DEFER_TASKRUN};
    r->fd=syscall(__NR_io_uring_setup,DEPTH,&p);
    if(r->fd<0)return -errno;
    r->sq_size=p.sq_off.array+p.sq_entries*sizeof(unsigned);
    r->cq_size=p.cq_off.cqes+p.cq_entries*sizeof(struct io_uring_cqe);
    if(p.features&IORING_FEAT_SINGLE_MMAP){if(r->cq_size>r->sq_size)r->sq_size=r->cq_size;r->cq_size=r->sq_size;}
    r->sq=mmap(NULL,r->sq_size,PROT_READ|PROT_WRITE,MAP_SHARED,r->fd,IORING_OFF_SQ_RING);
    if(r->sq==MAP_FAILED)return -errno;
    r->cq=(p.features&IORING_FEAT_SINGLE_MMAP)?r->sq:mmap(NULL,r->cq_size,PROT_READ|PROT_WRITE,MAP_SHARED,r->fd,IORING_OFF_CQ_RING);
    if(r->cq==MAP_FAILED)return -errno;
    r->entries_size=p.sq_entries*sizeof(struct io_uring_sqe);
    r->entries=mmap(NULL,r->entries_size,PROT_READ|PROT_WRITE,MAP_SHARED,r->fd,IORING_OFF_SQES);
    if(r->entries==MAP_FAILED)return -errno;
    r->head=(unsigned*)((char*)r->sq+p.sq_off.head);r->tail=(unsigned*)((char*)r->sq+p.sq_off.tail);
    r->mask=(unsigned*)((char*)r->sq+p.sq_off.ring_mask);r->array=(unsigned*)((char*)r->sq+p.sq_off.array);
    r->cq_head=(unsigned*)((char*)r->cq+p.cq_off.head);r->cq_tail=(unsigned*)((char*)r->cq+p.cq_off.tail);
    r->cq_mask=(unsigned*)((char*)r->cq+p.cq_off.ring_mask);r->cqes=(void*)((char*)r->cq+p.cq_off.cqes);
    return 0;
}
static size_t slot(struct reader *r,int64_t id){uint64_t x=(uint64_t)id;x^=x>>16;return (x*11400714819323198485ULL)&(r->slots-1);}
static int reap(struct reader *r,struct ring *q) {
    int error=0;unsigned h=acquire(q->cq_head),t=acquire(q->cq_tail);
    while(h!=t){struct io_uring_cqe *c=&q->cqes[h&*q->cq_mask];if(c->res!=r->width)error=c->res<0?c->res:-EIO;if(q->pending)q->pending--;h++;}
    release(q->cq_head,h);return error;
}
// Destruction refuses to free memory still owned by pending kernel reads.
int gb10_ple_reader_destroy(void *handle) {
    struct reader *r=handle;if(!r)return 0;
    for(unsigned i=0;i<r->nrings;i++)if(r->rings[i].pending){reap(r,&r->rings[i]);if(r->rings[i].pending)return -EBUSY;}
    for(unsigned i=0;i<r->nrings;i++)ring_close(&r->rings[i]);
    if(r->fd>=0)close(r->fd);
    free(r->sources);free(r->rings);free(r->items);free(r->unique_ids);free(r->destinations);free(r->miss_ids);free(r->scratch);free(r->cache);free(r->tags);free(r);return 0;
}
void *gb10_ple_reader_create(const char *path,int64_t rows,int64_t width,int64_t capacity,uint64_t cache_bytes,int *error) {
    *error=-EINVAL;
    if(!path||rows<1||width<1||width>65536||capacity<1||capacity>1048576||rows>INT64_MAX/width||capacity>INT64_MAX/width)return NULL;
    struct reader *r=calloc(1,sizeof(*r));if(!r){*error=-ENOMEM;return NULL;}r->fd=-1;
    r->rows=rows;r->width=width;r->capacity=capacity;
    r->fd=open(path,O_RDONLY|O_CLOEXEC);if(r->fd<0){*error=-errno;goto fail;}
    struct stat st;if(fstat(r->fd,&st)||!S_ISREG(st.st_mode)||st.st_size!=rows*width){*error=-EINVAL;goto fail;}
    r->nrings=(capacity+DEPTH-1)/DEPTH;r->rings=calloc(r->nrings,sizeof(*r->rings));
    if(!r->rings){r->nrings=0;*error=-ENOMEM;goto fail;}
    for(unsigned i=0;i<r->nrings;i++)r->rings[i].fd=-1;
    for(unsigned i=0;i<r->nrings;i++){*error=ring_open(&r->rings[i]);if(*error)goto fail;}
    r->item_slots=1;while(r->item_slots<(size_t)capacity*2)r->item_slots*=2;
    r->items=calloc(r->item_slots,sizeof(*r->items));r->sources=malloc(capacity*sizeof(*r->sources));r->unique_ids=malloc(capacity*sizeof(int64_t));r->destinations=malloc(capacity*sizeof(int64_t));r->miss_ids=malloc(capacity*sizeof(int64_t));r->scratch=malloc(capacity*width);
    if(!r->sources||!r->items||!r->unique_ids||!r->destinations||!r->miss_ids||!r->scratch){*error=-ENOMEM;goto fail;}
    size_t maxslots=cache_bytes/(width+sizeof(int64_t));
    if(maxslots){r->slots=1;while(r->slots<=maxslots/2)r->slots*=2;r->cache=malloc(r->slots*width);r->tags=malloc(r->slots*sizeof(int64_t));if(!r->cache||!r->tags){*error=-ENOMEM;goto fail;}memset(r->tags,0xff,r->slots*sizeof(int64_t));}
    *error=0;return r;
fail:gb10_ple_reader_destroy(r);return NULL;
}
// Caller serializes this handle. All unique cache misses are submitted before
// polling/reaping the first completion. Kernel I/O workers may run concurrently.
static int submit_batch(void *handle,const int64_t *ids,int64_t count,uint8_t *out,unsigned timeout_ms,unsigned sqe_flags) {
    struct reader *r=handle;if(!r||!ids||!out||count<0||count>r->capacity||!timeout_ms)return -EINVAL;
    if(r->poisoned)return -EIO;
    if(r->inflight)return -EBUSY;
    for(int64_t i=0;i<count;i++)if(ids[i]<0||ids[i]>=r->rows)return -ERANGE;
    if(++r->generation==0){memset(r->items,0,r->item_slots*sizeof(*r->items));r->generation=1;}
    int64_t n=0,misses=0,hits=0;
    for(int64_t i=0;i<count;i++){
        int64_t id=ids[i];uint64_t hash=(uint64_t)id;hash^=hash>>16;hash*=11400714819323198485ULL;
        size_t k=hash&(r->item_slots-1);
        while(r->items[k].generation==r->generation&&r->items[k].id!=id)k=(k+1)&(r->item_slots-1);
        if(r->items[k].generation!=r->generation){
            r->items[k]=(struct item){id,r->generation,(uint32_t)n};
            r->unique_ids[n]=id;
            size_t cache_slot=r->slots?hash&(r->slots-1):0;
            if(r->slots&&r->tags[cache_slot]==id){
                // Cache entries remain immutable until every output is copied.
                r->sources[n]=r->cache+cache_slot*r->width;hits++;
            }else{
                r->sources[n]=r->scratch+n*r->width;r->miss_ids[misses++]=n;
            }
            n++;
        }
        r->destinations[i]=r->items[k].index;
    }
    int failure=0;uint64_t submitted=0;
    // The normal io_uring issue path first tries a nonblocking read inline.
    // Do not force IOSQE_ASYNC: that schedules a kernel worker even for hot
    // page-cache reads. Reads that would block are deferred by io_uring.
    // Fill each bounded ring, then submit it. Never reap until all rings submit.
    for(int64_t base=0;base<misses;base+=DEPTH){
        struct ring *q=&r->rings[base/DEPTH];unsigned total=misses-base<DEPTH?misses-base:DEPTH;
        unsigned tail=acquire(q->tail);
        for(unsigned j=0;j<total;j++){
            unsigned ix=(tail+j)&*q->mask;int64_t dest=r->miss_ids[base+j];
            struct io_uring_sqe *e=&q->entries[ix];memset(e,0,sizeof(*e));e->opcode=IORING_OP_READ;e->flags=sqe_flags;e->fd=r->fd;e->off=r->unique_ids[dest]*r->width;e->addr=(uint64_t)(r->scratch+dest*r->width);e->len=r->width;e->user_data=dest;q->array[ix]=ix;
        }
        release(q->tail,tail+total);
        unsigned remaining=total;
        while(remaining){int rc=syscall(__NR_io_uring_enter,q->fd,remaining,0,0,NULL,0);if(rc<0&&errno==EINTR)continue;if(rc<=0){failure=rc<0?-errno:-EIO;goto drain;}q->pending+=rc;submitted+=rc;remaining-=rc;}
    }
drain:
    r->stats[0]++;r->stats[1]+=misses;r->stats[2]=submitted;r->stats[3]=submitted;r->stats[4]+=hits;
    r->inflight=1;r->submit_error=failure;r->active_count=count;
    r->active_misses=misses;r->active_submitted=submitted;r->active_out=out;
    r->deadline=now_ms()+timeout_ms;
    if(failure)r->poisoned=1;
    return failure;
}
int gb10_ple_reader_submit(void *handle,const int64_t *ids,int64_t count,uint8_t *out,unsigned timeout_ms) {
    return submit_batch(handle,ids,count,out,timeout_ms,0);
}
int gb10_ple_reader_submit_async(void *handle,const int64_t *ids,int64_t count,uint8_t *out,unsigned timeout_ms) {
    return submit_batch(handle,ids,count,out,timeout_ms,IOSQE_ASYNC);
}
// Same caller completes the batch after it has launched independent GPU work.
int gb10_ple_reader_complete(void *handle) {
    struct reader *r=handle;
    if(!r||!r->inflight)return -EINVAL;
    int failure=r->submit_error;
    int64_t count=r->active_count,misses=r->active_misses;
    uint64_t submitted=r->active_submitted,deadline=r->deadline;
    uint8_t *out=r->active_out;
    for(unsigned i=0;i<r->nrings;i++){
        struct ring *q=&r->rings[i];
        while(q->pending){
            int err=reap(r,q);if(err)failure=err;if(!q->pending)break;
            uint64_t now=now_ms();
            if(now>=deadline){r->poisoned=1;return -ETIMEDOUT;}
            uint64_t remaining=deadline-now;
            struct __kernel_timespec ts={.tv_sec=remaining/1000,.tv_nsec=(remaining%1000)*1000000};
            struct io_uring_getevents_arg arg={.ts=(uint64_t)&ts};
            // All reads are already submitted. Sleep for the remaining batch,
            // not one poll wakeup per arriving completion.
            int rc=syscall(__NR_io_uring_enter,q->fd,0,q->pending,
                           IORING_ENTER_GETEVENTS|IORING_ENTER_EXT_ARG,&arg,sizeof(arg));
            if(rc<0&&errno!=EINTR){r->poisoned=1;return errno==ETIME?-ETIMEDOUT:-errno;}
        }
    }
    r->inflight=0;
    if(failure){r->poisoned=1;return failure;}
    if(submitted!=(uint64_t)misses){r->poisoned=1;return -EIO;}
    // Commit output only after every completion has succeeded.
    for(int64_t i=0;i<count;i++)memcpy(out+i*r->width,r->sources[r->destinations[i]],r->width);
    if(r->slots)for(int64_t j=0;j<misses;j++){int64_t i=r->miss_ids[j];size_t k=slot(r,r->unique_ids[i]);memcpy(r->cache+k*r->width,r->scratch+i*r->width,r->width);r->tags[k]=r->unique_ids[i];}
    return 0;
}
int gb10_ple_reader_gather(void *handle,const int64_t *ids,int64_t count,uint8_t *out,unsigned timeout_ms) {
    int rc=gb10_ple_reader_submit(handle,ids,count,out,timeout_ms);
    return rc?rc:gb10_ple_reader_complete(handle);
}
int gb10_ple_reader_stats(void *handle,uint64_t *out){if(!handle||!out)return -EINVAL;memcpy(out,((struct reader*)handle)->stats,5*sizeof(uint64_t));return 0;}
