"""Bounded two-rank reader/phase traces with interval metrics, no serving edits."""
import argparse,concurrent.futures,json,pathlib,subprocess,time,urllib.request
ap=argparse.ArgumentParser();ap.add_argument('output',type=pathlib.Path);ap.add_argument('--seconds',type=int,default=30);ap.add_argument('--detail',action='store_true');ap.add_argument('--split',action='store_true');ap.add_argument('--gpu-wait',action='store_true');ap.add_argument('--async-ab','--policy-trace',dest='async_ab',action='store_true');ap.add_argument('--gate-timing',action='store_true');ap.add_argument('--async-policy',choices=['forced-worker','sqpoll','prefetch'],default='forced-worker');args=ap.parse_args()
out=args.output;out.mkdir(parents=True,exist_ok=False)
def metrics(name):
 try:
  with urllib.request.urlopen('http://127.0.0.1:30001/metrics',timeout=10) as r:(out/name).write_bytes(r.read())
 except Exception as e:(out/(name+'.err')).write_text(str(e))
def capture(rank):
 host=[] if rank==0 else ['ssh','192.168.100.11']
 lines=subprocess.check_output(host+['docker','top',f'qwen38-kv-paging-r{rank}','-eo','pid,comm'],text=True).splitlines()
 pid=int(next(l.split()[0] for l in lines if 'VLLM::Worker' in l));root=f'/proc/{pid}/root/opt/gb10'
 script=f'''uprobe:{root}/libple_batch_reader.so:gb10_ple_reader_gather /pid=={pid}/ {{ @start[tid]=nsecs; @rows[tid]=arg2; @handle[tid]=arg0; @miss0[tid]=*(uint64*)uptr(arg0+120); @hit0[tid]=*(uint64*)uptr(arg0+144); }}
uretprobe:{root}/libple_batch_reader.so:gb10_ple_reader_gather /pid=={pid} && @start[tid]/ {{ $h=@handle[tid]; printf("reader %llu %llu %llu %llu %llu %lld\\n",@start[tid],@rows[tid],nsecs-@start[tid],*(uint64*)uptr($h+120)-@miss0[tid],*(uint64*)uptr($h+144)-@hit0[tid],retval); delete(@start[tid]); delete(@rows[tid]); delete(@handle[tid]); delete(@miss0[tid]); delete(@hit0[tid]); }}
'''
 if args.split:
  script=script.replace('gb10_ple_reader_gather /pid==', 'gb10_ple_reader_submit /pid==', 1)
  script=script.replace('uretprobe:'+root+'/libple_batch_reader.so:gb10_ple_reader_gather', 'uretprobe:'+root+'/libple_batch_reader.so:gb10_ple_reader_complete')
  script+=f'''uretprobe:{root}/libple_batch_reader.so:gb10_ple_reader_submit /pid=={pid} && @start[tid]/ {{ printf("submit %llu %llu %lld\\n",@start[tid],nsecs-@start[tid],retval); }}
uprobe:{root}/libple_batch_reader.so:gb10_ple_reader_complete /pid=={pid}/ {{ @complete[tid]=nsecs; }}
uretprobe:{root}/libple_batch_reader.so:gb10_ple_reader_complete /pid=={pid} && @complete[tid]/ {{ printf("complete %llu %llu %lld\\n",@complete[tid],nsecs-@complete[tid],retval); delete(@complete[tid]); }}
'''
 if args.async_ab:
  script=script.replace(':gb10_ple_reader_submit ', ':gb10_ple_reader_submit* ')
  script=script.replace('@start[tid]=nsecs;', '@start[tid]=nsecs; printf("policy %llu %s %u %u\\n",@start[tid],probe,cpu,@sync[tid]);')
  script+=f'''uprobe:{root}/libple_batch_reader.so:gb10_ple_reader_gather /pid=={pid}/ {{ @sync[tid]=1; }}
uretprobe:{root}/libple_batch_reader.so:gb10_ple_reader_gather /pid=={pid}/ {{ delete(@sync[tid]); }}
'''
 if args.detail:
  script+=f'''uprobe:{root}/libmapped_wait.so:gb10_release /pid=={pid}/ {{ printf("release %llu %llu %llu\\n",nsecs,arg0,arg1); }}
uprobe:{root}/libple_gather.so:gb10_ple_hash /pid=={pid}/ {{ @hashstart[tid]=nsecs; @tokens[tid]=arg1; @reqs[tid]=arg3; }}
uretprobe:{root}/libple_gather.so:gb10_ple_hash /pid=={pid} && @hashstart[tid]/ {{ printf("hash %llu %llu %llu %llu %lld\\n",@hashstart[tid],nsecs,@tokens[tid],@reqs[tid],retval); delete(@hashstart[tid]); delete(@tokens[tid]); delete(@reqs[tid]); }}
'''
 if args.gpu_wait:
  script+=f'''uprobe:{root}/libmapped_wait.so:gb10_release /pid=={pid}/ {{ printf("gpu_wait %llu %llu %llu %llu %llu %llu\\n",nsecs,arg0,arg1,*(uint64*)uptr(arg0+128),*(uint64*)uptr(arg0+72),*(uint64*)uptr(arg0+80)); }}
'''
 if args.detail and subprocess.run(host+['sudo','-n','test','-f',root+'/libple_hash_gpu.so'],capture_output=True).returncode==0:
  script+=f'''uprobe:{root}/libple_hash_gpu.so:gb10_ple_hash_gpu* /pid=={pid}/ {{ @gstart[tid]=nsecs; @gtokens[tid]=arg1; @greqs[tid]=arg3; }}
uretprobe:{root}/libple_hash_gpu.so:gb10_ple_hash_gpu* /pid=={pid} && @gstart[tid]/ {{ printf("hash_gpu %llu %llu %llu %llu %lld\\n",@gstart[tid],nsecs,@gtokens[tid],@greqs[tid],retval); delete(@gstart[tid]); delete(@gtokens[tid]); delete(@greqs[tid]); }}
'''
 if args.gate_timing:
  script+=f'''uprobe:{root}/libmapped_wait.so:gb10_release /pid=={pid}/ {{ printf("gpu_gate %llu %llu %llu %llu %llu %llu\\n",nsecs,arg0,arg1,*(uint64*)uptr(arg0+128),*(uint64*)uptr(arg0+112),*(uint64*)uptr(arg0+96)); }}
'''
 script+=f'''uretprobe:{root}/libple_batch_reader.so:gb10_ple_reader_prefetch /pid=={pid} && @start[tid]/ {{ printf("prefetch %llu %lld\\n",@start[tid],retval); }}
'''
 script+=f'interval:s:{args.seconds} {{exit();}}\n'
 bt=out/f'rank{rank}.bt';bt.write_text(script)
 remote=f'/tmp/ple-campaign-{out.name}-r{rank}.bt'
 if rank:subprocess.run(['scp',str(bt),f'192.168.100.11:{remote}'],check=True,stdout=subprocess.DEVNULL)
 cmd=['ssh','192.168.100.11',f'sudo -n bpftrace -q {remote}'] if rank else ['sudo','-n','bpftrace','-q',str(bt)]
 with (out/f'rank{rank}.txt').open('w') as stdout,(out/f'rank{rank}.err').open('w') as stderr:subprocess.run(cmd,stdout=stdout,stderr=stderr,check=True,timeout=args.seconds+60)
 return {'rank':rank,'pid':pid,'mode':'detail' if args.detail else 'reader','split':args.split,'gpu_wait':args.gpu_wait,'async_ab':args.async_ab,'gate_timing':args.gate_timing}
metrics('metrics-before.txt');start=time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:metadata=list(pool.map(capture,(0,1)))
metrics('metrics-after.txt');(out/'capture.json').write_text(json.dumps({'start':start,'end':time.time(),'ranks':metadata,'async_policy':args.async_policy},indent=2)+'\n')
print(out,flush=True)
