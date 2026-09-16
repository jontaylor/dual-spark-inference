"""Create a bounded native-reader trace after startup; no serving edits."""
import json,pathlib,subprocess,sys
root=pathlib.Path(__file__).parent
rank=int(sys.argv[1]) if len(sys.argv)>1 else 0
host=[] if rank==0 else ['ssh','192.168.100.11']
lines=subprocess.check_output(host+['docker','top',f'qwen38-kv-paging-r{rank}','-eo','pid,comm'],text=True).splitlines()
pid=int(next(l.split()[0] for l in lines if 'VLLM::Worker' in l))
path=f'/proc/{pid}/root/opt/gb10/libple_batch_reader.so'
script=f'''uprobe:{path}:gb10_ple_reader_gather /pid=={pid}/ {{ @start[tid]=nsecs; @rows[tid]=arg2; }}
uretprobe:{path}:gb10_ple_reader_gather /pid=={pid} && @start[tid]/ {{ printf("%llu %llu %lld\\n",@rows[tid],nsecs-@start[tid],retval); delete(@start[tid]); delete(@rows[tid]); }}
interval:s:20 {{ exit(); }}
'''
(root/f'live-r{rank}.bt').write_text(script)
if rank:
 subprocess.run(['scp',str(root/f'live-r{rank}.bt'),f'192.168.100.11:/tmp/ple-live-r{rank}.bt'],check=True)
 cmd=['ssh','192.168.100.11',f'sudo -n bpftrace -q /tmp/ple-live-r{rank}.bt']
else:cmd=['sudo','-n','bpftrace','-q',str(root/f'live-r{rank}.bt')]
with (root/f'live-r{rank}.txt').open('w') as out,(root/f'live-r{rank}.err').open('w') as err:
 subprocess.run(cmd,stdout=out,stderr=err,check=True)
