import pathlib,sys
root=pathlib.Path(__file__).resolve().parents[2];src=root/'experiments/ple-latency-campaign-20260916/run_gpu_check.py';b=root/'experiments/ple-resident-20260916'
code=src.read_text();anchor='for dst,src in mounts.items():'
code=code.replace(anchor,"mounts['/opt/gb10/libple_batch_reader.so']=pathlib.Path("+repr(str(b/'libple_batch_reader.so'))+")\nmounts['/bench.py']=pathlib.Path("+repr(str(b/'gpu_benchmark.py'))+")\ncmd += ['-v', "+repr(str(root/'results/ple-resident-20260916')+':/results')+"]\n"+anchor)
code=code.replace("f'seccomp={old}/seccomp-uring.json'",repr('seccomp='+str(b/'seccomp.json')))
code=code.replace('result=subprocess.run(cmd,','cmd[-1] += " && /tmp/ple-check-venv/bin/python /bench.py"\nresult=subprocess.run(cmd,')
code=code.replace("log.write_text(result.stdout)","log=pathlib.Path("+repr(str(root/'results/ple-resident-20260916/gpu-integration.log'))+");log.write_text(result.stdout)")
sys.argv=[str(src),'--overlap','--telemetry','--gate-timing','--native-variant','17-fixed-policy','--connector-variant','17-fixed-policy','--submission-policy','sqpoll']
exec(compile(code,str(src),'exec'),{'__file__':str(src),'__name__':'__main__'})
