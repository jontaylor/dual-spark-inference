"""No service mutation: refuse until serving workers are gone, then benchmark."""
import json,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;name='gb10-fine-gemm-lab';image='vllm-gb10:v029-roce-qsa55122'
def run(args,**kwargs):return subprocess.run(args,check=True,**kwargs)
# A disconnected API is insufficient: confirm no GPU compute processes exist.
for command in (['docker','inspect','qwen38-kv-paging-r0','--format','{{.State.Running}}'],):
 r=subprocess.run(command,capture_output=True,text=True)
 if r.returncode==0 and r.stdout.strip()=='true':raise RuntimeError('Serving container still running')
procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
if procs:raise RuntimeError('GPU compute processes remain: '+procs)
if subprocess.run(['docker','inspect',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:raise RuntimeError('Existing lab container; inspect before proceeding')
record={'started':time.time(),'image':subprocess.check_output(['docker','image','inspect',image,'--format','{{.Id}}'],text=True).strip()}
try:
 with (p/'gemm-fine-tuning.log').open('w') as log:
  result=subprocess.run(['docker','run','--name',name,'--gpus','all','--network','host','--ipc','host','-v',str(p)+':/experiment:ro','--entrypoint','timeout',image,'240','python3','-u','/experiment/gemm_fine_offline.py'],stdout=log,stderr=subprocess.STDOUT,timeout=270)
 record['exit_code']=result.returncode
 copied=subprocess.run(['docker','cp',name+':/tmp/gb10-fine-gemm-tuning.json',str(p/'gemm-fine-tuning.json')],capture_output=True,text=True);record['evidence_copied']=copied.returncode==0
 if result.returncode==0 and copied.returncode==0:
  run(['python3',str(p/'select_fine_gemm.py')]);record['candidate_prepared']=True
 else:record['candidate_prepared']=False
finally:
 subprocess.run(['docker','rm','-f',name],check=False)
 record['finished']=time.time();(p/'gemm-fine-run.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
if not record.get('candidate_prepared'):raise SystemExit(1)
