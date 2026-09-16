"""Start/stop the authorised bounded serving profiler without logging credentials."""
import argparse,json,pathlib,time,urllib.request
p=argparse.ArgumentParser();p.add_argument('action',choices=['start','stop']);p.add_argument('output',type=pathlib.Path);args=p.parse_args()
r=pathlib.Path(__file__).resolve().parents[2];cfg=json.loads((r/'deploy_config.json').read_text());key=pathlib.Path(cfg['api_key_file']).read_text().strip()
req=urllib.request.Request(f"http://127.0.0.1:{cfg['port']}/{args.action}_profile",data=b'',headers={'Authorization':'Bearer '+key},method='POST')
with urllib.request.urlopen(req,timeout=60) as response:result={'time':time.time(),'action':args.action,'status':response.status,'body':response.read().decode()}
args.output.write_text(json.dumps(result,indent=2)+'\n');print(result)
