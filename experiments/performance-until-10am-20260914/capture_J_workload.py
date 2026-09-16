"""Read the actual workload controller and J cycle; never change admission."""
import json
import subprocess
import time
from pathlib import Path

p = Path(__file__).resolve().parent
code = '''from pathlib import Path
import json,time
p=Path('/home/jon/artifact-agent-orchestration/.artifact-agent/baseline-comparison/20260914-temperature-until-1000')
controller=json.loads((p/'status.json').read_text())
f=Path('/proc')/str(controller['pid'])/'stat'
fields=f.read_text().rsplit(') ',1)[1].split() if f.exists() else None
cycle=p/'batch-004'
campaign=json.loads((cycle/'campaign.json').read_text())
telemetry=json.loads((cycle/'telemetry.json').read_text())
print(json.dumps({'time':time.time(),'controller':controller,
 'controller_state':fields[0] if fields else 'gone',
 'controller_starttime':fields[19] if fields else None,
 'campaign':campaign,'telemetry':telemetry}))
'''
result = subprocess.run(['ssh', 'jon@192.168.0.167', 'python3 -'],
                        input=code, text=True, capture_output=True,
                        check=True, timeout=20)
d = json.loads(result.stdout)
(p / 'J-workload-status.json').write_text(json.dumps(d, indent=2))
rows = [{k: r.get(k) for k in ['id', 'status', 'generated_tokens',
                              'max_prompt_tokens', 'verification']}
        for r in d['telemetry']['rows']]
quality = {'time': time.time(), 'campaign_status': d['campaign']['status'],
           'controller_state': d['controller_state'],
           'current_cycle': d['controller']['current'], 'rows': rows,
           'scope': 'Actual workload verification state; unfinished checks are pending, never passes. Campaign server_context is inherited descriptive text, not live serving evidence.'}
(p / 'J-workload-quality-latest.json').write_text(json.dumps(quality, indent=2))
print(json.dumps(quality, indent=2))
