from pathlib import Path
import shutil
r=Path('/home/jon/dual-spark-inference-kv-paging'); e=r/'experiments/batch-invariant-20260913'
shutil.copy2(r/'deploy_config.json',e/'config.candidate.json')
shutil.copy2(r/'launch_rank.py',e/'launch_rank.candidate.py')
shutil.copy2(e/'config.clean.json',r/'deploy_config.json')
shutil.copy2(e/'launch_rank.before.py',r/'launch_rank.py')
