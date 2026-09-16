"""GPU wait timing must reflect delayed publication and preserve timeout status."""
import time
import torch
from vllm.v1.ple_offload.gb10_mapped import map_shared_tensor, library, publish_done, publish_expected
flag=torch.zeros(32,dtype=torch.int64).share_memory_()
dev=map_shared_tensor(flag,torch.device('cuda:0'))
stream=torch.cuda.current_stream()
lib=library()
# Warm launch/setup before the measured delayed publication.
publish_expected(flag,1);publish_done(flag,1)
assert lib.gb10_wait(dev.data_ptr(),1000000000,0,stream.cuda_stream)==0
stream.synchronize()
assert flag[16].item()==0 and flag[24].item()==1
assert flag[18].item()==0
for seq in range(2,7):
 publish_expected(flag,seq)
 assert lib.gb10_wait(dev.data_ptr(),1000000000,0,stream.cuda_stream)==0
 time.sleep(.005)
 publish_done(flag,seq)
 stream.synchronize()
 elapsed=flag[17].item()
 assert flag[16].item()==0 and flag[24].item()==seq
 assert flag[18].item()>0 and 1000000 < elapsed < 1000000000,(elapsed,flag.tolist())
publish_expected(flag,7)
assert lib.gb10_wait(dev.data_ptr(),2000000,0,stream.cuda_stream)==0
stream.synchronize()
assert flag[16].item()==-1 and flag[24].item()==7
assert flag[17].item()>=2000000 and flag[18].item()>0
print('PASS ready, delayed publication timing and nontrapping timeout telemetry',flush=True)
