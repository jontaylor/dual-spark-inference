"""CPU checks of actual parking ledger with finer token reservations."""
import json,math
from vllm.v1.core.sched.parking_policy import ParkingPolicy,Phase
cases=0
for page,unit in [(1920,7680),(1920,32768),(1600,6400),(1600,32768)]:
 cost=math.ceil(unit/page)
 # Every token footprint within supported context must be physically covered.
 for tokens in range(262145):
  units=max(1,math.ceil(tokens/unit));assert units*cost>=math.ceil(tokens/page);cases+=1
 p=ParkingPolicy(1564,ranks=2,unit=unit,unit_cost=cost,request_cost=34)
 for i in range(8):p.enqueue(str(i),1000);assert p.admit_head()==str(i)
 assert sum(t.phase==Phase.RUNNING for t in p.tickets.values())==8
 for n in [unit-1,unit,unit+1,unit*2+6]:
  for i in range(8):assert p.grow(str(i),n)
 before=p.free;gen=p.begin_save('0',unit*2,unit*2,64,0)
 assert not p.acknowledge('0',0,gen,Phase.SAVING) and p.free==before
 assert not p.acknowledge('0',1,gen-1,Phase.SAVING) and p.free==before
 assert p.acknowledge('0',1,gen,Phase.SAVING) and p.free>before
 assert p.admit_head()=='0';assert p.tickets['0'].phase==Phase.RESTORING
 held=p.free
 assert not p.acknowledge('0',0,gen,Phase.RESTORING) and p.free==held
 assert p.acknowledge('0',1,gen,Phase.RESTORING)
 # Completion snapshot reservations must constrain admissions before capacity is exceeded.
 p.cache_reserved=p.free
 p.enqueue('pressure',1000);assert p.admit_head() is None
 assert p.free==0
 p.cache_reserved-=p.cost(p.units(1000));assert p.admit_head()=='pressure' and p.free==0
print(json.dumps({'passed':True,'coverage_cases':cases,'concurrent_admission':8,'rank_acknowledgement_and_stale_generation_checks':True,'snapshot_pressure_admission_checks':True,'scope':'Actual CPU ParkingPolicy with fixed-cost fixture. No complete scheduler/model integration or observed write saving.'},indent=2))
