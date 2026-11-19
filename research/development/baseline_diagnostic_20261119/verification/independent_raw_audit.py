"""Independent stdlib reduction/selection/provenance audit; no model imports or fitting."""
import json, math, hashlib, pathlib, datetime, sys
ROOT=pathlib.Path(sys.argv[1]); OUT=pathlib.Path(sys.argv[2]); assertions=0

def check(v,msg):
 global assertions
 assertions+=1
 if not v: raise AssertionError(msg)
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def flat(x): return [z for row in x for z in row]
def metric(rows,a):
 owned={}
 for r in rows:
  b,c,y=map(flat,(r['base'],r['correction'],r['target']))
  e=math.fsum((x+a*z-t)**2 for x,z,t in zip(b,c,y,strict=True))/len(y)
  owned.setdefault(r['owner'],[]).append(e)
 return math.fsum(math.fsum(v)/len(v) for v in owned.values())/len(owned)
def close(a,b,msg): check(math.isclose(a,b,rel_tol=2e-13,abs_tol=2e-15),msg)
manifest=read(ROOT/'artifact_manifest.json'); summary=read(ROOT/'summary.json'); protocol=read(ROOT/'protocol.json')
for p,h in manifest['files'].items(): check(sha(ROOT/p)==h,'hash '+p)
check(protocol['seeds']==[101,202,303],'seeds'); check(sha(ROOT/'protocol.json')==summary['protocol_sha256'],'protocol hash')
BACKBONES=['persistence','linear','mlp']; CONV=['none','last_residual','fifo_mean','uniform_topk','local_ridge']; ALPHA=[0,.25,.5,1]
counts={'validation_cells':0,'test_cells':0,'forecast_rows':0,'admitted_donors':0};results=[]
for seed in summary['seeds']:
 root=ROOT/f'seed-{seed}'; selection=read(root/'selection.json');fixture=read(root/'raw_fixture.json'); recomputed={}
 check(len(set(fixture['train'])&set(fixture['validation']))==0,'owners train/validation')
 check(len(set(fixture['train'])&set(fixture['test']))==0,'owners train/test')
 for split in ['validation','test']:
  paths=sorted((root/split).glob('*.json'));check(len(paths)==27,'cell count')
  for path in paths:
   cell=read(path); rows=cell['records']; name=cell['cell']; counts[split+'_cells']+=1
   a=selection['validation_cells'][name]['selected_alpha']; close(a,cell['selected_alpha'],'alpha binding')
   val=metric(rows,a);close(val,cell['metric']['mse'],'metric '+str(path));counts['forecast_rows']+=len(rows)
   if split=='validation':
    best=min(ALPHA,key=lambda x:(metric(rows,x),x));close(a,best,'validation argmin');recomputed[name]=val
   else: check(cell['selection_sha256']==sha(root/'selection.json'),'selection binding')
   for serial,r in enumerate(rows):
    check(r['split']==split,'split identity'); check(r['target']==fixture[split][r['owner']][r['event']+1],'raw target identity')
    if split=='test':
     for pred,b,c in zip(flat(r['prediction']),flat(r['base']),flat(r['correction']),strict=True): close(pred,b+a*c,'prediction arithmetic')
    t=r['telemetry'];ss=t['donor_serials'];ww=t['donor_weights']
    for j,s in enumerate(ss):
     if ww and ww[j]==0: continue
     check(type(s) is int and 0<=s<serial,'prior donor serial')
     d=rows[s];check(d['owner']==r['owner'],'same-owner donor')
     check(d['event']==t['donor_events'][j],'donor event identity')
     check(d['event']+1==t['donor_available'][j]<=r['event'],'arrival chronology')
     counts['admitted_donors']+=1
 backbone=min(BACKBONES,key=lambda b:(recomputed[b+'--none'],BACKBONES.index(b)))
 check(selection['selected_backbone']==backbone,'backbone argmin')
 def key(name):
  r=selection['validation_cells'][name]
  return (recomputed[name],CONV.index(r['method']),r['penalty'])
 choices=[n for n,r in selection['validation_cells'].items() if r['backbone']==backbone and r['method'] in CONV]
 check(selection['comparator']==min(choices,key=key),'matched comparator argmin')
 choices=[n for n,r in selection['validation_cells'].items() if r['method'] in CONV]
 global_best=min(choices,key=lambda n:(recomputed[n],BACKBONES.index(selection['validation_cells'][n]['backbone']),*key(n)[1:]))
 check(selection['global_comparator']==global_best,'global comparator argmin')
 s=read(root/'summary.json');m=read(root/'test'/f"{selection['memory_cell']}.json");c=read(root/'test'/f"{selection['comparator']}.json");g=read(root/'test'/f'{global_best}.json')
 mm=metric(m['records'],m['selected_alpha']);cm=metric(c['records'],c['selected_alpha']);gm=metric(g['records'],g['selected_alpha'])
 close(mm,s['memory_mse'],'summary memory');close(cm,s['comparator_mse'],'summary comparator');close(mm-cm,s['paired_difference'],'difference')
 results.append({'seed':seed,'backbone':backbone,'comparator':selection['comparator'],'memory_mse':mm,'comparator_mse':cm,'difference':mm-cm,'global_comparator':global_best,'global_comparator_mse':gm})
close(math.fsum(r['difference'] for r in results)/3,summary['mean_paired_difference'],'mean')
receipt={'status':'PASS_INDEPENDENT_SAVED_ARTIFACT_AUDIT','reviewer':'root coordinator independent of diagnostic author','verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'assertions':assertions,'counts':counts,'files_hashed':len(manifest['files']),'protocol_sha256':sha(ROOT/'protocol.json'),'source_sha256':summary['source_sha256'],'results':results,'mean_paired_difference':math.fsum(r['difference'] for r in results)/3,'independent_retraining':False,'scope':'Independent stdlib scalar MSE reduction, validation argmin reconstruction, prediction arithmetic, raw target identity, all admitted donor chronology, and file hashes. No outcome-driven changes or reruns.','limitations':['Three synthetic development repetitions, seed101 previously exposed.','The checks do not establish external validity, novel mechanism, or confirmatory significance.']}
OUT.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k not in ['results','limitations','scope']},indent=2))
