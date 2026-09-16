#!/usr/bin/env python3
"""Fit smooth empirical capacity functions from raw 30-second observations.
Run-balanced log-throughput quantile regression, with leave-one-run-out CV.
Only positive-throughput observations estimate active-operation capacity.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import linprog


def design(context, spec):
    x = np.log1p(np.clip(np.asarray(context, dtype=float), spec['context_min'], spec['context_max'])/1000)
    z = (x-spec['log_min'])/(spec['log_max']-spec['log_min'])
    if spec['kind']=='polynomial':
        return np.array([z**j for j in range(spec['degree']+1)]).T
    return BSpline.design_matrix(z, spec['knots'], 3).toarray()


def fit(context, throughput, arms, spec, q):
    B = design(context, spec); y = np.log(throughput); n,k = B.shape
    # Each run has equal total weight, irrespective of duration.
    weights = np.array([1/np.sum(arms==a) for a in arms]); weights /= weights.sum()
    from scipy.sparse import csr_matrix, eye, hstack
    eq = hstack([csr_matrix(B), eye(n), -eye(n)], format='csr')
    obj = np.r_[np.zeros(k), q*weights, (1-q)*weights]
    result = linprog(obj, A_eq=eq, b_eq=y, bounds=[(None,None)]*k+[(0,None)]*(2*n), method='highs')
    if not result.success:
        raise RuntimeError(result.message)
    return result.x[:k]


def predict(context, model):
    return np.exp(design(np.atleast_1d(context), model)@np.array(model['coefficients']))


class WorkFunctions:
    """Context in tokens; rates in cluster tokens/second; units are relative."""
    def __init__(self, path=None):
        path = Path(path) if path else Path(__file__).with_name('smooth-v2')/'functions.json'
        self.models = json.loads(path.read_text())['models']

    def _rate(self, kind, context):
        if not np.isfinite(context) or context < 0:
            raise ValueError('context must be finite and nonnegative')
        return float(predict(context, self.models[kind])[0])

    def decode_per_second(self, context):
        return self._rate('decode', context)

    def prefill_per_second(self, context):
        return self._rate('prefill', context)

    def decode_units(self, context, decode_per_second):
        return self._units(decode_per_second, self.decode_per_second(context))

    def prefill_units(self, context, prefill_per_second):
        return self._units(prefill_per_second, self.prefill_per_second(context))

    @staticmethod
    def _units(rate, reference):
        if not np.isfinite(rate) or rate < 0:
            raise ValueError('rate must be finite and nonnegative')
        return rate/reference


def write_csv(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parent
    p.add_argument('--samples',type=Path,default=root/'all-reference-v2/samples.csv')
    p.add_argument('--output',type=Path,default=root/'smooth-v2')
    p.add_argument('--quantile',type=float,default=.9)
    args=p.parse_args()
    if not 0<args.quantile<1: p.error('quantile must be between zero and one')
    args.output.mkdir(exist_ok=False,parents=True)
    rows=list(csv.DictReader(args.samples.open()))
    valid=[r for r in rows if r['valid']=='True']
    models={}; diagnostics={}; candidates_report=[]; sensitivity={}
    for kind in ['decode','prefill']:
        rr=[r for r in valid if float(r[kind+'_tok_s'])>0]
        c=np.array([float(r['context_estimate']) for r in rr]); y=np.array([float(r[kind+'_tok_s']) for r in rr]); arms=np.array([r['arm'] for r in rr])
        base=dict(context_min=float(c.min()),context_max=float(c.max()),log_min=float(np.log1p(c.min()/1000)),log_max=float(np.log1p(c.max()/1000)))
        specs=[]
        for degree in [1,2]: specs.append(dict(base,kind='polynomial',degree=degree,name=f'log-polynomial-{degree}'))
        for interior in [0,2,4]:
            knots=[0.]*4+list(np.linspace(0,1,interior+2)[1:-1])+[1.]*4
            specs.append(dict(base,kind='spline',knots=knots,name=f'cubic-spline-{interior}-knots'))
        evaluated=[]
        for spec in specs:
            losses=[]; coverage=[]
            for arm in sorted(set(arms)):
                train=arms!=arm; test=~train
                coef=fit(c[train],y[train],arms[train],spec,args.quantile)
                residual=np.log(y[test])-design(c[test],spec)@coef
                losses.append(float(np.mean(np.maximum(args.quantile*residual,(args.quantile-1)*residual))))
                coverage.append(float(np.mean(residual<=0)))
            item=dict(operation=kind,candidate=spec['name'],cv_loss=float(np.mean(losses)),cv_se=float(np.std(losses,ddof=1)/np.sqrt(len(losses))),heldout_coverage=float(np.mean(coverage)),fold_losses=losses)
            candidates_report.append(item); evaluated.append((spec,item))
        best=min(evaluated,key=lambda z:z[1]['cv_loss']);threshold=best[1]['cv_loss']+best[1]['cv_se']
        # Simpler candidate within one standard error of lowest validation loss.
        chosen=next(z for z in evaluated if z[1]['cv_loss']<=threshold)
        spec=chosen[0]; coef=fit(c,y,arms,spec,args.quantile)
        models[kind]=dict(spec,coefficients=coef.tolist(),quantile=args.quantile,positive_samples=len(rr),zero_samples_excluded=len(valid)-len(rr))
        diagnostics[kind]=dict(chosen=spec['name'],lowest_loss=best[0]['name'],selection_threshold=threshold,heldout_coverage=chosen[1]['heldout_coverage'],median_model=dict(spec,coefficients=fit(c,y,arms,spec,.5).tolist()))
        sensitivity[kind]={str(q):dict(spec,coefficients=fit(c,y,arms,spec,q).tolist()) for q in [.85,.95]}
    data=dict(models=models,diagnostics=diagnostics,sensitivity=sensitivity,candidates=candidates_report,source=str(args.samples.resolve()),source_sha256=hashlib.sha256(args.samples.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'functions.json').write_text(json.dumps(data,indent=2)+'\n')
    f=WorkFunctions(args.output/'functions.json')
    scored=[]
    for r in valid:
        c=float(r['context_estimate']);d=float(r['decode_tok_s']);pr=float(r['prefill_tok_s'])
        if -1e-6 < pr < 0: pr=0.  # Floating-point subtraction noise only.
        du=f.decode_units(c,d);pu=f.prefill_units(c,pr)
        scored.append(dict(arm=r['arm'],start_seconds=int(r['start_seconds']),end_seconds=int(r['end_seconds']),context=c,decode_tok_s=d,prefill_tok_s=pr,decode_reference=f.decode_per_second(c),prefill_reference=f.prefill_per_second(c),decode_units_s=du,prefill_units_s=pu,combined_units_s=du+pu))
    scores=[]
    for arm in sorted(set(r['arm'] for r in scored)):
        for interval,lo,hi in [('0-35m',0,2100),('30-35m',1800,2100)]:
            ss=[r for r in scored if r['arm']==arm and r['start_seconds']>=lo and r['end_seconds']<=hi]
            duration=sum(r['end_seconds']-r['start_seconds'] for r in ss)
            avg=lambda field:sum(r[field]*(r['end_seconds']-r['start_seconds']) for r in ss)/duration if duration else None
            score=dict(arm=arm,interval=interval,scored_seconds=duration,decode_units_s=avg('decode_units_s'),prefill_units_s=avg('prefill_units_s'),combined_units_s=avg('combined_units_s'))
            for q in ['0.85','0.95']:
                score['combined_q'+q]=sum((r['decode_tok_s']/predict(r['context'],sensitivity['decode'][q])[0]+r['prefill_tok_s']/predict(r['context'],sensitivity['prefill'][q])[0])*(r['end_seconds']-r['start_seconds']) for r in ss)/duration if duration else None
            scores.append(score)
    write_csv(args.output/'scored-windows.csv',scored);write_csv(args.output/'scores.csv',scores)
    grid=np.linspace(min(m['context_min'] for m in models.values()),max(m['context_max'] for m in models.values()),400)
    write_csv(args.output/'reference-curves.csv',[dict(context=c,decode_per_second=f.decode_per_second(c),prefill_per_second=f.prefill_per_second(c)) for c in grid])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,10),sharex=True)
    for ax,kind in zip(axes,['decode','prefill']):
        for arm in sorted(set(r['arm'] for r in valid)):
            rr=[r for r in valid if r['arm']==arm]
            ax.scatter([float(r['context_estimate'])/1000 for r in rr],[float(r[kind+'_tok_s']) for r in rr],s=7,alpha=.25,label=arm[:2])
        ax.fill_between(grid/1000,predict(grid,sensitivity[kind]['0.85']),predict(grid,sensitivity[kind]['0.95']),color='black',alpha=.1,label='85–95% quantile sensitivity (not confidence band)')
        ax.plot(grid/1000,predict(grid,models[kind]),color='black',lw=2,label=f'Selected {args.quantile:.0%} reference')
        ax.plot(grid/1000,predict(grid,diagnostics[kind]['median_model']),color='black',ls='--',lw=1,label='Median diagnostic')
        ax.set_ylabel(kind.capitalize()+' tokens/s');ax.grid(alpha=.2);ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8,ncol=4);axes[1].set_xlabel('Estimated mean HTTP in-flight context (thousands of tokens)')
    fig.suptitle('Smooth empirical capacity: raw observations, run-balanced quantile fits')
    fig.tight_layout();fig.savefig(args.output/'curves.png',dpi=160);fig.savefig(args.output/'curves.svg');plt.close(fig)
    report=['# Smooth empirical work scoring','',f'Fits use raw 30-second observations, not bucket maxima. Reference is the {args.quantile:.0%} conditional quantile of positive throughput. Runs have equal total fitting weight. Zero-rate windows are excluded from capacity fitting but earn zero credit for that operation during scoring. All archived reference windows are used, including after tasks finish.','',
        'Candidate selection: leave one complete run out at a time; evaluate log-throughput quantile loss, averaged equally across held-out runs. Select the simplest candidate within one standard error of the lowest loss. This assesses throughput fit, not prediction of task success. The quantile is an adjustable policy choice, not physical peak capacity.','',
        '| Operation | Selected function | Lowest CV loss candidate | Held-out observations below reference |','|---|---|---|---:|']
    for kind in models:
        d=diagnostics[kind];report.append(f"| {kind} | {d['chosen']} | {d['lowest_loss']} | {d['heldout_coverage']:.1%} |")
    report+=['','## Functions','', 'Use `WorkFunctions("path/to/functions.json")` from `fit_functions.py`. It exposes `decode_per_second(context)`, `prefill_per_second(context)`, `decode_units(context, rate)` and `prefill_units(context, rate)`. Context is tokens; rate is cluster tokens/s. Units = observed rate / fitted reference; total = decode units + prefill units. Values above one are allowed.','',
        'Exact representation: z = (log(1 + context/1000) − log_min)/(log_max − log_min); reference = exp(B(z) · coefficients). B is either a polynomial basis or cubic B-spline basis. Coefficients and knots are in functions.json. Context is clamped to each operation’s observed domain; no unsupported extrapolation.','',
        '## Minutes 30–35','', '| Rank | Run | Decode units/s | Prefill units/s | Combined units/s | Scored seconds |','|---:|---|---:|---:|---:|---:|']
    for i,r in enumerate(sorted([s for s in scores if s['interval']=='30-35m' and s['combined_units_s'] is not None],key=lambda r:-r['combined_units_s']),1):
        report.append(f"| {i} | {r['arm']} | {r['decode_units_s']:.3f} | {r['prefill_units_s']:.3f} | {r['combined_units_s']:.3f} | {r['scored_seconds']} |")
    report+=['','## First 35 minutes','', '| Run | Combined units/s | Scored seconds |','|---|---:|---:|']
    for r in scores:
        if r['interval']=='0-35m' and r['combined_units_s'] is not None:report.append(f"| {r['arm']} | {r['combined_units_s']:.3f} | {r['scored_seconds']} |")
    report+=['','## Sensitivity','']
    for q in ['0.85','0.95']:
        ss=sorted([s for s in scores if s['interval']=='30-35m' and s['combined_units_s'] is not None],key=lambda s:-s['combined_q'+q]);report.append(f"{q} reference: "+' → '.join(s['arm'][:2] for s in ss)+'\n')
    report+=['','## Candidate validation','', '| Operation | Candidate | CV loss | Standard error |','|---|---|---:|---:|']
    for r in candidates_report:report.append(f"| {r['operation']} | {r['candidate']} | {r['cv_loss']:.4f} | {r['cv_se']:.4f} |")
    report+=['','![Smooth reference curves](curves.png)','',
        'Limitations inherited from source data: context is estimated from overlapping HTTP requests, including queue time and uniform output growth, not measured GPU context. Both operations share that context estimate. Calls missing final usage may be absent. Run 9 includes a profiler overrun. Initial windows lacking context coverage are unscored. Negative prefill rates smaller than 1e-6 tok/s in magnitude are floating-point subtraction noise and are clamped to zero. Sparse high-context coverage, batch occupancy and operation mix affect empirical fits. The 85–95% band shows reference-choice sensitivity, not statistical confidence. No inference services or campaign files were modified.','',
        'Audit: functions.json includes exact coefficients, validation folds and source hash; reference-curves.csv, scored-windows.csv and scores.csv provide numerical outputs.']
    (args.output/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({k:{'selected':v['name'],'positive_samples':v['positive_samples']} for k,v in models.items()},indent=2))
    print('Report:',args.output/'REPORT.md')


if __name__=='__main__':main()
