"""Compare IN schedules 1, floor, 2*floor,... using full PW-bound runs."""
# Set the same single-thread BLAS policy as the earlier batching experiment.
import experiments.benchmark_batching
import argparse
import json
import math
from pathlib import Path
from time import perf_counter, process_time
import numpy as np
from demo import example_body
from geometry import uniform_ball
from experiments.mixing import example_geometry
from sampler import in_and_out


def plot(record, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(record['h_values']), figsize=(11, 4.8), layout='constrained', squeeze=False)
    summaries = []
    for ax, h in zip(axes[0], record['h_values']):
        for index, floor in enumerate(record['floors']):
            rows = [r for r in record['runs'] if r['h']==h and r['floor']==floor]
            if not rows:
                continue
            values = [r['seconds'] for r in rows]
            median = float(np.median(values))
            ax.scatter([index]*len(values), values, color='#8aa0ab', alpha=.7)
            ax.plot([index-.18,index+.18], [median,median], lw=3, color='#007f83')
            ax.annotate(f'{median:.2f}s', (index,median), xytext=(5,6), textcoords='offset points')
            summaries.append(dict(h=h,floor=floor,trials=len(rows),median_seconds=median,
                                  min_seconds=min(values),max_seconds=max(values)))
        ax.set(xticks=range(len(record['floors'])), xticklabels=[str(f) for f in record['floors']],
               xlabel='Second batch size (first batch is always 1)', ylabel='Completed-run elapsed seconds',
               title=f'h = {h:g}', yscale='log', xlim=(-.5,len(record['floors'])-.3))
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('10D simplex: minimum IN batch size after one rejection')
    fig.supxlabel(f"{len(record['runs'])}/{len(record['h_values'])*len(record['floors'])*record['repeats']} trials | dots: individual times; bars: medians\n"
                  'PW TV target 1e-6; uncapped retries; max batch 2^20; one BLAS thread. Paths differ across schedules.',fontsize=9)
    fig.savefig(str(prefix)+'.png',dpi=170)
    plt.close(fig)
    record['summary']=summaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--floors',nargs='+',type=int,default=[2,8,16,32])
    parser.add_argument('--h-values',nargs='+',type=float,default=[5e-5,1e-4])
    parser.add_argument('--repeats',type=int,default=10)
    parser.add_argument('--seed',type=int,default=719)
    parser.add_argument('--output',default='in_floor_comparison')
    parser.add_argument('--resume',type=Path,help='Keep completed trials from this JSON while adding floors')
    args=parser.parse_args()
    if args.repeats<1 or any(f<2 for f in args.floors) or any(not math.isfinite(h) or h<=0 for h in args.h_values):
        parser.error('Require positive repeats/h and floors >= 2')
    hs,floors=sorted(set(args.h_values)),sorted(set(args.floors))
    center,radius,_,log_M=example_geometry('simplex',10)
    numerator=log_M+math.log1p(-math.exp(-log_M))-math.log(4)-2*math.log(1e-6)
    A,b,_,_=example_body('simplex',10)
    prefix=Path(args.output)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    record=dict(h_values=hs,floors=floors,repeats=args.repeats,seed=args.seed,
                dimension=10,body='simplex',batch_limit=1048576,blas_threads=1,
                requested_tv=1e-6,max_attempts=None,prefetch_out=False,
                schedule='1, floor, 2*floor, ...; reset at each proper step',
                timing='full sampler including allocations and RNG draws; initialization/plotting excluded',runs=[])
    if args.resume:
        previous=json.loads(args.resume.read_text())
        for key in record:
            if key not in ('runs','floors') and previous.get(key)!=record[key]:
                parser.error(f'Resume settings differ: {key}')
        if not set(previous['floors']).issubset(floors):
            parser.error('Include all previous floors when resuming')
        record['runs']=previous['runs']
    completed={(r['h'],r['floor'],r['repeat']) for r in record['runs']}
    order=np.random.default_rng([args.seed,999])
    for repeat in range(args.repeats):
        jobs=[(i,f) for i in range(len(hs)) for f in floors]
        order.shuffle(jobs)
        for index,floor in jobs:
            h=hs[index]; seed=[args.seed,index,repeat]
            if (h,floor,repeat+1) in completed:
                continue
            x0=uniform_ball(center,radius,np.random.default_rng(seed+[0]))
            out_rng=np.random.default_rng(seed+[1]); rng=np.random.default_rng(seed+[2])
            steps=math.ceil(numerator/math.log1p(h*math.pi**2/2))
            print(f'[h={h:g}, floor={floor}, trial={repeat+1}] starting {steps:,} steps',flush=True)
            start,cpu_start=perf_counter(),process_time(); next_report=start+10
            def progress(done,proposals,rejected):
                nonlocal next_report
                now=perf_counter()
                if now>=next_report:
                    print(f'  {done:,}/{steps:,} | {now-start:.1f}s | {proposals:,} proposals | current rejects {rejected:,}',flush=True)
                    next_report=now+10
            stats={}
            path,attempts=in_and_out(A,b,x0,h=h,steps=steps,max_attempts=None,
                                     rng=rng,out_rng=out_rng,prefetch_out=False,batch_in=True,batch_floor=floor,
                                     batch_limit=1048576,diagnostics=stats,progress=progress)
            elapsed,cpu=perf_counter()-start,process_time()-cpu_start
            assert len(attempts)==steps and np.all(path@A.T<=b)
            record['runs'].append(dict(h=h,floor=floor,repeat=repeat+1,seed=seed,steps=steps,
                                      seconds=elapsed,cpu_seconds=cpu,endpoint=path[-1].tolist(),
                                      proposals=int(attempts.sum()),generated_inward=stats['generated_inward'],
                                      unused_inward=stats['unused_inward']))
            print(f'  completed in {elapsed:.3f}s; generated {stats["generated_inward"]:,} IN vectors',flush=True)
            Path(str(prefix)+'.json').write_text(json.dumps(record,indent=2))
            plot(record,prefix)
            Path(str(prefix)+'.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record['summary'],indent=2),flush=True)


if __name__=='__main__':
    main()
