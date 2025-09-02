#!/usr/bin/env python

from pathlib import Path

from disbatchc.disBatch import DisBatcher

algorithms = ["stan", "gistum", "gistvb", "gistvd", "gistvm"]
models = ["corr-normal", "garch", "glmm-poisson", "hmm", "ill-normal", "lotka-volterra", "normal", "rosenbrock"]

SEEDS = [1013658876,  633348545, 1102916873,  709689242,  389419992,
       1193867527, 1332678212,  220488909,  858249842,  383565548,
        410939869, 1277383536,  912389988,  744855144, 1248953637,
       1671247797,  596496931, 1274683165,  640039115,  442971378,
       1019120790,  264853859,  177006548, 1855777743,  983295147,
       1858803055,  423753213,  318459698,  210740583, 1614140820,
       1749622824, 1731150822, 1709125655,   31749161, 1800441340,
       1936986389,  515068415,  211402730, 1207020782,  791344550]

SRCDIR = Path(__file__).parent.resolve()
OUTDIR = Path('/mnt/home/eroualdes/ceph/gist/output/algorithm_by_model/')

SCRIPT = Path(SRCDIR / 'run_experiments.py')

def get_task(algorithm, model, i):
    '''
    Generate a task for parameters algorithms and models, with seed index i.
    '''
    if i >= len(SEEDS):
        print(f'Warning: exceed max retry count for {algorithm=} {model=}')
        return None

    jobdir = OUTDIR # / f'{algorithm}_{model}'
    prefix = f'mkdir -p {jobdir}; cd {jobdir}'
    cmd = f'{prefix}; python {SCRIPT} --seed={SEEDS[i]} {algorithm} {model} &> {i}.log'
    task = dict(algorithm=algorithm, model=model, i=i, cmd=cmd)
    return task


def main(disbatcher):
    # Submit all the tasks with the first seed
    tasks = []
    i = 0
    for alg in algorithms:
        for mod in models:
            task = get_task(alg, mod, i)
            i += 1
            disbatcher.submit(task['cmd'])
            tasks += [task]
            print(f'Submitting task {task}')
            del task

    # Wait for tasks to complete. Resubmit as necessary.
    njob = len(tasks)
    ndone = 0
    while ndone < njob:
        status = disbatcher.wait_one_task()
        oldtask = tasks[status['TaskId']]
        if status['ReturnCode'] in (74, 84):
            # resubmit with new seed
            newi = oldtask['i'] + 1

            newtask = get_task(oldtask['algorithm'], oldtask['model'], newi)
            if newtask is None:
                continue
            disbatcher.submit(newtask['cmd'])
            tasks += [newtask]
            print(f'Resubmitting task {newtask}')
        else:
            # done task
            ndone += 1
            print(f'Finished task successfully: {oldtask}')


if __name__ == '__main__':
    disbatcher = DisBatcher(tasksname='dynamic-disBatch')
    try:
        main(disbatcher)
    finally:
        disbatcher.done()
