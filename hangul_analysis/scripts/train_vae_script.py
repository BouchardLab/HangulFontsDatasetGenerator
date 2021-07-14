from time import time
from subprocess import run, Popen

imf_list = ['i', 'm', 'f', 'imf']
beta = 'beta'
bottom = 1000
ex = 1
static = 'False'
count = 0
start = 0
end =   1
cuda = 1
pretrained = 'False'
path_h5 = '' # path to hangul dataset
for s in range(start, end):
    count = 1
    even = 0
    odd = 0
    with open(f'training/{ex}_{s}_train_output.out', 'a+') as outfile:
        for fold in range(1, 7):
            args = ['python', '-u', 'scripts/train_vae.py',
                   f'--experiment={ex}', f'--nfolds=7',
                   '--path_h5', f'{path_h5}',
                   f'--fold={fold}', '--device', f'cuda:{cuda}',
                   f'--seed={s}', '--beta', f'{beta}',
                   f'--static', static, '--pretrained',
                    pretrained]
            start = time()
            run(args, stdout=outfile)
            duration = time() - start
            print(f"Training time: {duration}")
            if count == 0 and duration < bottom:
                break
            count += 1
            break
    print(f"Number of folds: {count}")
    if count == 7:
        with open(f'training/{ex}_{s}_cv_output.out', 'w+') as outfile:
            for imf in imf_list:
                args = ['nohup', 'python', '-u', 'scripts/cross_valid.py', f'--experiment={ex}',
                       '--nfolds=7', '--path_h5', f'{path_h5}',
                       '--path_output', 'outputs', '--device', f'cuda:{cuda}', f'--seed={s}',
                       '--num_fonts=5', '--knn=5', '--imf', imf, '--path_model',
                       './']
                run(args, stdout=outfile)
print("All done")
