#!/bin/bash

pred_step=1
dropout_p=0.2
n_epoch=5

for test_frac in 0.1 0.2
do
    echo "eval... multiresult_testfrac-${test_frac}_predstep-${pred_step}_do-${dropout_p}.out"
    nohup python3 -u test_frac_experiment_multiclass.py --test_frac ${test_frac} --n_in 12 --n_out 6 --n_hist 4 --pred_step ${pred_step} --dropout_p ${dropout_p} --n_epoch ${n_epoch} > multiresult_testfrac-${test_frac}_predstep-${pred_step}_do-${dropout_p}.out 2>&1 &
done