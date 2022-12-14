#!/bin/bash
test_frac=0.2
pred_step=3
dropout_p=0.2
n_epoch=10

nohup python3 -u timewise_run.py --test_frac ${test_frac} --n_in 12 --n_out 6 --n_hist 4 --pred_step ${pred_step} --dropout_p ${dropout_p} --n_epoch ${n_epoch} > testfrac-${test_frac}_predstep-${pred_step}_do-${dropout_p}.out 2>&1 &

tail -f testfrac-${test_frac}_predstep-${pred_step}_do-${dropout_p}.out                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            