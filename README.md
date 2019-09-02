Evaluation script for instance detection
=====================================================

usage:
-------

``` shell
    python evaluate.py --res_file <pred-file> --gt_file <gt-file> --gt_key images/instances --out_dir output --background 0 --distance_limit 3
```

output:
--------
- evaluation metrics are written to toml-file
  - #fs: false-split, multiple predictions for one gt instance
  - #ns: non-split, one prediction for multiple gt instances
  - #tpP: number of pred instances with exactly one gt instance
  - #fpP: predicted cell for non existing ground truth instances
  - #tpGT: number of gt instances with pred instance
  - #fnGT: no predicted cell for ground truth cell
  - average precision: tpP/(tpP+fnGT+fpP)
  - average precision: tpP/(tpP+fpP)

(Pred -> GT: for each predicted instance, ..
GT -> Pred: for each ground truth instance, ..)
