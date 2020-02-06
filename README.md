Evaluation script for instance detection
=====================================================

Developed for nuclei instance detection in 2d and 3d.
!Needs ground truth segmentation (checks if detection within segmentation)!
Can evaluate detection performance wrt. point detections and dense segmentations (computes center of mass of segmentation).

usage:
-------

``` shell
    python evaluate.py --res_file <pred-file> --gt_file <gt-file> --gt_key images/instances --out_dir output --detection hoefener
```
