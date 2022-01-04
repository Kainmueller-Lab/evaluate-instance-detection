import argparse
import logging
import glob
import os

import h5py
import numpy as np
import scipy.ndimage
from scipy.optimize import linear_sum_assignment
import scipy.spatial
import toml
import zarr

logger = logging.getLogger(__name__)


def evaluate_file(**kwargs):
    logger.info("evaluating %s %s", kwargs['res_file'], kwargs['gt_file'])
    sample_pred = os.path.basename(kwargs['res_file'])

    # load labelling
    if kwargs['res_file'].endswith("zarr"):
        input_file = zarr.open(kwargs['res_file'], 'r')
    elif kwargs['res_file'].endswith("hdf"):
        input_file = h5py.File(kwargs['res_file'], 'r')
    else:
        raise NotImplementedError("invalid pred format")
    if kwargs.get('sparse', False):
        reg_max = np.squeeze(np.array(input_file['/volumes/markers']), axis=0)
        pred_cells = np.argwhere(reg_max > 0)
    else:
        labeling = np.squeeze(np.array(input_file[kwargs['res_key']]),
                              axis=0)
        # seeds = np.array(input_file[kwargs['res_key']])
        # labeling, cnt = scipy.ndimage.label(seeds)
        pred_cells = np.array(scipy.ndimage.measurements.center_of_mass(
            labeling > 0,
            labeling, sorted(list(np.unique(labeling)))[1:]))
    logger.info("%s: number pred cells: %s", sample_pred, pred_cells.shape)

    if pred_cells.shape[0] == 0:
        logger.error("no predicted cells found, aborting..")
        return None

    # load gt image
    sample_gt = os.path.basename(kwargs['gt_file'])
    if kwargs['gt_file'].endswith("hdf"):
        with h5py.File(kwargs['gt_file'], 'r') as gt:
            gt_labels = np.array(gt[kwargs['gt_key']])
    elif kwargs['gt_file'].endswith("zarr"):
        gt = zarr.open(kwargs['gt_file'], 'r')
        gt_labels = np.array(gt[kwargs['gt_key']])
    else:
        raise NotImplementedError("invalid gt format")
    gt_labels = np.squeeze(gt_labels, axis=0)
    logger.debug("%s: gt min %f, max %f",
                 sample_gt, gt_labels.min(), gt_labels.max())

    # load gt points
    gt_cells = []
    csvFn = os.path.splitext(kwargs['gt_file'])[0] + ".csv"
    with open(csvFn, 'r') as f:
        for ln in f:
            z, y, x, _ = ln.strip().split(",")
            z = float(z)-kwargs.get("padding", [0, 0, 0])[0]
            y = float(y)-kwargs.get("padding", [0, 0, 0])[1]
            x = float(x)-kwargs.get("padding", [0, 0, 0])[2]
            gt_cells.append((z,y,x))
    gt_cells = np.array(gt_cells)
    logger.info("%s: number gt cells: %s", sample_gt, gt_cells.shape)
    if kwargs['debug']:
        gt_labels_debug = np.array(input_file['/volumes/gt_labels_debug'])
    else:
        gt_labels_debug = None

    if pred_cells.shape[0] > gt_cells.shape[0] * 10:
        logger.error("far too many predicted cells, aborting.. ({} vs {})".format(
            pred_cells.shape[0], gt_cells.shape[0]))
        return None

    detection_fn = kwargs.get('detection', 'greedy')
    outFnBase = os.path.join(
        kwargs['out_dir'],
        os.path.splitext(os.path.basename(kwargs['res_file']))[0] + "_scores")
    if detection_fn == "linear" or detection_fn == "hungarian":
        outFnBase += "_linear"
    elif detection_fn == "greedy":
        outFnBase += "_greedy"
    elif detection_fn == "hoefener":
        outFnBase += "_hoefener"
    if not kwargs.get("from_scratch") and len(glob.glob(outFnBase + "*")) > 0:
        with open(outFnBase+".toml", 'r') as tomlFl:
            metrics = toml.load(tomlFl)
        if kwargs.get('metric', None) is None:
            return metrics
        try:
            metric = metrics
            for k in kwargs['metric'].split('.'):
                metric = metric[k]
            logger.info('Skipping evaluation for %s. Already exists!',
                        kwargs['res_file'])
            return metrics
        except KeyError:
            logger.info('Error (key %s missing) in existing evaluation for %s. Recomputing!',
                        kwargs['metric'], kwargs['res_file'])

    tomlFl = open(outFnBase + ".toml", 'w')
    results = {}

    if detection_fn == "linear" or detection_fn == "hungarian":
        res = compute_linear_sum_assignment(gt_cells, pred_cells,
                                            gt_labels, **kwargs)
        results['confusion_matrix'] = res
    elif detection_fn == "greedy":
        results['confusion_matrix'] = {}
        res = computeMetrics(gt_cells, pred_cells,
                             gt_labels, gt_labels_debug,
                             draw=kwargs['debug'], **kwargs)
        results['confusion_matrix']['gt_pred'] = res
        res = computeMetrics(pred_cells, gt_cells,
                             gt_labels, gt_labels_debug,
                             draw=False, reverse=True, **kwargs)
        results['confusion_matrix']['pred_gt'] = res
    elif detection_fn == "hoefener":
        results['confusion_matrix'] = {}
        res = computeMetricsH(gt_cells, pred_cells,
                              gt_labels, outFnBase, **kwargs)
        results['confusion_matrix']['hoefener'] = res
    else:
        raise RuntimeError("invalid detection method")
    toml.dump(results, tomlFl)
    return results


def compute_linear_sum_assignment(gt_cells, pred_cells, gt_labels, **kwargs):
    costMat = np.zeros((len(gt_cells), len(pred_cells)), dtype=np.float32)
    distance_limit = kwargs.get('distance_limit', 10)
    out_of_range = distance_limit
    costMat[:,:] = out_of_range

    # costMat = scipy.spatial.distance.cdist(gt_cells, pred_cells)
    gt_cells_tree = scipy.spatial.cKDTree(gt_cells, leafsize=4)
    nn_distances_p, nn_locations_p = gt_cells_tree.query(
        pred_cells, k=5000)
        # distance_upper_bound=distance_limit)
    for dists, gIDs, pID in zip(nn_distances_p, nn_locations_p,
                                range(pred_cells.shape[0])):
        # dists = [dists]
        # gIDs = [gIDs]
        # print(dists)
        for d, gID in zip(dists, gIDs):
            if d != np.inf:
                costMat[gID, pID] = d

    pred_cells_tree = scipy.spatial.cKDTree(pred_cells, leafsize=4)
    nn_distances_g, nn_locations_g = pred_cells_tree.query(
        gt_cells, k=5000)
        # distance_upper_bound=distance_limit)
    for dists, pIDs, gID in zip(nn_distances_g, nn_locations_g,
                                range(gt_cells.shape[0])):
        # dists = [dists]
        # pIDs = [pIDs]
        # print(dists)
        for d, pID in zip(dists, pIDs):
            if d != np.inf:
                costMat[gID, pID] = d

    gt_inds, pred_inds = linear_sum_assignment(costMat)
    tp = 0
    for gID, pID in zip(gt_inds, pred_inds):
        # print(nn_distances_g[gID], nn_distances_p[pID])
        # logger.info("gt: %s   pred: %s   (dist: %s) %s %s",
        #             [int(c) for c in gt_cells[gID]], pred_cells[pID], costMat[gID, pID],
        #             gt_labels[int(round(pred_cells[pID][0])),
        #                       int(round(pred_cells[pID][1])),
        #                       int(round(pred_cells[pID][2]))],
        #             gt_labels[int(round(gt_cells[gID][0])),
        #                       int(round(gt_cells[gID][1])),
        #                       int(round(gt_cells[gID][2]))])
        if costMat[gID, pID] < distance_limit and (
                costMat[gID, pID] < 3 or \
                gt_labels[int(round(pred_cells[pID][0])),
                          int(round(pred_cells[pID][1])),
                          int(round(pred_cells[pID][2]))] == \
                gt_labels[int(round(gt_cells[gID][0])),
                          int(round(gt_cells[gID][1])),
                          int(round(gt_cells[gID][2]))]):
            tp += 1
    fp = len(pred_cells) - tp
    fn = len(gt_cells) - tp

    results = {}
    results['Num_Matches'] = len(gt_inds)
    results['Num_GT'] = len(gt_cells)
    results['Num_Pred'] = len(pred_cells)
    results['TP'] = tp
    results['FP'] = fp
    results['FN'] = fn

    apDef = tp / (tp + fn + fp)
    results['AP'] = apDef
    logger.debug("AP: %s", results['AP'])
    apSD = tp / (tp+fp)
    results['AP_CV'] = apSD
    logger.debug("AP_CV: %s", results['AP_CV'])

    return results


def computeMetrics(source_cells, target_cells,
                   gt_labels, gt_labels_debug,
                   draw=False, reverse=False, **kwargs):
    source_cells_tree = scipy.spatial.cKDTree(source_cells, leafsize=4)

    nn_distances, nn_locations = source_cells_tree.query(target_cells, k=1)

    fpP = 0
    fnGT = 0
    tpP = 0
    tpGT = 0
    nsP = 0
    fsGT = 0
    fn = 0

    cntsSource = np.zeros(source_cells.shape[0], dtype=np.uint16)
    cntsTarget = np.zeros(target_cells.shape[0], dtype=np.uint16)

    for dist, sID, tID in zip(nn_distances, nn_locations,
                               range(target_cells.shape[0])):
        if kwargs['debug'] and tID >= 20:
            break
        logger.debug("checking nearest neighbor target cell: %s", tID)
        if sID < source_cells.shape[0]:
            logger.debug("%s %s %s", dist, source_cells[sID],
                         target_cells[tID])
            # if same label
            if gt_labels[int(round(target_cells[tID][0])),
                         int(round(target_cells[tID][1])),
                         int(round(target_cells[tID][2]))] == \
               gt_labels[int(round(source_cells[sID][0])),
                         int(round(source_cells[sID][1])),
                         int(round(source_cells[sID][2]))]:
                cntsSource[sID] += 1
                cntsTarget[tID] += 1
        else:
            logger.debug("no neighbor for %s", sID)
            # fpP += 1

    results = {}
    if reverse:
        tpP = np.count_nonzero(cntsSource==1)
        fpP = np.count_nonzero(cntsSource==0)
        # non-split
        nsP = np.count_nonzero(cntsSource>1)

        tpGT = np.count_nonzero(cntsTarget==1)
        fnGT = np.count_nonzero(cntsTarget==0)

        results['Num_GT'] = target_cells.shape[0]
        results['Num_Pred'] = source_cells.shape[0]
    else:
        fnGT = np.count_nonzero(cntsSource==0)
        tpGT = np.count_nonzero(cntsSource==1)
        # false-split
        fsGT = np.count_nonzero(cntsSource>1)

        tpP = np.count_nonzero(cntsTarget==1)
        fpP = np.count_nonzero(cntsTarget!=1)

        results['Num_GT'] = source_cells.shape[0]
        results['Num_Pred'] = target_cells.shape[0]

    logger.debug("Num GT: %s", results['Num_GT'])
    logger.debug("Num Pred: %s", results['Num_Pred'])
    results['TP_GT'] = tpGT
    logger.debug("TP GT: %s", results['TP_GT'])
    results['FN_GT'] = fnGT
    logger.debug("FN GT: %s", results['FN_GT'])
    results['FS_GT'] = fsGT
    logger.debug("FS GT: %s", results['FS_GT'])
    results['TP_Pred'] = tpP
    logger.debug("TP Pred: %s", results['TP_Pred'])
    results['FP_Pred'] = fpP
    logger.debug("FP Pred: %s", results['FP_Pred'])
    results['NS_Pred'] = nsP
    logger.debug("NS Pred: %s", results['NS_Pred'])

    apDef = tpP / (tpP + fnGT + fpP)
    results['AP'] = apDef
    logger.debug("AP: %s", results['AP'])
    apSD = tpP / (tpP+fpP)
    results['AP_CV'] = apSD
    logger.debug("AP_CV: %s", results['AP_CV'])

    return results


def computeMetricsH(gt_cells, pred_cells,
                    gt_labels, outFn, **kwargs):
    distance_limit = kwargs.get('distance_limit', 6)

    gt_cells_tree = scipy.spatial.cKDTree(gt_cells, leafsize=4)
    nn_distancesP, nn_locationsP = gt_cells_tree.query(pred_cells, k=1)

    fp = 0
    fn = 0
    tp = 0
    matchesGT = {}
    for dist, gtID, pID in zip(nn_distancesP, nn_locationsP,
                               range(pred_cells.shape[0])):
        if kwargs['debug'] and gtID >= 20:
            break
        logger.debug("checking nearest neighbor pred cell: %s", pID)
        if gtID < gt_cells.shape[0]:
            logger.debug("%s %s %s", dist, gt_cells[gtID],
                         pred_cells[pID])
            # if within distance and same label
            # if dist < distance_limit and \
            if gt_labels[int(round(pred_cells[pID][0])),
                         int(round(pred_cells[pID][1])),
                         int(round(pred_cells[pID][2]))] == \
               gt_labels[int(round(gt_cells[gtID][0])),
                         int(round(gt_cells[gtID][1])),
                         int(round(gt_cells[gtID][2]))]:
                matchesGT.setdefault(gtID, []).append(pID)
            else:
                fp += 1
        else:
            raise RuntimeError("shouldn't happen")
            logger.debug("no neighbor for %s", gtID)

    gtIDs_t = []
    pIDs_t = []
    for gtID, pIDs in matchesGT.items():
        pIDs_t.append(pIDs[0])
        gtIDs_t.append(gtID)
        tp += 1
        if len(pIDs) > 1:
            fp += len(pIDs) - 1
    fn += len(gt_cells) - len(matchesGT)
    results = {}

    results['Num_GT'] = gt_cells.shape[0]
    results['Num_Pred'] = pred_cells.shape[0]

    logger.debug("Num GT: %s", results['Num_GT'])
    logger.debug("Num Pred: %s", results['Num_Pred'])
    results['TP'] = tp
    logger.debug("TP: %s", results['TP'])
    results['FN'] = fn
    logger.debug("FN: %s", results['FN'])
    results['FP'] = fp
    logger.debug("FP: %s", results['FP'])

    apDef = tp / (tp + fn + fp)
    results['AP'] = apDef
    logger.debug("AP: %s", results['AP'])
    apSD = tp / (tp+fp)
    results['AP_CV'] = apSD
    logger.debug("AP_CV: %s", results['AP_CV'])

    # reversed order
    pred_cells_tree = scipy.spatial.cKDTree(pred_cells, leafsize=4)
    nn_distancesGT, nn_locationsGT = pred_cells_tree.query(gt_cells, k=1)

    fp = 0
    fn = 0
    tp = 0
    matchesP = {}
    for dist, pID, gtID in zip(nn_distancesGT, nn_locationsGT,
                               range(gt_cells.shape[0])):
        if kwargs['debug'] and pID >= 20:
            break
        logger.debug("checking nearest neighbor gt cell: %s", gtID)
        if pID < pred_cells.shape[0]:
            logger.debug("%s %s %s", dist, pred_cells[pID],
                         gt_cells[gtID])
            # if within distance and same label
            # if dist < distance_limit and \
            if gt_labels[int(round(pred_cells[pID][0])),
                         int(round(pred_cells[pID][1])),
                         int(round(pred_cells[pID][2]))] == \
               gt_labels[int(round(gt_cells[gtID][0])),
                         int(round(gt_cells[gtID][1])),
                         int(round(gt_cells[gtID][2]))]:
                matchesP.setdefault(pID, []).append(gtID)
            else:
                fn += 1
        else:
            raise RuntimeError("shouldn't happen")
            logger.debug("no neighbor for %s", pID)

    for pID, gtIDs in matchesP.items():
        tp += 1
        if len(gtIDs) > 1:
            fn += len(gtIDs) - 1
    fp += len(pred_cells) - len(matchesP)
    results['TP_rev'] = tp
    results['FN_rev'] = fn
    results['FP_rev'] = fp

    apDef = tp / (tp + fn + fp)
    results['AP_rev'] = apDef
    apSD = tp / (tp+fp)
    results['AP_CV_rev'] = apSD

    if kwargs['visualize']:
        vis_tp = np.zeros_like(gt_labels, dtype=np.float32)
        vis_tp2 = np.zeros_like(gt_labels, dtype=np.float32)
        vis_fp = np.zeros_like(gt_labels, dtype=np.float32)
        vis_fn = np.zeros_like(gt_labels, dtype=np.float32)
        for gti, pi, in zip(gtIDs_t, pIDs_t):
            vis_tp[int(round(pred_cells[pi][0])),
                   int(round(pred_cells[pi][1])),
                   int(round(pred_cells[pi][2]))] = 1
            vis_tp2[int(round(gt_cells[gti][0])),
                    int(round(gt_cells[gti][1])),
                    int(round(gt_cells[gti][2]))] = 1
        for pi in range(len(pred_cells)):
            if pi in pIDs_t:
                continue
            vis_fp[int(round(pred_cells[pi][0])),
                   int(round(pred_cells[pi][1])),
                   int(round(pred_cells[pi][2]))] = 1
        for gti in range(len(gt_cells)):
            if gti in gtIDs_t:
                continue
            vis_fn[int(round(gt_cells[gti][0])),
                   int(round(gt_cells[gti][1])),
                   int(round(gt_cells[gti][2]))] = 1
        sz = 1
        vis_tp = scipy.ndimage.gaussian_filter(vis_tp, sz, truncate=sz)
        vis_fp = scipy.ndimage.gaussian_filter(vis_fp, sz, truncate=sz)
        vis_fn = scipy.ndimage.gaussian_filter(vis_fn, sz, truncate=sz)
        vis_tp2 = scipy.ndimage.gaussian_filter(vis_tp2, sz, truncate=sz)

        vis_tp = vis_tp/np.max(vis_tp)
        vis_fp = vis_fp/np.max(vis_fp)
        vis_fn = vis_fn/np.max(vis_fn)
        vis_tp2 = vis_tp2/np.max(vis_tp2)
        with h5py.File(outFn + "_vis.hdf", 'w') as fi:
            fi.create_dataset(
                'volumes/vis_tp',
                data=vis_tp,
                compression='gzip')
            fi.create_dataset(
                'volumes/vis_tp2',
                data=vis_tp2,
                compression='gzip')
            fi.create_dataset(
                'volumes/vis_fp',
                data=vis_fp,
                compression='gzip')
            fi.create_dataset(
                'volumes/vis_fn',
                data=vis_fn,
                compression='gzip')
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_file', type=str,
                        help='path to res_file', required=True)
    parser.add_argument('--res_file_suffix', type=str,
                        help='res_file suffix (deprecated)')
    parser.add_argument('--res_key', type=str,
                        help='name labeling hdf key')
    parser.add_argument('--gt_file', type=str,
                        help='path to gt_file (hdf, gt segmentation, required csv companion file with center points of instances)', required=True)
    parser.add_argument('--gt_key', type=str,
                        help='name gt hdf key')
    parser.add_argument('--padding', type=int,
                        help='padding in hdf file (deprecated)')
    parser.add_argument('--out_dir', type=str,
                        help='output directory', required=True)
    parser.add_argument('--metric', type=str,
                        default="confusion_matrix.AP",
                        help='check if this metric already has been computed in possibly existing result files')
    parser.add_argument("--from_scratch",
                        help="recompute everything (instead of checking if results are already there)",
                        action="store_true")
    parser.add_argument("--detection", type=str,
                        help="which matching method to use (linear (Hungarian matching), greedy (greedily match both ways), hoefener (preferred, from: Deep learning nuclei detection: A simple approach can deliver state-of-the-art results))",
                        action="store_true")
    parser.add_argument("--distance_limit",
                        help="distance limit to match gt and predicted detection (deprecated, only for linear matching)", type=int,
                        default=10)
    parser.add_argument("--debug", help="",
                        action="store_true")
    parser.add_argument("--visualize", help="",
                        action="store_true")
    parser.add_argument("--no_sparse",
                        help="work on segmentation predictions (computes center of mass and uses that as detection)",
                        dest='sparse',
                        action="store_false")

    args = parser.parse_args()
    if args.use_gt_fg:
        logger.info("using gt foreground")

    evaluate_file(vars(args))
