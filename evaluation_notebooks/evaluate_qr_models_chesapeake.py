{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "492f5885",
   "metadata": {},
   "outputs": [],
   "source": [
    "import rasterio\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import torch\n",
    "import time\n",
    "import os\n",
    "import sys\n",
    "sys.path.append('/home/esther/lc-mapping/scripts')\n",
    "import landcover_definitions as lc\n",
    "import util\n",
    "sys.path.append('/home/esther/torchgeo')\n",
    "import run_model_forward_and_produce_tifs \n",
    "from importlib import reload\n",
    "reload(run_model_forward_and_produce_tifs)\n",
    "#import run_through_tiles\n",
    "\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "d81accf1",
   "metadata": {},
   "outputs": [],
   "source": [
    "def reindex_cc(array_in):\n",
    "    impervious_idxs_highres_orig = [4,5,6]\n",
    "    impervious_idx_condensed = 4\n",
    "    \n",
    "    reindexed_array = array_in.copy()\n",
    "    for c_idx in impervious_idxs_highres_orig:\n",
    "        reindexed_array[array_in == c_idx] = impervious_idx_condensed\n",
    "            \n",
    "    return reindexed_array - 1\n",
    "\n",
    "ignore_index = 4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "883484f0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, 3])"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.arange(0,ignore_index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "6b8ee62d",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "20\n",
      "6.03 seconds\n",
      "6.05 seconds\n",
      "5.96 seconds\n",
      "5.97 seconds\n",
      "6.06 seconds\n",
      "6.00 seconds\n",
      "6.02 seconds\n",
      "5.89 seconds\n",
      "5.94 seconds\n",
      "5.98 seconds\n",
      "6.07 seconds\n",
      "6.23 seconds\n",
      "6.21 seconds\n",
      "6.19 seconds\n",
      "6.28 seconds\n",
      "6.30 seconds\n",
      "6.33 seconds\n",
      "6.00 seconds\n",
      "6.20 seconds\n",
      "6.13 seconds\n",
      "6.44 seconds\n",
      "6.60 seconds\n",
      "6.59 seconds\n",
      "6.51 seconds\n",
      "6.42 seconds\n",
      "6.72 seconds\n",
      "6.62 seconds\n",
      "6.57 seconds\n",
      "6.54 seconds\n",
      "6.07 seconds\n",
      "6.17 seconds\n",
      "6.24 seconds\n",
      "6.24 seconds\n",
      "6.39 seconds\n",
      "6.34 seconds\n",
      "6.39 seconds\n",
      "6.53 seconds\n",
      "6.54 seconds\n",
      "6.55 seconds\n",
      "6.49 seconds\n",
      "For ny+pa test set with qr_forward loss:\n",
      "acc q: 0.8462133269028715\n",
      "mean iou q: 0.6767380615052179\n",
      "acc r: 0.8539573488604718\n",
      "mean iou r: 0.6939103086405636\n",
      "IoU per class over the tiles (q) is: \n",
      "[0.793293819989226, 0.8051973408589472, 0.6611304000023405, 0.44733068517035796]\n",
      "IoU per class over the tiles (r) is: \n",
      "[0.8105675574771809, 0.8136806613536584, 0.67431060562961, 0.47708241010180513]\n",
      "20\n",
      "6.04 seconds\n",
      "6.09 seconds\n",
      "5.92 seconds\n",
      "5.98 seconds\n",
      "6.02 seconds\n",
      "5.96 seconds\n",
      "5.99 seconds\n",
      "5.92 seconds\n",
      "5.94 seconds\n",
      "5.98 seconds\n",
      "6.03 seconds\n",
      "6.13 seconds\n",
      "6.10 seconds\n",
      "6.21 seconds\n",
      "6.17 seconds\n",
      "6.31 seconds\n",
      "6.28 seconds\n",
      "6.03 seconds\n",
      "6.14 seconds\n",
      "6.21 seconds\n",
      "For ny test set with qr_forward loss:\n",
      "acc q: 0.862002628750849\n",
      "mean iou q: 0.7095529415216242\n",
      "acc r: 0.8689901875245085\n",
      "mean iou r: 0.7242465968509084\n",
      "IoU per class over the tiles (q) is: \n",
      "[0.849974074865186, 0.81705479147044, 0.6964321409218723, 0.4747507588289981]\n",
      "IoU per class over the tiles (r) is: \n",
      "[0.8638831375305049, 0.8255045450743717, 0.7086668781071237, 0.4989318266916334]\n",
      "20\n",
      "6.41 seconds\n",
      "6.59 seconds\n",
      "6.55 seconds\n",
      "6.50 seconds\n",
      "6.41 seconds\n",
      "6.72 seconds\n",
      "6.66 seconds\n",
      "6.66 seconds\n",
      "6.41 seconds\n",
      "6.07 seconds\n",
      "6.24 seconds\n",
      "6.26 seconds\n",
      "6.25 seconds\n",
      "6.39 seconds\n",
      "6.30 seconds\n",
      "6.38 seconds\n",
      "6.49 seconds\n",
      "6.66 seconds\n",
      "6.41 seconds\n",
      "6.53 seconds\n",
      "For pa test set with qr_forward loss:\n",
      "acc q: 0.8419778583012127\n",
      "mean iou q: 0.66568681129511\n",
      "acc r: 0.849943910759229\n",
      "mean iou r: 0.6809969597822135\n",
      "IoU per class over the tiles (q) is: \n",
      "[0.6997373171060703, 0.7996298678541546, 0.6465305792919254, 0.51684948092829]\n",
      "IoU per class over the tiles (r) is: \n",
      "[0.7200085640924442, 0.8091720772610658, 0.6604402519196578, 0.534366945855686]\n",
      "writing results to /home/esther/torchgeo_predictions/pa_fcn_0.0001_qr_forward_from_cooccurrences_101_31_no_osm_no_buildings_additive_smooth_0.0001_prior_smooth_0.0001/.pkl\n",
      "20\n",
      "6.06 seconds\n",
      "6.04 seconds\n",
      "5.92 seconds\n",
      "5.96 seconds\n",
      "6.03 seconds\n",
      "6.00 seconds\n",
      "6.05 seconds\n",
      "5.92 seconds\n",
      "5.94 seconds\n",
      "5.95 seconds\n",
      "6.06 seconds\n",
      "6.15 seconds\n",
      "6.14 seconds\n",
      "6.16 seconds\n",
      "6.13 seconds\n",
      "6.30 seconds\n",
      "6.32 seconds\n",
      "6.01 seconds\n",
      "6.17 seconds\n",
      "6.16 seconds\n",
      "6.42 seconds\n",
      "6.58 seconds\n",
      "6.55 seconds\n",
      "6.47 seconds\n",
      "6.43 seconds\n",
      "6.64 seconds\n",
      "6.62 seconds\n",
      "6.56 seconds\n",
      "6.47 seconds\n",
      "6.12 seconds\n",
      "6.19 seconds\n",
      "6.26 seconds\n",
      "6.31 seconds\n",
      "6.41 seconds\n",
      "6.27 seconds\n",
      "6.53 seconds\n",
      "6.58 seconds\n",
      "6.50 seconds\n",
      "6.40 seconds\n",
      "6.45 seconds\n",
      "For ny+pa test set with qr_reverse loss:\n",
      "acc q: 0.7868529815416312\n",
      "mean iou q: 0.5937724939886397\n",
      "acc r: 0.7876169839432412\n",
      "mean iou r: 0.596522109740738\n",
      "IoU per class over the tiles (q) is: \n",
      "[0.7848138245697828, 0.7468804500514419, 0.5304815837420368, 0.3129141175912971]\n",
      "IoU per class over the tiles (r) is: \n",
      "[0.7893536651764964, 0.747356465821595, 0.53163384861408, 0.3177444593507808]\n",
      "20\n",
      "6.04 seconds\n",
      "6.06 seconds\n",
      "5.90 seconds\n",
      "5.91 seconds\n",
      "6.02 seconds\n",
      "6.02 seconds\n",
      "6.01 seconds\n",
      "5.97 seconds\n",
      "5.91 seconds\n",
      "5.98 seconds\n",
      "6.06 seconds\n",
      "6.17 seconds\n",
      "6.12 seconds\n",
      "6.07 seconds\n",
      "6.18 seconds\n",
      "6.30 seconds\n",
      "6.24 seconds\n",
      "5.91 seconds\n",
      "6.11 seconds\n",
      "6.15 seconds\n",
      "For ny test set with qr_reverse loss:\n",
      "acc q: 0.7896135479702976\n",
      "mean iou q: 0.614379826103965\n",
      "acc r: 0.7901514433002428\n",
      "mean iou r: 0.6161292647988353\n",
      "IoU per class over the tiles (q) is: \n",
      "[0.8292401464801954, 0.7354482459501027, 0.5546667733743265, 0.3381641386112351]\n",
      "IoU per class over the tiles (r) is: \n",
      "[0.8318140482424236, 0.7358244065500023, 0.5554260265336495, 0.3414525778692659]\n",
      "20\n",
      "6.45 seconds\n",
      "6.58 seconds\n",
      "6.60 seconds\n",
      "6.44 seconds\n",
      "6.46 seconds\n",
      "6.60 seconds\n",
      "6.64 seconds\n",
      "6.60 seconds\n",
      "6.47 seconds\n",
      "6.12 seconds\n",
      "6.15 seconds\n",
      "6.26 seconds\n",
      "6.29 seconds\n",
      "6.37 seconds\n",
      "6.26 seconds\n",
      "6.41 seconds\n",
      "6.47 seconds\n",
      "6.50 seconds\n",
      "6.40 seconds\n",
      "6.43 seconds\n",
      "For pa test set with qr_reverse loss:\n",
      "acc q: 0.8224240631228289\n",
      "mean iou q: 0.6337404668027408\n",
      "acc r: 0.822743632511529\n",
      "mean iou r: 0.6350698812870447\n",
      "IoU per class over the tiles (q) is: \n",
      "[0.6971112298606674, 0.7827303003340529, 0.600911407805695, 0.45420892921054745]\n",
      "IoU per class over the tiles (r) is: \n",
      "[0.6998875437670332, 0.7829749588750222, 0.6014086101144216, 0.45600841239170137]\n",
      "writing results to /home/esther/torchgeo_predictions/pa_fcn_0.0001_qr_reverse_from_cooccurrences_101_31_no_osm_no_buildings_additive_smooth_0.0001_prior_smooth_0.0001/.pkl\n"
     ]
    }
   ],
   "source": [
    "reload(util)\n",
    "\n",
    "set_this = 'test'\n",
    "compute_r = True\n",
    "results_by_state_q = {}\n",
    "results_by_state_r = {}\n",
    "\n",
    "states_to_eval = ['ny+pa','ny', 'pa']\n",
    "\n",
    "loss_to_eval_options = ['qr_forward', 'qr_reverse']\n",
    "\n",
    "prior_version = 'from_cooccurrences_101_31_no_osm_no_buildings'\n",
    "prior_name = f'prior_{prior_version}'\n",
    "p_add_smooth = 1e-4\n",
    "lc_type = 'chesapeake_4_no_zeros'\n",
    "compute_r = True\n",
    "for loss in loss_to_eval_options:\n",
    "#for loss in [1]:\n",
    "    results_by_state_q[loss] = {}\n",
    "    results_by_state_r[loss] = {}\n",
    "    \n",
    "    for state_str in states_to_eval:\n",
    "\n",
    "        data_dir = '/home/esther/torchgeo_data/cvpr_chesapeake_landcover'\n",
    "        \n",
    "        run_name = f'{state_str}_fcn_0.0001_{loss}_{prior_version}_additive_smooth_0.0001_prior_smooth_0.0001/'\n",
    "        \n",
    "        print(len(tile_ids))\n",
    "        \n",
    "        accs_q = []\n",
    "        ious_q = []\n",
    "        accs_r = []\n",
    "        ious_r = []\n",
    "        num_pix = []\n",
    "        \n",
    "        for state in state_str.split('+'):\n",
    "            state_identifier = f'{state}_1m_2013_extended-debuffered-test_tiles'\n",
    "            data_dir_this_set = os.path.join(data_dir,state_identifier)\n",
    "            pred_dir = f'/home/esther/torchgeo_predictions/{run_name}/cvpr_chesapeake_landcover'\n",
    "            pred_dir_this_set = os.path.join(pred_dir,state_identifier)\n",
    "        \n",
    "            fns = os.listdir(data_dir_this_set)\n",
    "            tile_ids = np.unique([x[:17] for x in fns])\n",
    "            \n",
    "            for tile_id in tile_ids:\n",
    "                fn_this = os.path.join(data_dir_this_set, f'{tile_id}_lc.tif')\n",
    "                pred_fn_this = os.path.join(pred_dir_this_set, f'{tile_id}_{loss}_pred_last.tif')\n",
    "                t1 = time.time()\n",
    "\n",
    "                # gather the data\n",
    "                with rasterio.open(fn_this) as f:\n",
    "                    hr_lc = f.read()[0]\n",
    "                # reindex\n",
    "                hr_lc = reindex_cc(hr_lc)\n",
    "\n",
    "                preds_this_soft = rasterio.open(pred_fn_this).read()\n",
    "\n",
    "                preds_this = preds_this_soft.argmax(0)\n",
    "                acc_this_q = (np.array([hr_lc == preds_this])[np.array([hr_lc!=ignore_index])]).mean()\n",
    "\n",
    "                # ignore 0\n",
    "                iou_this_q = util.per_class_iou(hr_lc, preds_this, np.arange(0,ignore_index))\n",
    "                accs_q.append(acc_this_q)\n",
    "                ious_q.append(iou_this_q)\n",
    "                num_pix.append((hr_lc != ignore_index).sum())\n",
    "\n",
    "                if compute_r:\n",
    "                    # now do r\n",
    "                    prior_this = rasterio.open(fn_this.replace('lc.tif',f'{prior_name}.tif')).read()\n",
    "\n",
    "                    # first normalize\n",
    "                    prior = prior_this / prior_this.sum(axis=0)\n",
    "                    # now add smoothing and renormalize \n",
    "\n",
    "                    prior = (prior + p_add_smooth) / (prior + p_add_smooth).sum(axis=0)\n",
    "\n",
    "                    # compute z and r\n",
    "                    z = (preds_this_soft.T / preds_this_soft.sum(axis=(1,2)) ).T\n",
    "                    preds_r = (prior*z).argmax(0)\n",
    "\n",
    "                    acc_this_r = (np.array([hr_lc == preds_r])[np.array([hr_lc!=ignore_index])]).mean()\n",
    "                    iou_this_r = util.per_class_iou(hr_lc, preds_r, np.arange(0,ignore_index))\n",
    "\n",
    "                    accs_r.append(acc_this_r)\n",
    "                    ious_r.append(iou_this_r)\n",
    "\n",
    "                t2 = time.time()\n",
    "                print(f'{t2-t1:.2f} seconds')\n",
    "\n",
    "        ious_aggregated_q = util.aggregate_ious([x[1] for x in ious_q], [x[2] for x in ious_q])\n",
    "        acc_aggregated_q = (np.array(accs_q) * np.array(num_pix)).sum() / np.sum(num_pix)\n",
    "        print(f'For {state_str} {set_this} set with {loss} loss:')\n",
    "        print(f'acc q: {acc_aggregated_q}')\n",
    "        print(f'mean iou q: {np.mean(ious_aggregated_q[0])}')\n",
    "        \n",
    "        if compute_r:\n",
    "            acc_aggregated_r = (np.array(accs_r) * np.array(num_pix)).sum() / np.sum(num_pix)\n",
    "            ious_aggregated_r = util.aggregate_ious([x[1] for x in ious_r], [x[2] for x in ious_r])\n",
    "            print(f'acc r: {acc_aggregated_r}')\n",
    "            print(f'mean iou r: {np.mean(ious_aggregated_r[0])}')\n",
    "        \n",
    "        print('IoU per class over the tiles (q) is: ')\n",
    "        print(ious_aggregated_q[0])\n",
    "        if compute_r:\n",
    "            print('IoU per class over the tiles (r) is: ')\n",
    "            print(ious_aggregated_r[0])\n",
    "        \n",
    "\n",
    "        results_by_state_q[loss][state_str] = {'accs': accs_q,\n",
    "                                           'ious': ious_q,\n",
    "                                           'num_pix':num_pix,\n",
    "                                           'ious_aggregated': ious_aggregated_q,\n",
    "                                           'acc_aggregated':acc_aggregated_q}\n",
    "        \n",
    "        if compute_r:\n",
    "            results_by_state_r[loss][state_str] = {'accs': accs_r,\n",
    "                                               'ious': ious_r,\n",
    "                                               'num_pix':num_pix,\n",
    "                                               'ious_aggregated': ious_aggregated_r,\n",
    "                                               'acc_aggregated':acc_aggregated_r}\n",
    "        \n",
    "    out_fn = f'/home/esther/torchgeo_predictions/{run_name.replace(\"_\"+state_str,\"\")}.pkl'\n",
    "    with open(out_fn, 'wb') as f:\n",
    "        print(f'writing results to {out_fn}')\n",
    "        pickle.dump({'results_by_state_q':results_by_state_q[loss],\n",
    "                     'results_by_state_r':results_by_state_r[loss]}, f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "68782b55",
   "metadata": {},
   "outputs": [],
   "source": [
    "states_in_reporting_order = ['pa', 'ny', 'ny+pa']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "d754bd26",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "qr_forward q \n",
      "& 84.20 & 66.57 & 86.20 & 70.96 & 84.62 & 67.67 \n",
      "qr_forward r \n",
      "& 84.99  & 68.10 & 86.90  & 72.42 & 85.40  & 69.39 \n",
      "qr_reverse q \n",
      "& 82.24 & 63.37 & 78.96 & 61.44 & 78.69 & 59.38 \n",
      "qr_reverse r \n",
      "& 82.27  & 63.51 & 79.02  & 61.61 & 78.76  & 59.65 \n"
     ]
    }
   ],
   "source": [
    "#for loss in loss_to_eval_options:\n",
    "\n",
    "for loss in ['qr_forward','qr_reverse']:\n",
    "    print(loss + \" q \")\n",
    "    \n",
    "    result_str = \"\"\n",
    "    for state in states_in_reporting_order:\n",
    "    \n",
    "        results_q = results_by_state_q[loss][state]\n",
    "        \n",
    "        result_str += f\"& {np.round(results_q['acc_aggregated']*100,2):.02f} \"\n",
    "        result_str += f\"& {np.round(np.mean(results_q['ious_aggregated'][0])*100,2):.02f} \"\n",
    "        \n",
    "    print(result_str)\n",
    "    \n",
    "    print(loss + \" r \")\n",
    "    \n",
    "    result_str = \"\"\n",
    "    for state in states_in_reporting_order:\n",
    "    \n",
    "        results_r = results_by_state_r[loss][state]\n",
    "        \n",
    "        result_str += f\"& {np.round(results_r['acc_aggregated']*100,2):.02f}  \"\n",
    "        result_str += f\"& {np.round(np.mean(results_r['ious_aggregated'][0])*100,2):.02f} \"\n",
    "        \n",
    "    print(result_str)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "4b94e58c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "' 0.8730694345017398  0.7879995803554547  0.786142681636133  0.7030621633237482 '"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result_str"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f78c25f8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# results_by_state_q[state][loss] = {'accs': accs_q,\n",
    "#                                        'ious': ious_q,\n",
    "#                                        'num_pix':num_pix,\n",
    "#                                        'ious_aggregated': ious_aggregated_q,\n",
    "#                                        'acc_aggregated':acc_aggregated_q}\n",
    "        \n",
    "        \n",
    "#         results_by_state_r[state][loss] = {'accs': accs_r,\n",
    "#                                        'ious': ious_r,\n",
    "#                                        'num_pix':num_pix,\n",
    "#                                        'ious_aggregated': ious_aggregated_r,\n",
    "#                                        'acc_aggregated':acc_aggregated_r}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f3630434",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "id": "8a12aaad",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(6386, 5261)"
      ]
     },
     "execution_count": 94,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "preds_this_soft.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "7a624c60",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.6723449723188379\n",
      "0.8802201336957464\n"
     ]
    }
   ],
   "source": [
    "accs_by_state = []\n",
    "ious_by_state = []\n",
    "num_pix_by_state = []\n",
    "\n",
    "for state in states:\n",
    "    accs_by_state.append(results_by_state[state]['accs'])\n",
    "    ious_by_state.append(results_by_state[state]['ious'])\n",
    "    num_pix_by_state.append(results_by_state[state]['num_pix'])\n",
    "    \n",
    "ious_both_states = np.vstack(ious_by_state)#.ravel()\n",
    "\n",
    "accs_both_states = np.array(accs_by_state).ravel()\n",
    "num_pix_both_states = np.array(num_pix_by_state).ravel()\n",
    "\n",
    "ious_both_states_aggregated = util.aggregate_ious([x[1] for x in ious_both_states], [x[2] for x in ious_both_states])\n",
    "acc_both_states_aggregated = (accs_both_states * num_pix_both_states).sum() / np.sum(num_pix_both_states)\n",
    "\n",
    "results_by_state['states_combined'] = {'accs':accs_both_states,\n",
    "                                       'ious': ious_both_states,\n",
    "                                       'num_pix': num_pix_both_states,\n",
    "                                       'ious_aggregated': ious_both_states_aggregated,\n",
    "                                       'acc_aggregated':acc_both_states_aggregated\n",
    "                                      }\n",
    "\n",
    "print(np.mean(results_by_state['states_combined']['ious_aggregated'][0]))\n",
    "print(np.mean(results_by_state['states_combined']['acc_aggregated']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4c9b74b1",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 236,
   "id": "7e43c73c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7012840321365993"
      ]
     },
     "execution_count": 236,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.mean(results_by_state['states_combined']['ious_aggregated'][0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 237,
   "id": "5d436cb2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8614745704451643"
      ]
     },
     "execution_count": 237,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.mean(results_by_state['states_combined']['acc_aggregated'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "id": "c5d6cd6d",
   "metadata": {},
   "outputs": [],
   "source": [
    "prior_name = 'prior_from_cooccurrences_0_0_no_osm_no_buildings'\n",
    "\n",
    "prior_this = rasterio.open(fn_this.replace('lc.tif',f'{prior_name}.tif')).read()\n",
    "prior = (prior_this / 255. + 1e-4) / (prior_this / 255. + 1e-4).sum(axis=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "id": "e3c6cf0c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(4, 6509, 4830)"
      ]
     },
     "execution_count": 146,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prior_this.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "7fc60a23",
   "metadata": {},
   "outputs": [
    {
     "ename": "RasterioIOError",
     "evalue": "/home/esther/torchgeo_data/cvpr_chesapeake_landcover/ny_1m_2013_extended-debuffered-test_tiles/m_4207554_nw_18_1_lc_last.tif: No such file or directory",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mCPLE_OpenFailedError\u001b[0m                      Traceback (most recent call last)",
      "\u001b[0;32mrasterio/_base.pyx\u001b[0m in \u001b[0;36mrasterio._base.DatasetBase.__init__\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;32mrasterio/_shim.pyx\u001b[0m in \u001b[0;36mrasterio._shim.open_dataset\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;32mrasterio/_err.pyx\u001b[0m in \u001b[0;36mrasterio._err.exc_wrap_pointer\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;31mCPLE_OpenFailedError\u001b[0m: /home/esther/torchgeo_data/cvpr_chesapeake_landcover/ny_1m_2013_extended-debuffered-test_tiles/m_4207554_nw_18_1_lc_last.tif: No such file or directory",
      "\nDuring handling of the above exception, another exception occurred:\n",
      "\u001b[0;31mRasterioIOError\u001b[0m                           Traceback (most recent call last)",
      "\u001b[0;32m/tmp/ipykernel_26663/2956609048.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0mpreds_this\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mrasterio\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mopen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0moutput_fns\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mread\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mhr_lc_this\u001b[0m  \u001b[0;34m=\u001b[0m \u001b[0mrasterio\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mopen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0moutput_fns\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mreplace\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'qr_forward_pred'\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m'lc'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mread\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      3\u001b[0m \u001b[0mimg_this\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mrasterio\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mopen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0moutput_fns\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mreplace\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'qr_forward_pred'\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m'naip-new'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mread\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0mcc4_lc_reformatted\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcc7_to_cc4\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mhr_lc_this\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/anaconda/envs/torchgeo/lib/python3.9/site-packages/rasterio/env.py\u001b[0m in \u001b[0;36mwrapper\u001b[0;34m(*args, **kwds)\u001b[0m\n\u001b[1;32m    435\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    436\u001b[0m         \u001b[0;32mwith\u001b[0m \u001b[0menv_ctor\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msession\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0msession\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 437\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mf\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    438\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    439\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0mwrapper\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/anaconda/envs/torchgeo/lib/python3.9/site-packages/rasterio/__init__.py\u001b[0m in \u001b[0;36mopen\u001b[0;34m(fp, mode, driver, width, height, count, crs, transform, dtype, nodata, sharing, **kwargs)\u001b[0m\n\u001b[1;32m    218\u001b[0m         \u001b[0;31m# None.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    219\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mmode\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m'r'\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 220\u001b[0;31m             \u001b[0ms\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mDatasetReader\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpath\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdriver\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdriver\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0msharing\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0msharing\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    221\u001b[0m         \u001b[0;32melif\u001b[0m \u001b[0mmode\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m\"r+\"\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    222\u001b[0m             s = get_writer_for_path(path, driver=driver)(\n",
      "\u001b[0;32mrasterio/_base.pyx\u001b[0m in \u001b[0;36mrasterio._base.DatasetBase.__init__\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;31mRasterioIOError\u001b[0m: /home/esther/torchgeo_data/cvpr_chesapeake_landcover/ny_1m_2013_extended-debuffered-test_tiles/m_4207554_nw_18_1_lc_last.tif: No such file or directory"
     ]
    }
   ],
   "source": [
    "preds_this = rasterio.open(output_fns[2]).read()\n",
    "hr_lc_this  = rasterio.open(output_fns[2].replace('qr_forward_pred','lc')).read()[0]\n",
    "img_this = rasterio.open(output_fns[2].replace('qr_forward_pred','naip-new')).read()\n",
    "\n",
    "cc4_lc_reformatted = cc7_to_cc4[hr_lc_this] \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 226,
   "id": "2af7b241",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(4, 6405, 4742)"
      ]
     },
     "execution_count": 226,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "img_this.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 220,
   "id": "c4381da4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8673200535615924"
      ]
     },
     "execution_count": 220,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(cc4_lc_reformatted == preds_this.argmax(0)).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "35220b02",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'img_this' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m/tmp/ipykernel_26663/1820687404.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0mw1\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mw2\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m1000\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m3000\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mfig\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0max\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msubplots\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m3\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mfigsize\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m30\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m20\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 4\u001b[0;31m \u001b[0max\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mimshow\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mimg_this\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;36m3\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mh1\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0mh2\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mw1\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0mw2\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mT\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mswapaxes\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      5\u001b[0m \u001b[0max\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mimshow\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlc\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvis_lc\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcc4_lc_reformatted\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mh1\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0mh2\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mw1\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0mw2\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'chesapeake_4_no_zeros'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mT\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mswapaxes\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m \u001b[0max\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mimshow\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlc\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvis_lc\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpreds_this\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mh1\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0mh2\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mw1\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0mw2\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m/\u001b[0m \u001b[0;36m255.\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'chesapeake_4_no_zeros'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mT\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mswapaxes\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'img_this' is not defined"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABrcAAARiCAYAAAAOZ6xTAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAAA1/UlEQVR4nOzdX6jneX3f8dc7u11oTBqlToLdXcm2rJq90KITI6VpTUPrrr1YAl6oIVIJLFI35FIpNL3ITXNRCKLJssgi3nQvGkk2ZdOlUBIL1nZnwX+rKNOVutMVXGOwoNBl9dOLc1pOx1nnO2fPmfm+5jwe8IP5/X5fZz7Ml53zwuf8zsxaKwAAAAAAANDgJ270AQAAAAAAAGArcQsAAAAAAIAa4hYAAAAAAAA1xC0AAAAAAABqiFsAAAAAAADUELcAAAAAAACocdW4NTOPzMy3ZuZLL/H+zMxHZubizHxhZt588scEAOhgOwEAbGc7AQDHseWTW59Icu+Pef++JHcfPh5I8ocv/1gAALU+EdsJAGCrT8R2AgCu0VXj1lrr00m+82MuuT/JJ9eBzyZ55cy85qQOCADQxHYCANjOdgIAjuMk/s2t25M8e+T5pcPXAAD4UbYTAMB2thMA8CNuPYGfY67w2rrihTMP5OAj5HnFK17xlje84Q0n8MsDAKfpqaee+vZa69yNPsdNxHYCgJuU3XQqbCcAuEm9nO10EnHrUpI7jzy/I8lzV7pwrfVwkoeT5Pz58+vChQsn8MsDAKdpZv7HjT7DTcZ2AoCblN10KmwnALhJvZztdBLflvCxJO+bA29L8t211jdP4OcFALgZ2U4AANvZTgDAj7jqJ7dm5t8meXuSV8/MpST/KslfS5K11kNJHk/yziQXk3w/yftP67AAAHtnOwEAbGc7AQDHcdW4tdZ6z1XeX0k+eGInAgAoZjsBAGxnOwEAx3ES35YQAAAAAAAArgtxCwAAAAAAgBriFgAAAAAAADXELQAAAAAAAGqIWwAAAAAAANQQtwAAAAAAAKghbgEAAAAAAFBD3AIAAAAAAKCGuAUAAAAAAEANcQsAAAAAAIAa4hYAAAAAAAA1xC0AAAAAAABqiFsAAAAAAADUELcAAAAAAACoIW4BAAAAAABQQ9wCAAAAAACghrgFAAAAAABADXELAAAAAACAGuIWAAAAAAAANcQtAAAAAAAAaohbAAAAAAAA1BC3AAAAAAAAqCFuAQAAAAAAUEPcAgAAAAAAoIa4BQAAAAAAQA1xCwAAAAAAgBriFgAAAAAAADXELQAAAAAAAGqIWwAAAAAAANQQtwAAAAAAAKghbgEAAAAAAFBD3AIAAAAAAKCGuAUAAAAAAEANcQsAAAAAAIAa4hYAAAAAAAA1xC0AAAAAAABqiFsAAAAAAADUELcAAAAAAACoIW4BAAAAAABQQ9wCAAAAAACghrgFAAAAAABADXELAAAAAACAGuIWAAAAAAAANcQtAAAAAAAAaohbAAAAAAAA1BC3AAAAAAAAqCFuAQAAAAAAUEPcAgAAAAAAoIa4BQAAAAAAQA1xCwAAAAAAgBriFgAAAAAAADXELQAAAAAAAGqIWwAAAAAAANQQtwAAAAAAAKghbgEAAAAAAFBD3AIAAAAAAKCGuAUAAAAAAEANcQsAAAAAAIAa4hYAAAAAAAA1xC0AAAAAAABqiFsAAAAAAADUELcAAAAAAACoIW4BAAAAAABQQ9wCAAAAAACghrgFAAAAAABADXELAAAAAACAGuIWAAAAAAAANcQtAAAAAAAAaohbAAAAAAAA1BC3AAAAAAAAqCFuAQAAAAAAUEPcAgAAAAAAoIa4BQAAAAAAQA1xCwAAAAAAgBriFgAAAAAAADXELQAAAAAAAGqIWwAAAAAAANQQtwAAAAAAAKghbgEAAAAAAFBD3AIAAAAAAKCGuAUAAAAAAEANcQsAAAAAAIAa4hYAAAAAAAA1xC0AAAAAAABqiFsAAAAAAADUELcAAAAAAACoIW4BAAAAAABQQ9wCAAAAAACghrgFAAAAAABADXELAAAAAACAGuIWAAAAAAAANcQtAAAAAAAAaohbAAAAAAAA1BC3AAAAAAAAqCFuAQAAAAAAUEPcAgAAAAAAoIa4BQAAAAAAQA1xCwAAAAAAgBriFgAAAAAAADXELQAAAAAAAGqIWwAAAAAAANQQtwAAAAAAAKghbgEAAAAAAFBD3AIAAAAAAKCGuAUAAAAAAEANcQsAAAAAAIAa4hYAAAAAAAA1xC0AAAAAAABqiFsAAAAAAADUELcAAAAAAACoIW4BAAAAAABQQ9wCAAAAAACghrgFAAAAAABADXELAAAAAACAGuIWAAAAAAAANcQtAAAAAAAAaohbAAAAAAAA1BC3AAAAAAAAqCFuAQAA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AAAAaohbAAAAAAAA1BC3AAAAAAAAqCFuAQAAAAAAUEPcAgAAAAAAoIa4BQAAAAAAQA1xCwAAAAAAgBriFgAAAAAAADXELQAAAAAAAGqIWwAAAAAAANQQtwAAAAAAAKghbgEAAAAAAFBD3AIAAAAAAKCGuAUAAAAAAEANcQsAAAAAAIAa4hYAAAAAAAA1xC0AAAAAAABqiFsAAAAAAADUELcAAAAAAACoIW4BAAAAAABQQ9wCAAAAAACghrgFAAAAAABADXELAAAAAACAGuIWAAAAAAAANcQtAAAAAAAAaohbAAAAAAAA1BC3AAAAAAAAqCFuAQAAAAAAUEPcAgAAAAAAoIa4BQAAAAAAQA1xCwAAAAAAgBriFgAAAAAAADXELQAAAAAAAGqIWwAAAAAAANQQtwAAAAAAAKghbgEAAAAAAFBD3AIAAAAAAKCGuAUAAAAAAEANcQsAAAAAAIAa4hYAAAAAAAA1xC0AAAAAAABqiFsAAAAAAADUELcAAAAAAACoIW4BAAAAAABQQ9wCAAAAAACghrgFAAAAAABADXELAAAAAACAGuIWAAAAAAAANcQtAAAAAAAAaohbAAAAAAAA1BC3AAAAAAAAqCFuAQAAAAAAUEPcAgAAAAAAoIa4BQAAAAAAQA1xCwAAAAAAgBriFgAAAAAAADXELQAAAAAAAGqIWwAAAAAAANQQtwAAAAAAAKghbgEAAAAAAFBD3AIAAAAAAKCGuAUAAAAAAEANcQsAAAAAAIAa4hYAAAAAAAA1xC0AAAAAAABqiFsAAAAAAADUELcAAAAAAACoIW4BAAAAAABQQ9wCAAAAAACghrgFAAAAAABADXELAAAAAACAGuIWAAAAAAAANcQtAAAAAAAAaohbAAAAAAAA1BC3AAAAAAAAqCFuAQAAAAAAUEPcAgAAAAAAoIa4BQAAAAAAQA1xCwAAAAAAgBriFgAAAAAAADXELQAAAAAAAGqIWwAAAAAAANQQtwAAAAAAAKghbgEAAAAAAFBD3AIAAAAAAKCGuAUAAAAAAEANcQsAAAAAAIAa4hYAAAAAAAA1xC0AAAAAAABqiFsAAAAAAADUELcAAAAAAACoIW4BAAAAAABQQ9wCAAAAAACghrgFAAAAAABADXELAAAAAACAGuIWAAAAAAAANcQtAAAAAAAAaohbAAAAAAAA1BC3AAAAAAAAqCFu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AAAAgBriFgAAAAAAADXELQAAAAAAAGqIWwAAAAAAANQQtwAAAAAAAKghbgEAAAAAAFBD3AIAAAAAAKCGuAUAAAAAAEANcQsAAAAAAIAa4hYAAAAAAAA1xC0AAAAAAABqiFsAAAAAAADUELcAAAAAAACoIW4BAAAAAABQQ9wCAAAAAACghrgFAAAAAABADXELAAAAAACAGuIWAAAAAAAANcQtAAAAAAAAaohbAAAAAAAA1BC3AAAAAAAAqCFuAQAAAAAAUEPcAgAAAAAAoIa4BQAAAAAAQA1xCwAAAAAAgBriFgAAAAAAADXELQAAAAAAAGqIWwAAAAAAANQQtwAAAAAAAKghbgEAAAAAAFBD3AIAAAAAAKCGuAUAAAAAAEANcQsAAAAAAIAa4hYAAAAAAAA1xC0AAAAAAABqiFsAAAAAAADUELcAAAAAAACoIW4BAAAAAABQQ9wCAAAAAACghrgFAAAAAABADXELAAAAAACAGuIWAAAAAAAANcQtAAAAAAAAaohbAAAAAAAA1BC3AAAAAAAAqCFuAQAAAAAAUEPcAgAAAAAAoIa4BQAAAAAAQA1xCwAAAAAAgBriFgAAAAAAADXELQAAAAAAAGqIWwAAAAAAANQQtwAAAAAAAKghbgEAAAAAAFBD3AIAAAAAAKCGuAUAAAAAAEANcQsAAAAAAIAa4hYAAAAAAAA1xC0AAAAAAABqiFsAAAAAAADUELcAAAAAAACoIW4BAAAAAABQQ9wCAAAAAACghrgFAAAAAABADXELAAAAAACAGuIWAAAAAAAANcQtAAAAAAAAaohbAAAAAAAA1BC3AAAAAAAAqCFuAQAAAAAAUEPcAgAAAAAAoIa4BQAAAAAAQA1xCwAAAAAAgBriFgAAAAAAADXELQAAAAAAAGqIWwAAAAAAANQQtwAAAAAAAKghbgEAAAAAAFBD3AIAAAAAAKCGuAUAAAAAAEANcQsAAAAAAIAa4hYAAAAAAAA1xC0AAAAAAABqiFsAAAAAAADUELcAAAAAAACoIW4BAAAAAABQQ9wCAAAAAACghrgFAAAAAABADXELAAAAAACAGuIWAAAAAAAANcQtAAAAAAAAaohbAAAAAAAA1BC3AAAAAAAAqCFuAQAAAAAAUEPcAgAAAAAAoIa4BQAAAAAAQA1xCwAAAAAAgBriFgAAAAAAADXELQAAAAAAAGqIWwAAAAAAANQQtwAAAAAAAKghbgEAAAAAAFBD3AIAAAAAAKCGuAUAAAAAAEANcQsAAAAAAIAa4hYAAAAAAAA1xC0AAAAAAABqiFsAAAAAAADUELcAAAAAAACoIW4BAAAAAABQQ9wCAAAAAACghrgFAAAAAABADXELAAAAAACAGuIWAAAAAAAANcQtAAAAAAAAaohbAAAAAAAA1BC3AAAAAAAAqLEpbs3MvTPz1Zm5ODMf/jHX/eLM/GBm3nVyRwQA6GI7AQBsYzcBAMdx1bg1M7ck+ViS+5Lck+Q9M3PPS1z3e0meOOlDAgC0sJ0AALaxmwCA49ryya23Jrm41npmrfVCkkeT3H+F634ryR8l+dYJng8AoI3tBACwjd0EABzLlrh1e5Jnjzy/dPja/zMztyf5tSQP/bifaGYemJkLM3Ph+eefv9azAgA0sJ0AALY5sd10eK3tBABnxJa4NVd4bV32/PeTfGit9YMf9xOttR5ea51fa50/d+7cxiMCAFSxnQAAtjmx3ZTYTgBwlty64ZpLSe488vyOJM9dds35JI/OTJK8Osk7Z+bFtdYfn8QhAQCK2E4AANvYTQDAsWyJW08muXtm7kryP5O8O8l7j16w1rrr//54Zj6R5N8bGQDAGWU7AQBsYzcBAMdy1bi11npxZh5M8kSSW5I8stZ6emY+cPj+Vb/nMQDAWWE7AQBsYzcBAMe15ZNbWWs9nuTxy1674sBYa/2zl38sAIBethMAwDZ2EwBwHD9xow8AAAAAAAAAW4lbAAAAAAAA1BC3AAAAAAAAqCFuAQAAAAAAUEPcAgAAAAAAoIa4BQAAAAAAQA1xCwAAAAAAgBriFgAAAAAAADXELQAAAAAAAGqIWwAAAAAAANQQtwAAAAAAAKghbgEAAAAAAFBD3AIAAAAAAKCGuAUAAAAAAEANcQsAAAAAAIAa4hYAAAAAAAA1xC0AAAAAAABqiFsAAAAAAADUELcAAAAAAACoIW4BAAAAAABQQ9wCAAAAAACghrgFAAAAAABADXELAAAAAACAGuIWAAAAAAAANcQtAAAAAAAAaohbAAAAAAAA1BC3AAAAAAAAqCFuAQAAAAAAUEPcAgAAAAAAoIa4BQAAAAAAQA1xCwAAAAAAgBriFgAAAAAAADXELQAAAAAAAGqIWwAAAAAAANQQtwAAAAAAAKghbgEAAAAAAFBD3AIAAAAAAKCGuAUAAAAAAEANcQsAAAAAAIAa4hYAAAAAAAA1xC0AAAAAAABqiFsAAAAAAADUELcAAAAAAACoIW4BAAAAAABQQ9wCAAAAAACghrgFAAAAAABADXELAAAAAACAGuIWAAAAAAAANcQtAAAAAAAAaohbAAAAAAAA1BC3AAAAAAAAqCFuAQAAAAAAUEPcAgAAAAAAoIa4BQAAAAAAQA1xCwAAAAAAgBriFgAAAAAAADXELQAAAAAAAGqIWwAAAAAAANQQtwAAAAAAAKghbgEAAAAAAFBD3AIAAAAAAKCGuAUAAAAAAEANcQsAAAAAAIAa4hYAAAAAAAA1xC0AAAAAAABqiFsAAAAAAADUELcAAAAAAACoIW4BAAAAAABQQ9wCAAAAAACghrgFAAAAAABADXELAAAAAACAGuIWAAAAAAAANcQtAAAAAAAAaohbAAAAAAAA1BC3AAAAAAAAqCFuAQAAAAAAUEPcAgAAAAAAoIa4BQAAAAAAQA1xCwAAAAAAgBriFgAAAAAAADXELQAAAAAAAGqIWwAAAAAAANQQtwAAAAAAAKghbgEAAAAAAFBD3AIAAAAAAKCGuAUAAAAAAEANcQsAAAAAAIAa4hYAAAAAAAA1xC0AAAAAAABqiFsAAAAAAADUELcAAAAAAACoIW4BAAAAAABQQ9wCAAAAAACghrgFAAAAAABADXELAAAAAACAGuIWAAAAAAAANcQtAAAAAAAAaohbAAAAAAAA1BC3AAAAAAAAqCFuAQAAAAAAUEPcAgAAAAAAoIa4BQAAAAAAQA1xCwAAAAAAgBriFgAAAAAAADXELQAAAAAAAGqIWwAAAAAAANQQtwAAAAAAAKghbgEAAAAAAFBD3AIAAAAAAKCGuAUAAAAAAEANcQsAAAAAAIAa4hYAAAAAAAA1xC0AAAAAAABqiFsAAAAAAADUELcAAAAAAACoIW4BAAAAAABQQ9wCAAAAAACghrgFAAAAAABADXELAAAAAACAGuIWAAAAAAAANcQtAAAAAAAAaohbAAAAAAAA1BC3AAAAAAAAqCFuAQAAAAAAUEPcAgAAAAAAoIa4BQAAAAAAQA1xCwAAAAAAgBriFgAAAAAAADXELQAAAAAAAGqIWwAAAAAAANQQtwAAAAAAAKghbgEAAAAAAFBD3AIAAAAAAKCGuAUAAAAAAEANcQsAAAAA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hrgFAAAAAABADXELAAAAAACAGuIWAAAAAAAANcQtAAAAAAAAaohbAAAAAAAA1BC3AAAAAAAAqCFuAQAAAAAAUEPcAgAAAAAAoIa4BQAAAAAAQA1xCwAAAAAAgBriFgAAAAAAADXELQAAAAAAAGqIWwAAAAAAANQQtwAAAAAAAKghbgEAAAAAAFBD3AIAAAAAAKCGuAUAAAAAAEANcQsAAAAAAIAa4hYAAAAAAAA1xC0AAAAAAABqiFsAAAAAAADUELcAAAAAAACoIW4BAAAAAABQQ9wCAAAAAACghrgFAAAAAABADXELAAAAAACAGuIWAAAAAAAANcQtAAAAAAAAaohbAAAAAAAA1BC3AAAAAAAAqCFuAQAAAAAAUEPcAgAAAAAAoIa4BQAAAAAAQA1xCwAAAAAAgBriFgAAAAAAADXELQAAAAAAAGqIWwAAAAAAANQQtwAAAAAAAKghbgEAAAAAAFBD3AIAAAAAAKCGuAUAAAAAAEANcQsAAAAAAIAa4hYAAAAAAAA1xC0AAAAAAABqiFsAAAAAAADUELcAAAAAAACoIW4BAAAAAABQQ9wCAAAAAACghrgFAAAAAABADXELAAAAAACAGuIWAAAAAAAANcQtAAAAAAAAaohbAAAAAAAA1BC3AAAAAAAAqCFuAQAAAAAAUEPcAgAAAAAAoIa4BQAAAAAAQA1xCwAAAAAAgBriFgAAAAAAADXELQAAAAAAAGqIWwAAAAAAANQQtwAAAAAAAKghbgEAAAAAAFBD3AIAAAAAAKCGuAUAAAAAAEANcQsAAAAAAIAa4hYAAAAAAAA1xC0AAAAAAABqiFsAAAAAAADUELcAAAAAAACoIW4BAAAAAABQQ9wCAAAAAACghrgFAAAAAABADXELAAAAAACAGuIWAAAAAAAANcQtAAAAAAAAaohbAAAAAAAA1BC3AAAAAAAAqCFuAQAAAAAAUEPcAgAAAAAAoIa4BQAAAAAAQA1xCwAAAAAAgBriFgAAAAAAADXELQAAAAAAAGqIWwAAAAAAANQQtwAAAAAAAKghbgEAAAAAAFBD3AIAAAAAAKCGuAUAAAAAAEANcQsAAAAAAIAa4hYAAAAAAAA1xC0AAAAAAABqiFsAAAAAAADUELcAAAAAAACoIW4BAAAAAABQQ9wCAAAAAACghrgFAAAAAABADXELAAAAAACAGuIWAAAA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\n",
      "text/plain": [
       "<Figure size 2160x1440 with 3 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "h1, h2 = 1000,3000\n",
    "w1, w2 = 1000,3000\n",
    "fig, ax = plt.subplots(1,3,figsize=(30,20))\n",
    "ax[2].imshow(img_this[:3,h1:h2,w1:w2].T.swapaxes(0,1))\n",
    "ax[0].imshow(lc.vis_lc(cc4_lc_reformatted[h1:h2,w1:w2], 'chesapeake_4_no_zeros').T.swapaxes(0,1))\n",
    "ax[1].imshow(lc.vis_lc(preds_this[:,h1:h2,w1:w2] / 255., 'chesapeake_4_no_zeros').T.swapaxes(0,1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 197,
   "id": "5d23a27a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(6509, 4830)"
      ]
     },
     "execution_count": 197,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "preds_this.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 207,
   "id": "1a01f6e5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "20\n",
      "5.22 seconds\n",
      "5.77 seconds\n",
      "5.29 seconds\n",
      "5.22 seconds\n",
      "5.14 seconds\n",
      "5.39 seconds\n",
      "5.32 seconds\n",
      "5.22 seconds\n",
      "5.15 seconds\n",
      "4.85 seconds\n",
      "4.92 seconds\n",
      "4.93 seconds\n",
      "5.01 seconds\n",
      "5.20 seconds\n",
      "5.15 seconds\n",
      "5.14 seconds\n",
      "5.25 seconds\n",
      "5.26 seconds\n",
      "5.25 seconds\n",
      "5.25 seconds\n",
      "For pa test set:\n",
      "IoU (averaged per class) over the tiles is: \n",
      "0.6981163200141631\n",
      "IoU per class over the tiles is: \n",
      "[0.6843255369425629, 0.830881035590918, 0.6982093573192907, 0.579049350203881]\n",
      "accuracy over the tiles is: \n",
      "0.8680345890590898\n"
     ]
    }
   ],
   "source": [
    "reload(util)\n",
    "states = ['pa']#, 'ny']\n",
    "year = 2013\n",
    "set_this = 'test'\n",
    "\n",
    "prior_name = 'prior_from_cooccurrences_0_0_no_osm_no_buildings'\n",
    "\n",
    "save_2019_map_layer = True\n",
    "\n",
    "results_r_by_state = {}\n",
    "for state in states:\n",
    "    \n",
    "    data_dir = '/home/esther/torchgeo_data/cvpr_chesapeake_landcover'\n",
    "    dir_this_set = os.path.join(data_dir,f'{state}_1m_{year}_extended-debuffered-{set_this}_tiles')\n",
    "    \n",
    "    fns = os.listdir(dir_this_set)\n",
    "    tile_ids = np.unique([x[2:17] for x in fns])\n",
    "    print(len(tile_ids))\n",
    "\n",
    "    accs = []\n",
    "    ious = []\n",
    "    num_pix = []\n",
    "\n",
    "    for tile_id in tile_ids:\n",
    "        fn_this = os.path.join(dir_this_set, f'm_{tile_id}_lc.tif')\n",
    "        t1 = time.time()\n",
    "\n",
    "        # gather the data\n",
    "        with rasterio.open(fn_this) as f:\n",
    "            cc7_lc = f.read()[0]\n",
    "            \n",
    "        preds_this = rasterio.open(fn_this.replace('lc.tif','qr_forward_pred_last.tif')).read() / 255.\n",
    "        prior_this = rasterio.open(fn_this.replace('lc.tif',f'{prior_name}.tif')).read()\n",
    "\n",
    "        prior = (prior_this / 255. + 1e-4) / (prior_this / 255. + 1e-4).sum(axis=0)\n",
    "        \n",
    "        z = (preds_this.T / preds_this.sum(axis=(1,2)) ).T\n",
    "        preds_r = prior*z\n",
    "     #   preds_r = (preds_r.T / preds_r.sum(axis=0) ).T.shape\n",
    "        preds_r = preds_r.argmax(axis=0)\n",
    "        # realign the classes - lc uses just for classes for this task\n",
    "        cc4_lc_reformatted = cc7_to_cc4[cc7_lc] \n",
    "\n",
    "        acc_this = (preds_r == cc4_lc_reformatted).mean()\n",
    "\n",
    "        # ignore 0\n",
    "        iou_this = util.per_class_iou(cc4_lc_reformatted, preds_r, np.arange(0,4))\n",
    "        accs.append(acc_this)\n",
    "        ious.append(iou_this)\n",
    "        num_pix.append(cc4_lc_reformatted.shape[0]*cc4_lc_reformatted.shape[1])\n",
    "\n",
    "        t2 = time.time()\n",
    "        print(f'{t2-t1:.2f} seconds')\n",
    "        \n",
    "    ious_aggregated = util.aggregate_ious([x[1] for x in ious], [x[2] for x in ious])\n",
    "\n",
    "    print(f'For {state} {set_this} set:')\n",
    "    print('IoU (averaged per class) over the tiles is: ')\n",
    "    print(np.mean(ious_aggregated[0]))\n",
    "    print('IoU per class over the tiles is: ')\n",
    "    print(ious_aggregated[0])\n",
    "    acc_aggregated = (np.array(accs) * np.array(num_pix)).sum() / np.sum(num_pix)\n",
    "    print('accuracy over the tiles is: ')\n",
    "    print(acc_aggregated)\n",
    "\n",
    "\n",
    "    results_r_by_state[state] = {'accs': accs,\n",
    "                                   'ious': ious,\n",
    "                                   'num_pix':num_pix,\n",
    "                                   'ious_aggregated': ious_aggregated,\n",
    "                                   'acc_aggregated':acc_aggregated}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 212,
   "id": "ddbc3802",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f832b890",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "torchgeo",
   "language": "python",
   "name": "conda-env-torchgeo-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
