import rasterio
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
import sys
sys.path.append('/home/esther/lc-mapping/scripts')
import landcover_definitions as lc
import util
sys.path.append('/home/esther/torchgeo')
import run_model_forward_and_produce_tifs 
from importlib import reload
reload(run_model_forward_and_produce_tifs)


torchgeo_output_dir = '/home/esther/torchgeo/output'

states_to_eval = [#'phoenix_az-2010_1m',
                     # 'austin_tx-2012_1m',
                     # 'durham_nc-2012_1m', 
                      'pittsburgh_pa-2010_1m'
                       ]
                       
                       
loss_to_eval_options = ['nll']
prior_version = 'from_cooccurrences_101_31'

include_prior_as_datalayer=False
    

run_dirs = ['learn_prior_ea_2']
            

for run_dir in run_dirs:
    for states_str in states_to_eval:
        for loss in loss_to_eval_options:
            t1 = time.time()

            
            run_name = f"{states_str}_fcn_larger_0.0001_nll_blur_sigma_31_learn_the_prior"
            model_kwargs = {'output_smooth':1e-8, 'classes':5, 'num_filters': 128, 'in_channels':9}
            prior_version = 'prior_from_cooccurrences_101_31_no_osm_no_buildings'

            ckpt_name = 'last.ckpt'
            model_ckpt_fp = os.path.join(torchgeo_output_dir,run_dir,run_name, ckpt_name)
          #  data_dir_this_state = f'/home/esther/torchgeo_data/enviroatlas/{states_str}-val5_tiles-debuffered'
            data_dir_this_state = f'/home/esther/torchgeo_data/enviroatlas/{states_str}-val_tiles-debuffered'
            image_fns = [os.path.join(data_dir_this_state,x) for x in os.listdir(data_dir_this_state) if x.endswith(f'{prior_version}.tif')]

            # reorder the output names
            output_fns = [x.replace(f'{prior_version}.tif',f'prior_learned_101_31.tif') for x in image_fns]
            
            extra_fns = []
            for img_fn in image_fns:
                extra_fns_this_img = []
                for data_type in ["e_buildings", "c_roads", "d2_waterbodies", "d1_waterways"]:
                    extra_fns_this_img.append(img_fn.replace(f'{prior_version}.tif', f'{data_type}.tif'))
            
                extra_fns.append(extra_fns_this_img)


            print(model_ckpt_fp)
            
          #  print(image_fns)
            print([len(x) for x in extra_fns])

            # run through tifs and save the output
            run_model_forward_and_produce_tifs.run_through_tiles(model_ckpt_fp,
                                                                 image_fns[:],
                                                                 output_fns[:],
                                                                 evaluating_learned_prior=True,
                                                                 model='fcn-larger',
                                                                 gpu = 1,
                                                                 overwrite=True,
                                                                 model_kwargs=model_kwargs,
                                                                 include_prior_as_datalayer=True,
                                                                 prior_fns=extra_fns
                                                                 )

            t2 = time.time()
            print(f'{t2-t1} seconds for ten tiles')