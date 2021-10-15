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
                       'austin_tx-2012_1m',
                       'durham_nc-2012_1m', 
                       'pittsburgh_pa-2010_1m'
                       ]
                       
                       
loss_to_eval_options = ['qr_forward']#, 'qr_reverse']
prior_version = 'from_cooccurrences_101_31'

include_prior_as_datalayer=False
    

run_dirs = ['ea_learned_prior']

# run_dirs = ['ea_from_pittsburgh_model',
#             'ea_from_scratch',
#             'hp_gridsearch_pittsburgh',
#             'hp_gridsearch_pittsburgh_with_prior_as_input']
            

for run_dir in run_dirs:
    for states_str in states_to_eval:
        for loss in loss_to_eval_options:
            t1 = time.time()

            if run_dir == 'ea_from_pittsburgh_model':
                run_name = f'pa_checkpoint_{states_str}_fcn_1e-05_{loss}_{prior_version}_additive_smooth_0.0001_prior_smooth_0.0001/'
                model_kwargs = {'output_smooth':1e-4, 'classes': 5, 'num_filters':128, 'in_channels': 4}

            elif run_dir == 'ea_from_scratch':
                if loss == 'qr_forward':
                    run_name = f'{states_str}_fcn_0.0001_{loss}_{prior_version}_additive_smooth_0.0001_prior_smooth_0.0001/'
                elif loss == 'qr_reverse':
                    run_name = f'{states_str}_fcn_0.001_{loss}_{prior_version}_additive_smooth_0.0001_prior_smooth_0.0001/'

                model_kwargs = {'output_smooth':1e-4, 'classes': 5, 'num_filters':128, 'in_channels': 4}

            elif run_dir == 'hp_gridsearch_pittsburgh':    
                run_name = 'pittsburgh_pa-2010_1m_fcn_0.001_nll/'
                model_kwargs = {'output_smooth':1e-8, 'classes': 5, 'num_filters':128, 'in_channels': 4}
            elif run_dir == 'hp_gridsearch_pittsburgh_with_prior_as_input':    
                run_name = 'pittsburgh_pa-2010_1m_fcn_0.001_nll_with_prior/'
                include_prior_as_datalayer=True
                prior_type = 'prior_from_cooccurrences_101_31'
                model_kwargs = {'output_smooth':1e-8, 'classes': 5, 'num_filters':128, 'in_channels': 9}
                
            elif run_dir == 'ea_learned_prior':
                prior_version = 'learned_101_31'
                run_name = f'pa_checkpoint_{states_str}_fcn_1e-05_{loss}_{prior_version}_additive_smooth_0.0001_prior_smooth_0.0001/'
                model_kwargs = {'output_smooth':1e-4, 'classes': 5, 'num_filters':128, 'in_channels': 4}

            ckpt_name = 'last.ckpt'
            model_ckpt_fp = os.path.join(torchgeo_output_dir,run_dir,run_name, ckpt_name)

            data_dir_this_state = f'/home/esther/torchgeo_data/enviroatlas/{states_str}-test_tiles-debuffered'
            image_fns = [os.path.join(data_dir_this_state,x) for x in os.listdir(data_dir_this_state) if x.endswith('a_naip.tif')]

            # reorder the output names
            output_fns = [x.replace('a_naip.tif',f'{loss}_pred_last.tif') for x in image_fns]
            output_fns = [x.replace(f'torchgeo_data/', f'torchgeo_predictions/{run_name}') for x in output_fns]

            # prior fns only matter if they're used as model input
            if include_prior_as_datalayer:
                 prior_fns = [x.replace('a_naip',f'prior_{prior_version}') for x in image_fns]
            else: 
                prior_fns = ['' for x in image_fns]

            # make all the output filepaths if they don't already exists
            if not os.path.exists(f'/home/esther/torchgeo_predictions'):
                os.mkdir(f'/home/esther/torchgeo_predictions')
                print(f'making dir /home/esther/torchgeo_predictions')
            if not os.path.exists(f'/home/esther/torchgeo_predictions/{run_name}'):
                os.mkdir(f'/home/esther/torchgeo_predictions/{run_name}')
                print(f'making dir /home/esther/torchgeo_predictions/{run_name}')
            if not os.path.exists(f'/home/esther/torchgeo_predictions/{run_name}/enviroatlas'):
                os.mkdir(f'/home/esther/torchgeo_predictions/{run_name}/enviroatlas')
                print(f'making dir /home/esther/torchgeo_predictions/{run_name}/enviroatlas')
            if not os.path.exists(f'/home/esther/torchgeo_predictions/{run_name}/enviroatlas/{states_str}-test_tiles-debuffered'):
                os.mkdir(f'/home/esther/torchgeo_predictions/{run_name}/enviroatlas/{states_str}-test_tiles-debuffered')
                print(f'making dir /home/esther/torchgeo_predictions/{run_name}/enviroatlas/{states_str}-test_tiles-debuffered')

            print(model_ckpt_fp)

            # run through tifs and save the output
            run_model_forward_and_produce_tifs.run_through_tiles(model_ckpt_fp,
                                                                  image_fns[:],
                                                                  output_fns[:],
                                                                  gpu = 1,
                                                                  overwrite=True,
                                                                 model_kwargs=model_kwargs,
                                                                 include_prior_as_datalayer=include_prior_as_datalayer,
                                                                 prior_fns=prior_fns
                                                                 )

            t2 = time.time()
            print(f'{t2-t1} seconds for ten tiles')