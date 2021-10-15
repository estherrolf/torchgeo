"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Runs the train script with a grid of hyperparameters.
"""
import itertools
import subprocess
from multiprocessing import Process, Queue

# list of GPU IDs that we want to use, one job will be started for every ID in the list
GPUS = [0,1,2] 
TEST_MODE = False  # if False then print out the commands to be run, if True then run

# Hyperparameter options
training_set_options = [#'phoenix_az-2010_1m', 
                        'durham_nc-2012_1m', 
                        'austin_tx-2012_1m',
                        'pittsburgh_pa-2010_1m'
                       ]

model_options = ["fcn_larger"]
lr_options = [1e-3, 1e-4, 1e-5]
loss_options = ["nll"]

additive_smooth_options = [1e-8]

train_set, val_set, test_set = ["val5", "val5", "val5"]

nlcd_blur_kernelsize = 101
nlcd_blur_sigma_options = [31]

def do_work(work, gpu_idx):
    while not work.empty():
        experiment = work.get()
        experiment = experiment.replace("GPU", str(gpu_idx))
        print(experiment)
        if not TEST_MODE:
            subprocess.call(experiment.split(" "))
    return True


def main():

    work = Queue()

    for (states_str, model, lr,loss, nlcd_blur_sigma, additive_smooth) in itertools.product(
        training_set_options,
        model_options,
        lr_options,
        loss_options,
        nlcd_blur_sigma_options,
        additive_smooth_options
    ):

        experiment_name = f"{states_str}_{model}_{lr}_{loss}_blur_sigma_{nlcd_blur_sigma}_learn_the_prior"

        output_dir = "output/learn_prior_ea_2/"

        command = (
            "python train.py program.overwrite=True config_file=conf/enviroatlas_learn_the_prior.yml"
            + f" experiment.name={experiment_name}"
            + f" experiment.module.segmentation_model={model}"
            + f" experiment.module.loss={loss}"
            + f" experiment.module.learning_rate={lr}"
            + f" experiment.module.num_filters=128"
            + f" experiment.datamodeul.batch_size=128"
            + f" experiment.datamodeul.patch_size=128"
            + f" experiment.module.output_smooth={additive_smooth}"
            + f" experiment.datamodule.nlcd_blur_kernelsize={nlcd_blur_kernelsize}"
            + f" experiment.datamodule.nlcd_blur_sigma={nlcd_blur_sigma}"
            + f" experiment.datamodule.states_str={states_str}"
            + f" experiment.datamodule.patches_per_tile=800"
            + f" experiment.datamodule.train_set={train_set}"
            + f" experiment.datamodule.val_set={val_set}"
            + f" experiment.datamodule.test_set={test_set}"
            + f" program.output_dir={output_dir}"
            + f" program.log_dir=logs/learn_prior_ea_2/learn_the_prior"
            + " trainer.gpus=[GPU]"
        )
        command = command.strip()

        work.put(command)

    processes = []
    for gpu_idx in GPUS:
        p = Process(target=do_work, args=(work, gpu_idx))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
