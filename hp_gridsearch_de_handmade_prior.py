"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Runs the train script with a grid of hyperparameters.
"""
import itertools
import subprocess
from multiprocessing import Process, Queue

# list of GPU IDs that we want to use, one job will be started for every ID in the list
GPUS = [0,1,2,3]#, 1] 
TEST_MODE = False  # if False then print out the commands to be run, if True then run

# Hyperparameter options
training_set_options = ["de"]
model_options = ['fcn']#, 'unet']
lr_options = [1e-4,1e-5]

loss_options = ['qr_forward']
prior_version_options = [
                    #   'from_cooccurrences_101_15_no_osm_no_buildings',
                    #   'from_cooccurrences_101_31_no_osm_no_buildings',
                       'from_cooccurrences_101_15',
                       'from_cooccurrences_101_31',
                        ]

additive_smooth_options = [1e-2,1e-4,1e-8] 
prior_smooth_options = [1e-8]
train_set, val_set, test_set = ['test', 'test', 'test']

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

    for (states_str, model, lr,loss, prior_version, additive_smooth, prior_smooth) in itertools.product(
        training_set_options,
        model_options,
        lr_options,
        loss_options,
        prior_version_options,
        additive_smooth_options,
        prior_smooth_options
    ):
        experiment_name = f"{states_str}_{model}_{lr}_{loss}_{prior_version}_additive_smooth_{additive_smooth}"

        output_dir = "output/hp_gridsearch_de_handmade_prior"

        command = (
            "python train.py program.overwrite=True config_file=conf/chesapeake_learn_on_prior.yml"
            + f" experiment.name={experiment_name}"
            + f" experiment.module.segmentation_model={model}"
            + f" experiment.module.learning_rate={lr}"
            + f" experiment.module.loss={loss}"
            + f" experiment.module.output_smooth={additive_smooth}"
            + f" experiment.datamodule.prior_version={prior_version}"
            + f" experiment.datamodule.prior_smoothing_constant={prior_smooth}"
            + f" experiment.datamodule.states_str={states_str}"
            + f" experiment.datamodule.train_set={train_set}"
            + f" experiment.datamodule.val_set={val_set}"
            + f" experiment.datamodule.test_set={test_set}"
            + f" program.output_dir={output_dir}"
            + f" program.log_dir=logs/hp_gridsearch_de_handmade_prior"
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
