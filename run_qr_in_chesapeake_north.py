"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Runs the train script with a grid of hyperparameters.
"""
import itertools
import subprocess
from multiprocessing import Process, Queue

# list of GPU IDs that we want to use, one job will be started for every ID in the list
GPUS = [0,1,2]#,3]
TEST_MODE = False  # if False then print out the commands to be run, if True then run

# Hyperparameter options
training_set_options = ["ny", "pa","ny+pa"]
model_options = ['fcn']#, 'unet']
lr_options = [1e-4]

loss_options = ['qr_forward']
prior_version_options = ['from_cooccurrences_0_0_no_osm_no_buildings']
additive_smooth_options = [1e-4]

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

    for (states_str, model, lr,loss, prior_version, additive_smooth) in itertools.product(
        training_set_options,
        model_options,
        lr_options,
        loss_options,
        prior_version_options,
        additive_smooth_options,
    ):
        experiment_name = f"{states_str}_{model}_{lr}_{loss}_{prior_version}_additive_smooth_{additive_smooth}"

        output_dir = "output/qr_forward_eval"

        command = (
            "python train.py program.overwrite=True config_file=conf/chesapeake_learn_on_prior.yml"
            + f" experiment.name={experiment_name}"
            + f" experiment.module.segmentation_model={model}"
            + f" experiment.module.learning_rate={lr}"
            + f" experiment.module.loss={loss}"
            + f" experiment.module.output_smooth={additive_smooth}"
            + f" experiment.datamodule.prior_version={prior_version}"
            + f" experiment.datamodule.prior_smoothing_constant={additive_smooth}"
            + f" experiment.datamodule.states_str={states_str}"
            + f" experiment.datamodule.train_set={train_set}"
            + f" experiment.datamodule.val_set={val_set}"
            + f" experiment.datamodule.test_set={test_set}"
            + f" program.output_dir={output_dir}"
            + f" program.log_dir=logs/qr_forward_eval"
            + " program.data_dir=/home/esther/torchgeo_data/cvpr_chesapeake_landcover"
            + " trainer.gpus='GPU'"
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