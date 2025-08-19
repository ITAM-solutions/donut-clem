# Preparing for fine-tuning

Follow this steps to make Donut work in a Vast.ai instance or any other server 
with a GPU. 

(It is recommended to use at least an RTX-4090 GPU to have a good performance during training)

## 1. Install Python 3.8

```shell
~$ sudo apt update
~$ sudo apt install python3.8 python3.8-dev python3.8-venv -y
~$ python3.8 --version  # Should output 3.8.20
```

## 2. Set-up virtual environment

You can place the virtual environment wherever you prefer. Personally, I like to leave it at the `workspace/` folder
so everything I do stays in that folder.

```shell
~$ cd your/fav_dir
fav_dir$ python3.8 -m venv py38  # You can choose any other name as well
fav_dir$ source py38/bin/activate
(py38) /fav_dir$ cd ./workspace/donut  # Go back to the workspace directory to keep working in there
(py38) /workspace/donut$ pip install -r requirements_fine_tuning.txt
(py38) /workspace/donut$ pip install .  # Installs donut as a package
```

## 3. Customize your training configuration

Go to the `config` folder, and modify or create a new configuration file in YAML format. 

Make sure that you set up **at least** the following parameters:

- `result_path`: make sure to locate where the results will be saved for later testing.
- `pretrained_model_name_or_path`: Can be a HF model from the model Hub, or a path to your own model.
- `datasate_name_or_paths`: your dataset(s). Make sure to use squared brackets even if you specify just one
dataset. The route must be relative to the `donut` main directory, so if your data is directly located in the
workspace directory, the route would look something like this: `../<your_dataset>`
- `train_batch_sizes` and `val_batch_sizes`: adjust to your machine. For 24Gb VRAM, the best values are 2 and 2, respectively.
- `lr`: adjust learning rate if you see that the training goes too fast/slow. Recommended value: `5e-6`.
- `max_epochs`: depends on how much samples your dataset has. You must run a couple of times first to find the optimal amount of epochs.
- `num_workers`: for 24Gb VRAM, the best number of workers is 4.
- `check_val_every_n_epoch`: determines how much epochs execute before running the validation step. If you want faster trainings, you can
set this value higher, but take in mind that then, when you want to check the results, you will have less data.

## 4. Run training

```shell
(py38) /workspace/donut$ python train.py --config config/<your_train_config>.yaml
```

If you are running in Vast.ai, make sure to run this command in a shell command line, and not in a Jupyter notebook, as
the latter works over a User interface, and if you close the browser/turn off your computer, the execution will stop.

## Executing and stopping instance (Just for Vast.ai)

If you are going to execute a long training, but don't want Vast.ai to leave the machine opened wasting
budget, you can make use of `train.sh`.

Instead of manually executing the training, after activating your virtual environment, open
`train.sh` to update the training configuration file to be used, then save. If you want the Vast.ai
instance to be turned off to stop wasting budget once the training finishes (even if it returned an error or succeeded),
execute this file like:

```shell
(py38) /workspace/donut$ source train.sh "stop_instance"
```

## 5. Create a jupyter kernel (optional)

After fine-tuning, if you want to use the jupyter notebook in this repository
to test the resulting model after training, then follow this steps:

```shell
(py38) fav_dir$ python -m ipykernel install --user --name=py38 --display-name "py38"
```

Then open the jupyter notebook, go to *Kernel* &#8594; *Change kernel* &#8594; *py38*

Leave the jupyter notebook open or close it until the fine-tuning finishes

## 6. Validation results and testing

### Check validation results

To check the validation results, in the jupyter notebook, go to the section `READ TENSORFLOW RESULTS LOG`. In the first cell,
you must set:

- `results_path`: pointing to the folder named after your selected configuration file inside the results folder you 
specified earlier. Example: `donut/results/train_config`
- `exp_name`: go inside the `results_path` folder to find each fine-tuning execution. Use the name of the most recent one in this variable.
- `tf_file_name`: go inside the `exp_name` folder. You will find a file using the following nomenclature: `events.out.tfevents.<>.<>`. This
is the TensorBoard log containing all the training info. Specify the full name in this variable.

Then, execute all the cells below, until the training graphics are displayed.

### Inference over test set

In the Jupyter notebook, go to the section `TESTING INFERENCE`. Below the **IMPORTANT** section, you will have to set
some variables:

- `exp_name`: go inside the `results_path` folder to find each fine-tuning execution. Use the name of the most recent one in this variable.
- `trained_model_path`: update the string with the name of your `results` folder.
- `dataset_path`: path to your dataset. Remember to use relative paths, in this case relative to where the jupyter notebook is located.
- `task`: must be the same name as the dataset folder.

Execute the rest of the cells in the section. When executing the `test()` function, you will see Predictions and their Ground-truth, and
once all the inferences are finished, the average TED and F1 scores will be displayed as well, letting you know how accurate
your model is.

## 7. **SAVE YOUR WEIGHTS**

If you like what you see, don't forget to save your results somewhere safe (in your local device, or uploaded to HF Hub). 
Remember that if you turn off the instance

### What if I turned off the machine and didn't save the results?

Multiple situations can happen:

- If you try to run again the instance and it succeeds, then you can simply go to your results path and download the results.
- If the machine goes to `scheduling` state, it means someone else rented the machine, and you are not able to connect until this
person turns it off. If you don't want to wait an undetermined amount of time, you can create a second instance, and once it is created,
click on the "copy" button of your previous instance, then "paste" to your new instance. You can specify which folder do you
want to copy. Just leave it default to copy the whole `workspace` folder.

