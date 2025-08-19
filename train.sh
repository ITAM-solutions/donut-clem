#!/bin/bash

echo "Checking current virtual environment..."
python --version

while true; do
  read -p "Do you wish to install this program? " yn
  case $yn in
    [Yy]* ) echo "Continuing with the training setup..."; break;;
    [Nn]* ) echo "Please activate the correct environment before executing this script."; return;;
    * ) echo "Please answer yer or no.";;
  esac
done

echo "Starting to execute the training script"
python train.py --config config/train_custom.yaml

echo "Training end."
if [ "$1" == "stop_instance" ]; then
	echo "Turning off the instance..."
	vastai stop instance "$(python retrieve_training_instance_id.py "$(vastai show instances --raw)")"
fi
