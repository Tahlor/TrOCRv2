#!/bin/bash

config_yaml=$1

sbatch --export=config_yaml="${config_yaml}" SbatchScript.sh