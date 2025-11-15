#!/bin/bash

deepspeed --master_port 25434 --include localhost:4,5,6,7\
    llava/train/new_reference_logps.py \