#!/bin/bash
echo "Process $$ started at $(date)"
command="${@}"
echo $command
eval $command
echo "Process $$ ended at $(date)"
