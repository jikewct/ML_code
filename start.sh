#!/bin/bash
echo $@
nohup  python ./main.py  $@  &