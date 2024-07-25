#!/bin/bash
echo $@
if [ $1 == "f" ]; then
    shift
    echo $@
    nohup  accelerate launch --mixed_precision fp16 ./main.py  $@   >/dev/null 2>&1 &
else
    echo $@
    nohup  accelerate launch ./main.py  $@   >/dev/null 2>&1 &

fi
#nohup  python ./main.py  $@   >/dev/null 2>&1 &