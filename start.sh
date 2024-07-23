#!/bin/bash
echo $@
nohup  python ./main.py  $@   >/dev/null 2>&1 &