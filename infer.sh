#!/bin/sh
CURDIR="`dirname $0`" #获取此脚本所在目录
echo $CURDIR
cd $CURDIR #切换到该脚本所在目录
ls
pip list
echo "Starting infer..."
python infer.py