#!/bin/bash
# 一键清理脚本
rm -rf models/lr_partner/*
rm -f spu.log
rm -f partner_share.csv
ray stop || true
echo "已完成清理。"
