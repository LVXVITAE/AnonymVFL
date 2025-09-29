#!/bin/bash
# 一键清理脚本
rm -rf models/lr_company/*
rm -f spu.log
rm -f company_share.csv
ray stop || true
echo "已完成清理。"
