#!/usr/bin/env bash
echo "========== 主机信息 =========="
hostname
uname -a

echo ""
echo "========== GPU 信息 =========="
nvidia-smi -L 2>/dev/null || echo "无 GPU"

echo ""
echo "========== 所有环境变量 =========="
env | sort

echo ""
echo "========== 网络接口 =========="
ip addr show 2>/dev/null | grep "inet "

echo ""
echo "========== InfiniBand =========="
which ibstat 2>/dev/null && ibstat 2>/dev/null | head -30 || echo "无 ibstat"

echo ""
echo "========== NCCL 相关 =========="
env | grep -iE "nccl|rdma|ib_" || echo "无 NCCL 相关变量"

echo ""
echo "========== 完成 =========="
