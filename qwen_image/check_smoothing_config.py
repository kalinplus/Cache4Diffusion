#!/usr/bin/env python3
import os
import sys

# Add project root to path
sys.path.insert(0, '/home/hkl/Cache4Diffusion')

print("=== TaylorSeer QWen Image 配置检查 ===")
print()

# 检查环境变量
print("1. 环境变量配置:")
print(f"   USE_SMOOTHING = {os.environ.get('USE_SMOOTHING', 'False (默认)')}")
print(f"   USE_HYBRID_SMOOTHING = {os.environ.get('USE_HYBRID_SMOOTHING', 'False (默认)')}")
print(f"   SMOOTHING_METHOD = {os.environ.get('SMOOTHING_METHOD', 'exponential (默认)')}")
print(f"   SMOOTHING_ALPHA = {os.environ.get('SMOOTHING_ALPHA', '0.8 (默认)')}")
print()

# 检查当前使用的是哪个版本
use_smoothing = os.environ.get("USE_SMOOTHING", "False").lower() in ("true", "1", "yes")
if use_smoothing:
    print("✅ 当前运行的是：**平滑版本** TaylorSeer")
    method = os.environ.get("SMOOTHING_METHOD", "exponential")
    alpha = os.environ.get("SMOOTHING_ALPHA", "0.8")
    print(f"   平滑方法：{method}")
    print(f"   平滑系数：{alpha}")
    if os.environ.get("USE_HYBRID_SMOOTHING", "False").lower() == "true":
        print("   混合平滑：启用")
else:
    print("❌ 当前运行的是：**普通版本** TaylorSeer (无平滑)")
print()

print("=== 如何切换到平滑版本 ===")
print("运行前设置环境变量：")
print("export USE_SMOOTHING=\"True\"")
print("export SMOOTHING_METHOD=\"exponential\"  # 或 \"moving_average\"")
print("export SMOOTHING_ALPHA=\"0.8\"")
print()