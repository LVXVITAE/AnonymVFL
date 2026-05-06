#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联邦学习Web UI后端服务
支持跨机器通信的分布式联邦学习界面
"""

import os
import sys
import json
import asyncio
import subprocess
import threading
import time
import re
import traceback
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from textwrap import dedent

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room
import psutil
import yaml

# Kubernetes API 客户端（可选，如果不可用则使用文件系统方式）
try:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False
    print("⚠️  Kubernetes client not available, install with: pip install kubernetes")

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 记录应用启动时间（用于区分预置模型和本次训练产出的模型）
APP_START_TIME = time.time()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'federated_learning_secret_key_2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 节点角色检测（通过环境变量）
NODE_ROLE = os.getenv("NODE_ROLE", "company").lower()  # "company" 或 "partner"
IS_READONLY = NODE_ROLE == "partner"  # Partner 节点为只读模式
print(f"🔧 节点角色: {NODE_ROLE.upper()}, 只读模式: {IS_READONLY}")

# Partner 专用：Company WebUI 的 URL，用于代理训练状态查询
# 示例值（在 values-cluster-b.yaml 的 webui.env 中配置）：http://192.168.49.2:30080
COMPANY_WEBUI_URL = os.getenv("COMPANY_WEBUI_URL", "").rstrip("/")

# Company 专用：Partner WebUI 的 URL，用于训练/推理时自动获取 Partner 已上传的数据集路径
# 示例值（在 values-cluster-a.yaml 的 webui.env 中配置）：http://192.168.58.2:30081
PARTNER_WEBUI_URL = os.getenv("PARTNER_WEBUI_URL", "").rstrip("/")


# 只读模式装饰器


def require_write_permission(f):
    """装饰器：要求写权限（仅 Company 节点可用）"""
    from functools import wraps

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if IS_READONLY:
            return jsonify({
                "status": "error",
                "message": "当前节点为 Partner（只读模式），无法执行此操作"
            }), 403
        return f(*args, **kwargs)
    return decorated_function


# 全局状态管理


class SystemState:
    def __init__(self):
        self.company_status = "未启动"
        self.partner_status = "未启动"
        self.training_status = "未开始"
        self.training_run_id = None
        self.training_started_at = None
        self.training_finished_at = None
        self.current_step = 0
        self.current_epoch = 0
        self.total_epochs = 0
        self.total_steps = 0
        self.accuracy = 0.0
        self.loss = 0.0
        self.processes = {}
        self.logs = []
        self.config = self.load_config()
        # 训练完成后写入的模型注册表（不依赖共享文件系统）
        self.trained_model_registry: list = []
        # 本次训练正在进行中的待确认模型信息
        self.pending_training_model: dict = {}

    def load_config(self):
        """加载配置文件"""
        config_path = project_root / "web_ui" / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # 默认配置
            return {
                'company': {
                    'ip': '210.28.133.104',
                    'port_ray': 20001,
                    'port_spu': 11001,
                    'path': 'company'
                },
                'partner': {
                    'ip': '210.28.133.104',
                    'port_spu': 11002,
                    'path': 'partner'
                },
                'coordinator': {
                    'ip': '210.28.133.104',
                    'port_spu': 11003
                },
                'training': {
                    'model': 'SSLR',
                    'default_epochs': 10,
                    'default_batch_size': 1000,
                    'default_lr': 0.1,
                    'validation_steps': 1,
                    'n_estimators': 2,
                    'max_depth': 2,
                    'K_quantiles': 20,
                    'reg_coef': 0.0
                }
            }

    def save_config(self):
        """保存配置文件"""
        config_path = project_root / "web_ui" / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False,
                      allow_unicode=True)


# 全局状态实例
state = SystemState()
SERVER_STARTED_AT = datetime.now().isoformat()


def _new_training_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


def _build_model_dirs(model: str, run_id: str):
    if model == "SSXGBoost":
        base = "xgb"
        model_type = "SSXGBoost"
    else:
        base = "lr"
        model_type = "SSLR (逻辑回归)"
    company_model_name = f"{base}_company_{run_id}"
    partner_model_name = f"{base}_partner_{run_id}"
    return {
        "model_type": model_type,
        "company_model_name": company_model_name,
        "partner_model_name": partner_model_name,
        "company_path": f"/app/company/models/{company_model_name}",
        "partner_path": f"/app/partner/models/{partner_model_name}",
    }


def log_message(message: str, level: str = "INFO"):
    """记录日志消息"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "level": level,
        "message": message
    }
    state.logs.append(log_entry)
    # 保持最近1000条日志
    if len(state.logs) > 1000:
        state.logs = state.logs[-1000:]

    # 通过WebSocket广播日志
    socketio.emit('log_update', log_entry)

    # 🔥 同时输出到 stderr（kubectl logs 能看到）
    print(f"[{timestamp}] [{level}] {message}", file=sys.stderr, flush=True)


def get_system_info():
    """获取系统信息"""
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "timestamp": datetime.now().isoformat()
    }


@app.route('/')
def index():
    """主页面"""
    return render_template('index.html',
                           node_role=NODE_ROLE,
                           is_readonly=IS_READONLY)


@app.route('/api/status')
def get_status():
    """获取系统状态（主动检查 Kubernetes 实际状态）"""
    # 如果 Kubernetes API 可用，主动检查实际状态
    if K8S_AVAILABLE:
        try:
            # 加载 Kubernetes 配置
            try:
                config.load_incluster_config()
            except:
                config.load_kube_config()

            apps_v1 = client.AppsV1Api()
            namespace = os.getenv('NAMESPACE', os.getenv(
                'POD_NAMESPACE', 'mpc-test'))

            # 检查 Company Deployment 实际状态
            try:
                deployments = apps_v1.list_namespaced_deployment(
                    namespace=namespace,
                    label_selector='app.kubernetes.io/component=company'
                )
                if deployments.items:
                    deployment = deployments.items[0]
                    current_replicas = deployment.spec.replicas or 0
                    ready_replicas = deployment.status.ready_replicas or 0

                    if current_replicas > 0:
                        if ready_replicas > 0:
                            state.company_status = "运行中"
                        else:
                            state.company_status = "启动中"
                    else:
                        state.company_status = "已停止"
            except Exception as e:
                # 静默失败，使用内存中的状态
                pass

            # 检查 Partner Deployment 实际状态
            try:
                deployments = apps_v1.list_namespaced_deployment(
                    namespace=namespace,
                    label_selector='app.kubernetes.io/component=partner'
                )
                if deployments.items:
                    deployment = deployments.items[0]
                    current_replicas = deployment.spec.replicas or 0
                    ready_replicas = deployment.status.ready_replicas or 0

                    if current_replicas > 0:
                        if ready_replicas > 0:
                            state.partner_status = "运行中"
                        else:
                            state.partner_status = "启动中"
                    else:
                        state.partner_status = "已停止"
            except Exception as e:
                # 静默失败，使用内存中的状态
                pass

        except Exception as e:
            # 如果 Kubernetes API 检查失败，使用内存中的状态
            pass

    # Partner 只读节点：从 Company WebUI 代理训练相关状态，避免两端状态漂移
    proxied_training_status = state.training_status
    proxied_current_step = state.current_step
    proxied_current_epoch = state.current_epoch
    proxied_total_epochs = state.total_epochs
    proxied_accuracy = state.accuracy
    proxied_loss = state.loss
    proxied_started_at = state.training_started_at
    proxied_finished_at = state.training_finished_at
    proxied_run_id = state.training_run_id
    if IS_READONLY and COMPANY_WEBUI_URL:
        try:
            import urllib.request as _urllib_req
            with _urllib_req.urlopen(
                f"{COMPANY_WEBUI_URL}/api/status", timeout=2
            ) as resp:
                company_data = json.loads(resp.read().decode())
                proxied_training_status = company_data.get(
                    "training_status", state.training_status
                )
                proxied_current_step = company_data.get(
                    "current_step", state.current_step
                )
                proxied_current_epoch = company_data.get(
                    "current_epoch", state.current_epoch
                )
                proxied_total_epochs = company_data.get(
                    "total_epochs", state.total_epochs
                )
                proxied_accuracy = company_data.get("accuracy", state.accuracy)
                proxied_loss = company_data.get("loss", state.loss)
                proxied_started_at = company_data.get(
                    "training_started_at", state.training_started_at
                )
                proxied_finished_at = company_data.get(
                    "training_finished_at", state.training_finished_at
                )
                proxied_run_id = company_data.get(
                    "training_run_id", state.training_run_id
                )
        except Exception:
            pass  # 网络不可达时使用本地状态

    # 运行时间基于服务启动时间计算（前端按该值动态刷新）
    try:
        uptime_seconds = int((datetime.now() - datetime.fromisoformat(SERVER_STARTED_AT)).total_seconds())
    except Exception:
        uptime_seconds = 0

    return jsonify({
        "company_status": state.company_status,
        "partner_status": state.partner_status,
        "training_status": proxied_training_status,
        "training_run_id": proxied_run_id,
        "training_started_at": proxied_started_at,
        "training_finished_at": proxied_finished_at,
        "current_step": proxied_current_step,
        "current_epoch": proxied_current_epoch,
        "total_epochs": proxied_total_epochs,
        "total_steps": state.total_steps,
        "accuracy": proxied_accuracy,
        "loss": proxied_loss,
        "system_info": get_system_info(),
        "logs": state.logs[-50:],  # 最近50条日志
        "server_started_at": SERVER_STARTED_AT,
        "uptime_seconds": uptime_seconds,
        "node_role": NODE_ROLE,
        "is_readonly": IS_READONLY
    })


@app.route('/api/logs/<node_type>')
def get_logs(node_type):
    """获取指定节点的日志内容（优先使用 Kubernetes API，否则使用文件系统）"""
    try:
        # 优先使用 Kubernetes API（如果在 Kubernetes 环境中）
        if K8S_AVAILABLE:
            try:
                # 加载 Kubernetes 配置
                try:
                    config.load_incluster_config()  # 在 Pod 内部使用
                except:
                    config.load_kube_config()  # 本地开发使用

                v1 = client.CoreV1Api()

                # 获取命名空间（从环境变量或配置）
                namespace = os.getenv('NAMESPACE', os.getenv(
                    'POD_NAMESPACE', 'mpc-test'))

                # 根据节点类型选择标签选择器
                if node_type == 'company':
                    label_selector = 'app.kubernetes.io/component=company'
                elif node_type == 'partner':
                    label_selector = 'app.kubernetes.io/component=partner'
                else:
                    return jsonify({"status": "error", "message": "未知的节点类型"})

                # 获取 Pod 列表
                pods = v1.list_namespaced_pod(
                    namespace=namespace,
                    label_selector=label_selector
                )

                if not pods.items:
                    return jsonify({
                        "status": "error",
                        "message": f"未找到 {node_type} Pod (namespace: {namespace})"
                    })

                # 获取第一个 Pod 的日志（通常只有一个）
                pod_name = pods.items[0].metadata.name
                # 不再记录信息性日志，避免系统日志被刷屏（前端频繁轮询会产生大量重复日志）

                logs = v1.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=namespace,
                    tail_lines=100,
                    timestamps=False
                )

                # 将日志按行分割
                log_lines = logs.split('\n') if logs else []
                return jsonify({"status": "success", "logs": log_lines})

            except ApiException as e:
                error_message = str(e)
                error_body = getattr(e, 'body', '')

                # 检查是否是 Pod 还在创建中或已删除的正常情况
                is_normal_case = (
                    "ContainerCreating" in error_message or
                    "waiting to start" in error_message or
                    "not found" in error_message.lower() or
                    ("ContainerCreating" in str(error_body) if error_body else False)
                )

                if is_normal_case:
                    # Pod 还在创建中或已删除，这是正常情况，返回空日志，不记录错误
                    return jsonify({"status": "success", "logs": []})
                else:
                    # 其他错误才记录
                    log_message(f"Kubernetes API 错误: {str(e)}", "ERROR")
                    # 如果 Kubernetes API 失败，回退到文件系统方式
                    return _get_logs_from_filesystem(node_type)
            except Exception as e:
                log_message(f"Kubernetes 连接错误: {str(e)}", "ERROR")
                # 如果连接失败，回退到文件系统方式
                return _get_logs_from_filesystem(node_type)
        else:
            # 如果没有 Kubernetes 客户端，使用文件系统方式
            return _get_logs_from_filesystem(node_type)

    except Exception as e:
        log_message(f"获取日志失败: {str(e)}", "ERROR")
        return jsonify({"status": "error", "message": str(e)})


def _get_logs_from_filesystem(node_type):
    """从文件系统读取日志（回退方案）"""
    try:
        if node_type == 'company':
            log_file = project_root / "company" / "company_run.log"
        elif node_type == 'partner':
            log_file = project_root / "partner" / "partner_run.log"
        else:
            return jsonify({"status": "error", "message": "未知的节点类型"})

        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # 返回最后100行
                return jsonify({"status": "success", "logs": lines[-100:]})
        else:
            return jsonify({"status": "success", "logs": []})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/training_metrics')
def get_training_metrics():
    """从日志解析并返回训练指标（优先使用 Kubernetes Pod 日志，否则使用文件系统）"""
    # Partner 只读节点：直接代理 Company WebUI 的训练指标
    if IS_READONLY and COMPANY_WEBUI_URL:
        try:
            import urllib.request as _urllib_req
            with _urllib_req.urlopen(
                f"{COMPANY_WEBUI_URL}/api/training_metrics", timeout=3
            ) as resp:
                data = json.loads(resp.read().decode())
                # 同步本地 training_status，供其他逻辑使用
                if 'status' in data:
                    state.training_status = data['status']
                if 'current_step' in data:
                    state.current_step = data['current_step']
                if 'current_epoch' in data:
                    state.current_epoch = data['current_epoch']
                if 'started_at' in data:
                    state.training_started_at = data['started_at']
                if 'finished_at' in data:
                    state.training_finished_at = data['finished_at']
                if 'run_id' in data:
                    state.training_run_id = data['run_id']
                return jsonify(data)
        except Exception:
            pass  # 网络不可达时降级到本地指标

    try:
        # 初始化指标
        metrics = {
            'current_epoch': 0,
            'total_epochs': state.total_epochs or 10,
            'current_step': 0,
            'accuracy': 0,
            'f1': 0,
            'for': 0,
            'final_accuracy': None,
            'final_f1': None,
            'final_for': None,
            'status': state.training_status  # 使用内存中的状态作为默认值
        }

        lines = []

        # 优先使用 Kubernetes API 读取 Pod 日志
        if K8S_AVAILABLE:
            try:
                try:
                    config.load_incluster_config()
                except:
                    config.load_kube_config()

                v1 = client.CoreV1Api()
                namespace = os.getenv('NAMESPACE', os.getenv(
                    'POD_NAMESPACE', 'mpc-test'))

                # 获取 Company Pod 日志
                pods = v1.list_namespaced_pod(
                    namespace=namespace,
                    label_selector='app.kubernetes.io/component=company'
                )

                if pods.items:
                    pod_name = pods.items[0].metadata.name
                    try:
                        logs = v1.read_namespaced_pod_log(
                            name=pod_name,
                            namespace=namespace,
                            tail_lines=500,  # 读取更多行以确保能解析到指标
                            timestamps=False
                        )
                        lines = logs.split('\n') if logs else []
                    except ApiException as e:
                        # Pod 可能还在创建中，使用空列表
                        error_message = str(e)
                        if "ContainerCreating" not in error_message and "waiting to start" not in error_message:
                            print(f"⚠️ 读取 Pod 日志失败: {str(e)}",
                                  file=sys.stderr, flush=True)
                        lines = []
            except Exception as e:
                print(
                    f"⚠️ Kubernetes API 不可用，回退到文件系统: {str(e)}", file=sys.stderr, flush=True)
                lines = []

        # 如果 Kubernetes 方式失败，回退到文件系统
        if not lines:
            log_file = project_root / "company" / "company_run.log"
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            else:
                return jsonify(metrics)

        # 仅解析当前训练 run 的日志，避免二次训练被旧日志污染
        run_marker = None
        run_id = state.training_run_id
        if not run_id and state.training_status in ('未开始', '已停止'):
            metrics['run_id'] = None
            metrics['started_at'] = state.training_started_at
            metrics['finished_at'] = state.training_finished_at
            return jsonify(metrics)
        for line in lines:
            line_str = line if isinstance(line, str) else str(line)
            marker_match = re.search(r'__TRAINING_RUN_ID__=([A-Za-z0-9_\-]+)', line_str)
            if marker_match:
                run_marker = marker_match.group(1)
                # 若当前内存未记录 run_id，则采用日志中的最新 marker
                if not run_id:
                    run_id = run_marker
                    state.training_run_id = run_id

        parse_lines = lines
        if run_id:
            # 从最后一个匹配 run_id 的 marker 开始解析
            marker_idx = -1
            for i, line in enumerate(lines):
                line_str = line if isinstance(line, str) else str(line)
                if f"__TRAINING_RUN_ID__={run_id}" in line_str:
                    marker_idx = i
            if marker_idx >= 0:
                parse_lines = lines[marker_idx:]

        # 从后往前读，找到最新的指标（避免重复日志干扰）
        max_epoch = 0
        max_step = -1

        for line in reversed(parse_lines):
            line_str = line if isinstance(line, str) else str(line)

            # 解析最终结果（优先级最高）
            if metrics['final_accuracy'] is None:
                final_acc_match = re.search(
                    r'[•\s]*最终准确率[:\s]+([0-9.]+)', line_str)
                if final_acc_match:
                    metrics['final_accuracy'] = float(final_acc_match.group(1))

            if metrics['final_f1'] is None:
                final_f1_match = re.search(
                    r'[•\s]*最终F1分数[:\s]+([0-9.]+)', line_str)
                if final_f1_match:
                    metrics['final_f1'] = float(final_f1_match.group(1))

            if metrics['final_for'] is None:
                final_for_match = re.search(
                    r'[•\s]*最终误漏率[:\s]+([0-9.]+)', line_str)
                if final_for_match:
                    metrics['final_for'] = float(final_for_match.group(1))

            # 解析Epoch（取最大值）
            epoch_match = re.search(r'Epoch\s+(\d+)', line_str)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                if epoch > max_epoch:
                    max_epoch = epoch

            # 解析Step指标（取最新的）
            step_match = re.search(
                r'Step\s+(\d+),\s+Accuracy:\s+([0-9.]+),\s+F1:\s+([0-9.]+),\s+FOR:\s+([0-9.]+)', line_str)
            if step_match:
                step = int(step_match.group(1))
                if step > max_step:
                    max_step = step
                    metrics['current_step'] = step
                    metrics['accuracy'] = float(step_match.group(2))
                    metrics['f1'] = float(step_match.group(3))
                    metrics['for'] = float(step_match.group(4))

        metrics['current_epoch'] = max_epoch
        if max_epoch > 0:
            metrics['total_epochs'] = max(metrics['total_epochs'], max_epoch)

        # 只有在找到 step 时才更新 current_step，否则保持之前的值（避免在训练过程中切换为0）
        if max_step >= 0:
            # 找到了 step 信息，更新
            metrics['current_step'] = max_step
            # 同时更新 state，以便下次没有找到 step 时能保持这个值
            state.current_step = max_step
        else:
            # 没有找到 step 信息，保持之前的值（如果之前有值）
            # 如果之前也没有值，保持为0
            if state.current_step > 0:
                metrics['current_step'] = state.current_step
            else:
                metrics['current_step'] = 0

        # 检查是否有新的训练开始（在判断最终结果之前）
        # 如果检测到新的训练开始，清除最终结果（允许重新训练）
        has_new_training = any('开始训练任务' in str(line) for line in parse_lines[-10:])
        if has_new_training and metrics['final_accuracy'] is not None:
            # 新训练开始，重置最终结果
            metrics['final_accuracy'] = None
            metrics['final_f1'] = None
            metrics['final_for'] = None
            # 重置 step 和 epoch
            metrics['current_step'] = 0
            metrics['current_epoch'] = 0
            state.current_step = 0
            state.current_epoch = 0

        # 判断训练状态（优先使用解析到的状态，否则使用内存中的状态）
        # 优先级：已完成 > 训练中 > PSI中 > 内存状态 > 未开始
        if metrics['final_accuracy'] is not None:
            # 如果已有最终结果，保持"已完成"状态，不再进行其他判断
            metrics['status'] = '已完成'
            state.training_status = '已完成'
            state.training_finished_at = datetime.now().isoformat()
            # 训练完成后，保留最后的 epoch 和 step 信息用于显示
            # 不重置 current_step，保持最后的值

            # 训练首次完成时，将待确认模型信息写入注册表
            if state.pending_training_model:
                pending = state.pending_training_model
                trained_at = datetime.now().isoformat()
                run_id = pending.get('run_id')
                # 避免重复写入同一 run
                already_registered = any(
                    e.get('version_id') == run_id for e in state.trained_model_registry
                )
                if not already_registered:
                    for side in ('company', 'partner'):
                        info = pending.get(side, {})
                        if info:
                            state.trained_model_registry.append({
                                'version_id': run_id,
                                'node_type': info['node_type'],
                                'model_name': info['model_name'],
                                'model_type': pending['model_type'],
                                'path': info['path'],
                                'files': [],          # 文件在对端 Pod，无法直接访问
                                'trained_at': trained_at,
                                'training_params': pending.get('training_params', {}),
                            })
                state.pending_training_model = {}

            # 🔥 新增：训练完成后，清理 Deployment 的 args，移除训练命令
            # 防止 Pod 重启后自动执行训练，同时允许第二次训练
            try:
                if K8S_AVAILABLE and not hasattr(state, '_deployment_cleaned_after_training'):
                    # 只清理一次，避免重复操作
                    try:
                        config.load_incluster_config()
                    except:
                        config.load_kube_config()

                    apps_v1 = client.AppsV1Api()
                    namespace = os.getenv('NAMESPACE', os.getenv(
                        'POD_NAMESPACE', 'mpc-test'))

                    # 查找 Company Deployment
                    deployments = apps_v1.list_namespaced_deployment(
                        namespace=namespace,
                        label_selector='app.kubernetes.io/component=company'
                    )

                    if deployments.items:
                        deployment_name = deployments.items[0].metadata.name
                        deployment = apps_v1.read_namespaced_deployment(
                            name=deployment_name,
                            namespace=namespace
                        )

                        container = deployment.spec.template.spec.containers[0]
                        current_args = container.args[0] if container.args else ""

                        # 检查 args 中是否包含训练命令（通过检查是否有 python company/truerun.py）
                        if "python company/truerun.py" in current_args or "exec python" in current_args:
                            # 提取端口信息
                            ray_port = 6379
                            partner_service_name = "mpc-test-mobile-mpc-project-partner-svc"

                            # 尝试从当前 args 中提取端口
                            ray_match = re.search(
                                r'--port=(\d+)', current_args)
                            if ray_match:
                                ray_port = int(ray_match.group(1))

                            partner_match = re.search(
                                r'PARTNER_SERVICE_NAME="([^"]+)"', current_args)
                            if partner_match:
                                partner_service_name = partner_match.group(1)

                            # 恢复为默认的启动脚本（不包含训练命令）
                            default_script = _get_default_company_startup_script(
                                ray_port, partner_service_name)
                            container.args[0] = default_script

                            # 更新 Deployment
                            apps_v1.patch_namespaced_deployment(
                                name=deployment_name,
                                namespace=namespace,
                                body=deployment
                            )

                            # 标记已清理，避免重复操作
                            state._deployment_cleaned_after_training = True

                            log_message(
                                "训练完成，已清理 Deployment 配置，允许第二次训练", "INFO")
            except Exception as e:
                # 静默失败，不影响训练状态显示
                log_message(f"清理 Deployment 配置失败（不影响功能）: {str(e)}", "WARNING")
        elif state.training_status == '已完成':
            # 🔥 修复：如果内存中状态是"已完成"，优先保持"已完成"状态
            # 避免因为 current_step > 0 而误判为"训练中"
            # 这个判断必须在 current_step 判断之前，确保已完成状态不会被覆盖
            metrics['status'] = '已完成'
        elif metrics['current_step'] > 0:
            # 如果有正在进行的 step，判断为"训练中"
            metrics['status'] = '训练中'
            state.training_status = '训练中'
        elif any('开始训练任务' in str(line) or 'PSI' in str(line) for line in lines[-50:]):
            # 只有在不是"已完成"状态时，才判断为"PSI中"
            # 检查最近的日志，避免旧日志干扰
            metrics['status'] = 'PSI中'
            state.training_status = 'PSI中'
        # 如果无法从日志判断，使用内存中的状态
        elif state.training_status != '未开始':
            metrics['status'] = state.training_status
        else:
            # 如果状态是"未开始"，确保重置指标
            metrics['status'] = '未开始'
            state.training_status = '未开始'

        # 同步内存状态，供 /api/status 与前端概览统一读取
        state.current_epoch = metrics.get('current_epoch', state.current_epoch)
        state.current_step = metrics.get('current_step', state.current_step)
        state.accuracy = metrics.get('accuracy', state.accuracy)
        metrics['run_id'] = state.training_run_id
        metrics['started_at'] = state.training_started_at
        metrics['finished_at'] = state.training_finished_at

        return jsonify(metrics)

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"❌ 获取训练指标失败: {str(e)}", file=sys.stderr, flush=True)
        print(error_traceback, file=sys.stderr, flush=True)
        return jsonify({"error": str(e), "status": "error"})


@app.route('/api/config', methods=['GET', 'POST'])
def config_api():
    """配置管理API"""
    if request.method == 'GET':
        return jsonify(state.config)
    elif request.method == 'POST':
        if IS_READONLY:
            return jsonify({"status": "error", "message": "当前节点为 Partner（只读模式），无法修改配置"}), 403
        new_config = request.json
        state.config.update(new_config)
        state.save_config()
        log_message("配置已更新", "INFO")
        return jsonify({"status": "success"})


def _get_default_company_startup_script(ray_port, partner_service_name):
    """获取默认的 Company 节点启动脚本（不包含训练命令）"""
    return dedent(f"""
                echo "🚀 启动 Company 节点..."

                # 清理旧的 Ray 资源（如果存在）
                echo "🧹 清理旧的 Ray 资源..."
                ray stop --force || true
                sleep 2

                # 使用 K8s Downward API 注入的 POD_IP（避免 hostname -i 多网卡问题）
                export POD_IP=${{POD_IP:-$(hostname -i | tr ' ' '\n' | head -1)}}
                echo "📍 Pod IP: $POD_IP"

                # 使用 Ray CLI 启动 Ray Head（指定端口 {ray_port}）
                echo "📍 启动 Ray Head (端口: {ray_port})..."
                ray start --head \
                    --port={ray_port} \
                    --num-cpus=8 \
                    --resources='{{"company": 10}}' \
                    --object-store-memory=2000000000 \
                    --include-dashboard=false \
                    --node-ip-address=$POD_IP &

                RAY_PID=$!
                echo "Ray Head 后台进程 PID: $RAY_PID"

                # Coordinator 地址优先使用环境变量
                if [ -z "$COORDINATOR_SPU_ADDR" ]; then
                    echo "⚠️  未设置 COORDINATOR_SPU_ADDR，回退到本地端口"
                    COORDINATOR_SPU_ADDR="$POD_IP:9396"
                fi

                # 等待 Ray Head 启动
                echo "⏳ 等待 Ray Head 准备就绪..."
                sleep 5

                # 验证 Ray Head 是否启动成功
                if ! ray status &>/dev/null; then
                    echo "⚠️  Ray Head 启动检查失败，继续执行..."
                else
                    echo "✅ Ray Head 已启动"
                fi

                # 解析 Partner Service 的 Pod IP
                echo "🔍 解析 Partner Service 地址..."
                PARTNER_SERVICE_NAME="{partner_service_name}"
                PARTNER_IP=$(getent hosts $PARTNER_SERVICE_NAME | awk '{{ print $1 }}' | head -1)
                if [ -z "$PARTNER_IP" ]; then
                    echo "⚠️  无法解析 Partner Service，使用 Service 名称"
                    PARTNER_SPU_ADDR="{partner_service_name}:9395"
                else
                    echo "✅ Partner IP: $PARTNER_IP"
                    PARTNER_SPU_ADDR="$PARTNER_IP:9395"
                fi

                # 默认不自动执行训练，等待 Web UI 动态注入训练命令
                echo "✅ Company 节点已就绪，等待 Web UI 启动训练任务..."
                echo "📍 Ray Head 地址: $POD_IP:{ray_port}"
                echo "📍 Partner SPU 地址: $PARTNER_SPU_ADDR"
                echo "📍 Coordinator SPU 地址: $COORDINATOR_SPU_ADDR"
                echo ""
                echo "💡 提示：训练任务将由 Web UI 通过 Kubernetes API 动态注入"
                echo "💡 提示：Web UI 会修改此 Deployment 的 args，添加训练命令并重启 Pod"

                # 保持 Pod 运行（等待训练命令）
                tail -f /dev/null
        """)


def _resolve_partner_datasets(prefer_infer: bool = False):
    """从 Partner WebUI 获取已上传的数据集路径；若无可用数据或网络不通，则返回内置默认路径。

    prefer_infer=False → 返回 (train_path, val_path)
    prefer_infer=True  → 返回 infer_path (单条路径字符串)
    """
    DEFAULT_TRAIN = '/app/partner/guest_train.csv'
    DEFAULT_VAL   = '/app/partner/guest_test.csv'
    DEFAULT_INFER = '/app/partner/guest_test.csv'

    if not PARTNER_WEBUI_URL:
        return DEFAULT_INFER if prefer_infer else (DEFAULT_TRAIN, DEFAULT_VAL)

    try:
        import urllib.request as _urllib_req
        with _urllib_req.urlopen(
            f"{PARTNER_WEBUI_URL}/api/datasets/list", timeout=3
        ) as resp:
            data = json.loads(resp.read().decode())

        files = data.get('datasets', {}).get('partner', [])
        if not files:
            raise ValueError("partner dataset list is empty")

        # 按文件名关键字分类
        train_files = [f for f in files if 'train' in f['name'].lower()]
        val_files   = [f for f in files
                       if any(k in f['name'].lower() for k in ('test', 'val'))]
        # 推理优先选 test/val 类文件
        infer_files = val_files if val_files else files

        if prefer_infer:
            # 取最近修改的 test/val 类文件，或直接取第一个可用文件
            candidates = sorted(infer_files,
                                key=lambda f: f.get('modified', ''), reverse=True)
            return candidates[0]['path']

        # 训练场景
        train_path = (sorted(train_files,
                             key=lambda f: f.get('modified', ''), reverse=True)[0]['path']
                      if train_files else files[0]['path'])
        val_path   = (sorted(val_files,
                             key=lambda f: f.get('modified', ''), reverse=True)[0]['path']
                      if val_files
                      else (files[1]['path'] if len(files) > 1 else files[0]['path']))
        return train_path, val_path

    except Exception as e:
        log_message(f"无法从 Partner WebUI 获取数据集列表，使用默认路径: {e}", "WARNING")
        return DEFAULT_INFER if prefer_infer else (DEFAULT_TRAIN, DEFAULT_VAL)


def _generate_training_command(model, n_epochs, batch_size, lr, val_steps,
                               n_estimators, max_depth, k_quantiles, reg_coef,
                               pod_ip_placeholder, partner_spu_addr_placeholder,
                               ray_port, spu_port,
                               run_id,
                               company_model_save_dir, partner_model_save_dir,
                               company_train_dataset=None, company_val_dataset=None,
                               partner_train_dataset=None, partner_val_dataset=None):
    """生成训练命令参数（根据模型类型）"""
    # Company 数据集：优先使用 WebUI 传入的选择值，否则使用默认路径
    _company_train = company_train_dataset or '/app/company/host_train.csv'
    _company_val   = company_val_dataset   or '/app/company/host_test.csv'

    # Partner 数据集：优先使用 WebUI 传入值；若未传入则从 Partner WebUI 自动获取，
    # 若 Partner WebUI 不可达则回退内置默认路径
    if partner_train_dataset or partner_val_dataset:
        _partner_train = partner_train_dataset or '/app/partner/guest_train.csv'
        _partner_val   = partner_val_dataset   or '/app/partner/guest_test.csv'
    else:
        _partner_train, _partner_val = _resolve_partner_datasets(prefer_infer=False)

    # 🔥 修改：移除 exec，改为 python（这样进程退出不会导致容器退出）
    base_cmd = f"""python company/truerun.py \\
              --mode multi_distributed \\
              --company_spu_addr $POD_IP:{spu_port} \\
              --partner_spu_addr $PARTNER_SPU_ADDR \\
              --coordinator_spu_addr $COORDINATOR_SPU_ADDR \\
              --ray_head_addr $POD_IP:{ray_port} \\
              --run_psi True \\
              --path_to_company_train_dataset {_company_train} \\
              --path_to_company_val_dataset {_company_val} \\
              --path_to_company_share /app/company/company_share.csv \\
              --share_y False \\
              --path_to_partner_train_dataset {_partner_train} \\
              --path_to_partner_val_dataset {_partner_val} \\
              --path_to_partner_share /app/partner/partner_share.csv"""

    if model == "SSLR":
        cmd = f"""{base_cmd} \\
              --path_to_company_model_save_dir {company_model_save_dir} \\
              --path_to_partner_model_save_dir {partner_model_save_dir} \\
              --model SSLR \\
              --n_epochs {n_epochs} \\
              --batch_size {batch_size} \\
              --val_steps {val_steps} \\
              --lr {lr}"""
    else:  # SSXGBoost
        cmd = f"""{base_cmd} \\
              --path_to_company_model_save_dir {company_model_save_dir} \\
              --path_to_partner_model_save_dir {partner_model_save_dir} \\
              --model SSXGBoost \\
              --n_estimators {n_estimators} \\
              --max_depth {max_depth} \\
              --reg_coef {reg_coef} \\
              --K_quantiles {k_quantiles}"""

    # 每次训练写入唯一标记，便于后端仅解析当前训练进度
    cmd = f"""echo "__TRAINING_RUN_ID__={run_id}"
{cmd}"""

    # 🔥 关键修改：在训练命令后添加 tail -f /dev/null，保持容器运行
    # 这样即使训练完成，容器也不会退出，避免自动重启
    cmd = cmd + "\n\necho '✅ 训练任务已完成，保持容器运行...'\ntail -f /dev/null"

    return cmd


@app.route('/api/start_company', methods=['POST'])
@require_write_permission
def start_company():
    """启动Company节点（使用 Kubernetes API）"""
    try:
        if not K8S_AVAILABLE:
            return jsonify({"status": "error", "message": "Kubernetes API 不可用，无法在 Kubernetes 环境中启动"})

        # 加载 Kubernetes 配置
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()

        apps_v1 = client.AppsV1Api()
        namespace = os.getenv('NAMESPACE', os.getenv(
            'POD_NAMESPACE', 'mpc-test'))

        # 通过标签查找 Deployment（更可靠的方式）
        deployments = apps_v1.list_namespaced_deployment(
            namespace=namespace,
            label_selector='app.kubernetes.io/component=company'
        )
        if not deployments.items:
            return jsonify({"status": "error", "message": f"未找到 Company Deployment (namespace: {namespace})"})
        deployment_name = deployments.items[0].metadata.name

        # 检查 Deployment 是否存在
        try:
            deployment = apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
        except ApiException as e:
            if e.status == 404:
                return jsonify({"status": "error", "message": f"Deployment {deployment_name} 不存在"})
            raise

        # 检查当前副本数
        current_replicas = deployment.spec.replicas or 0

        if current_replicas > 0:
            state.company_status = "运行中"
            return jsonify({"status": "success", "message": "Company节点已在运行"})

        # 启动 Deployment（设置副本数为1）
        deployment.spec.replicas = 1
        apps_v1.patch_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            body=deployment
        )

        state.company_status = "运行中"
        log_message(f"Company Deployment 已启动 (namespace: {namespace})", "INFO")
        return jsonify({"status": "success", "message": "Company节点启动中"})

    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"❌ 启动Company节点失败: {str(e)}", file=sys.stderr, flush=True)
        print(error_traceback, file=sys.stderr, flush=True)
        log_message(
            f"启动Company节点失败: {str(e)}\n堆栈信息:\n{error_traceback}", "ERROR")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/start_partner', methods=['POST'])
@require_write_permission
def start_partner():
    """启动Partner节点（使用 Kubernetes API）"""
    try:
        if not K8S_AVAILABLE:
            return jsonify({"status": "error", "message": "Kubernetes API 不可用，无法在 Kubernetes 环境中启动"})

        # 加载 Kubernetes 配置
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()

        apps_v1 = client.AppsV1Api()
        namespace = os.getenv('NAMESPACE', os.getenv(
            'POD_NAMESPACE', 'mpc-test'))

        # 通过标签查找 Deployment（更可靠的方式）
        deployments = apps_v1.list_namespaced_deployment(
            namespace=namespace,
            label_selector='app.kubernetes.io/component=partner'
        )
        if not deployments.items:
            return jsonify({"status": "error", "message": f"未找到 Partner Deployment (namespace: {namespace})"})
        deployment_name = deployments.items[0].metadata.name

        # 检查 Deployment 是否存在
        try:
            deployment = apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
        except ApiException as e:
            if e.status == 404:
                return jsonify({"status": "error", "message": f"Deployment {deployment_name} 不存在"})
            raise

        # 检查当前副本数
        current_replicas = deployment.spec.replicas or 0

        if current_replicas > 0:
            state.partner_status = "运行中"
            return jsonify({"status": "success", "message": "Partner节点已在运行"})

        # 启动 Deployment（设置副本数为1）
        deployment.spec.replicas = 1
        apps_v1.patch_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            body=deployment
        )

        state.partner_status = "运行中"
        log_message(f"Partner Deployment 已启动 (namespace: {namespace})", "INFO")
        return jsonify({"status": "success", "message": "Partner节点启动中"})

    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"❌ 启动Partner节点失败: {str(e)}", file=sys.stderr, flush=True)
        print(error_traceback, file=sys.stderr, flush=True)
        log_message(
            f"启动Partner节点失败: {str(e)}\n堆栈信息:\n{error_traceback}", "ERROR")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/stop_company', methods=['POST'])
@require_write_permission
def stop_company():
    """停止Company节点（使用 Kubernetes API）"""
    try:
        if not K8S_AVAILABLE:
            return jsonify({"status": "error", "message": "Kubernetes API 不可用，无法在 Kubernetes 环境中停止"})

        # 加载 Kubernetes 配置
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()

        apps_v1 = client.AppsV1Api()
        namespace = os.getenv('NAMESPACE', os.getenv(
            'POD_NAMESPACE', 'mpc-test'))

        # 通过标签查找 Deployment（更可靠的方式）
        deployments = apps_v1.list_namespaced_deployment(
            namespace=namespace,
            label_selector='app.kubernetes.io/component=company'
        )
        if not deployments.items:
            return jsonify({"status": "error", "message": f"未找到 Company Deployment (namespace: {namespace})"})
        deployment_name = deployments.items[0].metadata.name

        # 检查 Deployment 是否存在
        try:
            deployment = apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
        except ApiException as e:
            if e.status == 404:
                return jsonify({"status": "error", "message": f"Deployment {deployment_name} 不存在"})
            raise

        # 检查当前副本数
        current_replicas = deployment.spec.replicas or 0

        if current_replicas == 0:
            state.company_status = "已停止"
            return jsonify({"status": "success", "message": "Company节点已停止"})

        # 停止 Deployment（设置副本数为0）
        deployment.spec.replicas = 0
        apps_v1.patch_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            body=deployment
        )

        state.company_status = "已停止"
        log_message(f"Company Deployment 已停止 (namespace: {namespace})", "INFO")
        socketio.emit('status_update', {
            'company_status': state.company_status
        })
        return jsonify({"status": "success", "message": "Company节点已停止"})

    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"❌ 停止Company节点失败: {str(e)}", file=sys.stderr, flush=True)
        print(error_traceback, file=sys.stderr, flush=True)
        log_message(
            f"停止Company节点失败: {str(e)}\n堆栈信息:\n{error_traceback}", "ERROR")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/stop_partner', methods=['POST'])
@require_write_permission
def stop_partner():
    """停止Partner节点（使用 Kubernetes API）"""
    try:
        if not K8S_AVAILABLE:
            return jsonify({"status": "error", "message": "Kubernetes API 不可用，无法在 Kubernetes 环境中停止"})

        # 加载 Kubernetes 配置
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()

        apps_v1 = client.AppsV1Api()
        namespace = os.getenv('NAMESPACE', os.getenv(
            'POD_NAMESPACE', 'mpc-test'))

        # 通过标签查找 Deployment（更可靠的方式）
        deployments = apps_v1.list_namespaced_deployment(
            namespace=namespace,
            label_selector='app.kubernetes.io/component=partner'
        )
        if not deployments.items:
            return jsonify({"status": "error", "message": f"未找到 Partner Deployment (namespace: {namespace})"})
        deployment_name = deployments.items[0].metadata.name

        # 检查 Deployment 是否存在
        try:
            deployment = apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
        except ApiException as e:
            if e.status == 404:
                return jsonify({"status": "error", "message": f"Deployment {deployment_name} 不存在"})
            raise

        # 检查当前副本数
        current_replicas = deployment.spec.replicas or 0

        if current_replicas == 0:
            state.partner_status = "已停止"
            return jsonify({"status": "success", "message": "Partner节点已停止"})

        # 停止 Deployment（设置副本数为0）
        deployment.spec.replicas = 0
        apps_v1.patch_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            body=deployment
        )

        state.partner_status = "已停止"
        log_message(f"Partner Deployment 已停止 (namespace: {namespace})", "INFO")
        socketio.emit('status_update', {
            'partner_status': state.partner_status
        })
        return jsonify({"status": "success", "message": "Partner节点已停止"})

    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"❌ 停止Partner节点失败: {str(e)}", file=sys.stderr, flush=True)
        print(error_traceback, file=sys.stderr, flush=True)
        log_message(
            f"停止Partner节点失败: {str(e)}\n堆栈信息:\n{error_traceback}", "ERROR")
        return jsonify({"status": "error", "message": str(e)})


@require_write_permission
@app.route('/api/start_training', methods=['POST'])
def start_training():
    """开始训练（通过 Kubernetes API 修改 Deployment 配置）"""
    try:
        if not K8S_AVAILABLE:
            return jsonify({"status": "error", "message": "Kubernetes API 不可用，无法在 Kubernetes 环境中启动训练"})

        # 获取训练参数（如果前端传入）
        params = request.get_json() or {}
        training_defaults = state.config.get('training', {})
        n_epochs = params.get(
            'n_epochs', training_defaults.get('default_epochs', 10))
        batch_size = params.get(
            'batch_size', training_defaults.get('default_batch_size', 1000))
        lr = params.get('lr', training_defaults.get('default_lr', 0.1))
        val_steps = params.get(
            'val_steps', training_defaults.get('validation_steps', 1))
        model = params.get('model', training_defaults.get('model', 'SSLR'))
        n_estimators = params.get(
            'n_estimators', training_defaults.get('n_estimators', 2))
        max_depth = params.get(
            'max_depth', training_defaults.get('max_depth', 2))
        k_quantiles = params.get('K_quantiles', params.get(
            'k_quantiles', training_defaults.get('K_quantiles', 20)))
        reg_coef = params.get(
            'reg_coef', training_defaults.get('reg_coef', 0.0))

        # 数据集路径（可由前端传入，否则使用默认值）
        company_train_dataset = params.get('company_train_dataset') or None
        company_val_dataset = params.get('company_val_dataset') or None
        partner_train_dataset = params.get('partner_train_dataset') or None
        partner_val_dataset = params.get('partner_val_dataset') or None

        log_message(
            f"[INFO] 训练参数 - Model: {model}, Epochs: {n_epochs}, Batch: {batch_size}, LR: {lr}, ValSteps: {val_steps}, Trees: {n_estimators}, Depth: {max_depth}, Quantiles: {k_quantiles}, Reg: {reg_coef}", "INFO")

        # 加载 Kubernetes 配置
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()

        apps_v1 = client.AppsV1Api()
        namespace = os.getenv('NAMESPACE', os.getenv(
            'POD_NAMESPACE', 'mpc-test'))

        # 通过标签查找 Deployment（更可靠的方式）
        deployments = apps_v1.list_namespaced_deployment(
            namespace=namespace,
            label_selector='app.kubernetes.io/component=company'
        )
        if not deployments.items:
            return jsonify({"status": "error", "message": f"未找到 Company Deployment (namespace: {namespace})"})
        deployment_name = deployments.items[0].metadata.name

        # 获取当前 Deployment
        try:
            deployment = apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
        except ApiException as e:
            if e.status == 404:
                return jsonify({"status": "error", "message": f"Deployment {deployment_name} 不存在"})
            raise

        # 检查 Deployment 是否在运行
        if deployment.spec.replicas == 0:
            return jsonify({"status": "error", "message": "请先启动 Company 节点"})

        # 获取容器配置
        container = deployment.spec.template.spec.containers[0]
        current_args = container.args[0] if container.args else ""

        # 从当前配置中提取端口信息（如果存在）
        # 默认值（从 values.yaml 中获取）
        spu_port = 9394
        ray_port = 6379

        # 尝试从当前 args 中提取端口（如果存在）
        spu_match = re.search(
            r'--company_spu_addr \$POD_IP:(\d+)', current_args)
        if spu_match:
            spu_port = int(spu_match.group(1))

        ray_match = re.search(r'--ray_head_addr \$POD_IP:(\d+)', current_args)
        if ray_match:
            ray_port = int(ray_match.group(1))

        # 为本次训练生成唯一 run_id 与模型目录（多版本保留）
        run_id = _new_training_run_id()
        model_dirs = _build_model_dirs(model, run_id)
        training_params = {
            "model": model,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "val_steps": val_steps,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "k_quantiles": k_quantiles,
            "reg_coef": reg_coef,
            "company_train_dataset": company_train_dataset or '/app/company/host_train.csv',
            "company_val_dataset": company_val_dataset or '/app/company/host_test.csv',
            "partner_train_dataset": partner_train_dataset or '/app/partner/guest_train.csv',
            "partner_val_dataset": partner_val_dataset or '/app/partner/guest_test.csv',
        }

        # 生成新的训练命令
        training_cmd = _generate_training_command(
            model=model,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            val_steps=val_steps,
            n_estimators=n_estimators,
            max_depth=max_depth,
            k_quantiles=k_quantiles,
            reg_coef=reg_coef,
            pod_ip_placeholder="$POD_IP",
            partner_spu_addr_placeholder="$PARTNER_SPU_ADDR",
            ray_port=ray_port,
            spu_port=spu_port,
            run_id=run_id,
            company_model_save_dir=model_dirs["company_path"],
            partner_model_save_dir=model_dirs["partner_path"],
            company_train_dataset=company_train_dataset,
            company_val_dataset=company_val_dataset,
            partner_train_dataset=partner_train_dataset,
            partner_val_dataset=partner_val_dataset
        )

        # 构建完整的启动脚本
        # 从当前 args 中提取启动脚本的前半部分（Ray 启动部分）
        if "exec python" in current_args:
            # 旧格式：有 exec python，提取之前的部分
            script_prefix = current_args.split("exec python")[0]
            # 提取 PARTNER_SERVICE_NAME（如果存在）
            partner_service_match = re.search(
                r'PARTNER_SERVICE_NAME="([^"]+)"', current_args)
            partner_service_name = partner_service_match.group(
                1) if partner_service_match else "mpc-test-mobile-mpc-project-partner-svc"
            new_args = script_prefix + training_cmd
        elif "tail -f /dev/null" in current_args:
            # 新格式：有 tail -f /dev/null，替换为训练命令
            script_prefix = current_args.split("tail -f /dev/null")[0]
            # 提取 PARTNER_SERVICE_NAME（如果存在）
            partner_service_match = re.search(
                r'PARTNER_SERVICE_NAME="([^"]+)"', current_args)
            partner_service_name = partner_service_match.group(
                1) if partner_service_match else "mpc-test-mobile-mpc-project-partner-svc"
            new_args = script_prefix + training_cmd
        else:
            # 如果找不到 exec python，使用默认的启动脚本模板
            # 尝试从 Deployment 标签获取 release name
            release_name = deployment.metadata.labels.get(
                'app.kubernetes.io/instance', 'mpc-test')
            partner_service_name = f"{release_name}-mobile-mpc-project-partner-svc"

            script_template = dedent("""
                                echo "🚀 启动 Company 节点..."

                                # 清理旧的 Ray 资源（如果存在）
                                echo "🧹 清理旧的 Ray 资源..."
                                ray stop --force || true
                                sleep 2

                                # 使用 K8s Downward API 注入的 POD_IP（避免 hostname -i 多网卡问题）
                                export POD_IP=${POD_IP:-$(hostname -i | tr ' ' '\n' | head -1)}
                                echo "📍 Pod IP: $POD_IP"

                                # 使用 Ray CLI 启动 Ray Head（指定端口 {ray_port}）
                                echo "📍 启动 Ray Head (端口: {ray_port})..."
                                ray start --head \
                                    --port={ray_port} \
                                    --num-cpus=8 \
                                    --resources='{{"company": 10}}' \
                                    --object-store-memory=2000000000 \
                                    --include-dashboard=false \
                                    --node-ip-address=$POD_IP &

                                RAY_PID=$!
                                echo "Ray Head 后台进程 PID: $RAY_PID"

                                # 等待 Ray Head 启动
                                echo "⏳ 等待 Ray Head 准备就绪..."
                                sleep 5

                                # 验证 Ray Head 是否启动成功
                                if ! ray status &>/dev/null; then
                                    echo "⚠️  Ray Head 启动检查失败，继续执行..."
                                else
                                    echo "✅ Ray Head 已启动"
                                fi

                                # 解析 Partner Service 的 Pod IP
                                echo "🔍 解析 Partner Service 地址..."
                                PARTNER_SERVICE_NAME="{partner_service_name}"
                                PARTNER_IP=$(getent hosts $PARTNER_SERVICE_NAME | awk '{{ print $1 }}' | head -1)
                                if [ -z "$PARTNER_IP" ]; then
                                    echo "⚠️  无法解析 Partner Service，使用 Service 名称"
                                    PARTNER_SPU_ADDR="{partner_service_name}:9395"
                                else
                                    echo "✅ Partner IP: $PARTNER_IP"
                                    PARTNER_SPU_ADDR="$PARTNER_IP:9395"
                                fi

                                # Coordinator 地址优先使用环境变量
                                if [ -z "$COORDINATOR_SPU_ADDR" ]; then
                                    echo "⚠️  未设置 COORDINATOR_SPU_ADDR，回退到本地端口"
                                    COORDINATOR_SPU_ADDR="$POD_IP:9396"
                                fi

                                # 运行 Company 程序
                                echo "🎯 启动 Company MPC 节点..."
                                echo "📍 Ray Head 地址: $POD_IP:{ray_port}"
                                echo "📍 Coordinator SPU 地址: $COORDINATOR_SPU_ADDR"

                                {training_cmd}
                        """)

            new_args = script_template.format(
                ray_port=ray_port,
                partner_service_name=partner_service_name,
                training_cmd=training_cmd
            )

        # 更新 Deployment 配置
        container.args[0] = new_args

        # 添加注释标记，方便识别
        # 修复：确保 metadata 和 annotations 存在
        if deployment.spec.template.metadata is None:
            from kubernetes.client import V1ObjectMeta
            deployment.spec.template.metadata = V1ObjectMeta()

        if deployment.spec.template.metadata.annotations is None:
            deployment.spec.template.metadata.annotations = {}

        deployment.spec.template.metadata.annotations[
            'training-config'] = f"model={model},updated={datetime.now().isoformat()}"

        # 更新 Deployment（包括新的训练命令和注释）
        apps_v1.patch_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            body=deployment
        )

        # 重启 Deployment 以应用新配置
        # 使用 patch_namespaced_deployment 而不是 patch_namespaced_deployment_scale
        # 因为 RBAC 只授予了 deployments 的 patch 权限，没有 deployments/scale 权限
        current_replicas = deployment.spec.replicas or 1

        # 先缩容到0（如果当前不是0）
        if current_replicas > 0:
            # 只 patch spec.replicas 字段，使用字典方式更简单
            apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body={'spec': {'replicas': 0}}
            )
            # 等待 Pod 完全停止，确保 Ray 资源被释放
            log_message("等待 Pod 停止并释放 Ray 资源...", "INFO")
            time.sleep(5)  # 增加等待时间，确保 Ray 资源完全释放

        # 再扩容到1
        apps_v1.patch_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            body={'spec': {'replicas': 1}}
        )

        # 重置训练状态和指标（为新 run 清空旧状态）
        state.training_status = "PSI中"  # 初始状态设为PSI中，因为训练开始时会先执行PSI
        state.training_run_id = run_id
        state.training_started_at = datetime.now().isoformat()
        state.training_finished_at = None
        state.current_step = 0
        state.current_epoch = 0
        state.total_epochs = n_epochs if model == "SSLR" else 0
        state.accuracy = 0.0
        state.loss = 0.0

        # 记录本次训练的待确认模型信息（训练完成后写入注册表）
        state.pending_training_model = {
            "run_id": run_id,
            "model_type": model_dirs["model_type"],
            "training_params": training_params,
            "company": {"model_name": model_dirs["company_model_name"], "path": model_dirs["company_path"],
                        "node_type": "company"},
            "partner": {"model_name": model_dirs["partner_model_name"], "path": model_dirs["partner_path"],
                        "node_type": "partner"},
        }

        # 🔥 重置清理标志，允许新训练完成后再次清理配置
        if hasattr(state, '_deployment_cleaned_after_training'):
            delattr(state, '_deployment_cleaned_after_training')

        log_message(
            f"训练任务已启动 - run_id={run_id}, Model={model} (namespace: {namespace})", "INFO")
        return jsonify({
            "status": "success",
            "message": f"已启动 {model} 训练",
            "training_status": "PSI中",
            "run_id": run_id
        })

    except Exception as e:
        # 获取完整的异常堆栈
        error_traceback = traceback.format_exc()
        error_msg = f"启动训练失败: {str(e)}\n堆栈信息:\n{error_traceback}"

        # 输出到 stderr（kubectl logs 能看到）
        print(f"❌ 启动训练失败: {str(e)}", file=sys.stderr, flush=True)
        print(error_traceback, file=sys.stderr, flush=True)

        log_message(error_msg, "ERROR")
        return jsonify({"status": "error", "message": str(e)})


@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    log_message("客户端已连接", "INFO")
    emit('status_update', {
        'company_status': state.company_status,
        'partner_status': state.partner_status,
        'training_status': state.training_status,
        'current_step': state.current_step,
        'total_steps': state.total_steps,
        'accuracy': state.accuracy,
        'loss': state.loss
    })


@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开连接"""
    log_message("客户端已断开连接", "INFO")


if __name__ == '__main__':
    # 确保必要的目录存在
    os.makedirs(project_root / "web_ui" / "templates", exist_ok=True)
    os.makedirs(project_root / "web_ui" / "static", exist_ok=True)

    log_message("联邦学习Web UI服务启动", "INFO")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)


# ==================== 模型参数管理 API ====================

@app.route('/api/models/list')
def list_models():
    """获取所有训练完成的模型列表。
    优先使用本次会话的训练注册表（不依赖跨 Pod 文件系统访问）；
    若注册表为空，则扫描本地文件系统，但只返回应用启动后新写入的模型
    （通过文件 mtime 过滤，排除镜像内预置的旧模型）。
    """
    try:
        # Partner 端直接代理 Company 的模型注册视图，保证两端看到同一批训练结果
        if IS_READONLY and COMPANY_WEBUI_URL:
            try:
                import urllib.request as _urllib_req
                with _urllib_req.urlopen(
                    f"{COMPANY_WEBUI_URL}/api/models/list", timeout=3
                ) as resp:
                    data = json.loads(resp.read().decode())
                    if data.get('status') == 'success':
                        return jsonify(data)
            except Exception:
                pass

        # ── 优先：使用内存注册表（本次 session 训练结果）──────────────────
        if state.trained_model_registry:
            ordered = sorted(
                state.trained_model_registry,
                key=lambda x: x.get('trained_at', ''),
                reverse=True
            )
            return jsonify({
                'status': 'success',
                'models': [
                    {k: v for k, v in m.items() if not k.startswith('_')}
                    for m in ordered
                ]
            })

        # ── 降级：扫描本地文件系统，过滤掉镜像内预置模型 ─────────────────
        models = []
        for node_type in ['company', 'partner']:
            models_dir = project_root / node_type / 'models'
            if not models_dir.exists():
                continue

            for model_dir in models_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                # 获取目录内所有文件
                model_files_raw = []
                for file_path in model_dir.iterdir():
                    if file_path.is_file():
                        st = file_path.stat()
                        model_files_raw.append({
                            'name': file_path.name,
                            'size': st.st_size,
                            'modified': datetime.fromtimestamp(st.st_mtime).isoformat(),
                            '_mtime': st.st_mtime,
                        })

                if not model_files_raw:
                    continue

                # 只展示在应用启动之后写入的模型（排除镜像内预置的旧文件）
                newest_mtime = max(f['_mtime'] for f in model_files_raw)
                if newest_mtime <= APP_START_TIME:
                    continue

                model_type = 'unknown'
                if 'lr' in model_dir.name.lower():
                    model_type = 'SSLR (逻辑回归)'
                elif 'xgb' in model_dir.name.lower():
                    model_type = 'SSXGBoost'

                model_files = [
                    {k: v for k, v in f.items() if k != '_mtime'}
                    for f in model_files_raw
                ]
                models.append({
                    'node_type': node_type,
                    'model_name': model_dir.name,
                    'model_type': model_type,
                    'files': model_files,
                    'path': str(model_dir),
                })

        models = sorted(models, key=lambda x: x.get('path', ''), reverse=True)
        return jsonify({'status': 'success', 'models': models})

    except Exception as e:
        log_message(f"获取模型列表失败: {str(e)}", "ERROR")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/models/download/<node_type>/<model_name>/<filename>')
def download_model_file(node_type, model_name, filename):
    """下载指定的模型文件"""
    try:
        # 构建文件路径
        model_dir = project_root / node_type / 'models' / model_name
        file_path = model_dir / filename

        # 安全检查：确保文件在 models 目录下
        if not file_path.exists() or not file_path.is_file():
            return jsonify({
                'status': 'error',
                'message': '文件不存在'
            }), 404

        # 检查文件是否在允许的目录中
        if not str(file_path.resolve()).startswith(str(model_dir.resolve())):
            return jsonify({
                'status': 'error',
                'message': '非法的文件路径'
            }), 403

        # 返回文件
        return send_from_directory(
            model_dir,
            filename,
            as_attachment=True,
            download_name=f"{node_type}_{model_name}_{filename}"
        )

    except Exception as e:
        log_message(f"下载模型文件失败: {str(e)}", "ERROR")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/models/view/<node_type>/<model_name>/<filename>')
def view_model_file(node_type, model_name, filename):
    """查看模型文件内容（文本文件）"""
    try:
        # 构建文件路径
        model_dir = project_root / node_type / 'models' / model_name
        file_path = model_dir / filename

        # 安全检查
        if not file_path.exists() or not file_path.is_file():
            return jsonify({
                'status': 'error',
                'message': '文件不存在'
            }), 404

        if not str(file_path.resolve()).startswith(str(model_dir.resolve())):
            return jsonify({
                'status': 'error',
                'message': '非法的文件路径'
            }), 403

        # 只支持文本文件查看
        text_extensions = ['.json', '.csv', '.txt', '.log']
        if file_path.suffix.lower() not in text_extensions:
            return jsonify({
                'status': 'error',
                'message': '不支持查看此类型文件，请下载后查看'
            }), 400

        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 如果是 JSON 文件，格式化输出
        if file_path.suffix.lower() == '.json':
            try:
                json_data = json.loads(content)
                content = json.dumps(json_data, indent=2, ensure_ascii=False)
            except:
                pass

        return jsonify({
            'status': 'success',
            'filename': filename,
            'content': content,
            'size': file_path.stat().st_size
        })

    except Exception as e:
        log_message(f"查看模型文件失败: {str(e)}", "ERROR")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/datasets/list')
def list_datasets():
    """获取当前节点可用的 CSV 数据集列表（扫描节点根目录和持久化上传目录）。
    每端只返回自身角色对应的数据集，不做跨节点查询。
    """
    try:
        own_type = 'partner' if IS_READONLY else 'company'
        scan_dirs = [
            project_root / own_type,
            project_root / own_type / 'Datasets' / 'uploads',
        ]
        seen = set()
        csv_files = []
        for scan_dir in scan_dirs:
            if not scan_dir.exists():
                continue
            for f in sorted(scan_dir.iterdir()):
                if f.is_file() and f.suffix.lower() == '.csv' and f.name not in seen:
                    seen.add(f.name)
                    csv_files.append({
                        'name': f.name,
                        'path': str(f),
                        'size': f.stat().st_size,
                        'modified': datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                    })

        return jsonify({'status': 'success', 'datasets': {own_type: csv_files}})

    except Exception as e:
        log_message(f"获取数据集列表失败: {str(e)}", "ERROR")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/datasets/upload', methods=['POST'])
def upload_dataset():
    """上传 CSV 数据集文件到本节点的持久化目录（Datasets/uploads）。
    每端只能上传自身角色对应的数据集（Company 上传 company 类型，Partner 上传 partner 类型）。
    """
    try:
        own_type = 'partner' if IS_READONLY else 'company'
        node_type = request.form.get('node_type', own_type)

        # 每端只允许上传自己的数据集
        if node_type != own_type:
            return jsonify({
                'status': 'error',
                'message': f'当前节点只能上传 {own_type} 类型的数据集'
            }), 403

        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': '未找到上传文件'}), 400
        file = request.files['file']
        if not file.filename:
            return jsonify({'status': 'error', 'message': '文件名为空'}), 400
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'status': 'error', 'message': '只允许上传 CSV 文件'}), 400

        import re as _re
        safe_name = _re.sub(r'[^\w\-.]', '_', file.filename)

        save_dir = project_root / own_type / 'Datasets' / 'uploads'
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / safe_name
        file.save(str(save_path))

        log_message(
            f"数据集已上传: {own_type}/Datasets/uploads/{safe_name} ({save_path.stat().st_size} bytes)",
            "INFO"
        )
        return jsonify({
            'status': 'success',
            'message': f'文件 {safe_name} 已上传',
            'path': str(save_path),
            'filename': safe_name
        })

    except Exception as e:
        log_message(f"上传数据集失败: {str(e)}", "ERROR")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/models/delete/<node_type>/<model_name>', methods=['POST'])
@require_write_permission
def delete_model(node_type, model_name):
    """删除指定的模型（仅 Company 节点可用）"""
    try:
        # 从内存注册表中移除（同时覆盖同类型的两端模型，因为注册表是按训练批次写入的）
        before = len(state.trained_model_registry)
        state.trained_model_registry = [
            m for m in state.trained_model_registry
            if not (m['node_type'] == node_type and m['model_name'] == model_name)
        ]
        removed_from_registry = len(state.trained_model_registry) < before

        # 尝试删除本地文件系统中的目录（WebUI Pod 可能无法访问实际模型文件）
        model_dir = project_root / node_type / 'models' / model_name
        if model_dir.exists() and model_dir.is_dir():
            import shutil
            shutil.rmtree(model_dir)
            log_message(f"已删除模型文件: {node_type}/{model_name}", "INFO")

        if removed_from_registry or model_dir.exists() is False:
            log_message(f"已删除模型注册: {node_type}/{model_name}", "INFO")
            return jsonify({'status': 'success', 'message': f'模型 {model_name} 已删除'})

        return jsonify({'status': 'error', 'message': '模型不存在'}), 404

    except Exception as e:
        log_message(f"删除模型失败: {str(e)}", "ERROR")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ──────────────────────────────────────────────────────────────────────────────
# 推理相关接口
# ──────────────────────────────────────────────────────────────────────────────

# 推理任务全局状态
infer_state = {
    'status': 'idle',       # idle / running / success / error
    'result': None,         # 推理结果 list[{id, prediction}]
    'message': '',
    'started_at': None,
    'finished_at': None,
}


def _run_infer_task(model_type: str, company_model_path: str, partner_model_path: str,
                    company_data_path: str, partner_data_path: str):
    """
    在后台线程中以子进程方式执行推理（仿照训练的子进程方式）。
    调用 company/infer_run.py，从其 stdout 最后一行解析 JSON 结果。
    """
    import json as _json

    try:
        infer_state['status'] = 'running'
        infer_state['message'] = '推理任务启动中...'
        log_message("开始执行推理任务（子进程模式）", "INFO")

        # 优先使用 K8s 注入的环境变量（与训练脚本保持一致）
        # POD_IP / HOST_IP 由 commonEnv helper 注入；PARTNER_SPU_ADDR 由 values 注入
        import socket as _socket
        pod_ip = (os.environ.get('POD_IP') or
                  os.environ.get('HOST_IP') or
                  _socket.gethostbyname(_socket.gethostname()))

        # SPU/Ray 端口与 Helm values 保持一致（固定值）
        spu_port = 9394
        ray_port_num = 6379

        company_spu_addr = f'{pod_ip}:{spu_port}'
        ray_head_addr = f'{pod_ip}:{ray_port_num}'

        # Partner 地址：优先使用环境变量，否则从 config.yaml 拼接
        partner_spu_addr = os.environ.get('PARTNER_SPU_ADDR', '')
        if not partner_spu_addr:
            cfg = state.config
            partner_cfg = cfg.get('partner', {})
            partner_ip = partner_cfg.get('ip', '127.0.0.1')
            partner_spu_addr = f"{partner_ip}:9395"

        # Coordinator 地址优先使用环境变量，否则回退到 config.yaml
        coordinator_spu_addr = os.environ.get('COORDINATOR_SPU_ADDR', '')
        if not coordinator_spu_addr:
            cfg = state.config
            coordinator_cfg = cfg.get('coordinator', {})
            coordinator_ip = coordinator_cfg.get('ip', '127.0.0.1')
            coordinator_port = coordinator_cfg.get('port_spu', 9396)
            coordinator_spu_addr = f"{coordinator_ip}:{coordinator_port}"

        infer_script = str(project_root / 'company' / 'infer_run.py')
        cmd = [
            'python', infer_script,
            '--mode',                 'multi_distributed',
            '--ray_head_addr',        ray_head_addr,
            '--company_spu_addr',     company_spu_addr,
            '--partner_spu_addr',     partner_spu_addr,
            '--coordinator_spu_addr', coordinator_spu_addr,
            '--model',                model_type,
            '--company_model_path',   company_model_path,
            '--partner_model_path',   partner_model_path,
            '--company_data_path',    company_data_path,
            '--partner_data_path',    partner_data_path,
        ]
        log_message(f"推理命令: {' '.join(cmd)}", "INFO")

        infer_state['message'] = '子进程启动中，等待推理完成...'
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(project_root / 'company'),
        )

        # 将 stderr 写入日志，方便排查
        if proc.stderr:
            for line in proc.stderr.strip().splitlines():
                log_message(f"[infer_run] {line}", "INFO")

        if proc.returncode != 0:
            raise RuntimeError(
                f"infer_run.py 退出码 {proc.returncode}:\n{proc.stderr[-2000:]}")

        # 从 stdout 最后一行解析 JSON 结果
        stdout_lines = [l for l in proc.stdout.strip().splitlines()
                        if l.strip()]
        if not stdout_lines:
            raise RuntimeError("infer_run.py 无输出")
        last_line = stdout_lines[-1]
        output = _json.loads(last_line)

        if output.get('status') != 'success':
            raise RuntimeError(output.get('result', '推理失败'))

        result = output['result']
        infer_state['status'] = 'success'
        infer_state['result'] = result
        infer_state['message'] = f'推理完成，共 {len(result)} 条结果'
        infer_state['finished_at'] = datetime.now().isoformat()
        log_message(f"推理完成，共 {len(result)} 条结果", "INFO")

    except Exception as e:
        infer_state['status'] = 'error'
        infer_state['message'] = str(e)
        infer_state['finished_at'] = datetime.now().isoformat()
        log_message(f"推理失败: {str(e)}", "ERROR")


@app.route('/api/infer/start', methods=['POST'])
@require_write_permission
def start_infer():
    """启动推理任务"""
    global infer_state
    if infer_state['status'] == 'running':
        return jsonify({'status': 'error', 'message': '推理任务正在执行中，请等待'}), 400

    params = request.get_json() or {}
    model_type = params.get('model_type', 'SSLR')
    company_model_path = params.get('company_model_path', '')
    partner_model_path = params.get('partner_model_path', '')
    company_data_path = params.get('company_data_path', '')
    partner_data_path = params.get('partner_data_path', '') or ''

    # Partner 推理数据路径：未指定时自动从 Partner WebUI 获取，或使用内置默认路径
    if not partner_data_path:
        partner_data_path = _resolve_partner_datasets(prefer_infer=True)
        log_message(f"Partner 推理数据集自动解析为: {partner_data_path}", "INFO")

    if not all([company_model_path, partner_model_path, company_data_path]):
        return jsonify({'status': 'error', 'message': '请提供模型路径和 Company 数据路径'}), 400

    infer_state = {
        'status': 'running',
        'result': None,
        'message': '推理任务已提交',
        'started_at': datetime.now().isoformat(),
        'finished_at': None,
    }

    t = threading.Thread(
        target=_run_infer_task,
        args=(model_type, company_model_path, partner_model_path,
              company_data_path, partner_data_path),
        daemon=True
    )
    t.start()

    return jsonify({'status': 'success', 'message': '推理任务已启动'})


@app.route('/api/infer/status')
def get_infer_status():
    """查询推理任务状态和结果"""
    resp = {
        'status': infer_state['status'],
        'message': infer_state['message'],
        'started_at': infer_state['started_at'],
        'finished_at': infer_state['finished_at'],
    }
    if infer_state['status'] == 'success':
        resp['result'] = infer_state['result']
        resp['count'] = len(infer_state['result']
                            ) if infer_state['result'] else 0
    return jsonify(resp)


@app.route('/api/infer/reset', methods=['POST'])
@require_write_permission
def reset_infer():
    """重置推理状态"""
    global infer_state
    if infer_state['status'] == 'running':
        return jsonify({'status': 'error', 'message': '推理任务正在执行中，无法重置'}), 400
    infer_state = {
        'status': 'idle',
        'result': None,
        'message': '',
        'started_at': None,
        'finished_at': None,
    }
    return jsonify({'status': 'success', 'message': '推理状态已重置'})
