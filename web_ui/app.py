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
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room
import psutil
import yaml

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'federated_learning_secret_key_2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 全局状态管理
class SystemState:
    def __init__(self):
        self.company_status = "未启动"
        self.partner_status = "未启动"
        self.training_status = "未开始"
        self.current_step = 0
        self.total_steps = 0
        self.accuracy = 0.0
        self.loss = 0.0
        self.processes = {}
        self.logs = []
        self.config = self.load_config()
        
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
                    'path': '/home/dxn/mobile_project3/company'
                },
                'partner': {
                    'ip': '210.28.133.104',
                    'port_spu': 11002,
                    'path': '/home/dxn/mobile_project3/partner'
                },
                'coordinator': {
                    'ip': '210.28.133.104',
                    'port_spu': 11003
                }
            }
    
    def save_config(self):
        """保存配置文件"""
        config_path = project_root / "web_ui" / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

# 全局状态实例
state = SystemState()

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
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """获取系统状态"""
    return jsonify({
        "company_status": state.company_status,
        "partner_status": state.partner_status,
        "training_status": state.training_status,
        "current_step": state.current_step,
        "total_steps": state.total_steps,
        "accuracy": state.accuracy,
        "loss": state.loss,
        "system_info": get_system_info(),
        "logs": state.logs[-50:]  # 最近50条日志
    })

@app.route('/api/config', methods=['GET', 'POST'])
def config_api():
    """配置管理API"""
    if request.method == 'GET':
        return jsonify(state.config)
    elif request.method == 'POST':
        new_config = request.json
        state.config.update(new_config)
        state.save_config()
        log_message("配置已更新", "INFO")
        return jsonify({"status": "success"})

@app.route('/api/start_company', methods=['POST'])
def start_company():
    """启动Company节点"""
    try:
        if state.company_status == "运行中":
            return jsonify({"status": "error", "message": "Company节点已在运行"})
        
        # 构建启动命令
        config = state.config['company']
        cmd = [
            "bash", "startA.sh"
        ]
        
        # 修改启动脚本中的IP和端口配置
        start_script_path = Path(config['path']) / "startA.sh"
        if start_script_path.exists():
            # 读取并修改启动脚本
            with open(start_script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 替换IP和端口配置
            content = content.replace("A_IP=\"210.28.133.104\"", f"A_IP=\"{config['ip']}\"")
            content = content.replace("B_IP=\"210.28.133.104\"", f"B_IP=\"{state.config['partner']['ip']}\"")
            content = content.replace("PORT_RAY=\"20001\"", f"PORT_RAY=\"{config['port_ray']}\"")
            content = content.replace("PORT_COMPANY_SPU=\"11001\"", f"PORT_COMPANY_SPU=\"{config['port_spu']}\"")
            content = content.replace("PORT_PARTNER_SPU=\"11002\"", f"PORT_PARTNER_SPU=\"{state.config['partner']['port_spu']}\"")
            content = content.replace("PORT_COORD_SPU=\"11003\"", f"PORT_COORD_SPU=\"{state.config['coordinator']['port_spu']}\"")
            
            # 写回修改后的脚本
            with open(start_script_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # 启动进程
        process = subprocess.Popen(
            cmd,
            cwd=config['path'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        state.processes['company'] = process
        state.company_status = "启动中"
        
        # 启动监控线程
        def monitor_company():
            while process.poll() is None:
                line = process.stdout.readline()
                if line:
                    log_message(f"[Company] {line.strip()}", "INFO")
                    # 检查特定状态
                    if "多方安全计算节点启动信息" in line:
                        state.company_status = "运行中"
                        socketio.emit('status_update', {
                            'company_status': state.company_status
                        })
            state.company_status = "已停止"
            socketio.emit('status_update', {
                'company_status': state.company_status
            })
        
        threading.Thread(target=monitor_company, daemon=True).start()
        
        log_message("Company节点启动命令已执行", "INFO")
        return jsonify({"status": "success", "message": "Company节点启动中"})
        
    except Exception as e:
        log_message(f"启动Company节点失败: {str(e)}", "ERROR")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/start_partner', methods=['POST'])
def start_partner():
    """启动Partner节点"""
    try:
        if state.partner_status == "运行中":
            return jsonify({"status": "error", "message": "Partner节点已在运行"})
        
        config = state.config['partner']
        cmd = ["bash", "joinB.sh"]
        
        # 修改启动脚本
        join_script_path = Path(config['path']) / "joinB.sh"
        if join_script_path.exists():
            with open(join_script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 替换配置
            content = content.replace("A_IP=\"210.28.133.104\"", f"A_IP=\"{state.config['company']['ip']}\"")
            content = content.replace("PORT_RAY=\"20001\"", f"PORT_RAY=\"{state.config['company']['port_ray']}\"")
            
            with open(join_script_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        process = subprocess.Popen(
            cmd,
            cwd=config['path'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        state.processes['partner'] = process
        state.partner_status = "启动中"
        
        def monitor_partner():
            while process.poll() is None:
                line = process.stdout.readline()
                if line:
                    log_message(f"[Partner] {line.strip()}", "INFO")
                    if "加入Ray集群" in line or "Ray cluster" in line:
                        state.partner_status = "运行中"
                        socketio.emit('status_update', {
                            'partner_status': state.partner_status
                        })
            state.partner_status = "已停止"
            socketio.emit('status_update', {
                'partner_status': state.partner_status
            })
        
        threading.Thread(target=monitor_partner, daemon=True).start()
        
        log_message("Partner节点启动命令已执行", "INFO")
        return jsonify({"status": "success", "message": "Partner节点启动中"})
        
    except Exception as e:
        log_message(f"启动Partner节点失败: {str(e)}", "ERROR")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/stop_company', methods=['POST'])
def stop_company():
    """停止Company节点"""
    try:
        if 'company' in state.processes:
            process = state.processes['company']
            if process.poll() is None:
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
            del state.processes['company']
        
        state.company_status = "已停止"
        log_message("Company节点已停止", "INFO")
        socketio.emit('status_update', {
            'company_status': state.company_status
        })
        
        return jsonify({"status": "success"})
    except Exception as e:
        log_message(f"停止Company节点失败: {str(e)}", "ERROR")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/stop_partner', methods=['POST'])
def stop_partner():
    """停止Partner节点"""
    try:
        if 'partner' in state.processes:
            process = state.processes['partner']
            if process.poll() is None:
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
            del state.processes['partner']
        
        state.partner_status = "已停止"
        log_message("Partner节点已停止", "INFO")
        socketio.emit('status_update', {
            'partner_status': state.partner_status
        })
        
        return jsonify({"status": "success"})
    except Exception as e:
        log_message(f"停止Partner节点失败: {str(e)}", "ERROR")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/start_training', methods=['POST'])
def start_training():
    """开始训练"""
    try:
        if state.company_status != "运行中" or state.partner_status != "运行中":
            return jsonify({"status": "error", "message": "请先启动Company和Partner节点"})
        
        # 这里可以添加启动训练的逻辑
        state.training_status = "训练中"
        state.current_step = 0
        state.total_steps = 100  # 示例值
        
        log_message("开始联邦学习训练", "INFO")
        socketio.emit('status_update', {
            'training_status': state.training_status,
            'current_step': state.current_step,
            'total_steps': state.total_steps
        })
        
        return jsonify({"status": "success"})
    except Exception as e:
        log_message(f"启动训练失败: {str(e)}", "ERROR")
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
