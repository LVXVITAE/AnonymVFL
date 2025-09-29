#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络通信工具模块
支持跨机器通信的分布式联邦学习系统
"""

import socket
import time
import threading
import json
import requests
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class NetworkManager:
    """网络管理器，处理跨机器通信"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.connections = {}
        self.heartbeat_thread = None
        self.running = False
        
    def start(self):
        """启动网络管理器"""
        self.running = True
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        logger.info("网络管理器已启动")
        
    def stop(self):
        """停止网络管理器"""
        self.running = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5)
        logger.info("网络管理器已停止")
        
    def _heartbeat_loop(self):
        """心跳检测循环"""
        while self.running:
            try:
                self._check_connections()
                time.sleep(self.config.get('network', {}).get('heartbeat_interval', 10))
            except Exception as e:
                logger.error(f"心跳检测错误: {e}")
                
    def _check_connections(self):
        """检查所有连接状态"""
        for node_type, node_config in self.config.items():
            if isinstance(node_config, dict) and 'ip' in node_config:
                ip = node_config['ip']
                port = node_config.get('port_ray', 20001)
                
                is_alive = self._ping_node(ip, port)
                self.connections[node_type] = {
                    'ip': ip,
                    'port': port,
                    'alive': is_alive,
                    'last_check': time.time()
                }
                
    def _ping_node(self, ip: str, port: int, timeout: int = 5) -> bool:
        """ping节点检查连通性"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((ip, port))
            sock.close()
            return result == 0
        except Exception:
            return False
            
    def get_connection_status(self) -> Dict:
        """获取连接状态"""
        return self.connections.copy()
        
    def send_command(self, target_node: str, command: str, data: Dict = None) -> bool:
        """向目标节点发送命令"""
        try:
            if target_node not in self.connections:
                logger.error(f"未知节点: {target_node}")
                return False
                
            node_info = self.connections[target_node]
            if not node_info['alive']:
                logger.error(f"节点 {target_node} 不可达")
                return False
                
            # 这里可以实现具体的命令发送逻辑
            # 例如通过HTTP API或自定义协议
            return self._send_http_command(
                node_info['ip'], 
                node_info['port'], 
                command, 
                data
            )
            
        except Exception as e:
            logger.error(f"发送命令失败: {e}")
            return False
            
    def _send_http_command(self, ip: str, port: int, command: str, data: Dict = None) -> bool:
        """通过HTTP发送命令"""
        try:
            url = f"http://{ip}:{port}/api/{command}"
            response = requests.post(url, json=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"HTTP命令发送失败: {e}")
            return False

class NodeDiscovery:
    """节点发现服务"""
    
    def __init__(self, broadcast_port: int = 9999):
        self.broadcast_port = broadcast_port
        self.discovered_nodes = {}
        self.running = False
        
    def start_discovery(self):
        """启动节点发现"""
        self.running = True
        discovery_thread = threading.Thread(target=self._discovery_loop, daemon=True)
        discovery_thread.start()
        
    def _discovery_loop(self):
        """发现循环"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind(('', self.broadcast_port))
        
        while self.running:
            try:
                data, addr = sock.recvfrom(1024)
                node_info = json.loads(data.decode())
                self.discovered_nodes[addr[0]] = node_info
                logger.info(f"发现节点: {addr[0]} - {node_info}")
            except Exception as e:
                logger.error(f"节点发现错误: {e}")
                
    def broadcast_self(self, node_info: Dict):
        """广播自身信息"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            
            message = json.dumps(node_info).encode()
            sock.sendto(message, ('<broadcast>', self.broadcast_port))
            sock.close()
        except Exception as e:
            logger.error(f"广播失败: {e}")
            
    def get_discovered_nodes(self) -> Dict:
        """获取发现的节点"""
        return self.discovered_nodes.copy()

class NetworkMonitor:
    """网络监控器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.monitoring = False
        self.stats = {
            'total_packets': 0,
            'failed_packets': 0,
            'avg_latency': 0.0,
            'bandwidth_usage': 0.0
        }
        
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()
        
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                self._collect_network_stats()
                time.sleep(5)  # 每5秒收集一次统计信息
            except Exception as e:
                logger.error(f"网络监控错误: {e}")
                
    def _collect_network_stats(self):
        """收集网络统计信息"""
        # 这里可以实现具体的网络统计收集逻辑
        # 例如使用psutil或其他网络监控工具
        pass
        
    def get_network_stats(self) -> Dict:
        """获取网络统计信息"""
        return self.stats.copy()

def test_network_connectivity(ip: str, port: int) -> Tuple[bool, float]:
    """测试网络连通性和延迟"""
    start_time = time.time()
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((ip, port))
        sock.close()
        
        latency = (time.time() - start_time) * 1000  # 转换为毫秒
        return result == 0, latency
        
    except Exception as e:
        logger.error(f"网络测试失败: {e}")
        return False, 0.0

def get_local_ip() -> str:
    """获取本机IP地址"""
    try:
        # 创建一个UDP socket来获取本机IP
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        local_ip = sock.getsockname()[0]
        sock.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

def scan_network_range(base_ip: str, port: int, timeout: float = 1.0) -> List[str]:
    """扫描网络范围内的活跃节点"""
    active_nodes = []
    base_parts = base_ip.split('.')
    base_prefix = '.'.join(base_parts[:3])
    
    def check_ip(ip):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((ip, port))
            sock.close()
            if result == 0:
                active_nodes.append(ip)
        except Exception:
            pass
    
    # 扫描192.168.x.x网段
    threads = []
    for i in range(1, 255):
        ip = f"{base_prefix}.{i}"
        thread = threading.Thread(target=check_ip, args=(ip,))
        thread.start()
        threads.append(thread)
        
        # 限制并发数
        if len(threads) >= 50:
            for t in threads:
                t.join()
            threads = []
    
    # 等待剩余线程完成
    for thread in threads:
        thread.join()
        
    return active_nodes
