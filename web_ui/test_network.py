#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络连通性测试脚本
用于测试跨机器通信的联邦学习系统
"""

import sys
import time
import socket
import threading
import requests
from pathlib import Path
import yaml

def test_port_connectivity(ip: str, port: int, timeout: int = 5) -> bool:
    """测试端口连通性"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"连接测试失败: {e}")
        return False

def test_http_endpoint(url: str, timeout: int = 10) -> bool:
    """测试HTTP端点"""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except Exception as e:
        print(f"HTTP测试失败: {e}")
        return False

def load_config() -> dict:
    """加载配置文件"""
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

def test_network_connectivity():
    """测试网络连通性"""
    print("=" * 60)
    print("🌐 联邦学习系统网络连通性测试")
    print("=" * 60)
    
    config = load_config()
    if not config:
        print("❌ 未找到配置文件 config.yaml")
        return False
    
    all_tests_passed = True
    
    # 测试Company节点
    if 'company' in config:
        company_config = config['company']
        company_ip = company_config.get('ip', '127.0.0.1')
        ray_port = company_config.get('port_ray', 20001)
        spu_port = company_config.get('port_spu', 11001)
        
        print(f"\n📡 测试Company节点 ({company_ip})")
        print("-" * 40)
        
        # 测试Ray端口
        ray_status = test_port_connectivity(company_ip, ray_port)
        print(f"  Ray端口 {ray_port}: {'✅ 连通' if ray_status else '❌ 不通'}")
        if not ray_status:
            all_tests_passed = False
        
        # 测试SPU端口
        spu_status = test_port_connectivity(company_ip, spu_port)
        print(f"  SPU端口 {spu_port}: {'✅ 连通' if spu_status else '❌ 不通'}")
        if not spu_status:
            all_tests_passed = False
    
    # 测试Partner节点
    if 'partner' in config:
        partner_config = config['partner']
        partner_ip = partner_config.get('ip', '127.0.0.1')
        partner_spu_port = partner_config.get('port_spu', 11002)
        
        print(f"\n📡 测试Partner节点 ({partner_ip})")
        print("-" * 40)
        
        # 测试SPU端口
        partner_spu_status = test_port_connectivity(partner_ip, partner_spu_port)
        print(f"  SPU端口 {partner_spu_port}: {'✅ 连通' if partner_spu_status else '❌ 不通'}")
        if not partner_spu_status:
            all_tests_passed = False
    
    # 测试Coordinator节点
    if 'coordinator' in config:
        coordinator_config = config['coordinator']
        coordinator_ip = coordinator_config.get('ip', '127.0.0.1')
        coordinator_spu_port = coordinator_config.get('port_spu', 11003)
        
        print(f"\n📡 测试Coordinator节点 ({coordinator_ip})")
        print("-" * 40)
        
        # 测试SPU端口
        coordinator_spu_status = test_port_connectivity(coordinator_ip, coordinator_spu_port)
        print(f"  SPU端口 {coordinator_spu_port}: {'✅ 连通' if coordinator_spu_status else '❌ 不通'}")
        if not coordinator_spu_status:
            all_tests_passed = False
    
    # 测试Web UI服务
    web_ui_config = config.get('web_ui', {})
    web_ui_port = web_ui_config.get('port', 5000)
    
    print(f"\n🌐 测试Web UI服务 (localhost:{web_ui_port})")
    print("-" * 40)
    
    web_ui_status = test_port_connectivity('localhost', web_ui_port)
    print(f"  Web服务端口 {web_ui_port}: {'✅ 连通' if web_ui_status else '❌ 不通'}")
    if not web_ui_status:
        all_tests_passed = False
    
    # 测试HTTP端点
    if web_ui_status:
        http_status = test_http_endpoint(f'http://localhost:{web_ui_port}/api/status')
        print(f"  HTTP API: {'✅ 正常' if http_status else '❌ 异常'}")
        if not http_status:
            all_tests_passed = False
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 所有网络测试通过！系统可以正常运行。")
    else:
        print("⚠️  部分网络测试失败，请检查配置和网络连接。")
    print("=" * 60)
    
    return all_tests_passed

def test_script_permissions():
    """测试脚本权限"""
    print("\n🔧 测试脚本权限")
    print("-" * 40)
    
    project_root = Path(__file__).parent.parent
    scripts_to_check = [
        "company/startA.sh",
        "company/clean.sh", 
        "partner/joinB.sh",
        "partner/clean.sh",
        "web_ui/start_web_ui.sh"
    ]
    
    all_scripts_ok = True
    for script_path in scripts_to_check:
        full_path = project_root / script_path
        if full_path.exists():
            if full_path.stat().st_mode & 0o111:  # 检查执行权限
                print(f"  ✅ {script_path}")
            else:
                print(f"  ❌ {script_path} (无执行权限)")
                all_scripts_ok = False
        else:
            print(f"  ❌ {script_path} (文件不存在)")
            all_scripts_ok = False
    
    if not all_scripts_ok:
        print("\n🔧 修复脚本权限...")
        for script_path in scripts_to_check:
            full_path = project_root / script_path
            if full_path.exists():
                full_path.chmod(0o755)
                print(f"  ✅ 已修复 {script_path}")
    
    return all_scripts_ok

def test_python_dependencies():
    """测试Python依赖"""
    print("\n🐍 测试Python依赖")
    print("-" * 40)
    
    required_packages = [
        'flask', 'flask_socketio', 'psutil', 'yaml', 'requests'
    ]
    
    all_deps_ok = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (未安装)")
            all_deps_ok = False
    
    return all_deps_ok

def main():
    """主测试函数"""
    print("🚀 联邦学习Web UI系统测试")
    print("=" * 60)
    
    # 测试Python依赖
    deps_ok = test_python_dependencies()
    
    # 测试脚本权限
    scripts_ok = test_script_permissions()
    
    # 测试网络连通性
    network_ok = test_network_connectivity()
    
    print("\n📊 测试结果汇总")
    print("=" * 60)
    print(f"Python依赖: {'✅ 通过' if deps_ok else '❌ 失败'}")
    print(f"脚本权限: {'✅ 通过' if scripts_ok else '❌ 失败'}")
    print(f"网络连通: {'✅ 通过' if network_ok else '❌ 失败'}")
    
    if deps_ok and scripts_ok and network_ok:
        print("\n🎉 所有测试通过！系统已准备就绪。")
        print("💡 现在可以运行: bash start_web_ui.sh")
        return True
    else:
        print("\n⚠️  部分测试失败，请根据上述信息修复问题。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
