# 三节点分布式联邦学习部署指南

## 概述
本指南介绍如何将 AnonymVFL 部署到三个不同的节点上，实现真正的分布式联邦学习。

## 网络架构
- **Company 节点 (Ray Head)**: 210.28.133.104 - SPU端口 11001
- **Partner 节点**: 210.28.133.105 - SPU端口 11002  
- **Coordinator 节点**: 210.28.133.106 - SPU端口 11003
- **Ray 集群端口**: 20001

## 部署步骤

### 1. 环境准备
在每个节点上：
```bash
# 安装依赖
pip install secretflow ray

# 复制项目代码到每个节点
scp -r /home/dxn/mobile_project2/AnonymVFL user@210.28.133.105:~/
scp -r /home/dxn/mobile_project2/AnonymVFL user@210.28.133.106:~/
```

### 2. 网络配置
确保以下端口在防火墙中开放：
- Ray 集群通信端口: 20001
- SPU 通信端口: 11001, 11002, 11003
- Ray 内部通信端口: 10002-19999 (动态分配)

### 3. 启动顺序

#### 第一步: 启动 Company 节点 (210.28.133.104)
```bash
cd /home/dxn/mobile_project2/AnonymVFL
chmod +x multi_run_company.sh
./multi_run_company.sh
```

#### 第二步: 启动 Partner 节点 (210.28.133.105)
在第二台机器上：
```bash
cd ~/AnonymVFL
chmod +x multi_run_partner.sh
./multi_run_partner.sh
```

#### 第三步: 启动 Coordinator 节点 (210.28.133.106)
在第三台机器上：
```bash
cd ~/AnonymVFL
chmod +x multi_run_coordinator.sh
./multi_run_coordinator.sh
```

## 单机三节点模拟测试

如果您只有一台机器，可以使用不同的端口模拟三个节点：

### 修改后的单机配置
```bash
# 修改三个启动脚本中的IP地址都改为 210.28.133.104
# 并修改端口避免冲突：
# Company: 210.28.133.104:11001
# Partner: 210.28.133.104:11002
# Coordinator: 210.28.133.104:11003
```

## 监控和日志

### Ray 集群状态查看
```bash
ray status
ray list nodes
```

### 端口监控
```bash
netstat -tuln | grep -E ':(11001|11002|11003|20001)'
```

### 日志位置
- Ray 集群日志: `/tmp/ray/session_*/logs/`
- Python 程序输出: 终端直接输出

## 故障排除

### 常见问题

1. **连接超时**
   - 检查防火墙设置
   - 确认IP地址和端口配置正确
   - 验证网络连通性: `ping 目标IP`

2. **Ray 集群连接失败**
   - 确保 Company 节点的 Ray Head 已启动
   - 检查 20001 端口是否监听: `netstat -tuln | grep 20001`

3. **SPU 初始化失败**
   - 检查 11001-11003 端口是否被占用
   - 确认所有节点的时间同步

4. **资源不足**
   - 调整每个节点的 `--num-cpus` 参数
   - 检查内存使用情况

### 日志分析
```bash
# 查看详细的 Ray 日志
tail -f /tmp/ray/session_*/logs/raylet.out

# 查看 GCS 服务日志
tail -f /tmp/ray/session_*/logs/gcs_server.out
```

## 性能优化

1. **网络优化**
   - 使用高速网络连接
   - 确保节点间延迟较低

2. **资源分配**
   - 根据实际硬件调整 CPU 核心数
   - 合理分配内存资源

3. **批次大小调整**
   - 网络环境下可适当减小 batch_size
   - 根据网络带宽调整数据传输频率

## 安全注意事项

1. **网络安全**
   - 使用VPN或专用网络连接
   - 配置防火墙只允许必要端口通信

2. **数据安全**
   - 确保数据文件在传输前加密
   - 定期备份重要的模型和结果

3. **访问控制**
   - 限制对集群管理端口的访问
   - 使用强密码和密钥认证
