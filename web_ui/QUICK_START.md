# 快速启动指南

## 🚀 一键启动联邦学习Web UI

### 1. 环境检查
```bash
cd /home/dxn/mobile_project3/web_ui
python3 test_network.py
```

### 2. 安装依赖（如果需要）
```bash
pip3 install -r requirements.txt
```

### 3. 启动Web UI
```bash
bash start_web_ui.sh
```

### 4. 访问界面
打开浏览器访问：`http://localhost:5000`

## 📋 操作步骤

### 步骤1: 配置网络
1. 在Web界面中设置各节点IP地址
2. 点击"保存配置"

### 步骤2: 启动Company节点
1. 点击"启动Company"按钮
2. 等待状态变为"运行中"

### 步骤3: 启动Partner节点  
1. 点击"启动Partner"按钮
2. 等待状态变为"运行中"

### 步骤4: 开始训练
1. 点击"开始训练"按钮
2. 监控训练进度和性能

## 🔧 故障排除

### 问题1: 端口被占用
```bash
# 查看端口占用
netstat -tulpn | grep :5000
# 杀死占用进程
sudo kill -9 <PID>
```

### 问题2: 网络不通
```bash
# 测试连通性
ping <目标IP>
telnet <目标IP> <端口>
```

### 问题3: 权限问题
```bash
# 修复权限
chmod +x /home/dxn/mobile_project3/company/*.sh
chmod +x /home/dxn/mobile_project3/partner/*.sh
chmod +x /home/dxn/mobile_project3/web_ui/*.sh
```

## 📞 技术支持

如遇问题，请查看：
- 日志文件：`web_ui/logs/`
- 系统日志：Web界面中的日志面板
- 网络测试：运行 `python3 test_network.py`
