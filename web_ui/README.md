# 联邦学习Web UI - 跨机器通信系统

这是一个支持跨机器通信的分布式联邦学习Web界面系统，基于Flask和WebSocket技术构建。

## 功能特性

- 🌐 **跨机器通信**: 支持Company和Partner节点在不同机器上运行
- 📊 **实时监控**: 实时显示系统状态、训练进度和性能指标
- 🔧 **配置管理**: 灵活的IP地址和端口配置
- 📝 **日志系统**: 完整的操作日志和错误追踪
- 🎛️ **Web控制台**: 直观的Web界面控制所有节点
- 🔄 **状态同步**: 实时状态更新和同步

## 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Company节点   │    │   Partner节点    │    │  Web UI服务器   │
│   (机器A)       │◄──►│   (机器B)       │◄──►│   (任意机器)    │
│                 │    │                 │    │                 │
│ - Ray Head      │    │ - Ray Worker    │    │ - Flask Web     │
│ - SPU Node      │    │ - SPU Node      │    │ - WebSocket     │
│ - 训练数据      │    │ - 训练数据      │    │ - 状态管理      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 安装和配置

### 1. 环境要求

- Python 3.7+
- 已安装SecretFlow环境
- 网络连通性（各节点间可互相访问）

### 2. 安装依赖

```bash
cd /home/dxn/mobile_project3/web_ui
pip install -r requirements.txt
```

### 3. 配置网络

编辑 `config.yaml` 文件，设置各节点的IP地址和端口：

```yaml
company:
  ip: "192.168.1.100"    # Company节点IP
  port_ray: 20001
  port_spu: 11001

partner:
  ip: "192.168.1.101"    # Partner节点IP  
  port_spu: 11002

coordinator:
  ip: "192.168.1.100"    # Coordinator IP
  port_spu: 11003
```

## 使用方法

### 1. 启动Web UI服务

```bash
cd /home/dxn/mobile_project3/web_ui
bash start_web_ui.sh
```

或者直接运行：

```bash
python3 app.py
```

### 2. 访问Web界面

打开浏览器访问：`http://localhost:5000`

### 3. 操作流程

1. **配置网络参数**
   - 在Web界面中设置各节点的IP地址和端口
   - 点击"保存配置"按钮

2. **启动Company节点**
   - 在Company节点机器上，点击"启动Company"按钮
   - 系统会自动修改启动脚本中的网络配置
   - 等待状态变为"运行中"

3. **启动Partner节点**
   - 在Partner节点机器上，点击"启动Partner"按钮
   - 系统会自动连接到Company节点的Ray集群
   - 等待状态变为"运行中"

4. **开始训练**
   - 当两个节点都运行后，点击"开始训练"按钮
   - 系统会启动联邦学习训练过程
   - 实时监控训练进度和性能指标

## 网络配置说明

### 端口说明

- **Ray端口**: 用于Ray集群通信（默认20001）
- **SPU端口**: 用于SecretFlow SPU通信（11001-11003）
- **Web UI端口**: Web界面服务端口（默认5000）

### 防火墙设置

确保以下端口在各节点间可访问：

```bash
# Company节点
sudo ufw allow 20001  # Ray端口
sudo ufw allow 11001  # Company SPU端口
sudo ufw allow 11003  # Coordinator SPU端口

# Partner节点  
sudo ufw allow 11002  # Partner SPU端口
sudo ufw allow 54001-54103  # Ray Worker端口范围

# Web UI服务器
sudo ufw allow 5000   # Web服务端口
```

## 故障排除

### 1. 连接问题

- 检查网络连通性：`ping <目标IP>`
- 检查端口是否开放：`telnet <IP> <端口>`
- 查看防火墙设置

### 2. 启动失败

- 检查Python环境和依赖包
- 查看日志文件中的错误信息
- 确认脚本文件权限正确

### 3. 训练问题

- 确保两个节点都已成功启动
- 检查数据文件路径是否正确
- 查看训练日志了解具体错误

## 高级功能

### 1. 自定义配置

可以通过修改 `config.yaml` 文件来自定义各种参数：

```yaml
# 训练参数
training:
  default_epochs: 20
  default_batch_size: 2000
  default_lr: 0.05

# 网络参数
network:
  timeout: 60
  retry_count: 5
  heartbeat_interval: 15
```

### 2. 日志管理

- 日志文件位置：`web_ui/logs/`
- 最大日志条数：1000条
- 支持不同级别的日志过滤

### 3. 性能监控

- 实时CPU、内存、磁盘使用率
- 网络延迟和带宽监控
- 训练进度和准确率追踪

## 开发说明

### 项目结构

```
web_ui/
├── app.py              # 主应用文件
├── network_utils.py    # 网络通信工具
├── config.yaml         # 配置文件
├── requirements.txt    # 依赖包列表
├── start_web_ui.sh     # 启动脚本
├── templates/          # HTML模板
│   └── index.html     # 主页面
└── static/            # 静态资源
```

### 扩展开发

1. **添加新的API端点**：在 `app.py` 中添加新的路由
2. **自定义前端组件**：修改 `templates/index.html`
3. **网络通信优化**：扩展 `network_utils.py`
4. **监控功能增强**：添加新的监控指标

## 许可证

本项目基于MIT许可证开源。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 项目Issues: [GitHub Issues]
- 邮箱: [your-email@example.com]

---

**注意**: 使用前请确保已正确配置网络环境，并测试各节点间的连通性。
