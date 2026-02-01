#!/usr/bin/env python3
"""
Partner 节点入口脚本 - 加入 Ray 集群作为 Worker 节点
用于 Kubernetes 部署环境
"""
import argparse
import os
import sys
import time
import signal
import logging
from typing import Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class PartnerNode:
    """Partner 节点管理类"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.ray_process: Optional[object] = None
        self.running = True

        # 注册信号处理
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """处理终止信号"""
        logger.info(f"收到信号 {signum}，准备停止 Partner 节点...")
        self.running = False
        self.stop()
        sys.exit(0)

    def start(self):
        """启动 Partner 节点并加入 Ray 集群"""
        try:
            import ray

            # 设置环境变量
            os.environ['JAX_PLATFORMS'] = 'cpu'
            os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false'
            os.environ['OMP_NUM_THREADS'] = '1'

            logger.info("=" * 60)
            logger.info("🚀 Partner 节点启动中...")
            logger.info("=" * 60)
            logger.info(f"📍 Partner SPU 监听地址: {self.args.partner_spu_addr}")
            logger.info(f"🌐 Ray Head 地址:        {self.args.ray_head_addr}")
            logger.info(f"🔗 Company SPU 地址:     {self.args.company_spu_addr}")
            logger.info(f"📦 模式:                 {self.args.mode}")
            logger.info("=" * 60)

            # 如果是分布式模式，作为 worker 加入 Ray 集群
            if self.args.mode == 'multi_distributed':
                logger.info(
                    f"正在作为 Worker 节点加入 Ray 集群: {self.args.ray_head_addr}")

                # 等待 Ray head 准备好
                self._wait_for_ray_head()

                # 加入 Ray 集群
                ray.init(
                    address=self.args.ray_head_addr,
                    namespace='default',
                    runtime_env={
                        'env_vars': {
                            'JAX_PLATFORMS': 'cpu',
                            'XLA_FLAGS': '--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false',
                            'OMP_NUM_THREADS': '1',
                        }
                    },
                    # 连接已有集群时不能指定 num_cpus, resources, object_store_memory
                    logging_level=logging.INFO,
                )

                logger.info("✅ Partner 节点已成功加入 Ray 集群")
                logger.info(f"🔍 Ray 仪表板: {ray.get_dashboard_url()}")

                # 保持运行状态
                logger.info("⏳ Partner 节点保持运行中，等待任务分配...")
                self._keep_alive()

            elif self.args.mode == 'multi_sim':
                # 模拟模式 - 本地运行
                logger.info("模拟模式：Partner 节点在本地运行")
                ray.init(
                    namespace='default',
                    num_cpus=self.args.num_cpus,
                    resources={'partner': 10},
                )
                logger.info("✅ Partner 节点已启动（模拟模式）")
                self._keep_alive()

            else:
                logger.error(f"未知的运行模式: {self.args.mode}")
                sys.exit(1)

        except KeyboardInterrupt:
            logger.info("收到中断信号，正在停止...")
            self.stop()
        except Exception as e:
            logger.error(f"❌ Partner 节点启动失败: {str(e)}", exc_info=True)
            sys.exit(1)

    def _wait_for_ray_head(self, max_retries: int = 30, retry_interval: int = 2):
        """等待 Ray head 节点准备好"""
        import socket

        host, port = self.args.ray_head_addr.split(':')
        port = int(port)

        logger.info(f"等待 Ray Head 节点准备好 ({host}:{port})...")

        for i in range(max_retries):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                sock.close()

                if result == 0:
                    logger.info(f"✅ Ray Head 节点已准备好")
                    return
            except Exception as e:
                pass

            if i < max_retries - 1:
                logger.info(f"等待 Ray Head 节点... ({i+1}/{max_retries})")
                time.sleep(retry_interval)

        raise TimeoutError(f"无法连接到 Ray Head 节点 {self.args.ray_head_addr}，超时")

    def _keep_alive(self):
        """保持进程运行"""
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("收到停止信号")

    def stop(self):
        """停止 Partner 节点"""
        try:
            import ray
            if ray.is_initialized():
                logger.info("正在断开 Ray 连接...")
                ray.shutdown()
                logger.info("✅ Partner 节点已停止")
        except Exception as e:
            logger.error(f"停止时出错: {str(e)}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Partner 节点 - MPC Worker')

    parser.add_argument(
        '--mode',
        type=str,
        default='multi_distributed',
        choices=['multi_sim', 'multi_distributed'],
        help='运行模式：multi_sim（模拟） 或 multi_distributed（分布式）'
    )

    parser.add_argument(
        '--partner_spu_addr',
        type=str,
        default='0.0.0.0:9395',
        help='Partner SPU 监听地址 (格式: host:port)'
    )

    parser.add_argument(
        '--company_spu_addr',
        type=str,
        default='company-svc:9394',
        help='Company SPU 地址 (格式: host:port)'
    )

    parser.add_argument(
        '--coordinator_spu_addr',
        type=str,
        default='coordinator-svc:9396',
        help='Coordinator SPU 地址 (格式: host:port)'
    )

    parser.add_argument(
        '--ray_head_addr',
        type=str,
        default='company-svc:6379',
        help='Ray Head 节点地址 (格式: host:port)'
    )

    parser.add_argument(
        '--num_cpus',
        type=int,
        default=8,
        help='分配给 Ray 的 CPU 核心数'
    )

    parser.add_argument(
        '--object_store_memory',
        type=int,
        default=2000000000,  # 2GB
        help='Ray 对象存储内存大小（字节）'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建必要的目录
    os.makedirs('/app/partner/models', exist_ok=True)
    os.makedirs('/app/logs', exist_ok=True)

    # 启动 Partner 节点
    partner = PartnerNode(args)
    partner.start()


if __name__ == '__main__':
    main()
