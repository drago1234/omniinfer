# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import logging
import requests
import argparse
import sys
import json

logger = logging.getLogger(__name__)


def check_health(ip, port):
    url = f"http://{ip}:{port}/status"

    try:
        response = requests.get(url, timeout=5)

        # 检查HTTP状态码是否为2xx
        if not response.ok:
            logger.error(f"ERROR: Health check failed - HTTP {response.status_code}")
            return False

        # 解析JSON响应
        try:
            data = response.json()
            status = data.get('status')

            if status != 'running':
                logger.error(f"ERROR: Service status is '{status}', expected 'running'")
                return False

            logger.info("OK: Service is healthy and running")
            return True

        except json.JSONDecodeError:
            logger.error("ERROR: Invalid JSON response")
            return False

    except requests.exceptions.RequestException as e:
        logger.error(f"ERROR: Request failed - {str(e)}")
        return False


if __name__ == "__main__":
    # 设置命令行参数解析
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description='Service health checker')
    parser.add_argument('--ip', required=True, help='Service IP address')
    parser.add_argument('--port', required=True, help='Service port number')

    args = parser.parse_args()

    # 执行健康检查
    success = check_health(args.ip, args.port)

    # 返回适当的退出码
    sys.exit(0 if success else 1)
