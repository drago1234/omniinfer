# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
import logging
import os
import sys

logger = logging.getLogger(__name__)

# Keep machine-readable command output on stdout without log prefixes.
output_logger = logging.getLogger(f"{__name__}.output")
output_logger.setLevel(logging.INFO)
output_logger.propagate = False
if not output_logger.handlers:
    output_handler = logging.StreamHandler(sys.stdout)
    output_handler.setFormatter(logging.Formatter("%(message)s"))
    output_logger.addHandler(output_handler)

GLOBAL_RANK_TABLE_KEY_ENV = 'GLOBAL_RANK_TABLE_FILE_PATH_KEY'
GROUP_ID_ENV = 'SERVER_GROUP_ID'


def get_role_node_size():
    try:
        group_id = int(os.getenv(GROUP_ID_ENV))

        global_rank_table_path_key = os.getenv(GLOBAL_RANK_TABLE_KEY_ENV)
        global_rank_table_path = os.getenv(global_rank_table_path_key)
        with open(global_rank_table_path, 'r') as file:
            buf = file.read()
        rank_table = json.loads(buf)

        target_group = rank_table['server_group_list'][group_id]
        return len(target_group['server_list'])
    except Exception as e:
        logger.error("%s", e)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output_logger.info("%s", get_role_node_size())
