import contextlib
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml
from jinja2 import Environment, StrictUndefined


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from omni_cli import main as cli  # noqa: E402


def _host(role, address, master_node=None):
    env_role = {"P": "prefill", "D": "decode", "C": "proxy"}[role]
    host = {
        "ansible_host": address,
        "ansible_user": "root",
        "DOCKER_IMAGE_ID": "omni:test",
        "container_name": f"omni_{role.lower()}_{address.replace('.', '_')}",
        "ascend_rt_visible_devices": "0",
        "env": {
            "ROLE": env_role,
            "CODE_PATH": "/workspace/omniinfer",
            "LOG_PATH": "/tmp/omni-logs",
            "MODEL_PATH": "/models/test",
            "API_PORT": "8000",
            "MASTER_PORT": "8001",
            "VLLM_LLMDATADIST_ZMQ_PORT": "8002",
        },
        "args": {"extra-args": {}},
    }
    if master_node is not None:
        host["master_node"] = master_node
        host["host_ip"] = address
    return host


def _inventory():
    return {
        "all": {
            "children": {
                "P": {
                    "hosts": {
                        "p0": _host("P", "10.0.0.1", "p0"),
                        "p4": _host("P", "10.0.0.5", "p4"),
                    }
                },
                "D": {"hosts": {"d0": _host("D", "10.0.1.1", "d0")}},
                "C": {"hosts": {"c0": _host("C", "10.0.2.1")}},
            }
        }
    }


def _single_role_inventory(role):
    host_name = "p0" if role == "P" else "d0"
    address = "10.0.0.1" if role == "P" else "10.0.1.1"
    host = _host(role, address, host_name)
    host["env"]["SERVER_IP_LIST"] = "192.0.2.10,192.0.2.11"
    return {
        "all": {
            "children": {
                role: {"hosts": {host_name: host}},
            }
        }
    }


def _prefill_proxy_inventory():
    inventory = _single_role_inventory("P")
    proxy_host = _host("C", "10.0.0.1")
    proxy_host["use_omni_proxy"] = True
    proxy_host["env"]["API_PORT"] = "7200"
    proxy_host["env"].pop("MASTER_PORT")
    proxy_host["env"].pop("VLLM_LLMDATADIST_ZMQ_PORT")
    inventory["all"]["children"]["C"] = {
        "hosts": {"c0": proxy_host}
    }
    return inventory


class FakeProcess:
    def __init__(self):
        self.stdout = []
        self.stderr = []
        self.returncode = 0

    def wait(self):
        return self.returncode


class SingleRoleSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.inventory_path = Path(self.temp_dir.name) / "server_profiles.yml"
        with self.inventory_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(_inventory(), stream, sort_keys=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_start_full_inventory_processes_all_pd_hosts(self):
        with mock.patch.object(cli, "omni_ranktable"), \
                mock.patch.object(cli.omni_cli.proxy, "omni_run_proxy"), \
                mock.patch.object(cli, "execute_command", return_value=0) as execute:
            cli.omni_cli_start(
                str(self.inventory_path),
                skip_verify_config=True,
            )

        commands = [call.args[0] for call in execute.call_args_list]
        self.assertEqual(len(commands), 3)
        self.assertTrue(any("ansible p0 " in command for command in commands))
        self.assertTrue(any("ansible p4 " in command for command in commands))
        self.assertTrue(any("ansible d0 " in command for command in commands))

    def test_start_supports_single_role_inventory_without_proxy(self):
        for role, host_name in (("P", "p0"), ("D", "d0")):
            with self.subTest(role=role):
                inventory = _single_role_inventory(role)
                with self.inventory_path.open("w", encoding="utf-8") as stream:
                    yaml.safe_dump(inventory, stream, sort_keys=False)

                with mock.patch.object(cli, "omni_ranktable") as ranktable, \
                        mock.patch.object(cli.omni_cli.proxy, "omni_run_proxy") as proxy, \
                        mock.patch.object(cli, "execute_command", return_value=0) as execute:
                    cli.omni_cli_start(str(self.inventory_path))

                ranktable.assert_called_once_with(str(self.inventory_path))
                proxy.assert_not_called()
                execute.assert_called_once()
                self.assertIn(f"ansible {host_name} ", execute.call_args.args[0])

                with self.inventory_path.open("r", encoding="utf-8") as stream:
                    persisted = yaml.safe_load(stream)
                persisted_host = persisted["all"]["children"][role]["hosts"][host_name]
                self.assertEqual(
                    persisted_host["env"]["SERVER_IP_LIST"],
                    "192.0.2.10,192.0.2.11",
                )

    def test_proxy_only_supports_prefill_inventory_with_proxy(self):
        with self.inventory_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(_prefill_proxy_inventory(), stream, sort_keys=False)

        with mock.patch.object(cli, "omni_ranktable") as ranktable, \
                mock.patch.object(cli.omni_cli.proxy, "omni_run_proxy") as proxy, \
                mock.patch.object(cli, "execute_command") as execute:
            cli.omni_cli_start(str(self.inventory_path), proxy_only=True)

        ranktable.assert_not_called()
        execute.assert_not_called()
        proxy.assert_called_once_with(
            str(self.inventory_path),
            cloud_mode=False,
            allow_dummy_decode=True,
        )

    def test_prefill_start_does_not_auto_start_incomplete_proxy(self):
        with self.inventory_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(_prefill_proxy_inventory(), stream, sort_keys=False)

        with mock.patch.object(cli, "omni_ranktable") as ranktable, \
                mock.patch.object(cli.omni_cli.proxy, "omni_run_proxy") as proxy, \
                mock.patch.object(cli, "execute_command", return_value=0) as execute:
            cli.omni_cli_start(str(self.inventory_path))

        ranktable.assert_called_once_with(str(self.inventory_path))
        proxy.assert_not_called()
        execute.assert_called_once()
        self.assertIn("ansible p0 ", execute.call_args.args[0])

    def test_proxy_uses_dummy_decode_for_prefill_only_inventory(self):
        inventory = _prefill_proxy_inventory()
        endpoints = cli.omni_cli.proxy.register_values(
            inventory,
            allow_dummy_decode=True,
        )

        self.assertEqual(endpoints["PREFILL_API_SERVER_LIST"], "10.0.0.1:8000")
        self.assertEqual(
            endpoints["DECODE_API_SERVER_LIST"],
            cli.omni_cli.proxy.DUMMY_DECODE_API_SERVER,
        )

        original_dir = Path.cwd()
        try:
            os.chdir(self.temp_dir.name)
            cli.omni_cli.proxy.process_results(
                endpoints,
                inventory,
                self.inventory_path,
                cloud_mode=True,
            )
            script = Path("omni_start_c.sh").read_text(encoding="utf-8")
        finally:
            os.chdir(original_dir)

        self.assertIn("--prefill-endpoints 10.0.0.1:8000", script)
        self.assertIn("--decode-endpoints 127.0.0.1:1", script)

    def test_proxy_keeps_real_decode_endpoints_when_present(self):
        endpoints = cli.omni_cli.proxy.register_values(
            _inventory(),
            allow_dummy_decode=True,
        )

        self.assertEqual(endpoints["DECODE_API_SERVER_LIST"], "10.0.1.1:8000@1")

    def test_proxy_start_matrix_with_and_without_run_dev(self):
        for inventory in (_inventory(), _prefill_proxy_inventory()):
            has_decode = "D" in inventory["all"]["children"]
            with self.inventory_path.open("w", encoding="utf-8") as stream:
                yaml.safe_dump(inventory, stream, sort_keys=False)
            for dev, proxy_only in ((False, False), (True, False),
                                    (False, True), (True, True)):
                with self.subTest(has_decode=has_decode, dev=dev, proxy_only=proxy_only), \
                        mock.patch.object(cli, "omni_ranktable") as ranktable, \
                        mock.patch.object(cli.omni_cli.proxy, "omni_run_proxy") as proxy, \
                        mock.patch.object(cli, "execute_command", return_value=0) as execute:
                    cli.omni_cli_start(str(self.inventory_path), skip_verify_config=True,
                                       dev=dev, proxy_only=proxy_only)
                    self.assertEqual(ranktable.called, not dev and not proxy_only)
                    self.assertEqual(proxy.called, proxy_only or (not dev and has_decode))
                    self.assertEqual(execute.called, not proxy_only)

    def test_stop_full_inventory_processes_all_pd_hosts(self):
        with mock.patch.object(cli, "execute_command", return_value=0) as execute:
            cli.omni_cli_stop(str(self.inventory_path))

        commands = [call.args[0] for call in execute.call_args_list]
        self.assertEqual(len(commands), 3)
        self.assertTrue(any("ansible p0 " in command for command in commands))
        self.assertTrue(any("ansible p4 " in command for command in commands))
        self.assertTrue(any("ansible d0 " in command for command in commands))

    def test_stop_and_sync_only_process_hosts_from_p_only_inventory(self):
        with self.inventory_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(_single_role_inventory("P"), stream, sort_keys=False)

        with mock.patch.object(cli, "execute_command", return_value=0) as execute:
            cli.omni_cli_stop(str(self.inventory_path))

        execute.assert_called_once()
        self.assertIn("ansible p0 ", execute.call_args.args[0])

        captured = {}

        def fake_popen(command, **_kwargs):
            captured["script"] = Path(command[0]).read_text(encoding="utf-8")
            return FakeProcess()

        with mock.patch.object(cli.subprocess, "Popen", side_effect=fake_popen):
            cli.sync_code(str(self.inventory_path), code_path="/source")

        self.assertIn("on p0", captured["script"])
        self.assertNotIn("on d0", captured["script"])
        self.assertNotIn("on c0", captured["script"])


class ConfigPathCliTests(unittest.TestCase):
    def _run_main(self, argv):
        with mock.patch.object(sys, "argv", ["omni_cli", *argv]):
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main()

    def test_start_cli_forwards_config_path(self):
        with mock.patch.object(cli, "omni_cli_start") as start:
            self._run_main([
                "start",
                "--config_path", "/tmp/server_profiles.yml",
            ])

        self.assertEqual(
            start.call_args.kwargs["inventory_path"],
            "/tmp/server_profiles.yml",
        )

    def test_run_docker_cli_forwards_config_path(self):
        with mock.patch.object(cli, "run_docker_containers") as run_docker:
            self._run_main([
                "run_docker",
                "--config_path", "/tmp/server_profiles.yml",
            ])

        self.assertEqual(
            run_docker.call_args.kwargs["inventory_path"],
            "/tmp/server_profiles.yml",
        )

    def test_stop_cli_forwards_config_path(self):
        with mock.patch.object(cli, "omni_cli_stop") as stop:
            self._run_main([
                "stop",
                "--config_path", "/tmp/server_profiles.yml",
            ])

        self.assertEqual(
            stop.call_args.kwargs["inventory_path"],
            "/tmp/server_profiles.yml",
        )

    def test_sync_and_install_cli_forward_config_path(self):
        with mock.patch.object(cli, "sync_code") as sync:
            self._run_main([
                "sync_code",
                "--config_path", "/tmp/p_only.yml",
                "--code_path", "/source",
            ])
        self.assertEqual(sync.call_args.kwargs["inventory_path"], "/tmp/p_only.yml")

        with mock.patch.object(cli, "install_code") as install:
            self._run_main([
                "install_code",
                "--config_path", "/tmp/d_only.yml",
            ])
        self.assertEqual(install.call_args.kwargs["inventory_path"], "/tmp/d_only.yml")

    def test_legacy_deploy_path_is_still_supported(self):
        with mock.patch.object(cli, "sync_code") as sync:
            self._run_main([
                "sync_code",
                "--deploy_path", "/tmp/legacy.yml",
                "--code_path", "/source",
            ])

        self.assertEqual(sync.call_args.kwargs["inventory_path"], "/tmp/legacy.yml")

    def test_inspect_cli_forwards_config_path(self):
        with mock.patch.object(cli, "print_node_config") as inspect:
            self._run_main([
                "inspect",
                "p0",
                "--config_path", "/tmp/p_only.yml",
            ])

        inspect.assert_called_once_with("/tmp/p_only.yml", "p0")

    def test_non_inventory_command_does_not_resolve_inventory(self):
        with mock.patch.object(cli, "get_default_deploy_path") as get_default, \
                mock.patch.object(cli, "upgrade_packages") as upgrade:
            self._run_main(["upgrade"])

        get_default.assert_not_called()
        upgrade.assert_called_once_with()


class RanktableRegressionTests(unittest.TestCase):
    def setUp(self):
        path = REPO_ROOT / "tools/omni_cli/configs/generate_ranktable.yml"
        self.play = yaml.safe_load(path.read_text(encoding="utf-8"))[0]
        self.jinja = Environment(undefined=StrictUndefined)

    def test_decode_only_writes_merged_table_for_one_or_four_hosts(self):
        for count in (1, 4):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as tmp:
                destination = Path(tmp) / "global/collect_files_d"
                destination.mkdir(parents=True)
                inputs = []
                for index in range(count):
                    address = f"192.0.2.{index + 1}"
                    path = destination / f"local_ranktable_{address}_0.json"
                    table = {
                        "version": "1.0", "status": "completed", "server_count": 1,
                        "server_list": [{
                            "server_id": address, "server_ip": address,
                            "device": [{"device_id": "0", "rank_id": "0"}],
                        }],
                    }
                    path.write_text(json.dumps(table), encoding="utf-8")
                    inputs.append(f"collect_files_d/{path.name}")
                script = self.jinja.from_string(
                    self.play["vars"]["generate_global_ranktable_cmd"]
                ).render(DECODE_RANKTABLE_LIST=",".join(inputs),
                         PREFILL_MERGE_RANKTABLE_LIST="")
                # Run the real merge helper, without sourcing the user's shell.
                script = script.replace(". ~/.bashrc", ":")
                script = script.replace(
                    "python /workspace/omniinfer/tools/scripts/pd_ranktable_tools.py",
                    f"{shlex.quote(sys.executable)} "
                    f"{shlex.quote(str(REPO_ROOT / 'tools/scripts/pd_ranktable_tools.py'))}",
                )
                result = subprocess.run(
                    ["bash", "-c", script], capture_output=True, text=True, timeout=15,
                    env={**os.environ, "RANKTABLE_SAVE_PATH": tmp,
                         "DECODE_POD_NUM": str(count)},
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                merged = json.loads((destination / "local_ranktable_merge.json").read_text())
                self.assertEqual(len(merged["server_list"]), count)
                ranks = [server["device"][0]["rank_id"] for server in merged["server_list"]]
                self.assertEqual(ranks, [str(index) for index in range(count)])
                self.assertFalse((Path(tmp) / "global/global_ranktable_merge.json").exists())

    def test_coordinator_conditions_handle_missing_or_empty_pd_groups(self):
        cases = [({}, ""), ({"C": ["c0"]}, ""), ({"P": [], "D": []}, ""),
                 ({"D": ["d0", "d1"]}, "d0"),
                 ({"P": ["p0"], "D": ["d0"]}, "p0")]
        for groups, expected in cases:
            with self.subTest(groups=groups):
                values = {"groups": groups}
                for name in ("ranktable_hosts", "ranktable_coordinator"):
                    expression = self.play["vars"][name].strip()[2:-2]
                    values[name] = self.jinja.compile_expression(expression)(**values)
                self.assertEqual(values["ranktable_coordinator"], expected)
                conditions = [task["when"] for task in self.play["tasks"]
                              if "ranktable_coordinator" in str(task.get("when", ""))]
                self.assertTrue(conditions)
                for condition in conditions:
                    for host in (expected or "c0", "other"):
                        actual = self.jinja.compile_expression(condition)(
                            **values, inventory_hostname=host)
                        wanted = bool(expected) and (
                            host != expected if "!=" in condition else host == expected)
                        self.assertEqual(actual, wanted, condition)

    def test_four_host_templates_share_one_master(self):
        for role, suffix in (("P", "p_only_p0"), ("D", "d_only_d0")):
            with self.subTest(role=role):
                path = REPO_ROOT / "tools/omni_cli/template" / (
                    f"cli_glm5_moe_w4a8c16_a2_{suffix}.yaml")
                inventory = yaml.safe_load(path.read_text(encoding="utf-8"))
                hosts = inventory["all"]["children"][role]["hosts"]
                master_name = role.lower() + "0"
                master_ip = hosts[master_name]["ansible_host"]
                self.assertEqual(len(hosts), 4)
                for host in hosts.values():
                    self.assertEqual(host["master_node"], master_name)
                    self.assertEqual(host["host_ip"], master_ip)
                    self.assertEqual(host["env"]["HOST_IP"], master_ip)
                if role == "P":
                    endpoints = cli.omni_cli.proxy.register_values(inventory, allow_dummy_decode=True)
                    self.assertEqual(endpoints["PREFILL_API_SERVER_LIST"],
                                     f"{master_ip}:{hosts[master_name]['env']['API_PORT']}")
                    register = next(task["set_fact"] for task in self.play["tasks"]
                                    if "PREFILL_MERGE_RANKTABLE_LIST" in task.get("set_fact", {}))
                    files = self.jinja.from_string(register["PREFILL_MERGE_RANKTABLE_LIST"]).render(
                        groups={"P": list(hosts)}, hostvars=hosts).strip()
                    self.assertNotIn(";", files)
                    self.assertEqual(len(files.split(",")), 4)


if __name__ == "__main__":
    unittest.main()
