# OX 重启后 Decode 服务退出：代码与现场核对

## 范围与结论

修复提交分支：`fix/ox-restart-recovery-complete`，从用户更新后的 `f5fbd3ef` 新建，基线与远端验证版本一致。本次提交包含已验证的恢复逻辑、机制回归测试及本文档。原有两个模型 JSON 配置修改及其他部署文档保留在工作树，不包含在本次提交。远端仍是该基础提交加受控补丁快照；本次本地归档提交不改变远端分支或重启服务。

## 仓内 UT 位置与执行方式

恢复相关测试统一归入 `tests/`，与既有 `test_ox_log_forwarding.py`、`test_ox_restart_requests.py` 并列：`test_ox_monitor.py`、`test_ox_child_signals.py`、`test_ox_recovery.py`、`test_ox_abnormal_states.py`。生产 `omni/accelerators/pd/` 不保留这些测试文件。源码路径由测试文件所在仓库根目录定位，不要求修改工作目录至生产模块目录。

仓库根执行 `python3 -m unittest discover -s tests -p 'test_ox_*.py' -v`。下文历史隔离目录保存的是当时归档布局，不能据此把测试放回生产源码目录；移动后的仓内路径以本节为准。

目录整理只修改本地仓库及用于验证的独立镜像目录，不删除远端正在使用的补丁部署文件，不改变其启动manifest或服务。后续若按新Git提交重新部署，应同时对齐测试路径与部署清单，不能只删除旧路径文件导致启动摘要校验失败。

整理后在D0容器的独立仓库布局副本执行上述统一发现命令，28项全部通过，包含原有日志转发与请求失败交接测试。生产源码两文件摘要未变；证据在控制端 `/data/yyx/ox_features_20261102/recovery-ut-layout-20261103/results.log`。

## 生产部署记录

用户授权停止、部署、重启及只杀 OX 验证后，已于2026-11-03 20:24将补丁真正写入四机 `/data/yyx/ox_features_20261102/omniinfer/omni/accelerators/pd/`，并逐文件核对摘要。此前20:17的再次失败仍使用旧生产源码，不能计作候选验证结果。

七个受控文件为两个模型 JSON、两个生产 Python 模块、三个测试模块。四机 `source-overrides.json` 与控制端 `launch/service.py` 的固定文件白名单同步更新，保留基础提交、文件摘要和 editable 安装校验，未取消预检。Python 补丁不需要重新编译 native OX；原生二进制没有改动。

- 备份：四机 `recovery-deployment-backups/20261103T202409/`，包括原源码、manifest及停止后的旧故障日志。
- 部署收据：控制端 `recovery-deployment-receipt.json`。
- 启动预检：控制端 `launch/preflight-history/20261103T202423.258470.json`，五容器均通过。
- 启动记录：控制端 `recovery-production-launch.log`。
- 自动验证：控制端 `verify_ox_recovery_production_20261103.py`；进度和结果为 `recovery-production-validation.json`、`recovery-production-validation.log`。等待17个后端全部健康，先真实P→D请求，再仅SIGKILL D0原生OX；验证新OX父进程不变、八个API PID不变、八个Decode端口恢复真实P→D请求及最终17/17健康。

### 第一轮生产验证：单次请求通过，持续监控验收不充分

2026-11-03 20:35:26，仅向 D0 容器原生 OX PID `11378` 发送 SIGKILL，没有向 API/EngineCore/helper 发送人为停止信号。新 OX 为 `42268`，约 **3.143秒**出现，父 EngineCore 仍为 `4079`。八个 API PID `[452,453,454,455,456,457,458,459]` 在故障前后完全一致。

- 故障前真实 P→D 请求正常回答 `42`，携带6个 remote blocks。
- 故障后分别直接向 D0 `9100`～`9107` 八个端口发送新的真实 P→D 请求，八项均回答 `42`，每项携带6个 remote blocks；对应拉取完成日志覆盖八 rank。该结果覆盖 H2H 传输、地址处理、真实 H2D 与模型回答，不只是 HTTP health。
- 八 rank 各有1次 `OX_RECOVERY_IPC_REPLACED`，没有 `OX_RECOVERY_FAILED` 或 `Shutting down FastAPI HTTP server`。
- 故障恰逢 DP4 的一个真实在途传输，request `chatcmpl-a91845bc1ac347c9967110548541df2c` 被记录为 `KV transfer failed ... OX restarted`；20:35:29 的 `scheduler.py:1244` 随后明确 `Aborting request ... KV loading failure`，不是只清 pending 却把调度请求悬挂。此旧请求未自动重放，新请求继续执行。
- 当时代理检查17/17健康，原验证收据 `stage=PASS`。该脚本只确认新PID及请求，没有确认Popen已经返回，后续复现证明这个PASS不能代表持续重启能力通过。

本轮验证是在服务收到其他业务流量的情况下进行，Prefill 排队使验证总时长增加；3.143秒仅指被杀旧 OX 到观察到新 OX 的时间，不是所有失败请求自动恢复时间。旧失败请求需调用方重试。尚未做完整精度集或性能验收；P 原生 OX、D1 原生 OX 的分别故障注入也不在本轮覆盖范围。

### 后续复现：Popen被并发fork的helper持有exec错误管道卡住

用户随后发现新OX `42268` 再次成为僵尸。只读采集显示父EngineCore `4079` 及API仍活着；监控线程活动栈停在 `OxProcessManager._start_monitor.monitor -> _launch_and_check -> _lauch_process -> subprocess.Popen.__init__ -> _execute_child`。因此第一次Popen从未返回，后续退出无法再被检测。

直接内存及fd证据：

- 监控线程`Popen._execute_child`帧局部变量`errpipe_read=478`，对应`pipe:[1858077307]`。
- ZMQ helper `42338` 的fd `479` 持有同一管道的**写端**；活动帧确认其在`DecodeConnectorWorker._process_zmq`内。
- Address helper `42451` 也继承了读端。OX `42268` 的退出wait status为9，且仍由4079持有。
- 新OX能短时处理请求，是因为exec已经成功；Python父线程仍在等错误管道EOF。持有写端的fork helper未exec，该fd的CLOEXEC标记不能使其关闭。不是仅凭没有日志推断。
- 原始现场证据：控制端 `recovery-second-exit-evidence.json`。

精确修复：在进程管理模块提供同一进程创建锁，原生`subprocess.Popen`与两个helper的`Process.start()`共用，避免fork-only子进程跨入Popen临时exec错误管道存在的窗口。锁仅保护创建过程，不覆盖数据传输/等待，不修改框架全局subprocess或fork行为。

新增确定性UT在`_fork_exec`前设置门闩（此时Popen已建立errpipe），同时尝试启动真实helper：helper必须等native Popen完整返回才能fork；不能只测睡眠后偶然没有发生竞态。另加连续两次native进程被杀的监控回归，四机均16/16通过。

第二轮生产验收脚本为控制端 `verify_ox_recovery_production_two_kills.py`，结果为 `recovery-production-validation-two-kills.json`。必须连续杀两次OX，每次都等到对应`OX restarted, pid=<new>`接管日志，再发真实P→D请求；最终核对API PID不变、八端口请求及17/17健康。不再只以新PID出现作为恢复成功标准。

### 第二轮生产验收结果：PASS（连续两次退出与接管）

进程创建锁补丁于20:55写入四机正式代码路径，备份 `recovery-deployment-backups/20261103T205509/`。基础提交仍为f5fbd3ef，叠加受控未提交补丁。manager摘要为 `cab20ae477b5e6fbcb6f6e673df06208fd3ea11b37550daf1cf7796c9f23745d`，connector摘要为 `65e964ad65e21de78b412c0149aba807dd20c10991ad586de7fa89ff728394ed`。

21:04～21:05完成：

| 故障轮次 | 被杀OX | 重建OX | 父EngineCore | 等待监控接管 | 重建后真实P→D |
|---|---|---|---|---|---|
| 1 | 11376 | 40795 | 4043不变 | 2.574秒 | 回答42，6个remote blocks |
| 2 | 40795 | 41442 | 4043不变 | 2.842秒 | 回答42，6个remote blocks |

两次均明确出现Python监控器 `OX restarted, pid=...`，不是只观察到native PID。八个API PID `[452,453,454,455,456,457,458,459]` 全程不变；八rank各有2次IPC_REPLACED、0次RECOVERY_FAILED、0次API_SHUTDOWN。第二次之后八个Decode端口分别完成新P→D请求，八项均回答42；代理最终17/17健康。服务保持运行。

只读内存复核进一步确认：监控线程回到 `Condition.wait -> Event.wait -> OxProcessManager.monitor` 的正常轮询栈；旧PID11376、40795均已reaped，当前41442状态S。原始证据为 `recovery-two-kills-monitor-proof.json`，生产脚本结果为 `recovery-production-validation-two-kills.json`（stage=PASS）。这覆盖了第一次验证遗漏的持续监控/再次退出路径。

### 扩展异常状态验证：7项生产故障注入通过

用户要求测试其他异常状态后，在同一套仍运行的服务上完成以下故障注入，无需重新加载模型。仅向已核对可执行文件路径的原生OX发送信号；崩溃测试前仅把目标OX实例的RLIMIT_CORE设为0，不修改父模型或全局core策略。各项都要求父进程不变、监控器明确接管新PID、真实P→D请求回答42及17/17健康。

| 节点 | 注入 | 新OX PID | 观察到监控接管的时长 | 结果 |
|---|---|---|---|---|
| D0 7.150.10.244 | SIGTERM | 45794 | 3.39秒 | PASS |
| D0 7.150.10.244 | SIGABRT | 46678 | 3.72秒 | PASS |
| D0 7.150.10.244 | SIGSEGV | 47270 | 3.72秒 | PASS |
| D0 7.150.10.244 | SIGSTOP，实际观察到T | 48113 | 7.98秒 | PASS |
| D1 7.150.12.61 | SIGTERM | 42580 | 3.20秒 | PASS |
| P0 7.150.13.19 | SIGSTOP，实际观察到T | 21796 | 8.37秒 | PASS |
| P1 7.150.12.78 | SIGTERM | 9708 | 未准确记录 | PASS |

暂停态需要约8秒是实现边界：先TERM并等5秒，停止进程无法处理TERM后再KILL、回收、退避和重建。表中时间包含远端查询/日志检查，不是native单独的启动耗时。P1的ACK保存在本机Ray worker stderr，测试最初错误地等待P0聚合日志而超时；已在原生worker日志确认新PID接管并补做真实请求，没有重复注入或修改生产源码，也没有将日志定位超时记成生产恢复失败。精确恢复时长未保留，故不推测填表。

扩展隔离测试新增5项，四机均通过；D0完整21项通过，其他三机在此前16项通过基础上执行新增5项。包括SIGTERM、SIGABRT、SIGSEGV、SIGSTOP，以及“活着但不健康不会被当前代码重启”的负向契约。业务源码没有为这些测试再次修改。

原始结果：控制端 `recovery-abnormal-states-production.json`（stage=PASS）、`verify_ox_abnormal_states_production_20261103.py`、隔离目录 `abnormal-states-tests.log` / `<IP>-abnormal-states-tests.log`。第一次P1日志定位失败的原始结果保留在 `recovery-abnormal-states-backups/initial-log-routing-failure.json`，最终文件注明修正过程。

**未覆盖为自动恢复能力：进程仍为S/R但内部死锁、事件循环不推进、传输卡住，当前只有请求超时，不会触发OX重启。** `_ready_check`仅启动时执行；负向隔离测试明确验证这一点。D态不可中断内核等待、真实ptrace的t态也未做生产故障注入。不能把上述7项通过扩大成“所有异常均能自动重启”；若需支持活进程挂死，需要独立设计心跳/传输进度探针和避免负载误判的策略，不能遇到任意请求超时就重启整个共享OX。

现场为 7.150.10.244 / 7.150.12.61 Decode、7.150.13.19 / 7.150.12.78 Prefill。四机源码均为 `f5fbd3efd45db43064a793bd388564272d5b5e3d`，用户确认仅杀死 OX 进程，未停止模型服务。

**OX 的退出发现及重新拉起实际上执行了；业务恢复失败不能据此算作监控未运行。已复现的关键缺陷是重启回调终止 fork 的 ZMQ 子进程时，继承的信号 wakeup fd 可以把 SIGTERM 通知写回父进程的 asyncio/uvloop，触发 API 关闭。另有强杀消费者后复用 multiprocessing.Queue 的锁残留风险，已用真实 Queue 复现，尚不能证明它在本次父服务退出前实际发生。**

## 现场时间线

控制端 `/data/yyx/ox_features_20261102/logs-new-topology/d0/`：

- `server_0.log:20364`，17:46:46：`OX process is in state Z`。
- `server_0.log:20365`：旧 OX 退出码 `-9`，开始重启。
- `server_0.log:20375`：`Shutting down FastAPI HTTP server`。
- `server_0.log:20382`：重新创建 ZMQ 子进程。
- `server_0.log:20384`：`OX restarted, pid=42741`。
- `omni_cli.log`：API 2、4 首先以退出码 0 结束，启动器随后清理其余 API。
- 检查现场时，新 OX 42741 已成为 PPID=1 的僵尸；不能把它归因为新 OX 启动没有执行。

`server_1.log:18417` 的清理 traceback 指向 vLLM EngineCore 的 `signal_handler -> SystemExit`。该日志单独不能证明信号来源，但与下面的机制复现一致。API 的 `vllm/entrypoints/launcher.py` 使用 `loop.add_signal_handler(SIGTERM, signal_handler)` 取消 server/watchdog 任务，产生上述关闭日志。

## 问题一：fork 子进程的信号通知没有隔离

调用链：

```text
OxProcessManager.monitor 发现原生 OX 退出
  -> before_restart = DecodeConnectorWorker._announce_ox_restart
  -> 失败旧 pending，请求代际文件原子替换
  -> _check_ox_generation
  -> _resp_process.terminate()，向旧 ZMQ helper 发 SIGTERM
  -> helper 继承的 CPython signal wakeup fd 写入信号字节
  -> API asyncio/uvloop 读到 SIGTERM，执行自身关闭回调
```

Python 的 signal handler 和 wakeup fd 是两个机制。子进程安装自己的 Python handler，并不会自动脱离继承的 wakeup fd；有 `SystemExit` handler 仍可能唤醒父 loop。原生 OX 是 exec 的 subprocess，与该 fork helper 不同。

远端真实 Linux 机制实验：只 terminate 测试 helper，父 loop SIGTERM callback 从 **1 次**降到 **0 次**，唯一变量是在 helper 入口执行 `signal.set_wakeup_fd(-1)`。

本地修复：在 `ox_process_manager.py` 提供 `isolate_connector_child_signals()`，并在 ZMQ、Address 两个子进程入口的第一步调用。只切断继承的信号通知 fd，保留既有 Python handlers，不修改父进程信号、任务完成语义、DMA 或 KV 生命周期。

## 问题二：“清 pending”不等于“安全清空队列”

当前 `_fail_pending()` 只从共享 `_pending` 删除上下文并投递失败请求 ID。`_check_ox_generation()` 只终止 ZMQ helper；原 `_send_q`、`_recv_q`、`_h2d_q` 和 Address helper 仍被保留。

真实 Queue 复现：让一个进程在 `Queue.get(timeout=30)` 中持有读锁，只终止该进程，再向同一队列投递新对象。新的 `get(timeout=.2)` 得到 `queue.Empty`，而非已入队的对象，因为死进程持有的读锁未释放。生产 sender 的 `asyncio.to_thread(_send_q.get, True, .5)` 同样存在这个强杀窗口。不能通过循环 `get_nowait()` “清空”一个已经损坏的队列来修复锁。

此外，Address 已生成的 H2D batch 可以留在 `_h2d_q`；`_post_success()` 在真实 H2D 之后才检查 `_pending`。因此不能把“旧 request 不再成功回报”扩大成“所有旧地址工作都已取消”。这部分需要单独处理真实读写者及目标内存保活，不能靠提前 COMPLETE、取消等待或直接释放 KV 来解决。

本轮已实现的恢复边界：

1. 正常恢复先设置 stop event，利用 sender 的有限 queue timeout、receiver 的有限 poll timeout，让旧 ZMQ helper 自行结束并释放锁；必须 join 确认退出。
2. 若必须强杀，废弃受影响 IPC，而不是让新 helper 继续使用它。重建必须包含所有旧生产者/消费者，防止仅换父对象而旧 Address/H2D 仍读取旧队列。
3. 暂停新提交，先对旧代做失效与清理；已经进入 DMA 的任务必须真实完成或被安全隔离后才能放开目标块。
4. 以新的 generation 重建 transport，旧请求明确失败且不自动重放。最后才宣布恢复完成并接受新请求。
5. `_before_restart` 失败目前只记录异常仍继续启动 OX；应避免这种情况被记成完整恢复成功。进程 ready 与 connector/queue ready 必须分别记录。

恢复代码现在先协作停止 Address 与 ZMQ。Address 未退出时保留 pending 与旧代，报告错误后继续重试，不强杀可能正在生成/投递 DMA 描述符的 Address。ZMQ 可在协作退出超时后强制终止；确认旧生产者退出后替换 send/recv/H2D/failure 四个 IPC 队列。H2D 取队列与替换互斥，真实 DMA 和 pending 失效互斥；已经取出的旧批次在 DMA 前验证请求提交身份，不允许旧批次覆盖复用的 request ID。

四个旧队列不会通过 `Queue.get()` 反序列化清空，以免获取死消费者遗留的读锁或解析半截消息。只在旧读写进程已退出后，丢弃管道原始字节，允许父进程 feeder 消耗其结束标记并退出，随后关闭端点。清理失败保留退休队列引用并重试，避免每次恢复泄漏 feeder 线程。此处使用 CPython Linux Queue 的内部 reader/thread 属性，测试明确覆盖4MiB未发送完整帧；后续 Python/平台升级需要保留该机制回归。

本地 transport generation 同时覆盖 OX 重启和 helper 自身退出，阻止 executor 中排队的旧请求在恢复后自动重放。监控线程先检查生产者是否存活，再读取可能由它写入的失败队列。旧代更新只在清理成功后进行。native manager 的重启回调失败会退避重试，不再吞掉失败后继续拉起并宣告成功。

以上恢复重构**已本地实现、四机隔离验证并部署生产**。地址进程在任意指令处被外部强杀、ACL DMA 永久不返回等情况不属于本轮验证的成功恢复保证；遇到不能证明旧写者结束的情况保留所有权，不冒险释放目标块。

## “异常”与“退出”的不同

`OxProcessManager` 当前检查 `poll()` 和 Linux T/t/Z 状态。进程仍活着时的单请求错误、native ERROR 文本或内部任务挂起，不会自动触发这条重启路径。请求错误应失败对应请求；是否重启 OX 应依据进程/服务健康，不能看到任意 ERROR 就清空所有请求。

## 验证与交付状态

- 本地已取回远端5个后续提交的 Git 历史，从 `f5fbd3ef` 新建 `fix/ox-restart-recovery`，保留用户原有模型配置修改。修复没有覆盖后续 async ZMQ 的改动。
- `D:/codes/ox_features/ox_recovery_audit_20261103/` 保存原始远端代码与迁移前补丁；最终修复以新分支工作树为准。
- 第一阶段信号隔离四机各6/6通过；最终恢复机制测试覆盖14项：原故障正对照、父 loop 存活、真实 OX 替换 PID、cleanup 失败时禁止启动、强杀后队列更换、4MiB退休帧及 feeder 回收、stale batch 防写、DMA 完成前保留目标所有权、Address 超时后重试、旧 executor 任务不重放、真实 ZMQ/Address/H2D 链连续两次 helper SIGKILL 后继续处理新请求。
- 四机测试日志归档在控制端上述目录的 `<IP>-tests.log`。
- 最终四机均 **14/14 PASS**，五个源码/测试文件摘要逐一一致；收据为控制端该目录 `final-test-receipt.json`。210 等旧环境未参与本轮。已在 D0 加载 CANN 环境后完整导入候选 manager 和 connector，使用真实 vLLM/torch/torch_npu 依赖通过；隔离进程中将 manager 模块路径指向候选，没有改动生产安装。
- 测试直接编译实际 connector 中的类和方法，仅省略框架/NPU构造；真实使用 Linux fork、multiprocessing IPC、ZMQ sockets。DMA 使用完成门闩模拟验证所有权顺序，不代表真实 NPU 字节回归或完整业务故障注入。
- 隔离阶段不覆盖生产；随后按用户明确授权完成生产部署与重启，实际故障注入结果见上方生产部署记录及对应收据。

此前 UT 只证明原生 OX PID 会替换，没有把父 API 的信号唤醒管道纳入测试，也没有覆盖强杀后的队列锁残留；这就是原 UT 未暴露本次问题的具体缺口。
