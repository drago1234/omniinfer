# OX logs in files and the Python terminal

By default, the Python process manager appends P-side OX output to
`/data/ox_log/ox_log_p_server.log` and D-side OX output to
`/data/ox_log/ox_log_d_client.log`. Set `OX_LOG_PATH` to change the directory
for both files.

OX emits structured records with timestamp, level, component, PID, thread ID,
and an error ID for ERROR. Exception handlers call `ox::print_ox_backtrace`,
which writes stack frames as ERROR records with a shared `trace_id`. Ordinary
ERROR records and WARN records do not print a stack. Where an OX log file is configured,
`OxProcessManager` writes all native output to it. The Python terminal shows
only native ERROR records, unstructured error diagnostics (including libzmq
assertions and stack frames), and process failures.

The native level defaults to `INFO`. The OX log file therefore receives INFO,
WARN, and ERROR records by default. Native DEBUG records are suppressed before
either destination. To collect request-level DEBUG records in the OX log file,
set the level before starting omniinfer:

```sh
export OMNIINFER_OX_LOG_LEVEL=DEBUG
```

Even with that setting, Python still forwards only OX ERROR records and fatal
unstructured diagnostics. To reduce file volume, leave the native level at its
default or set it to `WARN` or `ERROR`.

The native logger's file sink is disabled by default because the Python
process manager already writes the child output to its OX log file. For a
standalone OX binary, set `OMNIINFER_OX_LOG_FILE=1` and optionally set
`OMNIINFER_OX_LOG_PATH`.

Checks:

```sh
python tests/test_ox_log_forwarding.py
g++ -std=c++20 -DNDEBUG -pthread -Iomni/accelerators/pd/ox \
  tests/test_ox_native_logger.cpp -o /tmp/test_ox_native_logger
/tmp/test_ox_native_logger
```
