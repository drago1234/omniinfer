# ZMQ stress test without starting omniinfer

Build on Linux with the same Boost, ZeroMQ/cppzmq and msgpack-cxx headers used by `ox`:

```sh
cd omni/accelerators/pd/ox
make rebuild
make -f Makefile.mock_zmq rebuild
cd ../../../..
python -m pip install msgpack pyzmq
python tests/test_zmq_stress.py --duration 60 --rate 900 --num-ox-threads 16
```

The mock uses the production `zmq_coroutine.hpp`, ROUTER request/response
coroutines, and multiple Asio threads. It simulates the P-side transfer, so
no P service, shared memory, or accelerator is required. The test checks
response IDs, success flags, process survival, and the final sent/received
count after draining. It exits with status 1 on failure. Use `--zmq-port` to
avoid a local port conflict and `--log-dir` to retain logs at a chosen path.

The existing `pd/mock_ox.cpp` and `pd/Makefile.mock_ox` test process
monitoring; they are separate from this ZMQ transport test.
