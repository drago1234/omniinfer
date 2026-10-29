# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Unit validation of notification ownership, deduplication and failure relay."""
from pathlib import Path
import sys
import types
import unittest
from dataclasses import dataclass,field
from unittest.mock import Mock,patch

@dataclass
class KVOutput:
    finished_sending:set|None=None
    finished_recving:set|None=None
    invalid_block_ids:set=field(default_factory=set)
    kv_connector_stats:object=None
    kv_connector_worker_meta:object=None

@dataclass
class ModelOutput:
    kv_connector_output:object=None
    sampled_token_ids:list=field(default_factory=list)

from omni_npu.connector import late_kv_completion as module
from omni_npu.connector import llmdatadist_connector_v1 as conn

class TestMerge(unittest.TestCase):
    def setUp(self):
        self.modules=patch.dict(sys.modules,{'vllm.v1.outputs':types.SimpleNamespace(KVConnectorOutput=KVOutput)})
        self.modules.start();self.addCleanup(self.modules.stop)
    def test_shared_empty_output_is_not_modified(self):
        shared=ModelOutput()
        reply=dict(finished_sending=None,finished_recving={'r'},invalid_block_ids=set())
        result=module.merge_llmdatadist_reply(shared,[reply])
        self.assertIsNone(shared.kv_connector_output)
        self.assertIsNot(shared,result)
        self.assertEqual(result.kv_connector_output.finished_recving,{'r'})
    def test_failure_flags_survive_in_the_same_scheduler_update(self):
        old=KVOutput(finished_recving={'old','same'},invalid_block_ids={1},kv_connector_stats='stats',kv_connector_worker_meta='meta')
        original=ModelOutput(old,[[7]])
        result=module.merge_llmdatadist_reply(original,dict(finished_sending={'s'},finished_recving={'failed','same'},invalid_block_ids={2}))
        self.assertEqual(result.kv_connector_output.invalid_block_ids,{1,2})
        self.assertEqual(result.kv_connector_output.finished_recving,{'old','same','failed'})
        self.assertEqual(old.finished_recving,{'old','same'})
        self.assertEqual(old.invalid_block_ids,{1})
        self.assertEqual(result.sampled_token_ids,[[7]])
        self.assertEqual(result.kv_connector_output.kv_connector_stats,'stats')
        self.assertEqual(result.kv_connector_output.kv_connector_worker_meta,'meta')
    def test_empty_poll_preserves_identity_and_unexpected_tp_fails(self):
        original=ModelOutput()
        self.assertIs(module.merge_llmdatadist_reply(original,None),original)
        self.assertIs(module.merge_llmdatadist_reply(original,[None]),original)
        with self.assertRaises(RuntimeError):module.merge_llmdatadist_reply(original,[None,None])
    def test_malformed_reply_fails_and_all_empty_reply_is_identity(self):
        original=ModelOutput()
        with self.assertRaisesRegex(RuntimeError,'Unexpected'):module.merge_llmdatadist_reply(original,dict(finished_recving={'r'}))
        empty=dict(finished_sending=None,finished_recving=set(),invalid_block_ids=set())
        self.assertIs(module.merge_llmdatadist_reply(original,empty),original)
    def test_error_only_reply_is_not_discarded(self):
        result=module.merge_llmdatadist_reply(ModelOutput(),dict(finished_sending=None,finished_recving=None,invalid_block_ids={8}))
        self.assertEqual(result.kv_connector_output.invalid_block_ids,{8})

class LLMD:
    def __init__(self):
        self.get_finished=Mock(side_effect=TypeError('step metadata is cleared'))
        self.poll_decode_completions=Mock(return_value=({'loaded'},{42}))

class TestPoll(unittest.TestCase):
    def config(self,role='kv_consumer',tp=1,pp=1):
        return types.SimpleNamespace(kv_transfer_config=types.SimpleNamespace(kv_role=role),parallel_config=types.SimpleNamespace(tensor_parallel_size=tp,pipeline_parallel_size=pp))
    def test_other_connectors_not_drained_and_errors_travel_with_completion(self):
        llmd=LLMD();other=types.SimpleNamespace(get_finished=Mock(side_effect=AssertionError('other child touched')))
        root=types.SimpleNamespace(_connectors=[other,types.SimpleNamespace(_connectors=[llmd])])
        mods={'vllm.distributed.kv_transfer':types.SimpleNamespace(get_kv_transfer_group=lambda:root,has_kv_transfer_group=lambda:True),
              'omni_npu.connector.llmdatadist_connector_v1':types.SimpleNamespace(LLMDataDistConnector=LLMD)}
        with patch.dict(sys.modules,mods):r=module.poll_llmdatadist_completion(self.config())
        self.assertEqual(r['finished_recving'],{'loaded'})
        self.assertEqual(r['invalid_block_ids'],{42})
        other.get_finished.assert_not_called()
        llmd.get_finished.assert_not_called()
        llmd.poll_decode_completions.assert_called_once_with()
    def test_prefill_tp_and_pp_are_noops(self):
        for config in [self.config(role='kv_producer'),self.config(tp=2),self.config(pp=2)]:
            self.assertIsNone(module.poll_llmdatadist_completion(config))
    def poll(self,root,has=True):
        mods={'vllm.distributed.kv_transfer':types.SimpleNamespace(get_kv_transfer_group=lambda:root,has_kv_transfer_group=lambda:has),
              'omni_npu.connector.llmdatadist_connector_v1':types.SimpleNamespace(LLMDataDistConnector=LLMD)}
        with patch.dict(sys.modules,mods):return module.poll_llmdatadist_completion(self.config())
    def test_no_transfer_group_or_no_llmd_child_is_a_noop(self):
        self.assertIsNone(self.poll(LLMD(),has=False))
        self.assertIsNone(self.poll(types.SimpleNamespace(_connectors=[types.SimpleNamespace()])))
    def test_two_llmd_children_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'one LLMD child'):
            self.poll(types.SimpleNamespace(_connectors=[LLMD(),LLMD()]))
    def test_empty_poll_returns_none(self):
        llmd=LLMD();llmd.poll_decode_completions.return_value=(set(),set())
        self.assertIsNone(self.poll(llmd))

class TestConnectorAPI(unittest.TestCase):
    poll=staticmethod(conn.LLMDataDistConnector.poll_decode_completions)
    def test_decode_can_poll_with_no_bound_metadata(self):
        worker=types.SimpleNamespace(get_finished=Mock(return_value=(None,{'r'})),get_block_ids_with_load_errors=Mock(return_value={9}))
        connector=types.SimpleNamespace(is_prefill=False,worker=worker,_connector_metadata=None)
        self.assertEqual(self.poll(connector),({'r'},{9}))
        worker.get_finished.assert_called_once_with(None)
    def test_prefill_and_scheduler_do_not_touch_completion_queue(self):
        for connector in [types.SimpleNamespace(is_prefill=True,worker=object()),types.SimpleNamespace(is_prefill=False,worker=None)]:
            self.assertEqual(self.poll(connector),(None,set()))



"""Run the candidate's actual engine-step code against deterministic futures."""
import ast
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any,cast
import sys
import unittest
from unittest.mock import patch
helper=module

path=Path(__file__).resolve().parents[3]/'omni/vllm_patches/patches/common/patch_step_with_batch_queue_async_dummy.py'
tree=ast.parse(path.read_text())
tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
scope=dict(time=NS(sleep=lambda _:None),Any=Any,cast=cast,
           DPEngineCoreProc=object,EngineCoreOutputs=dict,ModelRunnerOutput=ModelOutput,
           VLLMPatch=object,register_patch=lambda *a:lambda cls:cls)
class Future:
    def __init__(self,tag,result,events):self.tag,self.value,self.events=tag,result,events
    @classmethod
    def __class_getitem__(cls,typ):return cls
    def done(self):return True
    def result(self):
        self.events.append(self.tag)
        if isinstance(self.value,Exception):raise self.value
        return self.value
scope['Future']=Future
exec(compile(tree,str(path),'exec'),scope)
step=scope['StepWithBatchQueueAsyncDummyDPEnginePatch'].step_with_batch_queue

class TestDummyFlow(unittest.TestCase):
    def setUp(self):
        self.events=[];self.seen=[];self.shared=ModelOutput()
        self.reply=dict(finished_sending=None,finished_recving={'new'},invalid_block_ids={22})
        self.control=NS(total_num_scheduled_tokens=0)
        self.executor=NS(is_sleeping=False,output_rank=0)
        self.executor.execute_model=lambda *_args,**_kwargs:Future('control_result',self.shared,self.events)
        def rpc(name,**kwargs):
            self.events.append('rpc:'+name)
            self.assertEqual(kwargs,{'non_block':True,'unique_reply_rank':0})
            return Future('dummy_result',self.reply if name.endswith('_and_poll_kv') else None,self.events)
        self.executor.collective_rpc=rpc
        def update(s,o):self.events.append('scheduler_update');self.seen.append(o);return {}
        scheduler=NS(has_requests=lambda:True,schedule=lambda _:self.control,update_from_output=update)
        self.engine=NS(batch_queue=deque(),batch_queue_size=2,scheduler=scheduler,model_executor=self.executor,
                       is_ec_consumer=True,is_pooling_model=False,engines_running=True,
                       _should_throttle_prefills=lambda:False,log_error_detail=lambda _:nullcontext(),
                       log_iteration_details=lambda _:nullcontext(),_process_aborts_queue=lambda:self.events.append('aborts'),
                       vllm_config=NS(kv_transfer_config=NS(kv_role='kv_consumer'),parallel_config=NS(tensor_parallel_size=1,pipeline_parallel_size=1)))
        mods={'omni_npu.connector.late_kv_completion':helper,
              'vllm.v1.outputs':NS(KVConnectorOutput=KVOutput)}
        p=patch.dict(sys.modules,mods);p.start();self.addCleanup(p.stop)
    def test_poll_is_merged_before_the_only_scheduler_update(self):
        outputs,executed=step(self.engine)
        self.assertFalse(executed);self.assertEqual(len(self.seen),1)
        self.assertLess(self.events.index('dummy_result'),self.events.index('scheduler_update'))
        self.assertEqual(self.seen[0].kv_connector_output.finished_recving,{'new'})
        self.assertEqual(self.seen[0].kv_connector_output.invalid_block_ids,{22})
        self.assertIsNone(self.shared.kv_connector_output)
        self.assertTrue(self.engine._omni_dummy_submitted_in_step)
        self.assertEqual(self.events.count('dummy_result'),1)
    def test_non_decode_retains_existing_drain_then_wait_order(self):
        self.engine.vllm_config.kv_transfer_config.kv_role='kv_producer'
        step(self.engine)
        self.assertIn('rpc:execute_dummy_batch',self.events)
        self.assertLess(self.events.index('scheduler_update'),self.events.index('dummy_result'))
        self.assertIs(self.seen[0],self.shared)
    def test_older_real_batch_prevents_poll_path(self):
        old=NS(total_num_scheduled_tokens=4)
        f=Future('older_real_result',self.shared,self.events)
        self.engine.batch_queue.append((f,old,f))
        step(self.engine)
        self.assertIn('rpc:execute_dummy_batch',self.events)
        self.assertEqual(len(self.engine.batch_queue),1)
    def test_tp2_and_prefill_fall_back(self):
        for role,tp in [('kv_producer',1),('kv_consumer',2)]:
            self.engine.vllm_config.kv_transfer_config.kv_role=role
            self.engine.vllm_config.parallel_config.tensor_parallel_size=tp
            self.events.clear()
            step(self.engine)
            self.assertIn('rpc:execute_dummy_batch',self.events)
    def test_dummy_failure_cannot_publish_completion(self):
        self.reply=RuntimeError('dummy execution failed')
        with self.assertRaisesRegex(RuntimeError,'dummy execution failed'):step(self.engine)
        self.assertEqual(self.seen,[])
    def test_paused_wave_submits_no_dummy(self):
        self.engine.engines_running=False
        step(self.engine)
        self.assertFalse(any(e.startswith('rpc:') for e in self.events))

# Real DecodeWorker drain/publish methods, so reverting the atomic drain fails these tests.
from concurrent.futures import Future as _CF
class _Pull:
    SKIP_BLOCK_ID=[-1,0]
    def __init__(self,params,layer_ids):self.params=params
    def targets(self):return {'p0':NS(p_blocks=[1],d_blocks=self.params['d_blocks'],layer_ids=[0])}
    def done(self):return ('p0',self.params['req_id'],1)

class _Sync:
    def submit(self,fn):
        f=_CF();fn();f.set_result(None);return f

def new_decode_worker(pull_ok=True):
    # Real __init__; only the transport, port lookup and sender thread are stubbed.
    with patch.object(conn,'CacheManager'),patch.object(conn,'get_kv_port',return_value=0),\
         patch.object(conn,'start_daemon'):
        w=conn.DecodeWorker(NS())
    w.looping=False
    w.executor.shutdown(wait=False)
    w.manager=NS(register=lambda _:[0],pull_blocks=lambda *_:pull_ok)
    w.executor=_Sync()
    return w

class _PullStubbed(unittest.TestCase):
    def setUp(self):
        for p in (patch.object(conn,'SchemePull',_Pull),patch.object(conn.kv_transfer,'record_failure')):
            p.start();self.addCleanup(p.stop)

class TestWorkerPollRPC(unittest.TestCase):
    def test_dummy_runs_before_poll_and_reply_is_returned(self):
        from omni_npu.worker.npu_worker import NPUWorker
        calls=[];worker=NS(execute_dummy_batch=lambda:calls.append('dummy'),vllm_config='cfg')
        with patch.object(module,'poll_llmdatadist_completion',side_effect=lambda c:calls.append(c) or 'reply'):
            self.assertEqual(NPUWorker.execute_dummy_batch_and_poll_kv(worker),'reply')
        self.assertEqual(calls,['dummy','cfg'])

class TestDecodeWorkerAtomicDrain(_PullStubbed):
    def test_failure_and_completion_are_published_under_one_lock(self):
        w=new_decode_worker(pull_ok=False);held=[]
        put=w.pull_done.put
        w.pull_done.put=lambda r:(held.append(w._invalid_lock.locked()),put(r))
        w._pull_kv('R',dict(req_id='R',d_blocks=[22,0]))
        self.assertEqual(held,[True])
        self.assertEqual(w.get_finished(None),(None,{'R'}))
        self.assertEqual(w.get_block_ids_with_load_errors(),{22})  # SKIP_BLOCK_ID 0 dropped
    def test_failure_landing_between_the_two_queries_waits_for_its_completion(self):
        w=new_decode_worker(pull_ok=False)
        self.assertEqual(w.get_finished(None),(None,set()))
        w._pull_kv('R',dict(req_id='R',d_blocks=[22]))  # background thread interleaves here
        self.assertEqual(w.get_block_ids_with_load_errors(),set())
        self.assertEqual(w.get_finished(None),(None,{'R'}))
        self.assertEqual(w.get_block_ids_with_load_errors(),{22})
    def test_undrained_failures_accumulate_until_queried(self):
        w=new_decode_worker(pull_ok=False)
        w._pull_kv('A',dict(req_id='A',d_blocks=[5]));self.assertEqual(w.get_finished(None)[1],{'A'})
        w._pull_kv('B',dict(req_id='B',d_blocks=[6]));self.assertEqual(w.get_finished(None)[1],{'B'})
        self.assertEqual(w.get_block_ids_with_load_errors(),{5,6})
        self.assertEqual(w.get_block_ids_with_load_errors(),set())

class TestQueuedControlOrdering(_PullStubbed):
    """An older control batch left in the queue is popped after a newer one already drained.

    The poll then reports a later drain before an earlier one; a failure must still
    reach the scheduler no later than its completion.
    """
    def run_steps(self,tp):
        w=new_decode_worker(pull_ok=False);updates=[];inject=[True];self.rpcs=[]
        def control(*_a,**_k):
            fin=w.get_finished(None)[1]
            if inject:inject.pop();w._pull_kv('R',dict(req_id='R',d_blocks=[22]))
            err=w.get_block_ids_with_load_errors()
            return Done(ModelOutput(KVOutput(finished_recving=fin or None,invalid_block_ids=err)))
        def rpc(name,**_k):
            self.rpcs.append(name)
            if not name.endswith('_and_poll_kv'):return Done(None)
            fin,err=w.get_finished(None)[1],w.get_block_ids_with_load_errors()
            return Done(dict(finished_sending=None,finished_recving=fin or None,invalid_block_ids=err) if fin or err else None)
        sched=NS(has_requests=lambda:True,schedule=lambda _:NS(total_num_scheduled_tokens=0),
                 update_from_output=lambda s,o:updates.append(o.kv_connector_output) or {})
        engine=NS(batch_queue=deque([(Done(ModelOutput()),NS(total_num_scheduled_tokens=0),None)]),batch_queue_size=2,
                  scheduler=sched,model_executor=NS(is_sleeping=False,output_rank=0,execute_model=control,collective_rpc=rpc),
                  is_ec_consumer=True,is_pooling_model=False,engines_running=True,_should_throttle_prefills=lambda:False,
                  log_error_detail=lambda _:nullcontext(),log_iteration_details=lambda _:nullcontext(),_process_aborts_queue=lambda:None,
                  vllm_config=NS(kv_transfer_config=NS(kv_role='kv_consumer'),parallel_config=NS(tensor_parallel_size=tp,pipeline_parallel_size=1)))
        mods={'omni_npu.connector.late_kv_completion':helper,'vllm.v1.outputs':NS(KVConnectorOutput=KVOutput)}
        with patch.dict(sys.modules,mods):
            for _ in range(3):step(engine)
        first=lambda pred:next(i for i,kv in enumerate(updates) if kv is not None and pred(kv))
        return first(lambda kv:'R' in (kv.finished_recving or ())),first(lambda kv:22 in kv.invalid_block_ids)
    def test_failure_never_reaches_scheduler_after_completion(self):
        for tp in (2,1):  # TP2 falls back to the original path; TP1 polls after dummy
            with self.subTest(tp=tp):
                done_at,err_at=self.run_steps(tp)
                self.assertEqual('execute_dummy_batch_and_poll_kv' in self.rpcs,tp==1)
                self.assertLessEqual(err_at,done_at)

class Done:
    def __init__(self,value):self.value=value
    def done(self):return True
    def result(self):return self.value

if __name__=='__main__':unittest.main()
