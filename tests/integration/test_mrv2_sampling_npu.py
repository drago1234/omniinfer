"""Numerical and graph checks for MRv2 NPU fused filtering and uniform RNG."""
import pytest
import torch

torch_npu = pytest.importorskip('torch_npu')
from omni_npu.worker.npu.sampler import apply_top_k_top_p, gumbel_sample
from omni_npu.worker.npu.ops.rejection_sampler_utils import rejection_sample


@pytest.fixture(scope='module', autouse=True)
def npu_device():
    if not torch.npu.is_available():
        pytest.skip('NPU required')
    torch.npu.set_device(0)


@pytest.mark.parametrize('k_enabled,p_enabled', [(True,True),(True,False),(False,True),(False,False)])
def test_filter_matches_fp32_oracle(k_enabled,p_enabled):
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
    gen = torch.Generator().manual_seed(17)
    logits = torch.randn(3,1031,generator=gen)
    k = torch.tensor([1,100,1031],dtype=torch.int32) if k_enabled else None
    p = torch.tensor([0.8,0.6,1.]) if p_enabled else None
    expected = apply_top_k_top_p_pytorch(logits.clone(),k,p,allow_cpu_sync=True)
    result = apply_top_k_top_p(logits.npu(),None if k is None else k.npu(),
                               None if p is None else p.npu()).cpu()
    assert torch.equal(torch.isfinite(result),torch.isfinite(expected))
    assert torch.equal(result[torch.isfinite(result)],expected[torch.isfinite(expected)])


def test_gumbel_mixed_temperature_padding_and_processed_logits(monkeypatch):
    rows,vocab = 4,1031
    gen = torch.Generator().manual_seed(21)
    cpu = torch.randn(rows,vocab,generator=gen)
    noise_cpu = torch.rand(rows,vocab,generator=gen)*0.98+0.01
    noise = noise_cpu.npu()
    original_uniform = torch.Tensor.uniform_
    def fixed_uniform(tensor,*args,**kwargs):
        if tuple(tensor.shape) == (rows,vocab):
            return tensor.copy_(noise)
        return original_uniform(tensor,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'uniform_',fixed_uniform)
    mapping = torch.tensor([2,0,-1,1],device='npu',dtype=torch.int32)
    temp = torch.tensor([0.,1.,2.],device='npu')
    processed = torch.full((3,7,vocab),12345.,device='npu')
    column = torch.tensor(2,device='npu',dtype=torch.int64)
    sampled = gumbel_sample(cpu.npu(),mapping,temp,torch.zeros(3,device='npu',dtype=torch.int64),
                            torch.arange(rows,device='npu'),True,processed,column).cpu()
    expected_logits = cpu.clone()
    expected_logits[0] /= 2
    expected = expected_logits.clone()
    for row in (0,3):
        expected[row] += -torch.log(-torch.log(noise_cpu[row]+1e-20)+1e-20)
    assert torch.equal(sampled,expected.argmax(-1))
    result = processed.cpu()
    for row,req in ((0,2),(1,0),(3,1)):
        torch.testing.assert_close(result[req,2],expected_logits[row],rtol=1e-5,atol=1e-5)
    assert torch.all(result[:,:2] == 12345.) and torch.all(result[:,3:] == 12345.)


def test_gumbel_graph_replay_refreshes_noise_and_observes_inputs():
    rows,vocab = 3,1031
    logits = torch.zeros(rows,vocab,device='npu')
    logits[0,5] = 20
    mapping = torch.arange(rows,device='npu',dtype=torch.int32)
    temp = torch.tensor([0.,1.,1.],device='npu')
    seed = torch.zeros(rows,device='npu',dtype=torch.int64)
    pos = torch.zeros(rows,device='npu',dtype=torch.int64)
    gumbel_sample(logits,mapping,temp,seed,pos,True)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = gumbel_sample(logits,mapping,temp,seed,pos,True)
    samples=[]
    for _ in range(12):
        graph.replay()
        samples.append(out.cpu())
    samples=torch.stack(samples)
    assert torch.all(samples[:,0] == 5)
    assert torch.unique(samples[:,1]).numel() > 1
    temp.fill_(0)
    logits[:,7]=40
    graph.replay()
    assert out.cpu().tolist() == [7,7,7]


def test_gumbel_categorical_distribution():
    n=4096
    probs=torch.tensor([0.1,0.2,0.3,0.4])
    logits=probs.log().repeat(n,1).npu()
    out=gumbel_sample(logits,torch.zeros(n,device='npu',dtype=torch.int32),
                      torch.ones(1,device='npu'),torch.zeros(1,device='npu',dtype=torch.int64),
                      torch.arange(n,device='npu'),True)
    observed=torch.bincount(out.cpu(),minlength=4).float()/n
    assert torch.max(torch.abs(observed-probs)) < 0.045


def rejection_case(n,greedy=False,has_draft=True):
    p=torch.tensor([0.1,0.2,0.3,0.4])
    q=torch.tensor([0.4,0.3,0.2,0.1]) if has_draft else torch.tensor([1.,0.,0.,0.])
    gen=torch.Generator().manual_seed(22)
    chosen=torch.multinomial(q,n,replacement=True,generator=gen)
    draft_sampled=torch.zeros(n*2,dtype=torch.int64)
    draft_sampled[1::2]=chosen
    args=dict(target_logits=p.log().repeat(n*2,1).npu(),
              draft_logits=q.log().repeat(n,1,1).npu() if has_draft else None,
              draft_sampled=draft_sampled.npu(),cu_num_logits=torch.arange(0,n*2+1,2,device='npu',dtype=torch.int32),
              pos=torch.arange(n*2,device='npu'),idx_mapping=torch.arange(n,device='npu',dtype=torch.int32),
              expanded_idx_mapping=torch.arange(n,device='npu',dtype=torch.int32).repeat_interleave(2),
              expanded_local_pos=torch.tensor([0,1],device='npu',dtype=torch.int32).repeat(n),
              temperature=torch.full((n,),0. if greedy else 1.,device='npu'),
              seed=torch.zeros(n,device='npu',dtype=torch.int64),num_speculative_steps=1)
    return args,p,chosen


@pytest.mark.parametrize('has_draft',[False,True])
def test_rejection_distribution(has_draft):
    args,p,_=rejection_case(2048,has_draft=has_draft)
    sampled,count=rejection_sample(**args)
    observed=torch.bincount(sampled[:,0].cpu(),minlength=4).float()/2048
    assert torch.max(torch.abs(observed-p)) < 0.055
    assert set(count.cpu().tolist()) <= {1,2}


def test_rejection_greedy_and_graph_rng():
    args,_,chosen=rejection_case(64,greedy=True)
    sampled,count=rejection_sample(**args)
    assert torch.all(sampled[:,0].cpu()==3)
    assert torch.equal(count.cpu(),torch.where(chosen==3,2,1).to(torch.int32))
    args['temperature'].fill_(1)
    rejection_sample(**args)
    torch.npu.synchronize()
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        sampled,count=rejection_sample(**args)
    graph.replay()
    first=sampled[:,0].cpu().clone()
    graph.replay()
    assert not torch.equal(first,sampled[:,0].cpu())
