"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O

## 输出文件夹 out
out_dir = 'out'
## 评估间隔
eval_interval = 2000
## 日志间隔
log_interval = 1
## 评估迭代次数
eval_iters = 200
## 如果为True，则在第一次评估后退出脚本
eval_only = False # if True, script exits right after the first eval
## 如果为True，则始终在每次评估后保存检查点
always_save_checkpoint = True # if True, always save a checkpoint after each eval
## 初始化起点 scratch 从零开始 ， resume 从上次停止的地方开始， gpt2 从预训练模型开始
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
## 是否使用wandb
wandb_log = False # disabled by default
## wandb 项目名称
wandb_project = 'owt'
## wandb 运行名称
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data

## 数据集
dataset = 'openwebtext'
## q:used to simulate larger batch sizes
## a:用于模拟更大的批量大小
## 梯度累积步数
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
## 如果 gradient_accumulation_steps > 1，则为微批量大小
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
## 用于训练的块大小
block_size = 1024
# model

## self attention 层数
n_layer = 12
## mutilhead 数量
n_head = 12
## 隐藏层大小
n_embd = 768
## 丢弃率 ， 预训练为0，微调为0.1+
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
## 偏置，是否在LayerNorm和Linear层内使用偏置？
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer

## 学习率 ， 最大为 6e-4
# q: what is mean 6e-4?
# a: 6e-4 = 0.0006
learning_rate = 6e-4 # max learning rate
## 最大迭代次数 六十万次
max_iters = 600000 # total number of training iterations
## 权重衰减 1e-1 = 0.1
weight_decay = 1e-1
## beta1
beta1 = 0.9
## beta2
beta2 = 0.95
## 裁剪梯度
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings

## 是否衰减学习率
decay_lr = True # whether to decay the learning rate
## q:warm up
## a:预热
## 预热迭代次数
warmup_iters = 2000 # how many steps to warm up for
## 学习率衰减迭代次数，六十万次
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
## 最小学习率，应该是学习率的十分之一
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    assert gradient_accumulation_steps % torch.cuda.device_count() == 0
    gradient_accumulation_steps //= torch.cuda.device_count()
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
## q:以下代码是什么意思？
## a:计算每次迭代的tokens数量
## q:为什么要计算tokens数量？
## a:因为tokens数量是计算loss的重要参数
## q:为什么要乘gradient_accumulation_steps？
## a:因为梯度累积，每次迭代的tokens数量是batch_size * block_size
###
# 关于梯度累积的问题 看一下这个链接 https://zhuanlan.zhihu.com/p/454876670
###
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

## 如果是主进程，创建输出目录
if master_process:
    os.makedirs(out_dir, exist_ok=True)
## 设置随机种子
torch.manual_seed(1337 + seed_offset)
## 设置 矩阵乘法 时 使用 tf32
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
## 设置 cudnn 使用 tf32
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
## 设备类型
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
## q:float16 data type will automatically use a GradScaler
## a:float16数据类型将自动使用GradScaler
## q:什么是GradScaler
## a:GradScaler是一个用于动态调整梯度缩放因子的类，以便在训练过程中保持梯度值在一定范围内。
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
## q:以下代码什么意思？
## a:如果是cpu，使用nullcontext，否则使用torch.amp.autocast
## q:什么是nullcontext？
## a:https://docs.python.org/3/library/contextlib.html#contextlib.nullcontext
## q:什么是torch.amp.autocast？
## a:https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
# 加载数据
## q:以下代码什么意思？
## a:加载数据
## q:为什么要这样加载数据？
## a:因为数据太大，无法一次性加载到内存中
## q:为什么要使用np.memmap？
## a:https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

## q:以下代码什么意思？
## a:获取batch
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
# 初始化这些，如果init_from='resume'（即从检查点），可以覆盖
## 迭代次数
iter_num = 0
## 最佳验证损失
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
# 尝试从数据集中推导出vocab_size
## 获取权重元数据信息
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
# 模型初始化
## 模型参数
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line # 从命令行开始使用model_args

## 如果是从头开始训练
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    # 确定我们将用于从头开始训练的词汇量
    # 如果meta_vocab_size为None，则默认为GPT-2的vocab_size为50304（50257四舍五入为效率）
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    ## 创建模型参数
    gptconf = GPTConfig(**model_args)
    ## 根据参数创建模型
    model = GPT(gptconf)
## 如果是从检查点开始训练
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    # 从检查点恢复训练
    ## 加载检查点
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    ## 加载权重参数到设备 device=cuda
    checkpoint = torch.load(ckpt_path, map_location=device)
    ## 获取模型参数
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    # 强制这些配置属性相等，否则我们甚至无法恢复训练
    # 其余的属性（例如dropout）可以保持命令行中所需的状态
    # 从检查点的配置里 恢复 模型参数
    # 分别是 n_layer, n_head, n_embd, block_size, bias, vocab_size
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    ## 创建模型
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    # 修复状态字典的键:(，老实说，不知道检查点是如何获得这个前缀的，必须再调试一下
    
    ## 修复状态字典的键,删除_orig_mod.
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    ## 模型重新加载状态字典
    model.load_state_dict(state_dict)
    ## 读取迭代次数
    iter_num = checkpoint['iter_num']
    ## 读取最佳验证损失
    best_val_loss = checkpoint['best_val_loss']
## 如果是从GPT-2开始训练
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    # 从OpenAI GPT-2权重初始化
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
# 如果需要，使用模型手术将模型块大小裁剪下来？？？
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    ## 模型参数中的block_size设置为block_size
    model_args['block_size'] = block_size # so that the checkpoint will have the right value # 这样检查点就会有正确的值
# move the model to GPU
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
# 初始化GradScaler。如果enabled=False scaler是一个无操作
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
# 优化器
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
# 如果是从检查点开始训练
if init_from == 'resume':
    # 从检查点中加载优化器参数
    optimizer.load_state_dict(checkpoint['optimizer'])
# 释放内存
checkpoint = None # free up memory

# compile the model
# 如果使用编译模型
##  q:什么是编译模型?
##  a:编译模型是一种优化模型的方法，它可以提高模型的性能，减少模型的内存占用，加快模型的训练速度。
##  q:为什么要编译模型?
##  a:编译模型可以提高模型的性能，减少模型的内存占用，加快模型的训练速度。
##  q:编译模型的原理是什么?
##  a:编译模型的原理是将模型的计算图优化为一个计算
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
# 将模型包装到DDP容器中
# 如果开启ddp
## q:什么是ddp?
## a:DDP是分布式数据并行的缩写，是PyTorch提供的一种分布式训练方法，可以在多个GPU上进行训练，从而加快训练速度。
## q:为什么要使用ddp?
## a:DDP可以在多个GPU上进行训练，从而加快训练速度。
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
# 使用许多批次来估计任意精度的损失
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
## 获取第一个batch
X, Y = get_batch('train') # fetch the very first batch
## 记录开始时间
t0 = time.time()
## 记录迭代次数
local_iter_num = 0 # number of iterations in the lifetime of this process
## 模型模块 使用 ddp 或者 model
raw_model = model.module if ddp else model # unwrap DDP container if needed
## 运行时的 模型 flops 利用率 model flops utilization
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    ## 获取学习率
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    ## 每隔 eval_interval 评估一次
    if iter_num % eval_interval == 0 and master_process:
        ## 估算损失
        losses = estimate_loss()
        ## 打印日志
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        ## 上传 wandb
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        ## 保存模型 checkpoint
        ## 如果 val loss 降低了，或者 always_save_checkpoint 为 True
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                # 保存到 out_dir
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    ## 如果第一次只评估，不训练
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
