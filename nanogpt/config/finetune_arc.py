# Fine-tune GPT-2 small on ARC task descriptions (LARC)
# Usage (from nanoGPT root): python train.py config/finetune_arc.py

out_dir = 'out-arc'
eval_interval = 50
eval_iters = 50
log_interval = 10
always_save_checkpoint = False  # only save when val loss improves (best model wins)

wandb_log = False

dataset = 'arc'
batch_size = 8
gradient_accumulation_steps = 8    # effective batch = 64 sequences × 1024 tokens
block_size = 1024

# Fine-tune from GPT-2 small (124M params) pretrained weights
init_from = 'gpt2'

# Optimizer
learning_rate = 3e-5
max_iters = 200          # best checkpoint was at step 200; stop here
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# LR schedule: linear warmup then cosine decay to min_lr
decay_lr = True
warmup_iters = 50
lr_decay_iters = 200
min_lr = 3e-6

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True
