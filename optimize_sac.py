import os
import time
import wandb
import argparse
import torch
from stable_baselines3 import SAC
from config import create_opt
from net.envr import SingleOpt
from wandb.integration.sb3 import WandbCallback
from utils.callbacks import RewardLoggingCallback, BufferLoggingCallback

parser = argparse.ArgumentParser()
parser.add_argument('--protein', type=str, choices=['GFP', 'AAV','DHFR'], required=True)
parser.add_argument('--level', type=str, choices=['hard', 'medium'], required=True)
parser.add_argument('--device', type=str, required=True)
parser.add_argument('--run', type=int, default=0, help='Index of the run for the log') 
parser.add_argument('--max_step', type=int, default=None) 
parser.add_argument('--not_sparse', default=False, action='store_true')
parser.add_argument('--use_oracle', default=False, action='store_true')
parser.add_argument('--delta', type=float, default=None)
parser.add_argument('-M', '--step_mut', type=int, default=15)
parser.add_argument('-T', '--tag', type=str, default=None)
parser.add_argument('--ent_coef', type=str, default='auto', help='Entropy coefficient for exploration (can be auto or a float)')
parser.add_argument('--no_entropy', action='store_true', help='Disable entropy constraints')
parser.add_argument('--no_size', action='store_true', help='Disable size constraints')
parser.add_argument('--no_blosum', action='store_true', help='Disable BLOSUM constraints')

args = parser.parse_args()

if not os.path.exists('policy'):
    os.mkdir('policy')
if not os.path.exists('results'):
    os.mkdir('results')

args.seed = int(time.strftime('%H%M%S', time.localtime(time.time())))
project_name = 'SAC_{}_{}_{}'.format(args.protein, args.level, args.run)
if args.tag is not None:
    project_name += ('_' + args.tag)

save_dir = f"{project_name}_{time.strftime('%H_%M_%S', time.localtime(time.time()))}"

os.mkdir('policy/{}'.format(save_dir))
os.mkdir('results/{}'.format(save_dir))

os.environ["WANDB_MODE"] = "offline"
run = wandb.init(project="LatProtRL_SAC", name=project_name, mode="offline")

cfg = create_opt(args)
if args.delta != None:
    cfg.action_size = args.delta
if args.max_step != None:
    cfg.done_cond.max_step = args.max_step

cfg.use_entropy_constraint = not args.no_entropy
cfg.use_size_constraint = not args.no_size
cfg.use_blosum_constraint = not args.no_blosum

print("\n" + "="*80)
print(f"SAC 4-ROUND CHECK: Entropy={cfg.use_entropy_constraint}, Size={cfg.use_size_constraint}, BLOSUM={cfg.use_blosum_constraint}")
print("="*80 + "\n")

env = SingleOpt(cfg, seed=args.seed)

# Aligning SAC to PPO cadence (4 rounds of 4096 steps)
model = SAC("MlpPolicy", env, 
            ent_coef=args.ent_coef,
            verbose=1, 
            device=args.device, 
            tensorboard_log=None,
            learning_starts=1000,
            train_freq=4096,
            gradient_steps=4096)

# total_timesteps = 4 rounds * 4096 = 16384
model.learn(total_timesteps=16384, 
            callback=[WandbCallback(model_save_path='policy/'+save_dir), 
                     RewardLoggingCallback(), 
                     BufferLoggingCallback(cfg, 'results/'+save_dir, pth_dir='policy/'+save_dir)
    ])

wandb.finish()
