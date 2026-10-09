import os
import time
import wandb
import argparse
from net.ppo import PPO 
import torch
import argparse
torch.serialization.add_safe_globals([argparse.Namespace])
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
parser.add_argument('--ent_coef', type=float, default=0.0, help='Entropy coefficient for exploration')
parser.add_argument('--no_entropy', action='store_true', help='Disable entropy constraints')
parser.add_argument('--no_size', action='store_true', help='Disable size constraints')
parser.add_argument('--no_blosum', action='store_true', help='Disable BLOSUM constraints')

args = parser.parse_args()

if not os.path.exists('policy'):
    os.mkdir('policy')
if not os.path.exists('results'):
    os.mkdir('results')

args.seed = int(time.strftime('%H%M%S', time.localtime(time.time())))
project_name = '{}_{}_{}'.format(args.protein, args.level, args.run)
if args.tag is not None:
	project_name += ('_'+ args.tag)
save_dir = f"{project_name}_{time.strftime('%H_%M_%S', time.localtime(time.time()))}"

os.mkdir('policy/{}'.format(save_dir))
os.mkdir('results/{}'.format(save_dir))

# Set wandb to offline mode
os.environ["WANDB_MODE"] = "offline"
run = wandb.init(project="LatProtRL", name=project_name, mode="offline")
# log project configuration
run.config["protein"] = args.protein
run.config["level"] = args.level
run.config["protein_level"] = args.protein + "_" + args.level
run.config["use_oracle"] = args.use_oracle
run.config["description"] = args.tag if args.tag != None else "None"

cfg = create_opt(args)
run.config["step_mut"] = cfg.step_mut
run.config["topk"] = cfg.topk if cfg.topk != None else "None"
if args.delta != None:
	cfg.action_size = args.delta
	run.config["delta"] = args.delta
if args.max_step != None:
    cfg.done_cond.max_step = args.max_step
    run.config["max_step"] = args.max_step

# Handle constraints
cfg.use_entropy_constraint = not args.no_entropy
cfg.use_size_constraint = not args.no_size
cfg.use_blosum_constraint = not args.no_blosum

run.config["entropy_constraint"] = "ENABLED" if cfg.use_entropy_constraint else "DISABLED"
run.config["size_constraint"] = "ENABLED" if cfg.use_size_constraint else "DISABLED"
run.config["blosum_constraint"] = "ENABLED" if cfg.use_blosum_constraint else "DISABLED"

print("\n" + "="*80)
print(f"MODE: Entropy={cfg.use_entropy_constraint}, Size={cfg.use_size_constraint}, BLOSUM={cfg.use_blosum_constraint}")
print("="*80 + "\n")

env = SingleOpt(cfg, seed=args.seed)
n_calls = 256 

model = PPO("MlpPolicy", env, n_calls=n_calls, ent_coef=args.ent_coef*5,
			n_steps=4096, verbose=1, device=args.device, tensorboard_log=None)


model.learn(total_timesteps=40960,  # Increased from 2000 to 100k for thorough exploration
			callback=[WandbCallback(model_save_path='policy/'+save_dir), RewardLoggingCallback(), BufferLoggingCallback(cfg, 'results/'+save_dir, pth_dir='policy/'+save_dir)
	])

wandb.finish()