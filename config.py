import argparse

# --------------------------------------------------------------------
# Supported proteins and their configurations
# --------------------------------------------------------------------
SUPPORTED_PROTEINS = ["GFP", "AAV", "DHFR"]

PROTEIN_CONFIG = {
    "GFP": {
        "length": 237,
        "num_tokens": 239,
        "reduce_dim": 32,
        "embed_dim": 1280,
        "num_layers": 33,
        "hidden_dim": 256,
        "num_trainable_layers": 4,
        "min_fitness": 1.283419251,
        "max_fitness": 4.123108864,
        "action_size": 0.3,
        "topk": 18,
        "done_cond": dict(max_steps=5, max_mutation=15)
    },
    "AAV": {
        "length": 28,
        "num_tokens": 30,
        "reduce_dim": 16,
        "embed_dim": 1280,
        "num_layers": 33,
        "hidden_dim": 256,
        "num_trainable_layers": 4,
        "min_fitness": 0.0,
        "max_fitness": 19.53645667061,
        "action_size": 0.1,
        "topk": 8,
        "done_cond": dict(max_steps=3, max_mutation=15)
    },
    "DHFR": {
        "length": 219,           # update if your DHFR dataset has different length
        "num_tokens": 221,       # length + 2 (CLS/SEP)
        "reduce_dim": 16,
        "embed_dim": 1280,
        "num_layers": 33,
        "hidden_dim": 256,
        "num_trainable_layers": 4,
        "min_fitness": -5.215,   # raw dataset minimum
        "max_fitness": 4.437,    # raw dataset maximum
        "action_size": 0.3,      # Increased for more diverse actions
        "action_scale": 200.0,   # Increased to enable larger latent space moves
        "topk": 40,              # More amino acid options per position
        "start_percentile": 10,  # start pool drawn around the 10th percentile of fitness
        "step_mut": 10,          # Allow up to 10 mutations per step
        "done_cond": dict(max_steps=15, max_mutation=50)
    },
}

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def config_rep(device, protein, level, reduce_dim=None):
    assert protein in SUPPORTED_PROTEINS, f"protein must be one of {SUPPORTED_PROTEINS}"
    cfg = PROTEIN_CONFIG[protein]

    args = argparse.Namespace()
    args.name = protein
    args.device = device
    args.level = level
    args.embed_dim = cfg["embed_dim"]
    args.num_layers = cfg["num_layers"]
    args.hidden_dim = cfg["hidden_dim"]
    args.length = cfg["length"]
    args.num_tokens = cfg["num_tokens"]
    args.reduce_dim = reduce_dim if reduce_dim is not None else cfg["reduce_dim"]
    args.num_trainable_layers = cfg["num_trainable_layers"]
    return args


def get_fitness_info(protein):
    assert protein in SUPPORTED_PROTEINS, f"protein must be one of {SUPPORTED_PROTEINS}"
    cfg = PROTEIN_CONFIG[protein]
    return cfg["length"], cfg["min_fitness"], cfg["max_fitness"]


def create_base(args):
    assert args.protein in SUPPORTED_PROTEINS
    length, min_fit, max_fit = get_fitness_info(args.protein)

    opt = argparse.Namespace()
    opt.name = args.protein
    opt.device = args.device
    opt.level = args.level
    opt.length = length
    opt.min_fitness = min_fit
    opt.max_fitness = max_fit
    opt.seq_pretrained = f"saved/{args.protein}_{args.level}_LM.pt"
    opt.rew_pretrained = f"ckpt/{args.protein}/oracle.ckpt"
    opt.reduce_dim = None
    return opt


def create_opt(args):
    assert args.protein in SUPPORTED_PROTEINS
    cfg = PROTEIN_CONFIG[args.protein]
    length, min_fit, max_fit = get_fitness_info(args.protein)

    opt = argparse.Namespace()
    opt.name = args.protein
    opt.device = args.device
    opt.level = args.level
    opt.not_sparse = getattr(args, "not_sparse", False)
    opt.length = length
    opt.min_fitness = min_fit
    opt.max_fitness = max_fit
    opt.step_mut = getattr(args, "step_mut", cfg.get("step_mut", 1))
    opt.action_size = cfg["action_size"]
    opt.action_scale = cfg.get("action_scale", 200.0)  # default 200
    opt.topk = cfg["topk"]
    opt.start_percentile = cfg.get("start_percentile", 95)  # Default to elite (original behavior)

    # Done condition
    max_steps = cfg["done_cond"]["max_steps"]
    max_mutation = cfg["done_cond"]["max_mutation"]
    opt.done_cond = argparse.Namespace(max_steps=max_steps,
                                       max_mutation=max_mutation,
                                       step_mut=opt.step_mut)

    # Model checkpoints
    opt.seq_pretrained = f"saved/{args.protein}_{args.level}_LM.pt"
    use_oracle = getattr(args, "use_oracle", True)
    opt.rew_pretrained = (
        f"ckpt/{args.protein}/oracle.ckpt"
        if use_oracle else f"ckpt/{args.protein}/{args.level}.ckpt"
    )

    opt.reduce_dim = None
    return opt


def create_rep_from_opt(opt):
    return config_rep(opt.device, opt.name, opt.level, getattr(opt, "reduce_dim", None))
