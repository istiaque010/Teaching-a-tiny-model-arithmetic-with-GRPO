"""Phase 1: environment report + base checkpoint identity -> results/metrics/env.json"""
import os, platform, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grpo
import psutil, torch, tokenizers, numpy
from grpo.policy import load_policy, load_tokenizer
from src.model import count_parameters

m, ck = load_policy()
info = {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1), "torch": torch.__version__,
        "tokenizers": tokenizers.__version__, "numpy": numpy.__version__, "cuda_available": torch.cuda.is_available(),
        "base_checkpoint": os.path.relpath(grpo.BASE_CKPT, grpo.ROOT), "base_sha256": grpo.sha256(grpo.BASE_CKPT),
        "tokenizer_sha256": grpo.sha256(grpo.TOKENIZER), "model_config": m.cfg.to_dict(),
        "parameters": count_parameters(m), "vocab_size": load_tokenizer().vocab_size}
grpo.save_json(info, grpo.METRICS, "env.json")
print(info)
