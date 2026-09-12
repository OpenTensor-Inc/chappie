"""Training demo: train on synthetic curriculum, generate text.

Run: python examples/train_demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
from chappie import Chappie, ChappieConfig, ChappieTrainer
from chappie.core.config import GridConfig, TrainingConfig


def main():
    cfg = ChappieConfig(
        grid=GridConfig(n1=12, n2=12, n3=12),
        training=TrainingConfig(
            vocab_size=96, seq_len=16, batch_size=2,
            steps=30, lr=0.002, n_unroll=2,
            force_gain=0.3, param_every=2,
            val_every=10, checkpoint_every=30,
        ),
    )

    print("Building Chappie model...")
    model = Chappie(cfg, key=jax.random.PRNGKey(42))
    print(f"  Grid: {cfg.grid.n1}^3 = {cfg.grid.n1**3} points")
    print(f"  Vocab: {len(model.tokenizer)}")
    print(f"  Params: {model.param_count()}")

    # Pre-training NLL
    nll_pre = model.nll("the cat sat")
    print(f"\nPre-training NLL of 'the cat sat': {nll_pre:.4f}")

    # Train
    print(f"\nTraining for {cfg.training.steps} steps...")
    trainer = ChappieTrainer(model, cfg)
    final = trainer.train()

    # Post-training
    nll_post = model.nll("the cat sat")
    print(f"\nPost-training NLL of 'the cat sat': {nll_post:.4f}")
    print(f"Improvement: {nll_pre:.4f} -> {nll_post:.4f}")

    # Generate
    print("\nGenerating text...")
    for prompt in ["a", "the", "ab"]:
        text = model.generate(prompt, max_new_tokens=15,
                              key=jax.random.PRNGKey(0))
        print(f"  '{prompt}' -> {text!r}")

    print(f"\nFinal metrics: {final}")


if __name__ == "__main__":
    main()
