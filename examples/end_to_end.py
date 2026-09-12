"""End-to-end demo: text -> field -> logits -> generation (spec §33).

Run: python examples/end_to_end.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
from chappie import Chappie, ChappieConfig
from chappie.core.config import GridConfig


def main():
    # Small config for fast demo
    cfg = ChappieConfig(
        grid=GridConfig(n1=8, n2=8, n3=8),
    )
    print("Building Chappie model...")
    model = Chappie(cfg, key=jax.random.PRNGKey(42))
    print(f"  Grid: {cfg.grid.n1}^3 = {cfg.grid.n1**3} points")
    print(f"  Vocab: {len(model.tokenizer)}")
    print(f"  Params: {model.param_count()}")

    # Encode
    prompt = "hello"
    ids = model.encode(prompt)
    print(f"\nEncode '{prompt}' -> {ids.tolist()}")

    # Generate (untrained — random predictions)
    print(f"\nGenerating 20 tokens from '{prompt}' (untrained)...")
    text = model.generate(prompt, max_new_tokens=20, key=jax.random.PRNGKey(0))
    print(f"  Result: {text!r}")

    # NLL
    nll_val = model.nll("hello world")
    print(f"\nNLL of 'hello world': {nll_val:.4f}")

    # Train one step
    loss = model.train_step(ids)
    print(f"\nOne train step -> NLL = {loss:.4f}")

    # Save/load roundtrip
    import tempfile, os
    path = os.path.join(tempfile.gettempdir(), "chappie_demo.npz")
    model.save(path, step=1, metrics={"demo": True})
    model2 = Chappie.load(path)
    print(f"Save/load roundtrip OK (step={model2.step})")

    print("\nDone!")


if __name__ == "__main__":
    main()
