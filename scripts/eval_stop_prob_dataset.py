import argparse
import dataclasses
import logging

def _quantiles(x, qs: list[float]) -> dict[str, float]:
    import numpy as np

    if x.size == 0:
        return {f"q{int(q*100)}": float("nan") for q in qs}
    return {f"q{int(q*100)}": float(np.quantile(x, q)) for q in qs}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate stop_prob distribution on a behavior dataset batch stream.")
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--params-dir", required=True, help="Path to checkpoint params dir (e.g. .../<step>/params).")
    parser.add_argument(
        "--skill-list",
        nargs="*",
        default=None,
        help='Optional dataset skill_list override, e.g. "place on:1.0" "place under:1.0".',
    )
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-norm-stats", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logging.info("Importing deps...")
    import jax
    import jax.numpy as jnp
    import numpy as np

    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader
    from openpi.models import model as _model

    cfg = _config.get_config(args.config_name)
    cfg = dataclasses.replace(cfg, batch_size=args.batch_size, num_workers=args.num_workers)
    if args.skill_list is not None:
        if isinstance(cfg.data, list):
            raise ValueError("--skill-list override is not supported for multi-dataset configs.")
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, skill_list=list(args.skill_list)))

    logging.info("config_name=%s", args.config_name)
    logging.info("params_dir=%s", args.params_dir)
    if args.skill_list is not None:
        logging.info("skill_list=%s", args.skill_list)

    # Create model and load params.
    logging.info("Creating model...")
    rng = jax.random.PRNGKey(0)
    model = cfg.model.create(rng)
    logging.info("Restoring params...")
    params = _model.restore_params(args.params_dir, dtype=jnp.bfloat16)
    import flax.nnx as nnx

    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(params)
    model = nnx.merge(graphdef, state)
    model.eval()
    logging.info("Model ready.")

    # Build data loader and collect stats.
    logging.info("Building data loader (num_batches=%d batch_size=%d)...", args.num_batches, args.batch_size)
    dl = _data_loader.create_behavior_data_loader(
        cfg, shuffle=True, num_batches=args.num_batches, skip_norm_stats=args.skip_norm_stats
    )

    stop_probs_pos: list[np.ndarray] = []
    stop_probs_neg: list[np.ndarray] = []
    stop_probs_all: list[np.ndarray] = []
    masked_total = 0
    pos_total = 0
    neg_total = 0

    @jax.jit
    def predict(observation):
        return model.predict_stop_prob(observation)  # type: ignore[attr-defined]

    for i, (obs, _actions) in enumerate(dl):
        if getattr(obs, "stop_label", None) is None or getattr(obs, "stop_mask", None) is None:
            raise RuntimeError("Observation does not include stop_label/stop_mask. Check config/dataset wiring.")

        stop_prob = np.asarray(jax.device_get(predict(obs))).reshape(-1)
        stop_label = np.asarray(jax.device_get(obs.stop_label)).reshape(-1)
        stop_mask = np.asarray(jax.device_get(obs.stop_mask)).reshape(-1).astype(bool)

        mprob = stop_prob[stop_mask]
        mlabel = stop_label[stop_mask]
        masked_total += int(mprob.size)
        if mprob.size:
            stop_probs_all.append(mprob)

        pos = mprob[mlabel >= 0.5]
        neg = mprob[mlabel < 0.5]
        pos_total += int(pos.size)
        neg_total += int(neg.size)
        if pos.size:
            stop_probs_pos.append(pos)
        if neg.size:
            stop_probs_neg.append(neg)

        logging.info(
            "batch=%d masked=%d pos=%d neg=%d stop_prob_mean=%.4f",
            i,
            int(mprob.size),
            int(pos.size),
            int(neg.size),
            float(mprob.mean()) if mprob.size else float("nan"),
        )

    all_arr = np.concatenate(stop_probs_all, axis=0) if stop_probs_all else np.array([], dtype=np.float32)
    pos_arr = np.concatenate(stop_probs_pos, axis=0) if stop_probs_pos else np.array([], dtype=np.float32)
    neg_arr = np.concatenate(stop_probs_neg, axis=0) if stop_probs_neg else np.array([], dtype=np.float32)

    logging.info("masked_total=%d pos_total=%d neg_total=%d", masked_total, pos_total, neg_total)
    logging.info(
        "stop_prob_all mean=%.4f %s",
        float(all_arr.mean()) if all_arr.size else float("nan"),
        _quantiles(all_arr, [0.1, 0.5, 0.9]),
    )
    logging.info(
        "stop_prob_pos mean=%.4f %s",
        float(pos_arr.mean()) if pos_arr.size else float("nan"),
        _quantiles(pos_arr, [0.1, 0.5, 0.9]),
    )
    logging.info(
        "stop_prob_neg mean=%.4f %s",
        float(neg_arr.mean()) if neg_arr.size else float("nan"),
        _quantiles(neg_arr, [0.1, 0.5, 0.9]),
    )


if __name__ == "__main__":
    main()
