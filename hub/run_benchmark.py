import tensorflow as tf

tf.keras.backend.clear_session()

import argparse
import sys
import time

from model_mapping import MODEL_NAME_MAPPING

sys.path.append("..")
from utilities import get_device_name, get_flops, get_model_from_hub

BATCH_SIZE = 4
WARMUP_ITERATIONS = 10
NUM_ITERATIONS = 100
NUM_CHANNELS = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_family",
        type=str,
        required=True,
        choices=list(MODEL_NAME_MAPPING.keys()),
        help="Model family the variant belongs to.",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        required=True,
        choices=[
            variant for k in MODEL_NAME_MAPPING for variant in MODEL_NAME_MAPPING[k]
        ],
        help="Model variant to benchmark.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        help="Resolution to use for benchmarking.",
    )
    parser.add_argument(
        "--xla", action="store_true", help="XLA-compile the model variants."
    )
    parser.add_argument("--log_wandb", action="store_true", help="Log to WandB.")
    args = parser.parse_args()
    return args


def main(args):
    if args.log_wandb:
        try:
            import wandb
        except Exception:
            raise ImportError("wandb is not installed.")

    # Retrieve the current model variant.
    print(f"Running benchmark for {args.model_variant}...")
    all_model_variants = MODEL_NAME_MAPPING.get(args.model_family)
    if "deit" in args.model_variant or "swin" in args.model_variant:
        model_url = all_model_variants[args.model_variant]
    else:
        model_url, flops = all_model_variants[args.model_variant]

    # Determine the input spec with which to run the benchmark.
    if "deit" in model_url:
        args.resolution = int(model_url.split("/")[-2].split("_")[-1])
        assert args.resolution is not None
    if args.resolution is not None:
        input_spec_shape = [BATCH_SIZE] + [args.resolution, args.resolution, 3]

    # Initialize the model.
    model = get_model_from_hub(model_url, args.resolution)
    assert isinstance(model, tf.keras.Model)

    # XLA compilation.
    print(f"Compiling with XLA: {args.xla}...")
    if args.xla:
        model_xla = tf.function(model, jit_compile=True)

    # Determine the variable with which the benchmark is to be performed.
    benchmark_var = model_xla if args.xla else model

    # Generate a batch of random inputs and warm the model up.
    print("Warming up the model...")
    random_inputs = tf.random.normal(input_spec_shape)
    for _ in range(WARMUP_ITERATIONS):
        _ = benchmark_var(random_inputs, training=False)

    # Calculate throughput.
    print("Calculating throughput...")
    start_time = time.time()
    for _ in range(NUM_ITERATIONS):
        _ = benchmark_var(random_inputs, training=False)
    end_time = time.time()
    total_time = end_time - start_time
    throughput = NUM_ITERATIONS * BATCH_SIZE / total_time
    print("Throughput: {:.2f} samples per second".format(throughput))

    # Calculate FLOPs and number of parameters.
    num_params = model.count_params() / 1e6
    if "deit" in args.model_variant or "swin" in args.model_variant:
        flops = (get_flops(model, input_spec_shape)[0] / 1e9) / BATCH_SIZE
    print(f"Model parameters (million): {num_params:.2f}")
    print(f"FLOPs (giga): {flops:.2f}")

    # Log to WandB if specified.
    if args.log_wandb:
        device_name = get_device_name()
        run_name = f"{args.model_variant}@xla-{args.xla}@res-{args.resolution}@device-{device_name}"
        wandb.init(project="keras-xla-benchmarks", name=run_name, config=args)
        wandb.config.update(
            {
                "family": args.model_family,
                "variant": args.model_variant,
                "resolution": args.resolution,
            }
        )
        wandb.log(
            {
                "Throughput (samples/sec)": float(f"{throughput:.2f}"),
                "Num parameters (million)": float(f"{num_params:.2f}"),
                "FLOPs (giga)": float(f"{flops:.2f}"),
            }
        )
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
