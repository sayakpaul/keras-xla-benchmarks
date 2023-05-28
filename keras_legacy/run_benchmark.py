import argparse
import tensorflow as tf
from model_mapping import MODEL_NAME_MAPPING
import time
import wandb
from utilities import get_flops

BATCH_SIZE = 4
WARMUP_ITERATIONS = 10
NUM_ITERATIONS = 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_family",
        type=str,
        default="ResNet_V1",
        required=True,
        choices=list(MODEL_NAME_MAPPING.keys()),
        help="Model family to benchmark.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Resolution to use for benchmarking when model input spec is not fully defined.",
    )
    parser.add_argument(
        "--xla", action="store_true", help="XLA-compile the model variants."
    )
    parser.add_argument("--log_wandb", action="store_true", help="Log to WandB.")
    args = parser.parse_args()
    return args


def main(args):
    print(f"Running benchmark for {args.model_family}...")
    all_model_variants = MODEL_NAME_MAPPING.get(args.model_family)
    print(f"Total model variants: {len(all_model_variants)}.")

    for variant in all_model_variants:
        # Retrieve the current model variant.
        model = all_model_variants[variant]()
        assert isinstance(model, tf.keras.Model)

        # Determine the input spec with which to run the benchmark.
        input_spec_shape = [BATCH_SIZE] + model.inputs[0].shape[1:]
        if args.resolution is None:
            args.resolution = input_spec_shape[1]
        if input_spec_shape[1] is None and args.resolution is None:
            raise ValueError(
                "When model input spec is not fully defined, one must specify a valid `resolution`."
            )

        # XLA compilation.
        print(f"Compiling with XLA: {args.xla}...")
        if args.xla:
            model = tf.function(model, jit_compile=True)

        # Generate a batch of random inputs and warm the model up.
        print("Warming up the model...")
        random_inputs = tf.random.normal(input_spec_shape)
        for _ in range(WARMUP_ITERATIONS):
            _ = model(random_inputs, training=False)

        # Calculate throughput.
        print("Calculating throughput...")
        start_time = time.time()
        for _ in range(NUM_ITERATIONS):
            _ = model(random_inputs, training=False)
        end_time = time.time()
        total_time = end_time - start_time
        throughput = NUM_ITERATIONS * BATCH_SIZE / total_time
        print("Throughput: {:.2f} samples per second".format(throughput))

        # Calculate FLOPs and number of parameters.
        num_params = model.count_params() / 1e6
        flops = (get_flops(model, input_spec_shape) / 1e9) / BATCH_SIZE
        print(f"Model parameters (million): {num_params:.2f}")
        print(f"FLOPs (giga): {flops:.2f}")

        # Log to WandB if specified.
        if args.log_wandb:
            run_name = f"{variant}@xla-{args.xla}@res-{args.args.resolution}"
            wandb.init(project="keras-xla-benchmarks", nam=run_name, config=args)
            wandb.log(
                {
                    "Model family": args.model_family,
                    "Model variant": variant,
                    "XLA": args.xla,
                    "Throughput (samples/sec)": f"{throughput:.2f}",
                    "Num parameters (million)": f"{num_params:.2f}",
                    "FLOPs (giga)": f"{flops:.2f}",
                }
            )
            wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
