import tensorflow as tf


def get_device_name() -> str:
    """Retrieves the name of the GPU device on which the benchmark is being run."""
    gpu_devices = tf.config.list_physical_devices("GPU")
    if gpu_devices:
        details = tf.config.experimental.get_device_details(gpu_devices[0])
        name = details["device_name"].lower()

        if "tesla" not in name:
            return name.split(" ")[1].split("-")[0]
        else:
            return name.split(" ")[1]

    else:
        raise ValueError("No GPUs found!")
