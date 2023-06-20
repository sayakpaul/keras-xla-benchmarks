import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import \
    convert_variables_to_constants_v2_as_graph


def get_flops(model, input_shape=(4, 224, 224, 3)):
    """
    Calculate FLOPS for a `tf.keras.Model`.
    Ignore operations used in only training mode such as Initialization.
    Use tf.profiler of tensorflow v1 api.

    Args:
        regnety_instance: A regnety.models.model.RegNetY instance

    Returns:
        Tuple containing total float ops and paramenters

    Adapted from:
    https://github.com/AdityaKane2001/regnety/blob/1c0e3e6978e97bd8bc1c6eb3b100c012c2fc702a/regnety/utils/model_utils.py#L8
    """
    inputs = [tf.TensorSpec(input_shape, tf.float32)]
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )
    return flops.total_float_ops // 2, flops.parameters
