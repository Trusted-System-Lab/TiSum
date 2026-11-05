__all__ = ["run_onnx_model"]

import numpy as np
import onnxruntime as ort


def run_onnx_model(session: ort.InferenceSession, input_data: np.ndarray) -> np.ndarray:
    input_nodes = session.get_inputs()

    input_name = input_nodes[0].name
    ori_input_shape = input_nodes[0].shape
    input_shape = list(ori_input_shape)
    if len(input_shape) > 1:
        input_shape[0] = -1
    input_data = input_data.reshape(input_shape).astype(np.float32)

    output_data = []
    for i in range(input_data.shape[0]):
        x = input_data[[i]]
        y = session.run(None, {input_name: x})[0]
        output_data.append(y)

    output_data = np.concatenate(output_data, axis=0)

    return output_data
