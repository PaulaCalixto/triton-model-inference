import json
import numpy as np
import triton_python_backend_utils as pb_utils
from joblib import load

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])

        predicao_config = pb_utils.get_output_config_by_name(
            model_config, "PREDICAO")
        
        self.predicao_dtype = pb_utils.triton_string_to_numpy(
            predicao_config['data_type'])

        version_path =  args['model_repository'] + '/' + args['model_version']

        self.model = load(version_path + '/model.pickle')

    def execute(self, requests):
        responses = []

        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input")
            input_data = input_tensor.as_numpy()

            predicao = self.model.predict(input_data)

            predicao_tensor = pb_utils.Tensor(
                "PREDICAO", predicao.astype(self.predicao_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[predicao_tensor])
            responses.append(inference_response)

        return responses

    def finalize(self):
        print('Limpando...')