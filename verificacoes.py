# Criando um cliente para se comunicar com o Triton
import tritonclient.http as httpclient

triton_client = httpclient.InferenceServerClient(url="localhost:8000", verbose=True)

# Verificar se o servidor está ativo para receber solicitações
triton_client.is_server_live()

# Verificar se o Triton está pronto para receber inferências
triton_client.is_server_ready()

# Metadados do modelo 
triton_client.get_model_metadata("modelo_regressao")

{'name': 'modelo_regressao',
 'versions': ['1'],
 'platform': 'python',
 'inputs': [{'name': 'input', 'datatype': 'FP32', 'shape': [-1, -1]}],
 'outputs': [{'name': 'PREDICAO', 'datatype': 'BYTES', 'shape': [1]}]}