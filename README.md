# Triton Inference Server: Modelo de Regressão

## Pré-requisitos

Certifique-se de ter o Triton Inference Server instalado e configurado corretamente. Além disso, você precisará das seguintes bibliotecas Python:

- `scikit-learn==1.1.1`
- `triton-python-backend-utils`

Você pode instalar as dependências usando o arquivo requirements.txt fornecido:

`pip install -r requirements.txt`


## Como usar

1. Clone este repositório para o seu ambiente local.
2. Inicie o Triton Inference Server com o modelo de regressão fornecido.
3. Execute o script `inferencia.py` para realizar inferências sobre os dados de entrada.
4. Verifique a integridade do servidor Triton e os metadados do modelo usando o script `verificacoes.py`.

## Arquivos do Projeto

- `modelo_regressao/`: Contém o modelo de regressão treinado e os arquivos de configuração do Triton.
- `inferencia.py`: Script para realizar inferências sobre o modelo de regressão implantado.
- `verificacoes.py`: Script para verificar a integridade do servidor Triton e os metadados do modelo.
- `requirements.txt`: Arquivo contendo as dependências do Python necessárias para o projeto.
- `README.md`: Este arquivo que você está lendo agora, contendo informações sobre o projeto.

## Contribuição

Contribuições são bem-vindas! Se você encontrar algum problema ou tiver sugestões de melhorias, sinta-se à vontade para abrir uma issue ou enviar um pull request.
