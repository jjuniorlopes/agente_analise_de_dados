🕵️ Agente de Análise de Dados com IA (Gemini + LangChain)
Este projeto é uma aplicação web interativa que utiliza um agente de Inteligência Artificial para realizar Análise Exploratória de Dados (EDA) em qualquer arquivo CSV fornecido pelo usuário. A ferramenta foi construída com Streamlit para a interface, Pandas para a manipulação dos dados, e um agente poderoso baseado em LangChain e no modelo Gemini do Google.

✨ Principais Funcionalidades
Interface de Chat Interativa: Converse com o agente em linguagem natural para solicitar análises.

Análise de Qualquer CSV: Faça o upload de seus próprios arquivos .csv para uma análise personalizada.

Geração de Gráficos: Peça ao agente para criar visualizações de dados, como histogramas, gráficos de dispersão, boxplots e mapas de calor de correlação.

Análise Estatística: Obtenha insights sobre tipos de dados, valores ausentes, estatísticas descritivas (média, mediana), outliers e mais.

Modelo de Machine Learning: O agente pode aplicar modelos básicos de classificação (como Regressão Logística) e interpretar os resultados.

Conclusões Inteligentes: Peça ao agente para resumir as principais conclusões encontradas a partir da análise dos dados.

🚀 Demonstração ao Vivo (Live Demo)
Nota: Após implantar no Streamlit Community Cloud, substitua o link abaixo pelo URL da sua aplicação.


📸 Screenshot da Aplicação
Nota: Recomendo tirar um print da sua aplicação funcionando e adicioná-lo aqui para tornar seu README mais atraente.

🛠️ Tecnologias Utilizadas
Interface: Streamlit

Agente de IA: LangChain (create_pandas_dataframe_agent)

Modelo de Linguagem (LLM): Google Gemini (gemini-1.5-pro-latest)

Manipulação de Dados: Pandas

Visualização de Dados: Matplotlib

Machine Learning: Scikit-learn

⚙️ Configuração e Execução Local
Siga os passos abaixo para executar este projeto em sua máquina local.

Pré-requisitos
Python 3.9 ou superior

Pip (gerenciador de pacotes do Python)

Passos
1. Clone o Repositório

Bash

git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
2. Crie e Ative um Ambiente Virtual (Recomendado)

Bash

# Criar o ambiente
python -m venv .venv

# Ativar no Windows
.venv\Scripts\activate

# Ativar no macOS/Linux
source .venv/bin/activate
3. Instale as Dependências
Crie um arquivo chamado requirements.txt na raiz do seu projeto com o conteúdo abaixo e execute o comando pip install.

Arquivo requirements.txt:

streamlit
pandas
matplotlib
langchain-google-genai
langchain-core
langchain
langchain-experimental
scikit-learn
Comando de instalação:

Bash

pip install -r requirements.txt
4. Obtenha uma Chave de API do Google Gemini

Acesse o Google AI Studio.

Faça login com sua conta Google e clique em "Get API key" para gerar sua chave.

5. Execute a Aplicação
No terminal, com o ambiente virtual ativado, execute o seguinte comando:

Bash

streamlit run agente_analise_dados.py
A aplicação será aberta automaticamente no seu navegador. Insira sua chave de API na barra lateral para começar a usar.

☁️ Implantação no Streamlit Community Cloud
Para compartilhar sua aplicação com o mundo, siga estes passos simples:

Envie seu Projeto para o GitHub: Certifique-se de que seu repositório público no GitHub contém:

O arquivo principal da aplicação (ex: agente_analise_dados.py).

O arquivo requirements.txt.

Crie uma Conta no Streamlit Community Cloud:

Acesse share.streamlit.io e crie uma conta, conectando-a ao seu GitHub.

Implante a Aplicação:

No seu painel, clique em "New app".

Selecione seu repositório, o branch (geralmente main) e o caminho para o seu arquivo Python.

Clique em "Deploy!".

A plataforma irá instalar as dependências e hospedar sua aplicação, fornecendo um link público para acessá-la. A chave de API será solicitada diretamente na interface da aplicação, garantindo a segurança.
