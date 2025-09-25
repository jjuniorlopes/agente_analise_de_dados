# -*- Código Python para construção do agente de Inteligência Artificial genérico  -*-
"""
================================================================================
AGENTE DE ANÁLISE DE DADOS (EDA) COM GEMINI E LANGCHAIN
================================================================================

Objetivo Principal:
Este script implementa um agente de Inteligência Artificial genérico, projetado
para realizar Análise Exploratória de Dados (EDA) em qualquer conjunto de dados
fornecido em um arquivo CSV. O objetivo é criar uma ferramenta de EDA poderosa
e acessível, onde um usuário, mesmo sem conhecimento técnico em programação,
possa interagir em linguagem natural para obter insights, visualizações e
conclusões sobre seus dados.

Principais Funcionalidades:
- Carregamento de qualquer arquivo CSV através de uma interface web.
- Interface de chat interativa para realizar perguntas em linguagem natural.
- Geração de respostas textuais, tabelas e análises estatísticas.
- Criação de visualizações gráficas (histogramas, boxplots, mapas de calor, etc.).
- Aplicação de modelos básicos de machine learning para classificação.
- Apresentação de conclusões consolidadas a partir das análises realizadas.

Tecnologias Utilizadas:
- Interface Web: Streamlit
- Estrutura do Agente: LangChain e LangChain Experimental
- Modelo de Linguagem (LLM): Google Gemini (via langchain-google-genai)
- Manipulação de Dados: Pandas
- Visualização de Dados: Matplotlib
- Machine Learning: Scikit-learn

Lógica de Funcionamento:
1. O usuário acessa a interface criada com o Streamlit.
2. Na barra lateral, o usuário insere sua chave de API do Google Gemini e tem a
   opção de reiniciar a conversa.
3. O usuário faz o upload de um arquivo CSV de sua escolha.
4. O script carrega os dados do CSV em um DataFrame do Pandas.
5. Um agente do LangChain é inicializado, conectando o modelo Gemini a um
   conjunto de ferramentas capaz de executar código Python (especificamente
   comandos do Pandas e Matplotlib).
6. O usuário faz uma pergunta no chat (ex: "Qual a correlação entre as colunas?").
7. O agente recebe a pergunta, analisa o DataFrame e gera internamente o
   código Python necessário para encontrar a resposta.
8. O código é executado em um ambiente seguro, e o resultado (seja um texto,
   uma tabela ou um gráfico) é capturado.
9. O resultado final é formatado e exibido de forma clara na interface de
   chat para o usuário. A memória da conversa é mantida para permitir
   perguntas de acompanhamento.
"""

# Passo 1: Instalação e Importação das Bibliotecas
# -------------------------------------------------
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Layout Moderno com Cores ESG e Azul corporativo
# Substitua a sua função antiga por esta versão atualizada
def aplicar_estilo_moderno():
    """Aplica um CSS customizado para um layout moderno com cores ESG."""
    estilo_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');

        /* Corpo da página */
        .stApp {
            font-family: 'Source Sans Pro', sans-serif;
            background-color: #F0F2F6;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #E6EAF1;
        }

        /* Títulos */
        h1, h2 {
            color: #1A3A5A; /* Azul Corporativo Escuro */
        }
        
        /* --- MUDANÇA AQUI: Título "Configurações" na Sidebar --- */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
             background-color: #00A896; /* Fundo Verde ESG */
             color: white;              /* Texto branco */
             padding: 10px;             /* Espaçamento interno para criar a "caixa" */
             border-radius: 10px;       /* Bordas arredondadas */
             text-align: center;        /* Centralizar o texto */
             margin-bottom: 20px;       /* Espaço abaixo do título */
        }

        /* Botões */
        .stButton>button {
            border-radius: 20px;
            border: 1px solid #00A896; /* Verde ESG */
            background-color: #00A896;
            color: white;
            transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover {
            background-color: white;
            color: #00A896;
        }

        /* Chat bubbles */
        [data-testid="stChatMessage"] {
            border-radius: 20px;
            padding: 1em;
            margin-bottom: 1em;
        }
        
        [data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-assistant"]) {
            background-color: #FFFFFF;
            border: 1px solid #E6EAF1;
        }

        [data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-user"]) {
            background-color: #D6F1EE;
        }

        /* --- MUDANÇA AQUI: Texto (label) da caixa de input da API Key --- */
        [data-testid="stSidebar"] label {
            background-color: #00A896;    /* Fundo Verde ESG */
            color: white !important;      /* Texto branco (!important para garantir) */
            padding: 8px;                 /* Espaçamento interno */
            border-radius: 10px;          /* Bordas arredondadas */
            display: block;               /* Garante que o fundo ocupe toda a largura */
            width: 100%;
            text-align: center;           /* Centralizar o texto */
            margin-bottom: 10px;          /* Espaço antes da caixa de input */
        }
        
        [data-testid="stSidebar"] input[type="password"] {
            background-color: #D6F1EE;
            border: 1px solid #00A896;
            border-radius: 5px;
        }
    </style>
    """
    st.markdown(estilo_css, unsafe_allow_html=True)

# --- Configuração da Página do Streamlit ---
st.set_page_config(page_title="Agente de Análise de Dados", page_icon="🕵️", layout="wide")

# Aplica o estilo
aplicar_estilo_moderno()

st.title("🕵️ Agente de Análise de Dados com Gemini")
st.write(
    "**Você é um Analista de dados.** "
    "Sua tarefa é analisar os dados carregados e responder perguntas sobre eles "
    "e retornar textos, explicações e representações gráficas. Para começar, insira sua API Key do Google e faça o upload de um arquivo CSV."
)

# --- Funções Principais ---

@st.cache_resource
def get_llm(google_api_key):
    """
    Inicializa e retorna o modelo de linguagem (LLM) do Google Gemini.
    """
    # Modelo LLM Gemini 2.5 Flash
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0,
        convert_system_message_to_human=True
    )

@st.cache_data
def load_csv(uploaded_file):
    """
    Carrega o arquivo CSV enviado pelo usuário para o agente.
    """
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo CSV: {e}")
        return None

# Sistema de armazenamento de histórico de chat na memória
store = {}
def get_session_history(session_id: str):
    """Obtém ou cria um histórico de chat para uma sessão específica."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Função para Limpar o Chat
def limpar_chat():
    """Limpa o histórico de mensagens e reseta o estado do arquivo carregado."""
    st.session_state.messages = []
    if 'df_loaded' in st.session_state:
        del st.session_state['df_loaded']

# --- Interface do Streamlit ---

with st.sidebar:
    st.header("⚙️ Configurações")
    
 # Tenta carregar a chave de API dos segredos do Streamlit
    # Se não encontrar, mostra o campo para inserção manual (ideal para rodar localmente)
    try:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("✅ Chave de API carregada com segurança!")
    except (KeyError, FileNotFoundError):
        st.warning("🔑 Chave de API não encontrada nos segredos.")
        google_api_key = st.text_input(
            "Insira sua Chave de API do Google Gemini e pressione ENTER", 
            type="password",
            help="Você pode configurar a chave de forma permanente no arquivo .streamlit/secrets.toml"
        )
    
    # Botão para limpar o chat/reiniciar
    st.button("Limpar Chat e Reiniciar", on_click=limpar_chat, use_container_width=True)

    st.info(
    """
    **💡 Bem-vindo ao Agente de Análise de Dados!**
    
    📢 Para testar o agente, segue o link de um arquivo `csv` de fraudes em cartão de crédito pode ser baixado em: 
    
       ➡️ [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
       ou carregado dentro do seu computador.

    🔑 Para sua segurança, a chave necessária para ativar a inteligência do agente não ficará armazenada.

    1.  Carregue seu Arquivo CSV: 
        
        ✅ Clique no botão "Browse files" e selecione o arquivo CSV que você deseja analisar.
        
        ✅ O agente irá carregar os dados e exibir as primeiras linhas para confirmação.

    2.  **Faça suas Perguntas:**
        
        ✅ Utilize o campo de chat na parte inferior da página para fazer perguntas em linguagem natural sobre seus dados.

    **Exemplos de perguntas que você pode fazer:**
    
    ➡️ `Analise os tipos de dados informando o tipo de cada um deles, se tem valores ausentes ou valores repetidos. faça uma analise destes dados.`
    
    ➡️ `Qual a variável ou variáveis pode ou podem ser ou serem usada(s) para gerar um gráfico de outliers?`
    
    ➡️ `Gere o(s) gráfico(s) de outliers da variável ou variáveis que podem ser ou serem usada(s). Caso tenha mais de um coloque lado a lado.`
    
    ➡️ `Qual a correlação entre as colunas nos dados?`
    
    ➡️ `Gere um gráfico destas corelações.`
    
    ➡️ `Voce poderia aplicar algum metodo de classificação? Qual variáveil possivel? Aplique este método desta variável e diga como obtece esta conclusão.`
    
    ➡️ `Quais são as principais conclusões que podemos tirar destes dados e das analises geradas?`
    """
    )

# Verificação inicial da chave de API
if not google_api_key:
    st.warning("Por favor, insira sua chave de API na barra lateral para habilitar o agente.")
    st.stop()

uploaded_file = st.file_uploader("Faça o upload do seu arquivo CSV aqui para sua análise", type="csv")
if uploaded_file is None:
    st.info("Aguardando o carregamento de um arquivo CSV...")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe o histórico do chat
for message in st.session_state.messages:
    with st.chat_message(message.type):
        if "plot" in message.additional_kwargs:
            st.markdown(message.content)
            st.pyplot(message.additional_kwargs["plot"])
        else:
            st.markdown(message.content)

# Lógica principal
if uploaded_file is not None:
    df = load_csv(uploaded_file)

    if df is not None:
        if not hasattr(st.session_state, 'df_loaded') or not st.session_state.df_loaded:
             st.success("Arquivo CSV carregado com sucesso! Veja as cinco primeiras linhas:")
             st.dataframe(df.head()) #visualização das 5 primeiras linhas
             st.write("Estatísticas descritivas dos dados:") #estatísticas descritivas
             st.dataframe(df.describe(include='all').T) #transposta para melhor visualização
             st.session_state.df_loaded = True # Marca que o dataframe foi carregado

        try:
            llm = get_llm(google_api_key)
        except Exception as e:
            # Tratamento de erro para chave de API inválida
            if "API key is invalid" in str(e):
                st.error("Digite o token da LLM Gemini corretamente.")
            else:
                st.error(f"Ocorreu um erro ao inicializar o modelo: {e}")
            st.stop()
        
        agent_persona = """
        Você é um Analista de dados e especialista em Python, Pandas, Matplotlib e Seaborn. Sua tarefa é analisar os dados carregados e analisa-los.
        - Você deve responder a perguntas sobre o DataFrame (dados) `df` carregado.
        - Você tem acesso a uma ferramenta para executar código Python internamente.
        - Para responder, você DEVE gerar e executar um código Python válido usando a ferramenta `python_repl_ast`.
        - O DataFrame está disponível como a variável `df`.
        - **IMPORTANTE para gráficos**: Se a pergunta do usuário pedir um gráfico, GERE O CÓDIGO para criar o gráfico usando matplotlib ou seaborn. NÃO use `st.pyplot()` no código que você gera. Apenas gere o código de plotagem padrão. A interface se encarregará de exibi-lo.
        - Baseie-se nos dados carregados do CSV.
        - Para dados numéricos decimais retorne penas duas casas decimais.
        - Não ivente nada, se basear nos dados carregados.
        - Se ouver resposta textual em ingles, responda em português.
        - Seja conciso e direto ao ponto.
        - Sempre que possóvel, forneça respostas em formato de tabela para melhor visualização.
        - Use gráficos para ilustrar tendência, distribuições e correlações nos dados.
        - Os gráficos recomendados incluem histogramas, gráficos de barras, gráficos de linhas, gráficos de dispersão, mapas de calor ou outros gráficos.
        - Sempre que possível, utilize gráficos para complementar suas respostas.
        - Explique suas conclusões de forma clara.
        - Caso não saiba a informação, responda: "Não sei informar o que você pediu. Estou pronto para sua próxima pergunta ou instrução."
        - Após gerar um gráfico, forneça também uma breve explicação textual sobre o que o gráfico que você gerou.
        - Ao responder, não mostre o código Python gerado, a menos que seja explicitamente solicitado. Apenas mostre o resultado (texto, tabelas ou gráficos).
        - Seja um analista de dados crítico e detalhista.        
        """
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", agent_persona),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            prompt=prompt_template,
            verbose=False, # Alterado para False para uma UI mais limpa
            allow_dangerous_code=True,
            agent_executor_kwargs={"handle_parsing_errors": True}
        )
        
        agent_with_memory = RunnableWithMessageHistory(
            agent, get_session_history,
            input_messages_key="input", history_messages_key="chat_history",
        )
        
        if prompt := st.chat_input("Qual análise deseja realizar?"):
            st.session_state.messages.append(HumanMessage(content=prompt))
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Aguarde, analisando os dados..."):
                    try:
                        plt.clf() 
                        config = {"configurable": {"session_id": "user_session"}}
                        response = agent_with_memory.invoke({"input": prompt}, config=config)
                        output = response["output"]
                        st.markdown(output)

                        fig = plt.gcf()
                        if fig.get_axes():
                            st.pyplot(fig)
                            ai_message = AIMessage(content=output, additional_kwargs={"plot": fig})
                        else:
                            ai_message = AIMessage(content=output)
                        st.session_state.messages.append(ai_message)

                    except Exception as e:
                        # Tratamento de erro para cota excedida
                        if "429" in str(e):
                            st.error("Você excedeu sua cota de análise, tente mais tarde.")
                        else:
                            st.error(f"Ocorreu um erro: {e}")