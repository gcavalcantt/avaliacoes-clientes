import nltk
import re
import pandas as pd
import numpy as np
import gradio as gr
import tensorflow_hub as hub
from functools import partial
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from unidecode import unidecode
from sklearn.metrics.pairwise import cosine_similarity

# Baixa os pacotes necessários do NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Carrega o modelo Universal Sentence Encoder do TensorFlow Hub
modulo_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'
modelo = hub.load(modulo_url)

# Função para pré-processar o texto
def tratar_texto(texto):
    """
    Limpa e normaliza o texto:
    - Remove caracteres não alfanuméricos
    - Converte para minúsculas
    - Tokeniza as palavras
    - Remove stopwords, exceto 'não'
    - Remove acentos
    - Normaliza repetições excessivas de caracteres
    """
    
    # Remove caracteres não alfanuméricos e converte para minúsculas
    texto = re.sub(r'\W', ' ', texto.lower())
    
    # Tokeniza o texto
    word_tokenize_pt = partial(word_tokenize, language='portuguese')
    tokens = word_tokenize_pt(texto)
    
    # Define stopwords e remove todas, exceto 'não'
    stop_words = set(stopwords.words('portuguese'))
    stop_words.discard('não')
    tokens = [w for w in tokens if w not in stop_words]
    
    # Remove acentos
    texto_sem_acentos = unidecode(' '.join(tokens))
    
    # Normaliza repetições excessivas de caracteres, exceto 'rr' e 'ss'
    texto_normalizado = re.sub(r'(?!rr|ss)(.)\1+', r'\1', texto_sem_acentos)
    
    return texto_normalizado

# Função para processar o CSV e encontrar as reviews mais similares ao tema
def process_csv(arquivo, tema):
    """
    Processa um arquivo CSV contendo reviews e encontra as mais similares a um tema específico.
    
    - Aplica tratamento textual às reviews
    - Remove duplicatas e reviews vazias
    - Usa o Universal Sentence Encoder para transformar textos em embeddings
    - Calcula a similaridade cosseno entre o tema e as reviews
    - Retorna as 200 reviews mais similares
    """
    
    df = pd.read_csv(arquivo.name)
    
    # Aplica o tratamento de texto
    df['review_tratada'] = df['review'].apply(tratar_texto)
    
    # Remove reviews vazias e duplicadas
    df = df[df['review_tratada'] != '']
    df.drop_duplicates(subset=['review_tratada'], inplace=True)
    
    # Obtém embeddings das reviews e do tema
    reviews_emb = modelo(df['review_tratada'].tolist())
    tema_emb = modelo([tema])
    
    # Calcula a similaridade cosseno
    similaridades = cosine_similarity(tema_emb, reviews_emb).flatten()
    
    # Ordena os índices por similaridade decrescente
    top_indices = np.argsort(-similaridades)
    
    # Seleciona as reviews mais similares
    similar_reviews = df[['nota_review', 'review']].iloc[top_indices]
    similar_df = similar_reviews.head(200)
    
    # Salva os resultados em um arquivo CSV
    nome_arquivo = f'reviews_similares_{tema}.csv'
    similar_reviews.to_csv(nome_arquivo, index=False)
    
    return similar_df, nome_arquivo

# Interface Gradio para interagir com a aplicação
with gr.Blocks() as app:
    with gr.Row():
        gr.Markdown('## Encontrando as reviews mais similares ao tema')
    
    # Upload do arquivo CSV e entrada do tema
    csv_entrada = gr.File(label='Envie o CSV com as reviews', file_types=['.csv'])
    tema_entrada = gr.Textbox(label='Digite o tema para busca (ex: "entrega")')
    
    # Botão para processar as reviews
    botao = gr.Button('Clique para buscar as reviews')
    
    # Saídas: tabela com as top 200 reviews e link para baixar o CSV
    tabela_saida = gr.Dataframe(label='Top 200 reviews similares', headers=['Nota', 'Reviews'], interactive=False)
    arquivo_saida = gr.File(label='Baixar CSV ordenado com as reviews mais similares ao tema', interactive=False)
    
    # Aciona a função ao clicar no botão
    botao.click(process_csv, inputs=[csv_entrada, tema_entrada], outputs=[tabela_saida, arquivo_saida])
    
# Executa a aplicação
temp_app.launch()