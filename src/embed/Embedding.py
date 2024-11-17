import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import rdkit
import deepchem as dc
import tqdm
from sentence_transformers import SentenceTransformer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, BertTokenizerFast
from transformers import AutoTokenizer, AutoModel, LlamaForCausalLM
from transformers import BitsAndBytesConfig
from openai import OpenAI

from config import MODELS_PATH

device = torch.device("cuda")

client = OpenAI(api_key='')


class Embedding:
    def __init__(self):
        self.input = None
        self.tokenizer = None
        self.model = None
        self.models = {
            'bert': self.get_huggingface_models,
            'sbert': self.get_huggingface_models,
            'chemberta_simcse': self.get_huggingface_models,
            'chemberta_deepchem': self.get_huggingface_models,
            'bert_smiles': self.get_huggingface_models,
            'gpt': self.get_gpt,
            'doc2vec': self.doc2vec,
            'mol2vec': self.mol2vec
        }

    def identify_model(self, model_name):

        if model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            self.model = AutoModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        elif model_name == 'sbert':
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        elif model_name == 'chemberta_simcse':
            self.tokenizer = AutoTokenizer.from_pretrained(MODELS_PATH + "simcsesqrt-model")
            self.model = AutoModel.from_pretrained(MODELS_PATH + "simcsesqrt-model")
        elif model_name == 'bert_smiles':
            self.tokenizer = BertTokenizerFast.from_pretrained('unikei/bert-base-smiles')
            self.model = BertModel.from_pretrained('unikei/bert-base-smiles')
        elif model_name == 'chemberta_deepchem':
            self.tokenizer = RobertaTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
            self.model = RobertaModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        elif model_name in ('gpt', 'doc2vec', 'mol2vec'):
            return
        else:
            raise Exception(f"Model not found: {model_name}")

        self.model = self.model.to(device)


    # def llama_embedding(self, token, quantization_config, name='meta-llama/Llama-2-7b-chat-hf', device_map='auto'):
    #     model = LlamaForCausalLM.from_pretrained(
    #         name,
    #         output_hidden_states=True,
    #         quantization_config=quantization_config,
    #         token=token,
    #         device_map=device_map
    #     )

    #     tokenizer = AutoTokenizer.from_pretrained(
    #         name,
    #         quantization_config=quantization_config,
    #         token=token,
    #     )

    #     embeddings = []

    #     for text in self.input:
    #         tokens = tokenizer(text)
    #         input_ids = tokens['input_ids']

    #         with torch.no_grad():
    #             input_embeddings = model.get_input_embeddings()
    #             embedding = input_embeddings(torch.LongTensor([input_ids]))
    #             embedding = torch.mean(embedding[0], 0).detach().cpu().numpy()
    #             embeddings.append(embedding)

    #     return embeddings

    def get_huggingface_models(self, pooling='last_avg', max_len=64):
        embeddings = []

        for text in self.input:
            tokens = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                return_attention_mask=True,
                truncation=True,
                return_tensors='pt',
            )

            tokens = tokens.to(device)
            with torch.no_grad():
                outputs = self.model(**tokens, output_hidden_states=True)


            hidden_states = outputs.hidden_states

            if pooling == 'first_last_avg':
                sentence_embedding = (hidden_states[-1] + hidden_states[1])
            elif pooling == 'last_avg':
                sentence_embedding = (hidden_states[-1])
            elif pooling == 'last2avg':
                sentence_embedding = (hidden_states[-1] + hidden_states[-2])
            else:
                raise Exception(f"Pooling not found: {pooling}")

            attention_mask = tokens['attention_mask']
            sentence_embedding = ((sentence_embedding * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))

            embeddings.append(sentence_embedding.detach().cpu().numpy()[0])

        return embeddings

    def get_gpt(self):
        embeddings = []
        for sentence in self.input:
            embedding = client.embeddings.create(input=sentence, model='text-embedding-3-small').data[0].embedding
            embeddings.append(embedding)

        return embeddings

    def doc2vec(self):
        embeddings = []
        documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(self.input)]
        model = Doc2Vec(documents, vector_size=50, window=2, min_count=1, workers=4)
        embeddings = [model.infer_vector(text.split()) for text in self.input]
        return embeddings

    def mol2vec(self):
        embeddings = []
        featurizer = dc.feat.Mol2VecFingerprint()
        embeddings = featurizer.featurize(self.input)
        return embeddings

    def get_embeddings(self, model_name, texts, **kwargs):
        self.identify_model(model_name)
        self.input = texts
        embeddings = self.models[model_name]()
        return pd.DataFrame(embeddings)