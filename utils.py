from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st
openai.api_key = "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"
model = SentenceTransformer('all-MiniLM-L6-v2')
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from streamlit_chat import message
from langchain.vectorstores import Pinecone

# Specify your Pinecone API key
api_key = "b8eba605-67a5-4e77-9b19-6accfdabbda5"


pc = pinecone.Pinecone(
    api_key=api_key,
    environment="us-central1"
)
index = pc.Index('sample-index')

# Set up Pinecone client
api_key = "b8eba605-67a5-4e77-9b19-6accfdabbda5"
pc = pinecone.Pinecone(api_key=api_key, environment="us-central1")
index = pc.Index('sample-index')
model = SentenceTransformer('all-MiniLM-L6-v2')


# Initialize Langchain components
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="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")

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# Set up conversation prompt template
system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# Function to get conversation string
def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string

# Function to find match
def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(vector=input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']
