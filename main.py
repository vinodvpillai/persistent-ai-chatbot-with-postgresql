from typing import List
import atexit
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
from os.path import join, dirname
from dotenv import load_dotenv

import gradio as gr
import psycopg2

# Load environment variables
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Initialize the language model
llm = ChatGoogleGenerativeAI(model=os.environ.get('GOOGLE_MODEL'), api_key=os.environ.get('GOOGLE_API_KEY'))  # type: ignore

# PostgreSQL connection setup
conn = psycopg2.connect(
    dbname=os.environ.get('POSTGRES_DB'),
    user=os.environ.get('POSTGRES_USER'),
    password=os.environ.get('POSTGRES_PASSWORD'),
    host=os.environ.get('POSTGRES_HOST'),
    port=os.environ.get('POSTGRES_PORT')
)
cursor = conn.cursor()

# Define Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're an assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# Class to handle chat history with PostgreSQL
class PostgresChatHistory(BaseChatMessageHistory, BaseModel):
    user_id: str
    conversation_id: str
    messages: List[BaseMessage] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
        # Load existing messages from the database during initialization
        self.messages = self.get_all_messages()

    def add_message(self, message: BaseMessage) -> None:
        message_type = 'AI' if isinstance(message, AIMessage) else 'Human'
        cursor.execute(
            "INSERT INTO chat_history (user_id, conversation_id, message_type, content) VALUES (%s, %s, %s, %s)",
            (self.user_id, self.conversation_id, message_type, message.content)
        )
        conn.commit()
        # Also append to the local messages list
        self.messages.append(message)

    def get_all_messages(self) -> List[BaseMessage]:
        cursor.execute(
            "SELECT message_type, content FROM chat_history WHERE user_id = %s AND conversation_id = %s",
            (self.user_id, self.conversation_id)
        )
        rows = cursor.fetchall()
        messages = []
        for message_type, content in rows:
            if message_type == 'AI':
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
        return messages

    def clear(self) -> None:
        cursor.execute(
            "DELETE FROM chat_history WHERE user_id = %s AND conversation_id = %s",
            (self.user_id, self.conversation_id)
        )
        conn.commit()
        # Clear local messages list
        self.messages.clear()


# Function to retrieve or create a session history
def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    return PostgresChatHistory(user_id=user_id, conversation_id=conversation_id)

# Chain
chain = prompt | llm

# Session History
with_message_history = RunnableWithMessageHistory(
    chain, # type: ignore
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        ),
    ],
)

# Chat function for Gradio
def chat(username, user_query, history):
    user_id = username
    conversation_id = "1"  # For simplicity, assuming one conversation per user

    # Add the user query to the chat history
    history.append([f"{username}", f"{user_query}"])

    # Invoke the model with the user's query and retrieve session context
    response = with_message_history.invoke(
        {"question": user_query},
        config={"configurable": {"user_id": user_id, "conversation_id": conversation_id}}
    ).content

    # Add the AI response to the chat history
    history.append(["AI", response])
    
    return history

# Gradio interface
with gr.Blocks() as chat_interface:
    gr.Markdown("# Chat with AI Assistant")
    
    with gr.Row():
        username_input = gr.Textbox(label="Username", placeholder="Enter your username")
        query_input = gr.Textbox(label="Your Query", placeholder="Ask something...")
        
    submit_button = gr.Button("Submit")
    chatbot = gr.Chatbot(label="Chat History")
    submit_button.click(chat, inputs=[username_input, query_input, chatbot], outputs=chatbot)

# Launch Gradio interface
chat_interface.launch()

# Close PostgreSQL connection when done
atexit.register(lambda: conn.close())