# https://github.com/langchain-ai/langchain/issues/6918
# https://www.sqlitetutorial.net/sqlite-sample-database/

import os

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["OPENAI_API_KEY"] = "INPUT YOUR KEY"

from langchain_experimental.sql import SQLDatabaseChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


chat_template = """ Based on the schema given {info} write an executable query for the user input. 
Execute it in the database and get sql results. Make a response to user from sql results based on 
the question. 
Input: "user input"
SQL query: "SQL Query here"
"""
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", chat_template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# memory = ConversationBufferMemory()
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
db = SQLDatabase.from_uri("sqlite:///chinook.db")
table_info = db.table_info
m1 = ConversationBufferWindowMemory(k=4, return_messages=True)
db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db, verbose=True)

while True:
    query = input("human:")
    if query != "":
        chat = m1.load_memory_variables({})["history"]
        prompt = chat_prompt.format(info=table_info, history=chat, input=query)
        response = db_chain.run(prompt)
        m1.save_context({"input": query}, {"output": response})
    else:
        break

# Result:
# Human: artistsの数はいくつですか
# AI: There are 275 artists in the database.
# Human:  employeesの数はいくつですか
# AI: There are 8 employees in the database.
# Human: 1つ前の質問と2つ前の質問の合計はいくつですか
# SQLQuery:SELECT COUNT(*) FROM artists;
# SQLResult: [(275,)]
# Answer:There are 283 entries in total between artists and employees in the database.
# > Finished chain.
