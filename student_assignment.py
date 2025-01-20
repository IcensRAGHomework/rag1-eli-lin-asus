import json
import traceback
import requests

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import (ResponseSchema, StructuredOutputParser)
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.output_parsers import SimpleJsonOutputParser


from langchain import hub

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)
llm = AzureChatOpenAI(
    model=gpt_config['model_name'],
    deployment_name=gpt_config['deployment_name'],
    openai_api_key=gpt_config['api_key'],
    openai_api_version=gpt_config['api_version'],
    azure_endpoint=gpt_config['api_base'],
    temperature=gpt_config['temperature']
)

def get_date_schemas():
    return [
        ResponseSchema(
            name="date",
            description="該紀念日的日期",
            type="YYYY-MM-DD"),
        ResponseSchema(
            name="name",
            description="該紀念日的名稱")
    ]

def get_result_schemas():
    response_schemas = get_date_schemas()
    output_parser = StructuredOutputParser(response_schemas=response_schemas)
    format_instructions = output_parser.get_format_instructions()
    return [
        ResponseSchema(
            name="Result",
            description="一個結果的清單",
            type=format_instructions
        )
    ]

def get_date_chain():
    response_schemas = get_result_schemas()
    output_parser = StructuredOutputParser(response_schemas=response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages([
        ("system","你是一個使用台灣語言並回答問題且會遵照範例JSON格式輸出的紀念日專家,{format_instructions}"),
        ("human","{question}")])
    prompt = prompt.partial(format_instructions=format_instructions)
    return prompt | llm | output_parser


def generate_hw01(question):
    response_schemas = get_result_schemas()
    output_parser = StructuredOutputParser(response_schemas=response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages([
        ("system","你是一個使用台灣語言並回答問題且會遵照範例JSON格式輸出的紀念日專家,{format_instructions}"),
        ("human","{question}")])
    prompt = prompt.partial(format_instructions=format_instructions)
    # response = llm.invoke(prompt.format(question=question)).content
    response = get_date_chain().invoke({'question': question})
    
    return json.dumps(response, ensure_ascii=False)
    
def generate_hw02(question):
    examples = [
        {"input": "2024年台灣10月紀念日有哪些?", "output": '{\
            "Result": [\
                {\
                    "date": "2024-10-10",\
                    "name": "國慶日"\
                }\
            ]\
        }'}
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一個使用台灣語言且會遵照範例JSON格式輸出指定屬性且正確排版的答案的紀念日專家"),
            few_shot_prompt,
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ]
    )
    tools = [tool]
    agent = create_openai_functions_agent(llm, tools, final_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    response = agent_executor.invoke({"input": question}).get('output')
    return response

def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response

## for generate_hw02
def get_holidays_from_clendarific(year: int, month: int):
    api_key = "WqROpo9g1fRPEpFaSJgMOLXPuWsPXCA3"
    url = f"https://calendarific.com/api/v2/holidays?&api_key={api_key}&country=tw&year={year}&month={month}"
    response = requests.get(url).json().get('response')
    return response

class GetValue(BaseModel):
    year: int = Field(description="first number")
    month: int = Field(description="second number")

tool = StructuredTool.from_function(
    name="get_value",
    description="當詢問xx年台灣yy月紀念日有哪些?時, xx 為 year, yy 為 month",
    func=get_holidays_from_clendarific,
    args_schema=GetValue
)

print(generate_hw01("2024年台灣10月紀念日有哪些?"))