from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()


# 세션별 저장소
chat_histories = {}


def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]


def summarize_conversation(messages, model) -> str:
    """대화 내용을 요약합니다."""
    if len(messages) == 0:
        return ""

    conversation = "\n".join([
        f"{'사용자' if m.type == 'human' else 'AI'}: {m.text}"
        for m in messages
    ])

    prompt = f"다음 대화의 핵심 정보를 한 문장으로 요약해 주세요:\n{conversation}"
    response = model.invoke(prompt)
    return response.text


# 모델과 체인 설정
model = init_chat_model(model_provider="openai", model="gpt-5.2", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 HR 상담 어시스턴트입니다. 직원의 휴가, 급여, 복리후생 관련 질문에 답변합니다.\n{summary}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain_with_history = RunnableWithMessageHistory(
    prompt | model,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 대화 진행
session_id = "user-123"
config = {"configurable": {"session_id": session_id}}

chain_with_history.invoke({"input": "저는 개발팀 김철수입니다.", "summary": ""},
    config=config)
chain_with_history.invoke({"input": "연차 12일 남았고, 다음 달에 3일 휴가 계획 중이에요.", "summary": ""},
    config=config)

# 요약 생성 후 기록 초기화
summary = summarize_conversation(get_chat_history(session_id).messages, model)
print(f"요약: {summary}")

chat_histories[session_id] = InMemoryChatMessageHistory() # 초기화는 여기서 진행

# 요약을 활용한 새 대화
response = chain_with_history.invoke(
    {"input": "내 정보를 알려 줘.", "summary": f"이전 대화 요약: {summary}"},
    config=config
)

print(response.text)