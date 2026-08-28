from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()

# 세션별 대화 기록 저장소
chat_histories = {}


def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    """세션 ID에 해당하는 대화 기록을 반환합니다."""
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]


# 프롬프트 템플릿(대화 기록 포함)
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 HR 상담 어시스턴트입니다. 직원의 휴가, 복리후생, 인사 관련 질문에 답변합니다."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 모델 설정
model = init_chat_model(model_provider="openai", model="gpt-5.2", temperature=0.7)

# 체인 생성
chain = prompt | model

# 메시지 히스토리가 포함된 체인
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 대화 실행
config = {"configurable": {"session_id": "user-123"}}

response1 = chain_with_history.invoke(
    {"input": "저는 개발팀 김철수입니다. 연차 잔여 일수 확인하고 싶어요."},
    config=config
)

print(response1.text)

response2 = chain_with_history.invoke(
    {"input": "제 이름이 뭐였죠?"},
    config=config
)

print(response2.text)

history = get_chat_history("user-123")

print("=====저장된 내용=====")
for message in history.messages:
    role = "사용자" if message.type == "human" else "AI"
    print(f"{role}: {message.text}")