from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from dotenv import load_dotenv

load_dotenv()

# 세션별 대화 기록 저장소
chat_histories = {}


def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    """세션 ID에 해당하는 대화 기록을 반환합니다."""
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]

#  메시지 트리머 설정(최근 4개 메시지만 유지)
trimmer = trim_messages(
    max_tokens=4, # 최대 메시지 수
    strategy="last", # 최근 메시지 유지
    token_counter=len, # 메시지 수로 카운트
    include_system=True, # 시스템 메시지 포함
    start_on="human" # human 메시지부터 카운트
)

# 프롬프트 템플릿(대화 기록 포함)
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 HR 상담 어시스턴트입니다. 직원의 휴가, 복리후생, 인사 관련 질문에 답변합니다."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 모델 설정
model = init_chat_model(model_provider="openai", model="gpt-5.2", temperature=0.7)

# 체인 생성
chain = (
    {"history": lambda x: trimmer.invoke(x["history"]), "input": lambda x: x["input"]}
    | prompt 
    | model
)

# 메시지 히스토리가 포함된 체인
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 대화 실행
config = {"configurable": {"session_id": "user-123"}}

# 여러 차례 대화
print("=== 대화 1 ===")
response1 = chain_with_history.invoke({"input": "저는 개발팀 김철수입니다."},
    config=config)
print(response1.text)

print("\n=== 대화 2 ===")
response2 = chain_with_history.invoke({"input": "연차가 며칠 남았나요?"},
    config=config)
print(response2.text)

print("\n=== 대화 3 ===")
response3 = chain_with_history.invoke({"input": "다음 주에 3일 휴가 쓸 수 있을까요?"},
    config=config)
print(response3.text)

print("\n=== 대화 4 ===")
# 첫 번째 대화는 윈도우 밖으로 밀려남
response4 = chain_with_history.invoke({"input": "제가 어느 팀이었죠?"}, config=config)
print(response4.text)
