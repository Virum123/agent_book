from collections.abc import Iterator

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from my_agent.config import get_settings


def create_model(model_name: str | None=None) -> BaseChatModel:
    """LLM 모델을 생성합니다."""
    settings=get_settings()
    model_name=model_name or settings.default_model

    if not settings.openai_api_key:
        raise ValueError("키가 설정되지 않았습니다.")

    return init_chat_model(
        model_provider="openai",
        model=model_name,
        api_key=settings.openai_api_key.get_secret_value(),
    )

# 기본 시스템 프롬프트
DEFAULT_SYSTEM_PROMPT="""당신은 친절하고 도움이 되는 AI 어시스턴트입니다. 
사용자의 질문에 명확하고 간결히 답하세요.
확실치 않은 정보는 추측하지 마세요."""

class Agent:
    """AI 에이전트"""

    def __init__(self, model=None, system_prompt:str | None=None):
        self.model=model or create_model()
        self.system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT
        self.messages: list[BaseMessage]=[SystemMessage(content=self.system_prompt)]

    def chat(self, user_message: str) -> str:
        """사용자 메시지에 응답합니다."""
        self.messages.append(HumanMessage(content=user_message))
        response=self.model.invoke(self.messages)
        self.messages.append(response)
        return response.text

    def stream(self, user_message:str) -> Iterator[str]:
        """스트리밍으로 응답합니다."""
        self.messages.append(HumanMessage(content=user_message))
        full_response=""

        for chunk in self.model.stream(self.messages):
            if chunk.text:
                full_response += chunk.text
                yield chunk.text

        self.messages.append(AIMessage(content=full_response))

    def clear_history(self) -> None:
        """대화 기록을 초기화합니다."""
        self.messages=[SystemMessage(content=self.system_prompt)]


def create_agent(model_name: str | None=None, system_prompt: str | None=None):
    """AI 에이전트를 생성합니다."""
    model=create_model(model_name=model_name)
    return Agent(model=model, system_prompt=system_prompt)