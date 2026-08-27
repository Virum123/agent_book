from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """애플리케이션 설정"""

    model_config=SettingsConfigDict(
        env_file=".env",            # .env 파일에서 환경 변수 로드
        env_file_encoding="utf-8",
        case_sensitive=False,       # 대소문자 구분 안함
        extra="ignore",             # 정의되지 않은 환경 변수 무시
    )

    # LLM 설정
    openai_api_key: SecretStr=Field(..., description="OpenAI API 키")
    default_model: str=Field("gpt-5.2", description="기본 모델")

    # 애플리케이션 설정
    debug: bool=Field(False, description="디버그 모드")

def get_settings() -> Settings:
    """설정 인스턴스를 반환합니다."""
    return Settings()