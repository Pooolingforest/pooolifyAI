import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "dev")
    api_token: Optional[str] = os.getenv("API_TOKEN")

    database_url: Optional[str] = os.getenv("DATABASE_URL")

    # 고정 지원 모델만 사용하므로 기본 프로바이더 키는 더 이상 의미 없음
    llm_default_provider: str = os.getenv("LLM_DEFAULT_PROVIDER", "openai")

    # Provider-specific API keys
    llm_openai_api_key: Optional[str] = os.getenv("LLM_OPENAI_API_KEY")
    llm_claude_api_key: Optional[str] = (
        os.getenv("LLM_CLAUDE_API_KEY") or os.getenv("LLM_ANTHROPIC_API_KEY")
    )


settings = Settings()
