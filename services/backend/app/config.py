from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    # --- upstream services ---
    qdrant_url: str = "http://qdrant:6333"
    embedder_url: str = "http://embedder:8002"
    stt_url: str = "http://stt:8003"
    llm_url: str = "http://llm:8001"
    llm_model_name: str = "assistant"

    # --- audio / vad ---
    sample_rate: int = 16000
    frame_ms: int = 20
    vad_aggressiveness: int = 2
    vad_silence_ms: int = 700
    vad_min_speech_ms: int = 250
    vad_max_utterance_ms: int = 25000
    vad_preroll_ms: int = 300
    interim_transcription: bool = True
    interim_interval_ms: int = 1400
    # Default language for new sessions; the UI can override it per connection.
    # "auto" is deliberately not the default: whisper picks the language from
    # the first seconds of audio, and on 2-second utterances it regularly
    # mistakes Turkish for Persian or Arabic.
    stt_language: str = "tr"

    # --- rag ---
    docs_dir: str = "/data/docs"
    documents_collection: str = "documents"
    memories_collection: str = "memories"
    embedding_dim: int = 384
    chunk_chars: int = 450
    chunk_overlap: int = 80
    rag_top_k: int = 4
    # Threshold for always-on injection: strict, because every false positive
    # lands in front of a question that was never about the documents.
    rag_min_score: float = 0.82
    # Threshold for the explicit knowledge_search tool: looser. Here the model
    # already decided the question is about the user's files, it sees the score
    # next to each excerpt, and a miss ("nothing found") is worse than handing
    # it a marginal passage to judge for itself.
    rag_tool_min_score: float = 0.75
    # When false (default) the user's question reaches the model verbatim and
    # documents are only consulted if the model calls knowledge_search.
    always_on_rag: bool = False

    # --- llm behaviour ---
    assistant_language: str = "auto"
    temperature: float = 0.6
    # Deciding whether to call a tool is a classification, not a creative act,
    # and at 0.6 it is genuinely a coin flip near the decision boundary: the
    # same question got knowledge_search once and a made-up answer the next
    # time. Sampling is kept low until the tool results are in, then the normal
    # temperature takes over so the spoken answer still sounds natural.
    tool_decision_temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 1024
    # Every past turn lengthens the prompt, and prompt length measurably
    # degrades tool calling on this model. Three turns is the compromise
    # between follow-up questions working and tools staying reliable.
    history_turns: int = 3
    tools_enabled: bool = True
    max_tool_rounds: int = 4

    # --- web tools ---
    searxng_url: str = ""
    http_user_agent: str = "Mozilla/5.0 (X11; Linux aarch64) JetsonAssistant/1.0"
    web_fetch_max_chars: int = 4000
    web_search_results: int = 5

    @property
    def frame_bytes(self) -> int:
        """PCM16 mono bytes in one VAD frame."""
        return int(self.sample_rate * self.frame_ms / 1000) * 2


settings = Settings()
