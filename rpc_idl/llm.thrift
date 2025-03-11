enum ModelSelection {
    GPT4 = 1,
    GPT3.5 = 2,
    CLAUDE3 = 3,
    LLAMA3 = 4,
    QWEN = 5,
}

struct LlmGreetingRequest {
  1: i64 user_id = 0,
  2: i64 character_id = 0,
  3: str message,
  4: ModelSelection model,
}

struct LlmGreetingResponse {
  1: i64 user_id = 0,
  2: str message,
}