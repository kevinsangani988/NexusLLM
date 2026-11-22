def select_model(message : str):
    if len(message)<30:
        return {
            "model" : "ollama_chat/qwen2.5:1.5b",
            "route" : "fast",
            "reason" : "Not a long query",
            "fallback" : "openrouter/thinkingmachines/inkling-small"
        }
    elif 30<len(message)<100:
        return {
            "model" :"openrouter/thinkingmachines/inkling-small",
            "route" : "medium-fast",
            "reason": "suitable for everyday tasks",
            "fallback" : "ollama_chat/qwen2.5:1.5b"
        }
    elif len(message)>100:
        return {
                    "model" :"openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
                    "route" : "complex",
                    "reason": "suitable for complex tasks",
                    "fallback" : "openrouter/thinkingmachines/inkling-small"
                }