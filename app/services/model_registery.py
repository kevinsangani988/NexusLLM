MODEL_REGISTERY = {
    "fast" : {
        "model" : "ollama_chat/qwen2.5:1.5b",
        "description":"for simple and lightweight tasks"
    },
    "powerful" : {
        "model" : "openrouter/thinkingmachines/inkling-small",
        "description" : "more efficient member of the Inkling family and is suited for reasoning, coding, agentic workflows, retrieval-augmented generation, instruction following, and multilingual conversation."
    },
    "complex" : {
        "model" : "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
        "description" : "strong at multi-step reasoning and planning, with high-throughput inference designed for high-volume agent pipelines"

    }
}