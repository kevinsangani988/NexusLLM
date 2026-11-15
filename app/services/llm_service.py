from litellm import acompletion

async def completion(model, message):
    response = await acompletion(
        model=model,
        messages=[{"role": "user", "content": message}]
    )

    return response.choices[0].message.content