import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
import asyncio
import httpx
from app.services.providers.groq_script_provider import GroqScriptProvider

async def test():
    provider = GroqScriptProvider()
    script = "Heat a wok over high flame, add a splash of oil until it shimmers, then toss sliced carrots, bell peppers, broccoli florets, and snap peas, stirring rapidly. Sprinkle a pinch of salt, drizzle soy sauce, and cook until vegetables are crisp‑tender, releasing fragrant steam."
    
    orig_post = provider._post_groq
    async def debug_post(prompt, is_json=False, temperature=0.7):
        client = provider._get_client()
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": provider.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        if is_json:
            payload["response_format"] = {"type": "json_object"}
        resp = await client.post(provider.base_url, json=payload, headers=headers)
        print("DEBUG_POST Status:", resp.status_code)
        print("DEBUG_POST Body:", resp.text)
        return resp.json()["choices"][0]["message"]["content"]
    
    provider._post_groq = debug_post
    
    try:
        res = await provider.split_into_scenes(script, 20)
        print("Split result:", res)
    except Exception as e:
        print("Caught exception:", type(e), e)

asyncio.run(test())
