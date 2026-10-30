import sys
sys.path.insert(0, ".")
import httpx
import asyncio
from app.config import settings

async def test():
    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    script = "Heat a wok over high flame, add a splash of oil until it shimmers, then toss sliced carrots, bell peppers, broccoli florets, and snap peas, stirring rapidly. Sprinkle a pinch of salt, drizzle soy sauce, and cook until vegetables are crisp‑tender, releasing fragrant steam."
    
    prompt = (
        f"You are a master cinematic documentary director and visual editor.\n"
        f"Partition the following narration script into sequential, action-focused visual scene beats for a 20-second video.\n\n"
        f"FULL SCRIPT NARRATION:\n\"\"\"\n{script.strip()}\n\"\"\"\n\n"
        f"PARTITIONING RULES:\n"
        f"1. Target approximately 3 to 5 seconds of narration per visual beat (approx. 8 to 14 words per beat).\n"
        f"2. Each scene beat must depict one clearly identifiable, concrete visual action.\n"
        f"3. You MUST preserve the narration's exact original words and sequential order verbatim.\n\n"
        f"Return ONLY a valid JSON object matching this exact structure:\n"
        f"{{\n"
        f"  \"scenes\": [\n"
        f"    {{\"scene_order\": 1, \"scene_text\": \"...\"}}\n"
        f"  ]\n"
        f"}}\n"
    )

    payload = {
        "model": settings.GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
        print("Status:", resp.status_code)
        print("Body:", resp.text)

asyncio.run(test())
