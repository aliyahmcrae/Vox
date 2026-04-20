import asyncio, json
from openai import AsyncOpenAI

with open("secrets.json") as f:
  secrets = json.load(f)

client = AsyncOpenAI(api_key=secrets["openai"])

async def main():
    response = await client.responses.create(
        model="gpt-5.2",
        input="Tell me about the RMS Titanic"
    )

    print(response.output_text)

asyncio.run(main())
