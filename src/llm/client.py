from google import genai
from settings import settings

client = genai.Client(settings.gemini_api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words",
)

print(response.text)