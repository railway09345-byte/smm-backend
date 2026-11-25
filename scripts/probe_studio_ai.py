"""Small live provider check. Prints no credentials or provider response bodies."""
import asyncio
import json
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.disable(logging.CRITICAL)
import httpx
from app.core.config import settings


async def main():
    results = {}
    async with httpx.AsyncClient(timeout=30) as client:
        if settings.OPENAI_API_KEY:
            try:
                response = await client.post('https://api.openai.com/v1/chat/completions', headers={'Authorization': 'Bearer ' + settings.OPENAI_API_KEY}, json={'model': 'gpt-4o-mini', 'messages': [{'role': 'user', 'content': 'Reply with the word OK.'}], 'max_tokens': 8})
                results['openai'] = {'http_status': response.status_code, 'generated': response.is_success and bool(response.json().get('choices'))}
            except Exception as exc:
                results['openai'] = {'error_type': type(exc).__name__}
        else:
            results['openai'] = 'not_configured'
        if settings.GEMINI_API_KEY:
            try:
                response = await client.post('https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent', headers={'x-goog-api-key': settings.GEMINI_API_KEY}, json={'contents': [{'parts': [{'text': 'Reply with the word OK.'}]}]})
                results['gemini'] = {'http_status': response.status_code, 'generated': response.is_success and bool(response.json().get('candidates'))}
            except Exception as exc:
                results['gemini'] = {'error_type': type(exc).__name__}
        else:
            results['gemini'] = 'not_configured'
    print(json.dumps(results))


if __name__ == '__main__':
    asyncio.run(main())
