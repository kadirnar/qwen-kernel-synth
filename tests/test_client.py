import asyncio
import json

import httpx

from kernel_synth.client import AsyncOpenRouterClient


def test_client_normalizes_response_and_sends_privacy_controls() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "gen-1",
                "model": "fallback/model",
                "provider": "Example",
                "choices": [{"message": {"content": '{"ok":true}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.001},
            },
        )

    async def run() -> None:
        async with AsyncOpenRouterClient(
            api_key="test",
            transport=httpx.MockTransport(handler),
        ) as client:
            result = await client.complete(
                model="primary/model",
                fallbacks=("fallback/model",),
                messages=[{"role": "user", "content": "hello"}],
                response_format={"type": "json_object"},
                max_tokens=100,
                temperature=0.5,
                top_p=0.9,
                seed=7,
                reasoning_effort="high",
                zdr=True,
            )
        assert result.actual_model == "fallback/model"
        assert result.cost == 0.001

    asyncio.run(run())
    assert captured["models"] == ["primary/model", "fallback/model"]
    assert captured["reasoning"] == {"effort": "high", "exclude": True}
    assert captured["provider"] == {
        "allow_fallbacks": True,
        "require_parameters": True,
        "data_collection": "deny",
        "zdr": True,
    }
