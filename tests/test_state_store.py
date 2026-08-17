from __future__ import annotations

import json

from middot.state_store import RedisSessionStore


class FakePipeline:
    def __init__(self, client):
        self.client = client
        self.pending = None

    def watch(self, key):
        self.key = key

    def get(self, key):
        return self.client.values.get(key)

    def unwatch(self):
        return None

    def multi(self):
        return None

    def set(self, key, value, ex=None):
        self.pending = (key, value, ex)

    def execute(self):
        key, value, _ = self.pending
        self.client.values[key] = value
        return [True]

    def reset(self):
        return None


class FakeRedis:
    def __init__(self):
        self.values = {}

    def pipeline(self):
        return FakePipeline(self)


def test_redis_session_update_preserves_empty_json_arrays():
    client = FakeRedis()
    store = RedisSessionStore(client, 3600)
    key = store._key("session-one")
    client.values[key] = json.dumps(
        {"chat_history": [], "participants": [], "agent_task": {}},
        ensure_ascii=False,
    )

    assert store.update("session-one", {"city": "北京"}) is True
    restored = json.loads(client.values[key])

    assert restored["chat_history"] == []
    assert restored["participants"] == []
    assert restored["agent_task"] == {}
    assert restored["city"] == "北京"
