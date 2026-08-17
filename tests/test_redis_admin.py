import json

from middot.redis_admin import classify_key, mask_redis_key, redact_value, redis_key_page


class _FakePipeline:
    def __init__(self, client):
        self.client = client
        self.commands = []

    def type(self, key):
        self.commands.append(("type", key))

    def pttl(self, key):
        self.commands.append(("pttl", key))

    def memory_usage(self, key):
        self.commands.append(("memory_usage", key))

    def execute(self):
        return [getattr(self.client, command)(key) for command, key in self.commands]


class _FakeRedis:
    def __init__(self):
        self.values = {
            "middot:session:abcdef12": json.dumps({
                "conversation_id": "1234567890abcdef",
                "current_user_message": "我从精确地址出发",
                "participants": [{"name": "张三", "lat": 39.99291, "lng": 116.3109}],
                "api_key": "do-not-show",
            }, ensure_ascii=False)
        }

    def scan(self, cursor=0, match=None, count=None):
        del cursor, count
        keys = list(self.values)
        if match == "middot:session:*":
            keys = [key for key in keys if key.startswith("middot:session:")]
        return 0, keys

    def pipeline(self, transaction=False):
        del transaction
        return _FakePipeline(self)

    def type(self, key):
        return "string" if key in self.values else "none"

    def pttl(self, key):
        return 3599000 if key in self.values else -2

    def memory_usage(self, key):
        return len(self.values[key].encode())

    def strlen(self, key):
        return len(self.values[key])

    def getrange(self, key, start, end):
        return self.values[key][start:end + 1]


def test_key_categories_and_masks_identifiers():
    assert classify_key("middot:session:abcdef12") == "sessions"
    assert classify_key("middot:room:123456:presence:device") == "room_presence"
    assert classify_key("middot:room:123456:lease") == "room_leases"
    assert classify_key("middot:rooms:expiries") == "room_expiries"
    assert "abcdef12" not in mask_redis_key("middot:session:abcdef12")
    assert "123456" not in mask_redis_key("middot:room:123456:lease")


def test_redaction_hides_secrets_messages_ids_and_exact_coordinates():
    value = redact_value({
        "api_key": "secret",
        "conversation_id": "1234567890abcdef",
        "id": "room-device-identifier",
        "current_user_message": "一段用户原文",
        "address": "某个精确门牌号",
        "location": "116.3109,39.99291",
        "lat": 39.99291,
        "name": "张三",
    })
    assert value["api_key"] == "[已隐藏]"
    assert "1234567890abcdef" not in value["conversation_id"]
    assert "room-device-identifier" not in value["id"]
    assert value["current_user_message"] == "[文本 6 字]"
    assert value["address"] == "[精确地址已隐藏]"
    assert value["location"] == "[精确坐标已隐藏]"
    assert value["lat"] == 39.99
    assert value["name"] == "张**"


def test_key_page_returns_structured_privacy_aware_preview():
    page = redis_key_page(_FakeRedis(), category="sessions", limit=10)
    assert page["cursor"] == "0"
    assert len(page["items"]) == 1
    item = page["items"][0]
    assert item["category"] == "sessions"
    assert item["ttl_ms"] == 3599000
    assert item["preview"]["api_key"] == "[已隐藏]"
    assert item["preview"]["participants"][0]["name"] == "张**"
