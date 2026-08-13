"""地标城市推断、校区消歧和伪地理编码结果的回归测试。"""

import importlib.util
import os
import tempfile


root = tempfile.mkdtemp(prefix="middot-location-resolution-")
os.environ["MIDDOT_DB_PATH"] = os.path.join(root, "middot.db")
os.environ["MIDDOT_DEVICE_SECRET"] = "location-resolution-smoke-secret-20260813"
app_path = os.environ.get(
    "MIDDOT_APP_TEST_PATH", os.path.join(os.path.dirname(__file__), "app_v2.py")
)
spec = importlib.util.spec_from_file_location("middot_location_test_app", app_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


bootstrap = {"city": "北京", "participants": []}
assert module._infer_assistant_city("阿杰在浙大", bootstrap) == "杭州"
assert module._infer_assistant_city("阿杰在北京的浙江大学校友会", bootstrap) == "北京"

# 只说“浙大”不能产生任何坐标草稿，必须让用户选校区。
sid = module.session_create({
    "city": "北京",
    "participants": [
        {"id": "me", "name": "我", "lng": None, "lat": None},
        {"id": "friend", "name": "阿杰", "lng": None, "lat": None},
    ],
    "current_user_message": "阿杰在浙大",
    "current_utterance_parse": {
        "city_context": "杭州",
        "locations": [{
            "participant_index": 2, "owner": "阿杰", "expression": "浙大",
            "needs_disambiguation": True,
            "canonical_candidates": list(module._ZJU_CAMPUS_CHOICES),
        }],
    },
    "agent_task": {"status": "running", "completed": [], "failures": []},
})
result, patch = module._tool_set_participant_location(
    sid, {"index": 2, "place_name": "浙江大学", "city": "北京"}
)
assert result["ok"] and result.get("waiting_for_user"), result
assert patch["type"] == "choices" and patch["mode"] == "single", patch
assert [x["label"] for x in patch["options"]] == list(module._ZJU_CAMPUS_CHOICES)
assert module.session_get(sid)["city"] == "杭州"

# 模型若直接走“同名地点查询”旁路，也只能返回校区，不得搜出附属医院。
result, patch = module._tool_clarify_participant_location(
    sid, {"index": 2, "keyword": "浙大", "near_hint": "杭州"}
)
assert result["ok"] and patch["type"] == "choices", (result, patch)
assert [x["label"] for x in patch["options"]] == list(module._ZJU_CAMPUS_CHOICES)
assert patch["question"].startswith("阿杰在"), patch

# 即便模型错误传 city=北京，本轮解析出的杭州仍必须优先；具体校区才能落草稿。
calls = []
def fake_geocode(address, city=None):
    calls.append((address, city))
    return {
        "success": True, "lng": 120.091986, "lat": 30.297066,
        "formatted_address": "浙江省杭州市西湖区浙江大学紫金港校区",
        "city": "杭州市",
    }
module.amap_geocode = fake_geocode
module.session_update(sid, {
    "current_user_message": "浙江大学紫金港校区",
    "current_utterance_parse": {
        "city_context": "杭州",
        "locations": [{
            "participant_index": 2, "owner": "阿杰",
            "expression": "浙江大学紫金港校区", "needs_disambiguation": False,
        }],
    },
})
result, patch = module._tool_set_participant_location(
    sid, {"index": 2, "place_name": "浙江大学紫金港校区", "city": "北京"}
)
assert result["ok"] and patch["type"] == "draft", (result, patch)
assert calls[0][1] == "杭州", calls
assert patch["data"]["address"].startswith("浙江省杭州市"), patch

# “查询成功但只返回北京市”是伪成功，不能再写进参与者地址。
module.amap_geocode = lambda address, city=None: {
    "success": True, "lng": 116.407387, "lat": 39.904179,
    "formatted_address": "北京市", "city": "北京市",
}
bad = module._validated_place_geocode("浙大", "北京")
assert not bad["success"] and "市中心" in bad["error"], bad
print("LOCATION_RESOLUTION_SMOKE_OK")
