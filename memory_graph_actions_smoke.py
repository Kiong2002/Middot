"""图谱候选确认与正式关系修改/删除契约测试。"""

import importlib.util
import os
import tempfile


root=tempfile.mkdtemp(prefix="middot-graph-actions-")
os.environ["MIDDOT_DB_PATH"]=os.path.join(root,"middot.db")
os.environ["MIDDOT_DEVICE_SECRET"]="memory-graph-actions-secret-20260813"
app_path=os.environ.get("MIDDOT_APP_TEST_PATH",os.path.join(os.path.dirname(__file__),"app_v2.py"))
spec=importlib.util.spec_from_file_location("middot_graph_actions",app_path)
module=importlib.util.module_from_spec(spec);assert spec and spec.loader;spec.loader.exec_module(module)

did="7"*32;now=module._now();conn=module._db_connect()
conn.execute("INSERT INTO devices(device_id,created_at,last_seen_at) VALUES(?,?,?)",(did,now,now))
conn.execute("INSERT INTO memory_candidates(device_id,kind,entity_key,field_name,candidate_value,confidence,persistence_score,evidence_summary,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",(did,"person","阿杰","usual_place","浙江大学紫金港校区",.96,.9,"阿杰常从紫金港出发","candidate",now,now))
conn.commit();conn.close()

client=module.app.test_client();client.set_cookie(module.DEVICE_COOKIE,module._device_cookie_encode(did))
profile=client.get("/api/v2/memories").get_json()
candidate_edge=next(edge for edge in profile["graph"]["edges"] if edge["status"]=="candidate" and edge.get("candidate_ids"))
assert candidate_edge["value"]=="浙江大学紫金港校区"
confirmed=client.post("/api/v2/memories/candidate-group",json={"action":"confirm","candidate_ids":candidate_edge["candidate_ids"]})
assert confirmed.status_code==200,confirmed.get_json()
profile=confirmed.get_json()["profile"]
edge=next(edge for edge in profile["graph"]["edges"] if edge.get("predicate")=="usual_place" and edge["status"]=="confirmed")
assert edge["origin"]=="wiki" and isinstance(edge["action_id"],int),edge

edited=client.patch("/api/v2/memories/relation",json={"origin":edge["origin"],"id":edge["action_id"],"predicate":"usual_place","value":"浙江大学玉泉校区"})
assert edited.status_code==200,edited.get_json()
conn=module._db_connect()
assert conn.execute("SELECT value FROM memory_wiki_facts WHERE id=?",(edge["action_id"],)).fetchone()[0]=="浙江大学玉泉校区"
assert conn.execute("SELECT usual_place FROM memory_people WHERE device_id=? AND name='阿杰'",(did,)).fetchone()[0]=="浙江大学玉泉校区"
assert conn.execute("SELECT value FROM memory_wiki_fact_versions WHERE device_id=?",(did,)).fetchone()[0]=="浙江大学紫金港校区"
conn.close()

deleted=client.delete("/api/v2/memories/relation",json={"origin":"wiki","id":edge["action_id"],"predicate":"usual_place"})
assert deleted.status_code==200,deleted.get_json()
conn=module._db_connect()
assert not conn.execute("SELECT 1 FROM memory_wiki_facts WHERE id=?",(edge["action_id"],)).fetchone()
person=conn.execute("SELECT usual_place FROM memory_people WHERE device_id=? AND name='阿杰'",(did,)).fetchone()
assert person and person[0] is None

# 旧人物投影生成的关系同样可以只修改/删除单个字段，不删除人物实体。
conn.execute("UPDATE memory_people SET relation='朋友',updated_at=? WHERE device_id=? AND name='阿杰'",(now+10,did));conn.commit();conn.close()
profile=client.get("/api/v2/memories").get_json()
relation_edge=next(edge for edge in profile["graph"]["edges"] if edge.get("predicate")=="relation")
assert relation_edge["origin"]=="person"
changed=client.patch("/api/v2/memories/relation",json={"origin":"person","id":relation_edge["action_id"],"predicate":"relation","value":"同事"})
assert changed.status_code==200,changed.get_json()
removed=client.delete("/api/v2/memories/relation",json={"origin":"person","id":relation_edge["action_id"],"predicate":"relation"})
assert removed.status_code==200,removed.get_json()
conn=module._db_connect();row=conn.execute("SELECT relation FROM memory_people WHERE device_id=? AND name='阿杰'",(did,)).fetchone();assert row and row[0] is None;conn.close()
print("MEMORY_GRAPH_ACTIONS_SMOKE_OK")
