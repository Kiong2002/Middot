"""审计并可选应用 Middot 存量候选的实体/字段规范化。"""

import argparse
import json

import app_v2


parser = argparse.ArgumentParser()
parser.add_argument("--device-id", required=True)
parser.add_argument("--apply", action="store_true", help="只应用高置信且通过类型校验的修正")
args = parser.parse_args()
print(json.dumps({
    "candidate_audit": app_v2.memory_audit_unresolved_candidates(args.device_id, apply=args.apply),
    "duplicate_entity_audit": app_v2.memory_audit_entity_duplicates(args.device_id),
}, ensure_ascii=False, indent=2))
