"""保守的划分隔离：共享来源或复合水体名称的记录不跨集合。"""

import hashlib
import re


def assign_split_groups(frame):
    frame = frame.copy()
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for _, row in frame.iterrows():
        name = "water:" + row.waterbody
        find(name)
        for component in re.split(r"\s*/\s*", row.waterbody):
            if component:
                union(name, "water:" + component)
        source = row.get("source_id")
        if isinstance(source, str) and source.strip():
            union(name, "source:" + source.strip().casefold())
    roots = [find("water:" + name) for name in frame.waterbody]
    frame["split_group"] = ["sg_" + hashlib.sha256(root.encode()).hexdigest()[:16] for root in roots]
    mapping = frame[["waterbody", "split_group"]].drop_duplicates().sort_values(["split_group", "waterbody"])
    audit = {
        "policy": "waterbody/source connected components v1",
        "mapping": mapping.to_dict(orient="records"),
        "component_count": frame.split_group.nunique(),
        "limitations": "来源整体隔离可能过于保守；名称连接不构成完整水体实体或空间独立性证明",
    }
    return frame, audit
