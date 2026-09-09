"""SafeFullCache 极端场景测试。"""
import sys, time
sys.path.insert(0, "d:\\pycharm_project\\practiceLLM")
from src.core.cache import SafeFullCache

# 1. 空数据键
c = SafeFullCache(max_size=10, default_ttl=3600)
assert c.get("") is None, "空键"
assert c.get(None) is None, "None键"
print("[OK] 空/None键安全")

# 2. 超大值
c.set("large", "x" * 1000000)
v = c.get("large")
assert len(v) == 1000000
print("[OK] 1MB值正常")

# 3. TTL过期
c = SafeFullCache(max_size=10, default_ttl=3600)
c.set("k", "v", ttl=0.01)
time.sleep(0.02)
assert c.get("k") is None, "TTL过期"
print("[OK] TTL过期正常淘汰")

# 4. 容量淘汰
c = SafeFullCache(max_size=3, default_ttl=3600)
c.set("a", "1"); c.set("b", "2"); c.set("c", "3")
c.set("d", "4")
assert c.get("a") is None, "a应被淘汰"
assert c.get("d") == "4", "d应存在"
print(f"[OK] 容量淘汰: size={c.stats()['size']}")

# update 不存在键（SafeFullCache.update 仅更新已有键，不新增）
c = SafeFullCache(max_size=10, default_ttl=3600)
c.update("nonexistent", "val")
assert c.get("nonexistent") is None, "update不存在键不新增"
c.set("exist", "old")
c.update("exist", "new")
assert c.get("exist") == "new", "update存在键应生效"
print("[OK] update存在键更新/不存在键不新增")

# 6. get_or_set异常
c = SafeFullCache(max_size=10, default_ttl=3600)
def fails():
    raise RuntimeError("生成失败")
try:
    c.get_or_set("err", fails)
    print("[FAIL] 应抛出异常")
except RuntimeError:
    print("[OK] get_or_set异常正确透传")

# 7. clear
c.clear()
assert c.stats()["size"] == 0
print("[OK] clear后size=0")

# 8. 批量淘汰
c = SafeFullCache(max_size=5, default_ttl=3600)
for i in range(100):
    c.set(f"k{i}", f"v{i}")
assert c.stats()["size"] <= 5
print(f"[OK] 100次写入后size={c.stats()['size']} (<=5)")

# 9. stats字段完整性
s = c.stats()
for field in ["size", "max_size", "hits", "misses", "hit_rate", "expired_count"]:
    assert field in s, f"缺少字段: {field}"
print("[OK] stats字段完整")

# 10. 1容量（max_size=0 自动提升为 1）
c = SafeFullCache(max_size=0, default_ttl=3600)
c.set("k", "v")
assert c.stats()["size"] == 1
print("[OK] max_size=0 自动提升为 1")

print("--- SafeFullCache 极端场景完成 ---")