"""
多模态嵌入测试：验证 qwen3-vl-embedding 的文本、图片、融合嵌入能力。
"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.memory.memory_tool import EmbeddingClient


def _create_test_image(width=64, height=64, color=(255, 0, 0)):
    """创建测试用简易 PNG 图片（纯色矩形）。"""
    import struct
    import zlib

    def _chunk(chunk_type, data):
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))

    raw = b""
    for y in range(height):
        raw += b"\x00"  # filter byte
        for x in range(width):
            raw += bytes(color)

    idat = _chunk(b"IDAT", zlib.compress(raw))
    iend = _chunk(b"IEND", b"")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(sig + ihdr + idat + iend)
        return f.name


def test_text_embedding():
    """测试 1：纯文本嵌入（向后兼容）"""
    client = EmbeddingClient()
    if not client.is_available:
        print("⚠  API Key 不可用，跳过测试")
        return

    vec = client.embed("Hello, world! 你好，世界！")
    assert vec is not None, "纯文本嵌入应返回向量"
    assert len(vec) > 0, "向量不应为空"
    print(f"  ✓ 文本嵌入成功，维度: {len(vec)}")
    print(f"  ✓ 前 5 维: {vec[:5]}")

    # 验证缓存命中
    vec2 = client.embed("Hello, world! 你好，世界！")
    assert vec == vec2, "相同文本缓存应返回相同向量"
    print("  ✓ 缓存命中成功")


def test_image_embedding():
    """测试 2：图片嵌入"""
    client = EmbeddingClient()
    if not client.is_available:
        return

    img_path = _create_test_image()
    try:
        vec = client.embed(images=[img_path])
        assert vec is not None, "图片嵌入应返回向量"
        assert len(vec) > 0, "向量不应为空"
        print(f"  ✓ 图片嵌入成功，维度: {len(vec)}")
        print(f"  ✓ 前 5 维: {vec[:5]}")
    finally:
        os.unlink(img_path)


def test_text_image_fusion():
    """测试 3：图文融合嵌入（enable_fusion=True）"""
    client = EmbeddingClient()
    if not client.is_available:
        return

    img_path = _create_test_image()
    try:
        vec = client.embed(
            text="一张红色的图片",
            images=[img_path],
            enable_fusion=True,
        )
        assert vec is not None, "图文融合嵌入应返回向量"
        assert len(vec) > 0, "向量不应为空"
        print(f"  ✓ 图文融合嵌入成功，维度: {len(vec)}")
        print(f"  ✓ 前 5 维: {vec[:5]}")
    finally:
        os.unlink(img_path)


def test_cosine_similarity():
    """测试 4：余弦相似度计算"""
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [0.0, 1.0, 0.0]
    vec_c = [0.5, 0.5, 0.0]

    sim_ab = EmbeddingClient.cosine_similarity_dense(vec_a, vec_b)
    sim_ac = EmbeddingClient.cosine_similarity_dense(vec_a, vec_c)

    print(f"  ✓ 正交向量 (a,b) 相似度: {sim_ab:.4f}（应为 0.0）")
    print(f"  ✓ 相似向量 (a,c) 相似度: {sim_ac:.4f}（应为 ~0.707）")

    assert abs(sim_ab - 0.0) < 1e-6, "正交向量相似度应为 0"
    assert abs(sim_ac - 0.7071) < 0.001, "45° 角相似度应为 ~0.707"


if __name__ == "__main__":
    print("=" * 60)
    print("多模态嵌入功能验证")
    print("=" * 60)

    print("\n1️⃣  纯文本嵌入（向后兼容）")
    test_text_embedding()

    print("\n2️⃣  图片嵌入")
    test_image_embedding()

    print("\n3️⃣  图文融合嵌入")
    test_text_image_fusion()

    print("\n4️⃣  余弦相似度计算")
    test_cosine_similarity()

    print("\n" + "=" * 60)
    print("所有测试完成 ✅")
    print("=" * 60)