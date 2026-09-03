"""
测试结构化索引的边界情况：纯结构化数据 vs 无属性纯文本
"""
import sys
sys.path.insert(0, '.')
from src.tools.rag.llamaindex.structured_index import MetadataExtractor
from src.tools.rag.llamaindex.query_parser import QueryParser
from llama_index.core.schema import Document

extractor = MetadataExtractor()
parser = QueryParser()

print("=" * 60)
print("场景1: 纯结构化数据（用户信息表）")
print("=" * 60)
doc1 = Document(
    text="张三  13800138000  北京市海淀区中关村大街1号\n"
         "李四  13900139000  上海市浦东新区陆家嘴金融中心\n"
         "王五  13700137000  深圳市南山区科技园南区"
)
doc1.metadata = {"file_name": "user_info.csv"}
meta1 = extractor.extract(doc1)
print(f"  doc_type: {meta1.get('doc_type')}")
print(f"  tags: {meta1.get('tags')}")
print(f"  section: {meta1.get('section')}")

print()
print("=" * 60)
print("场景2: 无属性纯文本（小说节选）")
print("=" * 60)
doc2 = Document(
    text="那是一个闷热的下午。祥子拉着空车在街上走，心里盘算着今天的买卖。\n"
         "街上的柳树像病了似的，叶子挂着层灰土在枝上打着卷；枝条一动也懒得动，无精打采地低垂着。\n"
         "马路上一个水点也没有，干巴巴地发着白光。\n"
         "便道上尘土飞起多高，跟天上的灰气联接起来，结成一片毒恶的灰沙阵，烫着行人的脸。"
)
doc2.metadata = {"file_name": "camel_xiangzi.txt"}
meta2 = extractor.extract(doc2)
print(f"  doc_type: {meta2.get('doc_type')}")
print(f"  tags: {meta2.get('tags')}")
print(f"  section: {meta2.get('section')}")

print()
print("=" * 60)
print("场景3: 用户查询解析")
print("=" * 60)
for q in ["查找张三的电话", "骆驼祥子中关于天气的描写"]:
    filters, cleaned = parser.parse(q)
    print(f"  查询: {q}")
    print(f"  filters: {filters}")
    print(f"  cleaned: {cleaned}")
    print()