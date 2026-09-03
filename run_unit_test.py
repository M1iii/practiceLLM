#!/usr/bin/env python3
"""直接运行单元测试，绕过 pytest 配置问题。"""

import sys
import os

# 确保项目根目录在 sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests.unit.test_structured_data_tool import (
    TestColumnInfo,
    TestTableInfo,
    TestDatabaseInspectorStatic,
    TestSQLExecutorExtraction,
    TestStructuredDataTool,
)

def run_tests():
    """手动运行测试。"""
    tests = [
        TestColumnInfo("test_basic_construction"),
        TestColumnInfo("test_summary_output"),
        TestTableInfo("test_primary_key_extraction"),
        TestTableInfo("test_column_names"),
        TestTableInfo("test_to_prompt_context"),
        TestDatabaseInspectorStatic("test_excluded_tables_config"),
        TestDatabaseInspectorStatic("test_get_related_tables_keyword_match"),
        TestDatabaseInspectorStatic("test_format_schema_context"),
        TestSQLExecutorExtraction("test_extract_pure_sql"),
        TestSQLExecutorExtraction("test_extract_from_markdown_code_block"),
        TestSQLExecutorExtraction("test_extract_cannot_answer"),
        TestSQLExecutorExtraction("test_extract_add_limit_when_missing"),
        TestSQLExecutorExtraction("test_extract_keep_existing_limit"),
        TestSQLExecutorExtraction("test_extract_with_trailing_semicolon"),
        TestStructuredDataTool("test_basic_properties"),
        TestStructuredDataTool("test_get_parameters"),
        TestStructuredDataTool("test_lazy_initialization"),
        TestStructuredDataTool("test_error_missing_question"),
        TestStructuredDataTool("test_error_unknown_action"),
    ]

    passed = 0
    failed = 0

    print("=" * 60)
    print("Running StructuredDataTool unit tests")
    print("=" * 60)
    print()

    for test in tests:
        test_name = f"{test.__class__.__name__}.{test._testMethodName}"
        try:
            # pytest style: call the method
            if hasattr(test, "setup"):
                test.setup()
            getattr(test, test._testMethodName)()
            print(f"✅ {test_name}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Summary: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
