"""
Test Main Graph with Skill Integration

Tests the complete skill-based architecture with routing.
"""

import sys
sys.path.insert(0, r"c:\Users\71949\Desktop\text2sql")

from langchain_community.chat_models import ChatTongyi
from langchain.messages import HumanMessage

from agent.config import get_config
from agent.skill_graph_builder import create_skill_based_graph

def test_skill_routing():
    """Test query routing to different skills"""
    
    print("="*70)
    print("SKILL-BASED MAIN GRAPH - INTEGRATION TEST")
    print("="*70)
    
    # Initialize
    print("\n[1/4] Initializing Skill-based graph...")
    config = get_config()
    llm = ChatTongyi(model=config.llm.model, dashscope_api_key=config.llm.api_key)
    
    try:
        graph = create_skill_based_graph(config, llm)
        print("[OK] Graph created successfully")
    except Exception as e:
        print(f"[FAIL] Graph creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test queries for each skill type
    test_queries = [
        {
            "type": "SIMPLE",
            "query": "List the first 5 users from tb_user table",
            "expected_skill": "simple"
        },
        {
            "type": "COMPLEX", 
            "query": "Find the top 3 shops by voucher count and show their average voucher values",
            "expected_skill": "complex"
        },
        {
            "type": "ANALYSIS",
            "query": "Analyze user engagement trends and provide insights with visualization recommendations",
            "expected_skill": "analysis"
        }
    ]
    
    print(f"\n[2/4] Testing {len(test_queries)} query types...")
    
    results = []
    for i, test_case in enumerate(test_queries, 1):
        query_type = test_case["type"]
        query = test_case["query"]
        expected = test_case["expected_skill"]
        
        print(f"\n--- Test {i}/{len(test_queries)}: {query_type} Query ---")
        print(f"Query: {query[:60]}...")
        
        try:
            initial_state = {"messages": [HumanMessage(content=query)]}
            result = graph.invoke(initial_state)
            
            # Check routing
            query_type_result = result.get("query_type", "unknown")
            routed_correctly = query_type_result == expected
            
            status = "[OK]" if routed_correctly else "[WARN]"
            print(f"{status} Routed to: {query_type_result} (expected: {expected})")
            
            # Check execution
            messages = result.get("messages", [])
            print(f"[OK] Execution completed ({len(messages)} messages)")
            
            results.append({
                "type": query_type,
                "routed_to": query_type_result,
                "expected": expected,
                "correct": routed_correctly,
                "executed": len(messages) > 0
            })
            
        except Exception as e:
            print(f"[FAIL] Test failed: {e}")
            results.append({
                "type": query_type,
                "routed_to": "error",
                "expected": expected,
                "correct": False,
                "executed": False
            })
    
    # Summary
    print("\n[3/4] Test Results Summary")
    print("-"*70)
    
    correct_routing = sum(1 for r in results if r["correct"])
    successful_execution = sum(1 for r in results if r["executed"])
    
    print(f"Routing Accuracy: {correct_routing}/{len(results)}")
    print(f"Execution Success: {successful_execution}/{len(results)}")
    
    for result in results:
        status = "[OK]" if result["correct"] else "[FAIL]"
        print(f"{status} {result['type']}: routed to '{result['routed_to']}'")
    
    # Overall result
    print("\n[4/4] Overall Assessment")
    print("-"*70)
    
    all_passed = correct_routing == len(results) and successful_execution == len(results)
    
    if all_passed:
        print("[SUCCESS] All tests passed!")
        print("\nSkill Integration:")
        print("  [OK] Query Router working")
        print("  [OK] Simple Query Skill integrated")
        print("  [OK] Complex Query Skill integrated")
        print("  [OK] Data Analysis Skill integrated")
        print("  [OK] Conditional routing working")
    else:
        print("[PARTIAL] Some tests failed")
        print(f"  Routing: {correct_routing}/{len(results)}")
        print(f"  Execution: {successful_execution}/{len(results)}")
    
    print("\n" + "="*70)
    
    return all_passed

if __name__ == "__main__":
    success = test_skill_routing()
    sys.exit(0 if success else 1)
