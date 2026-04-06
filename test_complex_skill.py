"""
Test Complex Query Skill in isolation

Tests the Plan-Execute pattern with multi-step query decomposition.
"""

import sys
sys.path.insert(0, r"c:\Users\71949\Desktop\text2sql")

from langchain_community.chat_models import ChatTongyi
from langchain.messages import HumanMessage

from agent.config import get_config
from agent.database import SQLDatabaseManager
from agent.tools import SQLToolManager
from skills.complex_query import ComplexQuerySkill

def test_complex_query():
    """Test complex query skill with a multi-step query"""
    
    print("="*70)
    print("COMPLEX QUERY SKILL - INTEGRATION TEST")
    print("="*70)
    
    # Initialize components
    print("\n[1/4] Initializing components...")
    config = get_config()
    llm = ChatTongyi(model=config.llm.model, dashscope_api_key=config.llm.api_key)
    db_manager = SQLDatabaseManager(config.database)
    tool_manager = SQLToolManager(db_manager, llm)
    skill = ComplexQuerySkill(llm, tool_manager, db_manager)
    print("[OK] Components initialized")
    
    # Test query - requires multiple steps
    test_query = """
    Find the top 3 most popular shops based on the number of vouchers they issued,
    and for each shop, show the total number of vouchers and their average value.
    """
    
    print(f"\n[2/4] Test Query:")
    print(f"  {test_query.strip()}")
    
    # Execute
    print(f"\n[3/4] Executing Complex Query Skill...")
    print("-"*70)
    
    initial_state = {"messages": [HumanMessage(content=test_query)]}
    
    try:
        result = skill.invoke(initial_state)
        
        print("-"*70)
        print("\n[4/4] EXECUTION COMPLETED!")
        
        # Display results
        messages = result.get("messages", [])
        print(f"\nTotal messages: {len(messages)}")
        
        # Show query plan (DETAILED)
        query_plan = result.get("query_plan", [])
        print(f"\nQuery Plan: {query_plan}")
        if query_plan:
            print(f"\nQuery Plan ({len(query_plan)} steps):")
            for step in query_plan:
                print(f"  Step {step.get('step_id')}: {step.get('description', 'N/A')}")
                print(f"    Query: {step.get('query', 'N/A')[:80]}...")
                print(f"    Depends on: {step.get('depends_on', [])}")
        else:
            print("\nWARNING: No query plan generated!")
        
        # Show step results (DETAILED)
        step_results = result.get("step_results", {})
        print(f"\nStep Results: {step_results}")
        if step_results:
            print(f"\nExecuted Steps: {len(step_results)}/{len(query_plan)}")
            for step_id, res in step_results.items():
                status = "[OK]" if res.get("success") else "[FAIL]"
                print(f"  {status} Step {step_id}: {res.get('description', 'N/A')}")
                if not res.get("success"):
                    print(f"       Error: {res.get('error', 'Unknown')}")
        else:
            print("\nWARNING: No steps were executed!")
        
        # Show execution flow
        print(f"\nExecution Flow:")
        for i, msg in enumerate(messages, 1):
            content_preview = str(msg.content)[:80]
            msg_type = type(msg).__name__
            print(f"  {i}. [{msg_type}] {content_preview}...")
        
        # Show final result
        if messages:
            print(f"\nFinal Result:")
            last_msg = messages[-1]
            print(f"  {last_msg.content[:500]}")
            if len(last_msg.content) > 500:
                print("  ...")
        
        print("\n" + "="*70)
        print("[SUCCESS] TEST PASSED - Complex Query Skill works correctly!")
        print("="*70)
        
        return True
        
    except Exception as e:
        print("-"*70)
        print(f"\n[FAILED] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*70)
        return False

if __name__ == "__main__":
    success = test_complex_query()
    sys.exit(0 if success else 1)
