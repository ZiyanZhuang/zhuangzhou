
import json
import re
from zhuangzhou.nlp.agent import AgentExecutor

def test_deepseek_parsing():
    print("Running DeepSeek-R1 Parsing Stress Test...")
    
    # Simulate a messy output from DeepSeek-R1
    mock_text = """
    <think>
    User wants to analyze a protein. I need to check files first.
    I will use the list_directory tool.
    </think>
    
    Certainly! I will list the files for you.
    Action: {"name": "list_directory", "input": {"dir_path": "./data"}}
    
    Please wait for the results.
    """
    
    # We need a dummy executor
    executor = AgentExecutor(None, None)
    tool_calls = executor._parse_manual_tool_calls(mock_text)
    
    if not tool_calls:
        print("FAIL: No tool calls parsed.")
        return False
        
    tc = tool_calls[0]
    print(f"Parsed Tool: {tc.name}")
    print(f"Arguments: {tc.arguments}")
    
    assert tc.name == "list_directory"
    assert tc.arguments.get("dir_path") == "./data"
    print("SUCCESS: DeepSeek-R1 Parsing Test Passed.")
    return True

if __name__ == "__main__":
    try:
        test_deepseek_parsing()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
