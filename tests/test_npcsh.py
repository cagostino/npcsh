import pexpect
import sys
import time
import re

def test_npcsh():
    # Start the npcsh process
    npcsh = pexpect.spawn('npcsh', encoding='utf-8', timeout=30)
    npcsh.logfile = sys.stdout  # Log output to console for visibility

    # Wait for the prompt
    npcsh.expect('npcsh>')

    # Test 1: Compile the foreman NPC
    npcsh.sendline('/compile foreman.npc')
    npcsh.expect('Compiled NPC profile:')
    npcsh.expect('npcsh>')

    # Test 2: Switch to foreman NPC
    npcsh.sendline('/foreman')
    npcsh.expect('Switched to NPC: foreman')
    npcsh.expect('foreman>')

    # Test 3: Test weather_tool
    npcsh.sendline("What's the weather in Tokyo?")
    # Expect the assistant to provide a weather update
    npcsh.expect('The weather in .* is', timeout=60)
    npcsh.expect('foreman>')
    print("Test 3 passed: weather_tool executed successfully.")
    time.sleep(1)

    # Test 4: Test calculator tool
    npcsh.sendline("Calculate the sum of 2 and 3.")
    npcsh.expect('The result of .* is 5', timeout=30)
    npcsh.expect('foreman>')
    print("Test 4 passed: calculator tool executed successfully.")
    time.sleep(1)

    # Test 5: Test database_query tool
    npcsh.sendline("Find all users with the role 'admin'.")
    npcsh.expect('Here are the results:', timeout=30)
    npcsh.expect('foreman>')
    print("Test 5 passed: database_query tool executed successfully.")
    time.sleep(1)

    # Exit npcsh
    npcsh.sendline('/exit')
    npcsh.expect(pexpect.EOF)

if __name__ == '__main__':
    test_npcsh()