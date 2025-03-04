curl -N -X POST http://localhost:5337/api/stream \
    -H "Content-Type: application/json" \
    -d '{
    "commandstr": "Hello, how are you?",
    "conversationId": "test-conv-123",
    "model": "gpt-4o-mini",
    "provider": "openai",
    "saveToSqlite3": false,
    "npc": "sibiji",
    "currentPath": "/home/user",
    "messages": [],
    "attachments": []
}'