#!/bin/bash

echo "Testing health endpoint..."
curl -s http://localhost:5337/api/health | jq '.'

echo -e "\nTesting execute endpoint..."
curl -s -X POST http://localhost:5337/api/execute \
  -H "Content-Type: application/json" \
  -d '{"commandstr": "hello world", "currentPath": "/media/caug/extradrive1/npcww/npcsh", "conversationId": "test124"}' | jq '.'

echo -e "\nTesting conversations endpoint..."
curl -s "http://localhost:5337/api/conversations?path=/tmp" | jq '.'

echo -e "\nTesting conversation messages endpoint..."
curl -s http://localhost:5337/api/conversation/test123/messages | jq '.'

