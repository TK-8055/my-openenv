#!/bin/bash

echo "===== FINAL PROJECT CHECK ====="

echo ""
echo "1. Hidden urgency check"
grep -n "true_urgency" server/my_env_environment.py || echo "❌ MISSING true_urgency"

echo ""
echo "2. URGENCY_MAP check"
grep -n "URGENCY_MAP" server/my_env_environment.py || echo "❌ MISSING URGENCY_MAP"

echo ""
echo "3. Dependency check (task chain)"
grep -n "tasks\[0\].done" server/my_env_environment.py || echo "❌ MISSING dependency task0"
grep -n "tasks\[1\].done" server/my_env_environment.py || echo "❌ MISSING dependency task1"

echo ""
echo "4. Uncertainty check"
grep -n "random.uniform" server/my_env_environment.py || echo "❌ MISSING randomness"

echo ""
echo "5. Pressure check"
grep -n "\*\* 2" server/my_env_environment.py || echo "❌ MISSING nonlinear pressure"

echo ""
echo "6. Critical failure check"
grep -n "critical" server/my_env_environment.py | grep "return 0.0" || echo "❌ MISSING critical failure"

echo ""
echo "7. Prompt reasoning check"
grep -n "Explain your reasoning" inference.py || echo "❌ WEAK PROMPT"

echo ""
echo "8. OpenAI proxy check"
grep -n "API_BASE_URL" inference.py || echo "❌ MISSING API_BASE_URL"
grep -n "API_KEY" inference.py || echo "❌ MISSING API_KEY"

echo ""
echo "9. Localhost check (clean)"
grep -r "localhost" server inference.py client.py 2>/dev/null && echo "❌ FOUND localhost" || echo "✔ CLEAN (no localhost in your code)"

echo ""
echo "===== DONE ====="