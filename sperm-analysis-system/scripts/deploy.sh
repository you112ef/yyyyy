#!/bin/bash
# 🚀 Complete Deployment Guide
# Developer: Youssef Shitiwi (يوسف شتيوي)

# =============================================================================
# 1. QUICK DEPLOYMENT - Get Started in 5 Minutes
# =============================================================================

echo "🔍 Checking prerequisites..."
command -v docker >/dev/null 2>&1 || { echo "❌ Docker required. Install from https://docker.com"; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose required"; exit 1; }

echo "🚀 Starting complete system..."
docker-compose up -d

echo "⏳ Waiting for services to start..."
sleep 30

echo "🏥 Checking system health..."
curl -s http://localhost:8000/api/v1/health/simple || echo "⚠️ API not ready yet"

echo "✅ Deployment complete!"
echo "🌐 API: http://localhost:8000"
echo "📚 Docs: http://localhost:8000/docs"
echo "💾 Database: localhost:5432"
echo "🔄 Redis: localhost:6379"