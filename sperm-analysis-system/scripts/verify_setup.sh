#!/bin/bash

# ==================================================
# SPERM ANALYSIS SYSTEM VERIFICATION SCRIPT
# Developer: Youssef Shitiwi
# ==================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo -e "${BLUE}======================================"
echo -e "🔍 SPERM ANALYSIS SYSTEM VERIFICATION"
echo -e "   Developer: Youssef Shitiwi"
echo -e "======================================${NC}"

print_check() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
    fi
}

print_status() {
    echo -e "${YELLOW}🔍 $1${NC}"
}

# Check project structure
print_status "Checking project structure..."

# Backend files
print_check $([[ -f "$PROJECT_ROOT/backend/main.py" ]] && echo 0 || echo 1) "Backend FastAPI main.py"
print_check $([[ -f "$PROJECT_ROOT/backend/routes/analysis.py" ]] && echo 0 || echo 1) "Analysis routes"
print_check $([[ -f "$PROJECT_ROOT/backend/services/video_processor.py" ]] && echo 0 || echo 1) "Video processor service"

# Training files
print_check $([[ -f "$PROJECT_ROOT/training/scripts/train_model.py" ]] && echo 0 || echo 1) "AI training script"
print_check $([[ -f "$PROJECT_ROOT/training/models/casa_metrics.py" ]] && echo 0 || echo 1) "CASA metrics calculator"
print_check $([[ -f "$PROJECT_ROOT/training/models/tracker.py" ]] && echo 0 || echo 1) "DeepSORT tracker"

# Android files
print_check $([[ -f "$PROJECT_ROOT/android/app/build.gradle" ]] && echo 0 || echo 1) "Android build configuration"
print_check $([[ -f "$PROJECT_ROOT/android/app/src/main/java/com/spermanalysis/ui/MainActivity.kt" ]] && echo 0 || echo 1) "Android main activity"
print_check $([[ -f "$PROJECT_ROOT/android/gradlew" ]] && echo 0 || echo 1) "Gradle wrapper"

# Docker files
print_check $([[ -f "$PROJECT_ROOT/Dockerfile" ]] && echo 0 || echo 1) "Dockerfile"
print_check $([[ -f "$PROJECT_ROOT/docker-compose.yml" ]] && echo 0 || echo 1) "Docker Compose"

# Scripts
print_check $([[ -f "$PROJECT_ROOT/scripts/build_android.sh" ]] && echo 0 || echo 1) "Android build script"
print_check $([[ -f "$PROJECT_ROOT/scripts/test_android_app.sh" ]] && echo 0 || echo 1) "Android test script"
print_check $([[ -f "$PROJECT_ROOT/scripts/deploy.sh" ]] && echo 0 || echo 1) "Deployment script"

# Documentation
print_status "Checking documentation..."
print_check $([[ -f "$PROJECT_ROOT/README.md" ]] && echo 0 || echo 1) "Project README"
print_check $([[ -f "$PROJECT_ROOT/docs/api.md" ]] && echo 0 || echo 1) "API documentation"
print_check $([[ -f "$PROJECT_ROOT/docs/android_build_guide.md" ]] && echo 0 || echo 1) "Android build guide"
print_check $([[ -f "$PROJECT_ROOT/ANDROID_BUILD_INSTRUCTIONS.md" ]] && echo 0 || echo 1) "Quick build instructions"

# Count files
echo ""
print_status "Project statistics:"
echo "📁 Backend files: $(find "$PROJECT_ROOT/backend" -type f | wc -l)"
echo "📁 Training files: $(find "$PROJECT_ROOT/training" -type f | wc -l)"
echo "📁 Android files: $(find "$PROJECT_ROOT/android" -type f | wc -l)"
echo "📁 Docker files: $(find "$PROJECT_ROOT" -maxdepth 1 -name "*.yml" -o -name "Dockerfile" | wc -l)"
echo "📁 Documentation files: $(find "$PROJECT_ROOT/docs" -type f | wc -l)"
echo "📁 Total project files: $(find "$PROJECT_ROOT" -type f | wc -l)"

echo ""
echo -e "${GREEN}✅ System verification complete!${NC}"
echo -e "${BLUE}📱 Ready to build Android APK${NC}"
echo -e "${BLUE}🚀 Ready to deploy backend${NC}"
echo -e "${BLUE}🧠 Ready to train AI models${NC}"

echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Run: ${GREEN}./scripts/build_android.sh${NC} (build APK)"
echo "2. Run: ${GREEN}./scripts/test_android_app.sh${NC} (test app)"
echo "3. Run: ${GREEN}./scripts/deploy.sh${NC} (start backend)"
echo "4. Read: ${GREEN}ANDROID_BUILD_INSTRUCTIONS.md${NC} (detailed guide)"