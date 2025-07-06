#!/bin/bash

# ==================================================
# SPERM ANALYSIS ANDROID APK BUILD SCRIPT
# Developer: Youssef Shitiwi
# ==================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANDROID_DIR="$PROJECT_ROOT/android"
APK_OUTPUT_DIR="$PROJECT_ROOT/apk_builds"

echo -e "${BLUE}======================================"
echo -e "🧬 SPERM ANALYSIS ANDROID APK BUILDER"
echo -e "   Developer: Youssef Shitiwi"
echo -e "======================================${NC}"

# Function to print colored status
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Android project exists
if [ ! -d "$ANDROID_DIR" ]; then
    print_error "Android project directory not found at $ANDROID_DIR"
    exit 1
fi

# Create APK output directory
mkdir -p "$APK_OUTPUT_DIR"

# Change to Android directory
cd "$ANDROID_DIR"

# Function to check Android environment
check_android_env() {
    print_status "Checking Android development environment..."
    
    # Check for Java/JDK
    if ! command -v java &> /dev/null; then
        print_error "Java is not installed. Please install OpenJDK 11 or higher."
        echo "Ubuntu/Debian: sudo apt update && sudo apt install openjdk-11-jdk"
        echo "CentOS/RHEL: sudo yum install java-11-openjdk-devel"
        exit 1
    fi
    
    # Check Java version
    JAVA_VERSION=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}' | cut -d'.' -f1)
    if [ "$JAVA_VERSION" -lt 11 ]; then
        print_error "Java 11 or higher is required. Current version: $JAVA_VERSION"
        exit 1
    fi
    
    print_status "✓ Java $JAVA_VERSION detected"
    
    # Check for Android SDK (optional, Gradle wrapper will handle it)
    if [ -n "$ANDROID_HOME" ]; then
        print_status "✓ Android SDK found at: $ANDROID_HOME"
    else
        print_warning "ANDROID_HOME not set. Gradle will use embedded SDK."
    fi
    
    # Make gradlew executable
    if [ -f "./gradlew" ]; then
        chmod +x ./gradlew
        print_status "✓ Gradle wrapper is ready"
    else
        print_error "Gradle wrapper not found. This script must be run from the android directory."
        exit 1
    fi
}

# Function to clean project
clean_project() {
    print_status "Cleaning previous builds..."
    ./gradlew clean
    print_status "✓ Project cleaned"
}

# Function to build debug APK
build_debug_apk() {
    print_status "Building DEBUG APK..."
    ./gradlew assembleDebug
    
    # Find and copy debug APK
    DEBUG_APK=$(find . -name "*debug*.apk" -type f | head -n1)
    if [ -n "$DEBUG_APK" ]; then
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        cp "$DEBUG_APK" "$APK_OUTPUT_DIR/sperm-analysis-debug-$TIMESTAMP.apk"
        print_status "✓ Debug APK built: $APK_OUTPUT_DIR/sperm-analysis-debug-$TIMESTAMP.apk"
        echo -e "${BLUE}Debug APK size: $(du -h "$DEBUG_APK" | cut -f1)${NC}"
    else
        print_error "Debug APK not found after build"
        exit 1
    fi
}

# Function to build release APK (unsigned)
build_release_apk() {
    print_status "Building RELEASE APK (unsigned)..."
    ./gradlew assembleRelease
    
    # Find and copy release APK
    RELEASE_APK=$(find . -name "*release*.apk" -type f | head -n1)
    if [ -n "$RELEASE_APK" ]; then
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        cp "$RELEASE_APK" "$APK_OUTPUT_DIR/sperm-analysis-release-unsigned-$TIMESTAMP.apk"
        print_status "✓ Release APK built: $APK_OUTPUT_DIR/sperm-analysis-release-unsigned-$TIMESTAMP.apk"
        echo -e "${BLUE}Release APK size: $(du -h "$RELEASE_APK" | cut -f1)${NC}"
    else
        print_error "Release APK not found after build"
        exit 1
    fi
}

# Function to generate keystore and sign APK
sign_release_apk() {
    print_status "Setting up APK signing..."
    
    KEYSTORE_DIR="$PROJECT_ROOT/keystore"
    KEYSTORE_FILE="$KEYSTORE_DIR/sperm-analysis.keystore"
    
    mkdir -p "$KEYSTORE_DIR"
    
    if [ ! -f "$KEYSTORE_FILE" ]; then
        print_status "Generating new keystore..."
        
        # Generate keystore with default values for automation
        keytool -genkey -v -keystore "$KEYSTORE_FILE" \
            -alias sperm-analysis \
            -keyalg RSA \
            -keysize 2048 \
            -validity 10000 \
            -dname "CN=Youssef Shitiwi, OU=Development, O=Sperm Analysis, L=Unknown, S=Unknown, C=US" \
            -storepass "spermanalysis123" \
            -keypass "spermanalysis123"
        
        print_status "✓ Keystore generated at: $KEYSTORE_FILE"
        
        # Create signing config
        cat > "$KEYSTORE_DIR/signing.properties" << EOF
storeFile=$KEYSTORE_FILE
storePassword=spermanalysis123
keyAlias=sperm-analysis
keyPassword=spermanalysis123
EOF
        print_status "✓ Signing configuration saved"
    fi
    
    # Find unsigned APK
    UNSIGNED_APK=$(find "$APK_OUTPUT_DIR" -name "*release-unsigned*.apk" -type f | tail -n1)
    if [ -z "$UNSIGNED_APK" ]; then
        print_error "No unsigned release APK found. Build release APK first."
        return 1
    fi
    
    # Sign the APK
    SIGNED_APK="${UNSIGNED_APK/unsigned/signed}"
    print_status "Signing APK..."
    
    jarsigner -verbose -sigalg SHA1withRSA -digestalg SHA1 \
        -keystore "$KEYSTORE_FILE" \
        -storepass "spermanalysis123" \
        -keypass "spermanalysis123" \
        -signedjar "$SIGNED_APK" \
        "$UNSIGNED_APK" \
        sperm-analysis
    
    if [ $? -eq 0 ]; then
        print_status "✓ APK signed successfully: $SIGNED_APK"
        echo -e "${BLUE}Signed APK size: $(du -h "$SIGNED_APK" | cut -f1)${NC}"
        
        # Verify signature
        jarsigner -verify -verbose -certs "$SIGNED_APK"
        print_status "✓ APK signature verified"
    else
        print_error "Failed to sign APK"
        return 1
    fi
}

# Function to optimize APK with zipalign
optimize_apk() {
    print_status "Optimizing APK with zipalign..."
    
    # Find signed APK
    SIGNED_APK=$(find "$APK_OUTPUT_DIR" -name "*signed*.apk" -type f | tail -n1)
    if [ -z "$SIGNED_APK" ]; then
        print_error "No signed APK found. Sign APK first."
        return 1
    fi
    
    OPTIMIZED_APK="${SIGNED_APK/signed/optimized}"
    
    # Use zipalign if available
    if command -v zipalign &> /dev/null; then
        zipalign -v 4 "$SIGNED_APK" "$OPTIMIZED_APK"
        if [ $? -eq 0 ]; then
            print_status "✓ APK optimized: $OPTIMIZED_APK"
            echo -e "${BLUE}Optimized APK size: $(du -h "$OPTIMIZED_APK" | cut -f1)${NC}"
        else
            print_warning "zipalign failed, using signed APK"
            cp "$SIGNED_APK" "$OPTIMIZED_APK"
        fi
    else
        print_warning "zipalign not found, copying signed APK"
        cp "$SIGNED_APK" "$OPTIMIZED_APK"
    fi
}

# Function to generate build summary
generate_build_summary() {
    print_status "Generating build summary..."
    
    SUMMARY_FILE="$APK_OUTPUT_DIR/build_summary.md"
    
    cat > "$SUMMARY_FILE" << EOF
# Sperm Analysis Android Build Summary

**Developer:** Youssef Shitiwi  
**Build Date:** $(date)  
**Build Environment:** $(uname -a)  

## Build Results

### Generated APKs

EOF
    
    # List all APKs in output directory
    for apk in "$APK_OUTPUT_DIR"/*.apk; do
        if [ -f "$apk" ]; then
            APK_NAME=$(basename "$apk")
            APK_SIZE=$(du -h "$apk" | cut -f1)
            echo "- **$APK_NAME** ($APK_SIZE)" >> "$SUMMARY_FILE"
        fi
    done
    
    cat >> "$SUMMARY_FILE" << EOF

### Installation Instructions

#### Debug APK (for testing)
\`\`\`bash
adb install sperm-analysis-debug-*.apk
\`\`\`

#### Release APK (production)
\`\`\`bash
adb install sperm-analysis-*-optimized.apk
\`\`\`

### Features Included
- Video upload and analysis
- Real-time progress tracking
- CASA metrics visualization
- Results export (CSV/JSON)
- Modern Material Design UI
- Offline capability

### System Requirements
- Android 7.0 (API level 24) or higher
- Camera permission for video recording
- Storage permission for file access
- Network permission for API communication

### Backend Integration
The app connects to the FastAPI backend at:
- Default: http://10.0.2.2:8000 (Android emulator)
- Production: Configure in app settings

### Support
For issues or questions, contact: Youssef Shitiwi

---
Built with ❤️ for advanced sperm analysis research
EOF
    
    print_status "✓ Build summary generated: $SUMMARY_FILE"
}

# Function to setup development environment
setup_dev_env() {
    print_status "Setting up Android development environment..."
    
    # Install Android SDK dependencies
    if command -v apt-get &> /dev/null; then
        print_status "Installing dependencies (Ubuntu/Debian)..."
        sudo apt-get update
        sudo apt-get install -y openjdk-11-jdk wget unzip
    elif command -v yum &> /dev/null; then
        print_status "Installing dependencies (CentOS/RHEL)..."
        sudo yum install -y java-11-openjdk-devel wget unzip
    fi
    
    # Download Android command line tools if not available
    if [ ! -d "$HOME/android-sdk" ]; then
        print_status "Downloading Android SDK command line tools..."
        mkdir -p "$HOME/android-sdk/cmdline-tools"
        cd "$HOME/android-sdk/cmdline-tools"
        
        wget -q https://dl.google.com/android/repository/commandlinetools-linux-8512546_latest.zip
        unzip -q commandlinetools-linux-8512546_latest.zip
        mv cmdline-tools latest
        
        # Set environment variables
        export ANDROID_HOME="$HOME/android-sdk"
        export PATH="$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools"
        
        # Add to bashrc for persistence
        echo "export ANDROID_HOME=$HOME/android-sdk" >> ~/.bashrc
        echo "export PATH=\$PATH:\$ANDROID_HOME/cmdline-tools/latest/bin:\$ANDROID_HOME/platform-tools" >> ~/.bashrc
        
        print_status "✓ Android SDK installed at: $HOME/android-sdk"
        print_warning "Please restart your terminal or run: source ~/.bashrc"
    fi
}

# Main execution function
main() {
    echo -e "${BLUE}Select build option:${NC}"
    echo "1) Quick debug build"
    echo "2) Full release build (signed & optimized)"
    echo "3) Setup development environment"
    echo "4) Clean project"
    echo "5) Build all variants"
    echo "6) Exit"
    
    read -p "Enter your choice (1-6): " choice
    
    case $choice in
        1)
            check_android_env
            build_debug_apk
            generate_build_summary
            ;;
        2)
            check_android_env
            clean_project
            build_release_apk
            sign_release_apk
            optimize_apk
            generate_build_summary
            ;;
        3)
            setup_dev_env
            ;;
        4)
            check_android_env
            clean_project
            ;;
        5)
            check_android_env
            clean_project
            build_debug_apk
            build_release_apk
            sign_release_apk
            optimize_apk
            generate_build_summary
            ;;
        6)
            echo -e "${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            print_error "Invalid choice. Please select 1-6."
            main
            ;;
    esac
}

# Install requirements if needed
install_requirements() {
    print_status "Checking build requirements..."
    
    # Check for required commands
    REQUIRED_COMMANDS=("java" "keytool" "jarsigner")
    MISSING_COMMANDS=()
    
    for cmd in "${REQUIRED_COMMANDS[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            MISSING_COMMANDS+=("$cmd")
        fi
    done
    
    if [ ${#MISSING_COMMANDS[@]} -ne 0 ]; then
        print_error "Missing required commands: ${MISSING_COMMANDS[*]}"
        echo -e "${YELLOW}Would you like to install missing dependencies? (y/n)${NC}"
        read -p "Choice: " install_choice
        
        if [[ $install_choice =~ ^[Yy]$ ]]; then
            setup_dev_env
        else
            print_error "Cannot proceed without required dependencies."
            exit 1
        fi
    fi
}

# Enhanced error handling
trap 'echo -e "${RED}Build failed at line $LINENO${NC}"; exit 1' ERR

# Script entry point
echo -e "${GREEN}Checking system requirements...${NC}"
install_requirements

# Show available APKs if any exist
if [ -d "$APK_OUTPUT_DIR" ] && [ "$(ls -A $APK_OUTPUT_DIR/*.apk 2>/dev/null)" ]; then
    echo -e "${BLUE}Existing APKs found:${NC}"
    ls -la "$APK_OUTPUT_DIR"/*.apk
    echo ""
fi

# Run main function
main

echo -e "${GREEN}======================================"
echo -e "✅ ANDROID BUILD PROCESS COMPLETED!"
echo -e "   APKs available in: $APK_OUTPUT_DIR"
echo -e "   Developer: Youssef Shitiwi"
echo -e "======================================${NC}"