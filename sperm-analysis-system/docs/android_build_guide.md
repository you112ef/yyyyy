# 📱 Android APK Building - Complete Guide
# Developer: Youssef Shitiwi (يوسف شتيوي)

## 1. PREREQUISITES & SETUP

### Install Required Tools
```bash
# Install Android Studio and SDK
# Download from: https://developer.android.com/studio

# OR install command line tools only
wget https://dl.google.com/android/repository/commandlinetools-linux-8512546_latest.zip
unzip commandlinetools-linux-8512546_latest.zip
export ANDROID_HOME=$HOME/Android/Sdk
export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin
export PATH=$PATH:$ANDROID_HOME/platform-tools

# Install required SDK components
sdkmanager "platforms;android-34"
sdkmanager "build-tools;34.0.0"
sdkmanager "platform-tools"
```

### Verify Setup
```bash
# Check Android environment
cd sperm-analysis-system/android
./gradlew --version

# Check available tasks
./gradlew tasks
```

## 2. QUICK APK BUILD

### Debug APK (For Testing)
```bash
cd sperm-analysis-system/android

# Build debug APK
./gradlew assembleDebug

# APK location:
# app/build/outputs/apk/debug/app-debug.apk

echo "✅ Debug APK built: app/build/outputs/apk/debug/app-debug.apk"
```

### Install Debug APK
```bash
# Install on connected device/emulator
adb install app/build/outputs/apk/debug/app-debug.apk

# Or install via USB
# Enable USB debugging on your phone first
adb devices  # Check device is connected
adb install app/build/outputs/apk/debug/app-debug.apk
```

## 3. RELEASE APK PRODUCTION BUILD

### Step 1: Create Keystore (First Time Only)
```bash
# Generate release keystore
keytool -genkey -v -keystore sperm-analysis-keystore.jks \
    -keyalg RSA -keysize 2048 -validity 10000 \
    -alias sperm-analysis-key \
    -dname "CN=Youssef Shitiwi, OU=Development, O=Sperm Analysis, L=City, S=State, C=US"

# Enter a secure password when prompted
# Store this keystore safely - needed for app updates!
```

### Step 2: Configure Signing
```gradle
// Add to android/app/build.gradle
android {
    signingConfigs {
        release {
            storeFile file('../sperm-analysis-keystore.jks')
            storePassword 'your_keystore_password'
            keyAlias 'sperm-analysis-key'
            keyPassword 'your_key_password'
        }
    }
    
    buildTypes {
        release {
            signingConfig signingConfigs.release
            minifyEnabled true
            shrinkResources true
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }
}
```

### Step 3: Build Release APK
```bash
# Build signed release APK
./gradlew assembleRelease

# APK location:
# app/build/outputs/apk/release/app-release.apk

echo "✅ Release APK built: app/build/outputs/apk/release/app-release.apk"
```

## 4. ADVANCED BUILD CONFIGURATIONS

### Multiple Build Variants
```gradle
// android/app/build.gradle
android {
    flavorDimensions "version"
    
    productFlavors {
        free {
            dimension "version"
            applicationIdSuffix ".free"
            versionNameSuffix "-free"
            
            buildConfigField "String", "API_BASE_URL", '"https://api-free.spermanalysis.com/api/v1/"'
            buildConfigField "boolean", "IS_PREMIUM", "false"
        }
        
        premium {
            dimension "version"
            applicationIdSuffix ".premium"
            versionNameSuffix "-premium"
            
            buildConfigField "String", "API_BASE_URL", '"https://api-premium.spermanalysis.com/api/v1/"'
            buildConfigField "boolean", "IS_PREMIUM", "true"
        }
        
        research {
            dimension "version"
            applicationIdSuffix ".research"
            versionNameSuffix "-research"
            
            buildConfigField "String", "API_BASE_URL", '"https://api-research.spermanalysis.com/api/v1/"'
            buildConfigField "boolean", "IS_PREMIUM", "true"
            buildConfigField "boolean", "ENABLE_ADVANCED_METRICS", "true"
        }
    }
}
```

### Build All Variants
```bash
# Build all APK variants
./gradlew assembleRelease

# This creates:
# app-free-release.apk
# app-premium-release.apk
# app-research-release.apk
```

### Environment-Specific Builds
```bash
# Development build
./gradlew assembleDevelopmentDebug

# Staging build
./gradlew assembleStagingRelease

# Production build
./gradlew assembleProductionRelease
```

## 5. APP BUNDLE (RECOMMENDED FOR PLAY STORE)

### Build App Bundle
```bash
# Build Android App Bundle (preferred for Play Store)
./gradlew bundleRelease

# Bundle location:
# app/build/outputs/bundle/release/app-release.aab

echo "✅ App Bundle built: app/build/outputs/bundle/release/app-release.aab"
```

### App Bundle Benefits
- Smaller download sizes (Google Play's Dynamic Delivery)
- Automatic APK generation for different device configurations
- Better optimization for user devices

## 6. OPTIMIZATION & PERFORMANCE

### ProGuard Configuration
```proguard
# android/app/proguard-rules.pro

# Keep model classes
-keep class com.spermanalysis.data.model.** { *; }

# Keep API service interfaces
-keep interface com.spermanalysis.data.remote.** { *; }

# Retrofit
-keepattributes Signature
-keepattributes *Annotation*
-keep class retrofit2.** { *; }

# Gson
-keep class com.google.gson.** { *; }
-keep class * implements com.google.gson.TypeAdapterFactory
-keep class * implements com.google.gson.JsonSerializer
-keep class * implements com.google.gson.JsonDeserializer

# Room database
-keep class * extends androidx.room.RoomDatabase
-keep @androidx.room.Entity class *
-dontwarn androidx.room.paging.**

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Developer attribution - keep this visible
-keep class com.spermanalysis.BuildConfig { *; }
```

### Build Optimization
```gradle
// android/app/build.gradle
android {
    buildTypes {
        release {
            // Enable code shrinking
            minifyEnabled true
            shrinkResources true
            
            // Enable build cache
            buildConfigField "boolean", "ENABLE_LOGGING", "false"
            
            // Optimize for size
            android.packagingOptions {
                exclude 'META-INF/DEPENDENCIES'
                exclude 'META-INF/LICENSE'
                exclude 'META-INF/LICENSE.txt'
                exclude 'META-INF/NOTICE'
                exclude 'META-INF/NOTICE.txt'
            }
        }
    }
    
    // Enable build cache
    buildCache {
        local {
            enabled = true
        }
    }
}
```

## 7. AUTOMATED BUILD SCRIPTS

### Build Script
```bash
#!/bin/bash
# build-apk.sh - Automated APK building
# Developer: Youssef Shitiwi

set -e

echo "🏗️ Building Sperm Analysis APK..."
echo "Developer: Youssef Shitiwi (يوسف شتيوي)"

# Configuration
BUILD_TYPE=${1:-debug}  # debug or release
FLAVOR=${2:-}           # optional flavor

cd android

# Clean previous builds
echo "🧹 Cleaning previous builds..."
./gradlew clean

# Build APK
if [ "$BUILD_TYPE" = "release" ]; then
    if [ -z "$FLAVOR" ]; then
        echo "🚀 Building release APK..."
        ./gradlew assembleRelease
        APK_PATH="app/build/outputs/apk/release/app-release.apk"
    else
        echo "🚀 Building $FLAVOR release APK..."
        ./gradlew assemble${FLAVOR^}Release
        APK_PATH="app/build/outputs/apk/$FLAVOR/release/app-$FLAVOR-release.apk"
    fi
else
    echo "🛠️ Building debug APK..."
    ./gradlew assembleDebug
    APK_PATH="app/build/outputs/apk/debug/app-debug.apk"
fi

# Verify APK exists
if [ -f "$APK_PATH" ]; then
    APK_SIZE=$(du -h "$APK_PATH" | cut -f1)
    echo "✅ APK built successfully!"
    echo "📍 Location: $APK_PATH"
    echo "📏 Size: $APK_SIZE"
    
    # Get APK info
    echo "📋 APK Information:"
    aapt dump badging "$APK_PATH" | grep -E "(package|application-label|versionName|versionCode)"
else
    echo "❌ APK build failed!"
    exit 1
fi

# Optional: Install if device connected
if adb devices | grep -q "device$"; then
    read -p "📱 Device detected. Install APK? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📲 Installing APK..."
        adb install -r "$APK_PATH"
        echo "✅ APK installed successfully!"
    fi
fi

echo "🎉 Build process completed!"
```

### Continuous Integration Script
```yaml
# .github/workflows/build-apk.yml
name: Build Android APK
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up JDK 17
      uses: actions/setup-java@v3
      with:
        java-version: '17'
        distribution: 'temurin'
    
    - name: Setup Android SDK
      uses: android-actions/setup-android@v2
    
    - name: Cache Gradle packages
      uses: actions/cache@v3
      with:
        path: |
          ~/.gradle/caches
          ~/.gradle/wrapper
        key: ${{ runner.os }}-gradle-${{ hashFiles('**/*.gradle*', '**/gradle-wrapper.properties') }}
        restore-keys: |
          ${{ runner.os }}-gradle-
    
    - name: Make gradlew executable
      run: chmod +x android/gradlew
    
    - name: Build Debug APK
      run: |
        cd android
        ./gradlew assembleDebug
    
    - name: Upload APK artifact
      uses: actions/upload-artifact@v3
      with:
        name: sperm-analysis-debug-apk
        path: android/app/build/outputs/apk/debug/app-debug.apk
```

## 8. APK ANALYSIS & VALIDATION

### APK Analysis
```bash
# Analyze APK size and content
./gradlew analyzeDebugBundle

# Or use bundletool
bundletool validate --bundle=app/build/outputs/bundle/release/app-release.aab

# APK analyzer (if Android Studio installed)
# $ANDROID_HOME/tools/bin/apkanalyzer
```

### APK Testing
```bash
# Install and test APK
#!/bin/bash
# test-apk.sh

APK_PATH=${1:-app/build/outputs/apk/debug/app-debug.apk}

echo "🧪 Testing APK: $APK_PATH"

# Install APK
adb install -r "$APK_PATH"

# Launch app
adb shell am start -n com.spermanalysis.debug/com.spermanalysis.ui.MainActivity

# Check app is running
sleep 5
if adb shell pidof com.spermanalysis.debug > /dev/null; then
    echo "✅ App launched successfully!"
else
    echo "❌ App failed to launch!"
    exit 1
fi

# Run basic UI tests
echo "🔄 Running basic tests..."
adb shell input tap 500 1000  # Tap somewhere on screen
sleep 2

# Check for crashes
if adb shell pidof com.spermanalysis.debug > /dev/null; then
    echo "✅ App stable after interaction!"
else
    echo "⚠️ App may have crashed!"
fi

echo "🎉 APK testing completed!"
```

## 9. DISTRIBUTION PREPARATION

### Play Store Preparation
```bash
# 1. Build App Bundle (recommended)
./gradlew bundleRelease

# 2. Generate upload key (if first time)
keytool -genkey -v -keystore upload-keystore.jks \
    -keyalg RSA -keysize 2048 -validity 25000 \
    -alias upload-key

# 3. Create store listing assets
mkdir -p store-assets/{screenshots,graphics}

# Screenshots needed:
# - Phone: 16:9 or 9:16 ratio, 2-8 screenshots
# - Tablet: 16:10 or 10:16 ratio, 1-8 screenshots
# - Feature graphic: 1024 x 500 PNG or JPG
# - App icon: 512 x 512 PNG
```

### Direct Distribution
```bash
# Create distribution package
#!/bin/bash
# package-for-distribution.sh

VERSION=$(grep versionName android/app/build.gradle | cut -d'"' -f2)
PACKAGE_DIR="sperm-analysis-v$VERSION"

echo "📦 Creating distribution package v$VERSION..."

mkdir -p "$PACKAGE_DIR"

# Copy APK
cp android/app/build/outputs/apk/release/app-release.apk \
   "$PACKAGE_DIR/sperm-analysis-v$VERSION.apk"

# Copy documentation
cp README.md "$PACKAGE_DIR/"
cp docs/android.md "$PACKAGE_DIR/installation-guide.md"

# Create installation instructions
cat > "$PACKAGE_DIR/INSTALL.txt" << EOF
Sperm Analysis App Installation
Developer: Youssef Shitiwi (يوسف شتيوي)

Installation Steps:
1. Enable "Unknown Sources" in Android Settings > Security
2. Copy APK to your device
3. Tap the APK file to install
4. Grant necessary permissions when prompted

System Requirements:
- Android 7.0+ (API level 24+)
- 100MB free storage
- Camera permission for video recording
- Internet connection for analysis

Support:
- Email: support@spermanalysis.com
- Developer: Youssef Shitiwi
EOF

# Create checksums
cd "$PACKAGE_DIR"
sha256sum * > checksums.txt
cd ..

# Create archive
tar -czf "$PACKAGE_DIR.tar.gz" "$PACKAGE_DIR"

echo "✅ Distribution package created: $PACKAGE_DIR.tar.gz"
```

## 10. TROUBLESHOOTING

### Common Build Issues

#### Memory Issues
```bash
# Increase Gradle memory
echo "org.gradle.jvmargs=-Xmx4096m -XX:MaxPermSize=512m" >> android/gradle.properties
```

#### SDK Issues
```bash
# Update SDK
sdkmanager --update

# Install missing components
sdkmanager "platforms;android-34" "build-tools;34.0.0"
```

#### Dependency Issues
```bash
# Clean and rebuild
cd android
./gradlew clean
./gradlew build --refresh-dependencies
```

#### Signing Issues
```bash
# Verify keystore
keytool -list -v -keystore sperm-analysis-keystore.jks

# Create new keystore if corrupted
keytool -genkey -v -keystore new-keystore.jks -keyalg RSA -keysize 2048 -validity 10000 -alias new-key
```

### Debug APK Installation
```bash
# Check device connection
adb devices

# Enable USB debugging
# Settings > Developer Options > USB Debugging

# Install with debugging
adb install -r -d app/build/outputs/apk/debug/app-debug.apk

# View app logs
adb logcat | grep "SpermAnalysis"
```

### Performance Optimization
```bash
# Profile APK size
./gradlew assembleDebug
bundletool build-apks --bundle=app/build/outputs/bundle/debug/app-debug.aab --output=debug.apks
bundletool get-size total --apks=debug.apks

# Analyze build
./gradlew assembleDebug --scan
```

## 11. QUICK REFERENCE COMMANDS

```bash
# Essential build commands

# Debug build (fastest)
./gradlew assembleDebug

# Release build (optimized)
./gradlew assembleRelease

# App bundle (Play Store)
./gradlew bundleRelease

# Clean build
./gradlew clean assembleRelease

# Install to device
adb install -r app/build/outputs/apk/debug/app-debug.apk

# Uninstall from device
adb uninstall com.spermanalysis

# Build specific flavor
./gradlew assemblePremiumRelease

# Run tests
./gradlew test

# Lint check
./gradlew lint

# Build with verbose output
./gradlew assembleDebug --info

# Build offline (no network)
./gradlew assembleDebug --offline
```

## 📱 APK OUTPUT INFORMATION

**Debug APK:**
- Location: `android/app/build/outputs/apk/debug/app-debug.apk`
- Size: ~25-35 MB
- Signed with: Debug keystore (auto-generated)
- Permissions: All development permissions enabled

**Release APK:**
- Location: `android/app/build/outputs/apk/release/app-release.apk`
- Size: ~15-25 MB (optimized)
- Signed with: Your release keystore
- Permissions: Production permissions only

**App Bundle:**
- Location: `android/app/build/outputs/bundle/release/app-release.aab`
- Size: ~20-30 MB
- Format: Android App Bundle (for Play Store)
- Benefits: Dynamic delivery, smaller downloads

## 🎯 SUCCESS CHECKLIST

- [ ] APK builds without errors
- [ ] APK installs on test device
- [ ] App launches successfully
- [ ] All core features work
- [ ] No memory leaks or crashes
- [ ] Proper developer attribution visible
- [ ] Release APK properly signed
- [ ] App bundle ready for Play Store

**Developer: Youssef Shitiwi (يوسف شتيوي)**  
**App Package:** com.spermanalysis  
**Minimum Android:** 7.0 (API 24)  
**Target Android:** 14 (API 34)