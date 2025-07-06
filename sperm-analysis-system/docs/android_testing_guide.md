# 📱 Android APK Build & Testing Guide

> **Developer:** Youssef Shitiwi  
> **Sperm Analysis Mobile Application**

This guide provides step-by-step instructions for building and testing the Android application, including troubleshooting common issues.

## 🛠️ Prerequisites

### System Requirements

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y openjdk-11-jdk wget unzip

# CentOS/RHEL
sudo yum install -y java-11-openjdk-devel wget unzip

# macOS
brew install openjdk@11
```

### Environment Setup

```bash
# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
echo 'export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64' >> ~/.bashrc

# Verify Java installation
java -version
javac -version
```

### Android SDK (Optional)

```bash
# Download Android Command Line Tools
mkdir -p ~/android-sdk/cmdline-tools
cd ~/android-sdk/cmdline-tools
wget https://dl.google.com/android/repository/commandlinetools-linux-8512546_latest.zip
unzip commandlinetools-linux-8512546_latest.zip
mv cmdline-tools latest

# Set Android environment
export ANDROID_HOME=~/android-sdk
export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools
echo 'export ANDROID_HOME=~/android-sdk' >> ~/.bashrc
echo 'export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools' >> ~/.bashrc
```

## 🏗️ Building the APK

### Method 1: Using Build Script

```bash
cd sperm-analysis-system/scripts
./build_android.sh

# Select option:
# 1) Quick debug build
# 2) Full release build (signed & optimized)
# 3) Setup development environment
```

### Method 2: Manual Build

```bash
cd sperm-analysis-system/android

# Clean previous builds
./gradlew clean

# Build debug APK
./gradlew assembleDebug

# Build release APK
./gradlew assembleRelease

# Build all variants
./gradlew assemble
```

### Build Variants

| Variant | Purpose | Signing | Optimization |
|---------|---------|---------|--------------|
| Debug | Development/Testing | Debug key | None |
| Release | Production | Release key | ProGuard/R8 |
| Staging | Pre-production | Debug key | Partial |

## 📦 APK Outputs

### Build Locations

```
android/app/build/outputs/apk/
├── debug/
│   └── app-debug.apk                    # Debug build
├── release/
│   └── app-release-unsigned.apk         # Unsigned release
└── signed/
    └── app-release-signed.apk           # Signed release
```

### APK Information

```bash
# Check APK details
aapt dump badging app-debug.apk

# APK size analysis
aapt list -v app-debug.apk | grep "\.dex"

# Installation verification
adb install app-debug.apk
```

## 🧪 Testing Framework

### Unit Testing

```bash
# Run unit tests
./gradlew test

# Run specific test class
./gradlew test --tests "*SpermAnalysisTest"

# Generate test report
./gradlew testDebugUnitTest
# Report: app/build/reports/tests/testDebugUnitTest/index.html
```

### Instrumentation Testing

```bash
# Run connected tests (requires device/emulator)
./gradlew connectedAndroidTest

# Run specific instrumentation test
./gradlew connectedAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.spermanalysis.ExampleInstrumentedTest
```

### Manual Testing Checklist

#### 📱 Core Functionality
- [ ] App launches successfully
- [ ] UI renders correctly on different screen sizes
- [ ] Video selection from gallery works
- [ ] Camera video recording works
- [ ] Video upload to API succeeds
- [ ] Real-time progress updates display
- [ ] Analysis results display correctly
- [ ] Results can be exported (CSV/JSON/PDF)

#### 🔗 API Integration
- [ ] Network connectivity handling
- [ ] API authentication (if enabled)
- [ ] Error handling and user feedback
- [ ] Retry mechanisms for failed requests
- [ ] Offline mode functionality

#### 📋 User Interface
- [ ] Navigation between screens
- [ ] Form validation and error messages
- [ ] Loading states and progress indicators
- [ ] Touch interactions and gestures
- [ ] Accessibility features

#### 🔒 Permissions & Security
- [ ] Camera permission request
- [ ] Storage permission request
- [ ] Network permission usage
- [ ] Data encryption in transit
- [ ] Secure storage of sensitive data

## 🎯 Testing Scenarios

### Video Analysis Workflow

```kotlin
// Test case example
@Test
fun testVideoAnalysisWorkflow() {
    // 1. Select video file
    val videoFile = createTestVideoFile()
    assertNotNull(videoFile)
    
    // 2. Upload video
    val uploadResult = apiService.uploadVideo(videoFile)
    assertTrue(uploadResult.isSuccessful)
    
    // 3. Start analysis
    val analysisId = uploadResult.body()?.analysisId
    val analysisResult = apiService.startAnalysis(analysisId)
    assertTrue(analysisResult.isSuccessful)
    
    // 4. Check status
    var status: AnalysisStatus
    do {
        Thread.sleep(1000)
        status = apiService.getAnalysisStatus(analysisId).body()!!
    } while (status.status == "processing")
    
    // 5. Get results
    assertEquals("completed", status.status)
    val results = apiService.getAnalysisResults(analysisId)
    assertNotNull(results.body()?.casaMetrics)
}
```

### Network Error Handling

```kotlin
@Test
fun testNetworkErrorHandling() {
    // Simulate network error
    mockWebServer.enqueue(MockResponse().setResponseCode(500))
    
    val result = apiService.uploadVideo(testVideoFile)
    assertFalse(result.isSuccessful)
    
    // Verify error handling in UI
    onView(withId(R.id.error_message))
        .check(matches(isDisplayed()))
        .check(matches(withText(containsString("Network error"))))
}
```

### Performance Testing

```kotlin
@Test
fun testLargeVideoUpload() {
    // Create large test video (100MB)
    val largeVideoFile = createLargeTestVideo(100 * 1024 * 1024)
    
    val startTime = System.currentTimeMillis()
    val result = apiService.uploadVideo(largeVideoFile)
    val uploadTime = System.currentTimeMillis() - startTime
    
    assertTrue(result.isSuccessful)
    assertTrue("Upload should complete within 5 minutes", uploadTime < 300000)
}
```

## 🔍 Device Testing

### Emulator Setup

```bash
# Create Android Virtual Device
avdmanager create avd -n "SpermAnalysis_Test" -k "system-images;android-30;google_apis;x86_64"

# Start emulator
emulator -avd SpermAnalysis_Test

# Install APK on emulator
adb install app-debug.apk
```

### Physical Device Testing

```bash
# Enable USB debugging on device
# Connect device via USB

# Verify device connection
adb devices

# Install APK
adb install app-debug.apk

# View logs
adb logcat | grep "SpermAnalysis"

# Capture screenshots
adb shell screencap -p /sdcard/screenshot.png
adb pull /sdcard/screenshot.png
```

### Device Compatibility Matrix

| Device Type | Screen Size | Android Version | Memory | Status |
|-------------|-------------|-----------------|--------|--------|
| Phone | 5.0" - 6.9" | 7.0+ (API 24+) | 3GB+ | ✅ Supported |
| Tablet | 7.0" - 13" | 7.0+ (API 24+) | 4GB+ | ✅ Supported |
| Foldable | Variable | 10.0+ (API 29+) | 6GB+ | ⚠️ Limited |

## 📊 Performance Monitoring

### Memory Usage

```bash
# Monitor memory during testing
adb shell dumpsys meminfo com.spermanalysis

# Heap dump for analysis
adb shell am dumpheap com.spermanalysis /data/local/tmp/heap.hprof
adb pull /data/local/tmp/heap.hprof
```

### CPU Profiling

```bash
# CPU profiling during video analysis
adb shell top -p $(adb shell pidof com.spermanalysis)

# Method tracing
adb shell am profile start com.spermanalysis /data/local/tmp/profile.trace
# ... perform operations ...
adb shell am profile stop com.spermanalysis
```

### Network Monitoring

```bash
# Monitor network traffic
adb shell dumpsys netstats detail

# Packet capture
adb shell tcpdump -i any -w /data/local/tmp/capture.pcap
```

## 🐛 Debugging & Troubleshooting

### Common Build Issues

#### Java/Gradle Issues
```bash
# Java version mismatch
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
./gradlew --stop
./gradlew clean

# Gradle daemon issues
./gradlew --stop
rm -rf ~/.gradle/daemon/
./gradlew clean build

# Dependency conflicts
./gradlew dependencies
./gradlew dependencyInsight --dependency retrofit
```

#### Android SDK Issues
```bash
# Missing SDK components
sdkmanager --install "platforms;android-34"
sdkmanager --install "build-tools;34.0.0"

# License acceptance
sdkmanager --licenses
```

### Runtime Debugging

#### LogCat Filtering
```bash
# Filter by package
adb logcat | grep "com.spermanalysis"

# Filter by tag
adb logcat -s "SpermAnalysis"

# Filter by priority
adb logcat "*:E"  # Errors only
```

#### Crash Analysis
```bash
# Get crash logs
adb logcat -b crash

# ANR (Application Not Responding) logs
adb shell dumpsys activity processes | grep -A 10 "ANR"
```

### Performance Issues

#### Memory Leaks
```kotlin
// Use LeakCanary (already included in debug builds)
// Check for common leaks:
// 1. Activity context in static variables
// 2. Listener not unregistered
// 3. AsyncTask holding activity reference
```

#### UI Performance
```bash
# GPU overdraw debugging
adb shell setprop debug.hwui.overdraw show

# Layout inspection
adb shell setprop debug.layout true
```

## 📈 Automated Testing

### CI/CD Pipeline

```yaml
# .github/workflows/android.yml
name: Android CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up JDK 11
      uses: actions/setup-java@v3
      with:
        java-version: '11'
        distribution: 'temurin'
        
    - name: Cache Gradle dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.gradle/caches
          ~/.gradle/wrapper
        key: ${{ runner.os }}-gradle-${{ hashFiles('**/*.gradle*', '**/gradle-wrapper.properties') }}
        
    - name: Run tests
      working-directory: ./android
      run: ./gradlew test
      
    - name: Build APK
      working-directory: ./android
      run: ./gradlew assembleDebug
      
    - name: Upload APK
      uses: actions/upload-artifact@v3
      with:
        name: debug-apk
        path: android/app/build/outputs/apk/debug/app-debug.apk
```

### Test Automation Scripts

```bash
#!/bin/bash
# automated_testing.sh

echo "🧬 Starting Sperm Analysis App Testing"
echo "Developer: Youssef Shitiwi"

# Build APK
cd android
./gradlew clean assembleDebug

# Install on connected device
adb install -r app/build/outputs/apk/debug/app-debug.apk

# Run automated UI tests
./gradlew connectedAndroidTest

# Generate test reports
./gradlew jacocoTestReport

echo "✅ Testing completed!"
echo "📊 Test reports available in: app/build/reports/"
```

## 📋 Quality Assurance

### Code Quality Checks

```bash
# Lint analysis
./gradlew lint
# Report: app/build/reports/lint-results.html

# Code coverage
./gradlew jacocoTestReport
# Report: app/build/reports/jacoco/test/html/index.html

# Static analysis with SpotBugs
./gradlew spotbugsMain
```

### Security Testing

```bash
# APK security analysis
analyze_apk app-release.apk

# Network security testing
# Test with Charles Proxy or similar tools
# Verify HTTPS implementation
# Check certificate pinning
```

### Accessibility Testing

```bash
# Accessibility scanner
# Install from Google Play Store
# Run on physical device
# Check for:
# - Content descriptions
# - Touch target sizes
# - Color contrast
# - Navigation order
```

## 📤 APK Distribution

### Release Preparation

```bash
# Generate signed release APK
./gradlew assembleRelease

# Sign APK manually
jarsigner -verbose -sigalg SHA1withRSA -digestalg SHA1 \
  -keystore sperm-analysis.keystore \
  app-release-unsigned.apk \
  sperm-analysis

# Optimize APK
zipalign -v 4 app-release-unsigned.apk app-release.apk
```

### Distribution Channels

1. **Google Play Store**
   - Upload to Play Console
   - Internal/Alpha/Beta testing tracks
   - Production release

2. **Enterprise Distribution**
   - Direct APK distribution
   - MDM (Mobile Device Management)
   - Corporate app stores

3. **Side-loading**
   - Direct APK installation
   - QR code distribution
   - Web download

## 🔧 Development Tools

### Recommended Tools

1. **Android Studio** - Primary IDE
2. **ADB (Android Debug Bridge)** - Device communication
3. **Gradle** - Build system
4. **LeakCanary** - Memory leak detection
5. **Stetho** - Network debugging
6. **Firebase Crashlytics** - Crash reporting

### Useful Commands

```bash
# Package info
adb shell dumpsys package com.spermanalysis

# Clear app data
adb shell pm clear com.spermanalysis

# Force stop app
adb shell am force-stop com.spermanalysis

# Start activity
adb shell am start -n com.spermanalysis/.MainActivity

# Take screenshot
adb shell screencap -p > screenshot.png

# Record screen
adb shell screenrecord /data/local/tmp/demo.mp4
```

## 📝 Testing Documentation

### Test Reports

Generate comprehensive test reports including:
- Unit test results
- Integration test results
- Code coverage metrics
- Performance benchmarks
- Security scan results

### Bug Reporting Template

```markdown
## Bug Report

**App Version:** 1.0.0
**Android Version:** 11 (API 30)
**Device:** Samsung Galaxy S21

### Description
Brief description of the issue

### Steps to Reproduce
1. Open app
2. Select video
3. Tap analyze

### Expected Behavior
Analysis should start

### Actual Behavior
App crashes

### Screenshots/Logs
[Attach screenshots and logcat output]

### Additional Context
Any additional information
```

---

This comprehensive testing guide ensures the **Sperm Analysis** mobile application by **Youssef Shitiwi** meets the highest quality standards for research and clinical use.