package com.spermanalysis

import android.app.Application
import android.content.Context
import androidx.work.Configuration
import androidx.work.WorkManager
import com.spermanalysis.data.local.AppDatabase
import com.spermanalysis.data.remote.ApiClient
import com.spermanalysis.data.repository.AnalysisRepository
import com.spermanalysis.utils.NotificationHelper
import com.spermanalysis.utils.PreferenceManager

/**
 * Sperm Analysis Application Class
 * Author: Youssef Shitiwi (يوسف شتيوي)
 * 
 * Main application class for initialization and dependency management
 */
class SpermAnalysisApplication : Application(), Configuration.Provider {

    companion object {
        @Volatile
        private var INSTANCE: SpermAnalysisApplication? = null
        
        fun getInstance(): SpermAnalysisApplication {
            return INSTANCE ?: throw IllegalStateException("Application not initialized")
        }
    }

    // Core components
    lateinit var database: AppDatabase
        private set
    
    lateinit var apiClient: ApiClient
        private set
    
    lateinit var analysisRepository: AnalysisRepository
        private set
    
    lateinit var preferenceManager: PreferenceManager
        private set
    
    lateinit var notificationHelper: NotificationHelper
        private set

    override fun onCreate() {
        super.onCreate()
        INSTANCE = this
        
        initializeComponents()
        initializeWorkManager()
        
        // Setup crash reporting in production
        setupCrashReporting()
    }

    private fun initializeComponents() {
        // Initialize preferences
        preferenceManager = PreferenceManager(this)
        
        // Initialize notification helper
        notificationHelper = NotificationHelper(this)
        
        // Initialize database
        database = AppDatabase.getDatabase(this)
        
        // Initialize API client
        apiClient = ApiClient(
            baseUrl = getApiBaseUrl(),
            context = this
        )
        
        // Initialize repository
        analysisRepository = AnalysisRepository(
            apiService = apiClient.apiService,
            analysisDao = database.analysisDao(),
            spermDataDao = database.spermDataDao()
        )
    }

    private fun initializeWorkManager() {
        val workManagerConfig = Configuration.Builder()
            .setMinimumLoggingLevel(android.util.Log.INFO)
            .build()
        
        WorkManager.initialize(this, workManagerConfig)
    }

    private fun setupCrashReporting() {
        // In production, you would initialize crash reporting here
        // e.g., Crashlytics, Bugsnag, etc.
    }

    private fun getApiBaseUrl(): String {
        return if (BuildConfig.DEBUG) {
            "http://10.0.2.2:8000/api/v1/"  // Local development
        } else {
            "https://your-api-domain.com/api/v1/"  // Production
        }
    }

    override fun getWorkManagerConfiguration(): Configuration {
        return Configuration.Builder()
            .setMinimumLoggingLevel(android.util.Log.INFO)
            .build()
    }

    fun getAppContext(): Context = applicationContext
}