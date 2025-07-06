package com.spermanalysis.data.remote

import android.content.Context
import com.google.gson.GsonBuilder
import com.spermanalysis.utils.DateDeserializer
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.*
import java.util.concurrent.TimeUnit

/**
 * API Client Configuration
 * Author: Youssef Shitiwi (يوسف شتيوي)
 */
class ApiClient(
    private val baseUrl: String,
    private val context: Context
) {
    
    private val gson = GsonBuilder()
        .setDateFormat("yyyy-MM-dd'T'HH:mm:ss")
        .registerTypeAdapter(Date::class.java, DateDeserializer())
        .create()
    
    private val okHttpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(120, TimeUnit.SECONDS) // For large video uploads
        .addInterceptor(createLoggingInterceptor())
        .addInterceptor(createAuthInterceptor())
        .build()
    
    private val retrofit = Retrofit.Builder()
        .baseUrl(baseUrl)
        .client(okHttpClient)
        .addConverterFactory(GsonConverterFactory.create(gson))
        .build()
    
    val apiService: ApiService = retrofit.create(ApiService::class.java)
    
    private fun createLoggingInterceptor(): HttpLoggingInterceptor {
        return HttpLoggingInterceptor().apply {
            level = if (com.spermanalysis.BuildConfig.DEBUG) {
                HttpLoggingInterceptor.Level.BODY
            } else {
                HttpLoggingInterceptor.Level.NONE
            }
        }
    }
    
    private fun createAuthInterceptor(): okhttp3.Interceptor {
        return okhttp3.Interceptor { chain ->
            val originalRequest = chain.request()
            
            // Add API key or authentication headers if needed
            val newRequest = originalRequest.newBuilder()
                .addHeader("Content-Type", "application/json")
                .addHeader("User-Agent", "SpermAnalysis-Android/1.0")
                .build()
            
            chain.proceed(newRequest)
        }
    }
}