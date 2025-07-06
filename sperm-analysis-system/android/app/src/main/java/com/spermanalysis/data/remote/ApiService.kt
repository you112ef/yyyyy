package com.spermanalysis.data.remote

import com.spermanalysis.data.model.*
import okhttp3.MultipartBody
import okhttp3.RequestBody
import okhttp3.ResponseBody
import retrofit2.Response
import retrofit2.http.*

/**
 * API Service Interface
 * Author: Youssef Shitiwi (يوسف شتيوي)
 * 
 * Retrofit interface for Sperm Analysis API
 */
interface ApiService {

    @Multipart
    @POST("upload")
    suspend fun uploadVideo(
        @Part file: MultipartBody.Part
    ): Response<FileUploadResponse>

    @Multipart
    @POST("analyze")
    suspend fun startAnalysis(
        @Part("upload_id") uploadId: RequestBody,
        @Part("analysis_name") analysisName: RequestBody?,
        @Part("config") config: RequestBody
    ): Response<AnalysisResponse>

    @Multipart
    @POST("analyze-direct")
    suspend fun analyzeDirectly(
        @Part file: MultipartBody.Part,
        @Part("analysis_name") analysisName: RequestBody?,
        @Part("fps") fps: RequestBody,
        @Part("pixel_to_micron") pixelToMicron: RequestBody,
        @Part("confidence_threshold") confidenceThreshold: RequestBody,
        @Part("iou_threshold") iouThreshold: RequestBody,
        @Part("min_track_length") minTrackLength: RequestBody,
        @Part("enable_visualization") enableVisualization: RequestBody,
        @Part("export_trajectories") exportTrajectories: RequestBody
    ): Response<AnalysisResponse>

    @GET("status/{analysis_id}")
    suspend fun getAnalysisStatus(
        @Path("analysis_id") analysisId: String
    ): Response<StatusResponse>

    @GET("results/{analysis_id}")
    suspend fun getAnalysisResults(
        @Path("analysis_id") analysisId: String
    ): Response<ResultsResponse>

    @GET("download/{analysis_id}/{format_type}")
    @Streaming
    suspend fun downloadResults(
        @Path("analysis_id") analysisId: String,
        @Path("format_type") formatType: String
    ): Response<ResponseBody>

    @GET("download/{analysis_id}")
    suspend fun getAvailableDownloads(
        @Path("analysis_id") analysisId: String
    ): Response<AvailableDownloadsResponse>

    @DELETE("analysis/{analysis_id}")
    suspend fun cancelAnalysis(
        @Path("analysis_id") analysisId: String
    ): Response<BasicResponse>

    @DELETE("results/{analysis_id}")
    suspend fun cleanupAnalysis(
        @Path("analysis_id") analysisId: String,
        @Query("keep_results") keepResults: Boolean = true
    ): Response<BasicResponse>

    @DELETE("upload/{upload_id}")
    suspend fun cleanupUpload(
        @Path("upload_id") uploadId: String
    ): Response<BasicResponse>

    @GET("queue")
    suspend fun getQueueStatus(): Response<QueueStatusResponse>

    @GET("health")
    suspend fun getHealth(): Response<HealthResponse>

    @GET("health/simple")
    suspend fun getSimpleHealth(): Response<SimpleHealthResponse>
}

// Additional response models
data class AvailableDownloadsResponse(
    @com.google.gson.annotations.SerializedName("analysis_id")
    val analysisId: String,
    
    val status: AnalysisStatus,
    
    @com.google.gson.annotations.SerializedName("available_downloads")
    val availableDownloads: List<DownloadInfo>,
    
    @com.google.gson.annotations.SerializedName("total_formats")
    val totalFormats: Int
)

data class DownloadInfo(
    val format: String,
    val url: String,
    val description: String,
    
    @com.google.gson.annotations.SerializedName("content_type")
    val contentType: String
)

data class BasicResponse(
    val message: String,
    
    @com.google.gson.annotations.SerializedName("analysis_id")
    val analysisId: String? = null,
    
    @com.google.gson.annotations.SerializedName("upload_id")
    val uploadId: String? = null
)

data class QueueStatusResponse(
    @com.google.gson.annotations.SerializedName("queue_status")
    val queueStatus: QueueStatus,
    
    @com.google.gson.annotations.SerializedName("total_uploads")
    val totalUploads: Int,
    
    @com.google.gson.annotations.SerializedName("disk_usage")
    val diskUsage: DiskUsage,
    
    val timestamp: String
)

data class QueueStatus(
    val pending: Int,
    val processing: Int,
    val completed: Int,
    val failed: Int,
    val total: Int
)

data class DiskUsage(
    @com.google.gson.annotations.SerializedName("uploads_mb")
    val uploadsMb: Float,
    
    @com.google.gson.annotations.SerializedName("results_mb")
    val resultsMb: Float,
    
    @com.google.gson.annotations.SerializedName("total_mb")
    val totalMb: Float
)

data class HealthResponse(
    val status: String,
    val timestamp: String,
    val version: String,
    val uptime: Float,
    
    @com.google.gson.annotations.SerializedName("database_connected")
    val databaseConnected: Boolean,
    
    @com.google.gson.annotations.SerializedName("model_loaded")
    val modelLoaded: Boolean,
    
    @com.google.gson.annotations.SerializedName("gpu_available")
    val gpuAvailable: Boolean,
    
    @com.google.gson.annotations.SerializedName("cpu_usage")
    val cpuUsage: Float,
    
    @com.google.gson.annotations.SerializedName("memory_usage")
    val memoryUsage: Float,
    
    @com.google.gson.annotations.SerializedName("disk_usage")
    val diskUsage: Float,
    
    @com.google.gson.annotations.SerializedName("pending_analyses")
    val pendingAnalyses: Int,
    
    @com.google.gson.annotations.SerializedName("processing_analyses")
    val processingAnalyses: Int
)

data class SimpleHealthResponse(
    val status: String,
    val timestamp: String,
    val service: String
)