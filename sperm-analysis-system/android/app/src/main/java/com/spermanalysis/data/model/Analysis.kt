package com.spermanalysis.data.model

import android.os.Parcelable
import androidx.room.Entity
import androidx.room.PrimaryKey
import com.google.gson.annotations.SerializedName
import kotlinx.parcelize.Parcelize
import java.util.*

/**
 * Data Models for Sperm Analysis
 * Author: Youssef Shitiwi (يوسف شتيوي)
 */

@Parcelize
@Entity(tableName = "analyses")
data class Analysis(
    @PrimaryKey
    val id: String,
    
    @SerializedName("analysis_name")
    val analysisName: String? = null,
    
    @SerializedName("video_filename")
    val videoFilename: String,
    
    val status: AnalysisStatus,
    
    @SerializedName("created_at")
    val createdAt: Date,
    
    @SerializedName("started_at")
    val startedAt: Date? = null,
    
    @SerializedName("completed_at")
    val completedAt: Date? = null,
    
    @SerializedName("processing_time")
    val processingTime: Float? = null,
    
    @SerializedName("error_message")
    val errorMessage: String? = null,
    
    // Configuration
    val fps: Float = 30.0f,
    
    @SerializedName("pixel_to_micron")
    val pixelToMicron: Float = 1.0f,
    
    @SerializedName("confidence_threshold")
    val confidenceThreshold: Float = 0.3f,
    
    @SerializedName("iou_threshold")
    val iouThreshold: Float = 0.5f,
    
    @SerializedName("min_track_length")
    val minTrackLength: Int = 10,
    
    @SerializedName("enable_visualization")
    val enableVisualization: Boolean = true,
    
    @SerializedName("export_trajectories")
    val exportTrajectories: Boolean = true,
    
    // Progress tracking
    val progress: Float = 0.0f,
    
    @SerializedName("current_frame")
    val currentFrame: Int? = null,
    
    @SerializedName("total_frames")
    val totalFrames: Int? = null,
    
    @SerializedName("processing_stage")
    val processingStage: String? = null,
    
    // Local file paths
    val localVideoPath: String? = null,
    val localResultsPath: String? = null
) : Parcelable

@Parcelize
enum class AnalysisStatus : Parcelable {
    @SerializedName("pending")
    PENDING,
    
    @SerializedName("processing")
    PROCESSING,
    
    @SerializedName("completed")
    COMPLETED,
    
    @SerializedName("failed")
    FAILED,
    
    @SerializedName("cancelled")
    CANCELLED
}

@Parcelize
data class AnalysisConfig(
    val fps: Float = 30.0f,
    
    @SerializedName("pixel_to_micron")
    val pixelToMicron: Float = 1.0f,
    
    @SerializedName("confidence_threshold")
    val confidenceThreshold: Float = 0.3f,
    
    @SerializedName("iou_threshold")
    val iouThreshold: Float = 0.5f,
    
    @SerializedName("min_track_length")
    val minTrackLength: Int = 10,
    
    @SerializedName("enable_visualization")
    val enableVisualization: Boolean = true,
    
    @SerializedName("export_trajectories")
    val exportTrajectories: Boolean = true
) : Parcelable

@Parcelize
data class SpermParameters(
    @SerializedName("track_id")
    val trackId: Int,
    
    @SerializedName("duration_frames")
    val durationFrames: Int,
    
    @SerializedName("duration_seconds")
    val durationSeconds: Float,
    
    // Motion classification
    @SerializedName("is_motile")
    val isMotile: Boolean,
    
    @SerializedName("is_progressive")
    val isProgressive: Boolean,
    
    @SerializedName("is_slow_progressive")
    val isSlowProgressive: Boolean,
    
    @SerializedName("is_non_progressive")
    val isNonProgressive: Boolean,
    
    @SerializedName("is_immotile")
    val isImmotile: Boolean,
    
    // Velocity parameters (μm/s)
    val vcl: Float, // Curvilinear velocity
    val vsl: Float, // Straight-line velocity
    val vap: Float, // Average path velocity
    
    // Motion parameters (%)
    val lin: Float, // Linearity
    val str: Float, // Straightness
    val wob: Float, // Wobble
    
    // Path parameters
    val alh: Float, // Amplitude of lateral head displacement (μm)
    val bcf: Float, // Beat cross frequency (Hz)
    
    // Distance parameters (μm)
    @SerializedName("total_distance")
    val totalDistance: Float,
    
    @SerializedName("net_distance")
    val netDistance: Float,
    
    // Trajectory data
    @SerializedName("trajectory_x")
    val trajectoryX: List<Float>? = null,
    
    @SerializedName("trajectory_y")
    val trajectoryY: List<Float>? = null
) : Parcelable

@Parcelize
data class PopulationStatistics(
    @SerializedName("total_sperm_count")
    val totalSpermCount: Int,
    
    // Counts
    @SerializedName("motile_count")
    val motileCount: Int,
    
    @SerializedName("progressive_count")
    val progressiveCount: Int,
    
    @SerializedName("slow_progressive_count")
    val slowProgressiveCount: Int,
    
    @SerializedName("non_progressive_count")
    val nonProgressiveCount: Int,
    
    @SerializedName("immotile_count")
    val immotileCount: Int,
    
    // Percentages
    @SerializedName("motility_percentage")
    val motilityPercentage: Float,
    
    @SerializedName("progressive_percentage")
    val progressivePercentage: Float,
    
    @SerializedName("slow_progressive_percentage")
    val slowProgressivePercentage: Float,
    
    @SerializedName("non_progressive_percentage")
    val nonProgressivePercentage: Float,
    
    @SerializedName("immotile_percentage")
    val immotilePercentage: Float,
    
    // Mean values
    @SerializedName("mean_vcl")
    val meanVcl: Float,
    
    @SerializedName("mean_vsl")
    val meanVsl: Float,
    
    @SerializedName("mean_vap")
    val meanVap: Float,
    
    @SerializedName("mean_lin")
    val meanLin: Float,
    
    @SerializedName("mean_str")
    val meanStr: Float,
    
    @SerializedName("mean_wob")
    val meanWob: Float,
    
    @SerializedName("mean_alh")
    val meanAlh: Float,
    
    @SerializedName("mean_bcf")
    val meanBcf: Float,
    
    // Standard deviations
    @SerializedName("std_vcl")
    val stdVcl: Float,
    
    @SerializedName("std_vsl")
    val stdVsl: Float,
    
    @SerializedName("std_vap")
    val stdVap: Float,
    
    @SerializedName("std_lin")
    val stdLin: Float
) : Parcelable

@Parcelize
data class AnalysisResults(
    @SerializedName("analysis_id")
    val analysisId: String,
    
    @SerializedName("analysis_name")
    val analysisName: String? = null,
    
    val timestamp: Date,
    
    @SerializedName("video_filename")
    val videoFilename: String,
    
    @SerializedName("video_duration")
    val videoDuration: Float,
    
    @SerializedName("total_frames")
    val totalFrames: Int,
    
    val fps: Float,
    
    val config: AnalysisConfig,
    
    @SerializedName("individual_sperm")
    val individualSperm: List<SpermParameters>,
    
    @SerializedName("population_statistics")
    val populationStatistics: PopulationStatistics,
    
    @SerializedName("processing_time")
    val processingTime: Float,
    
    @SerializedName("model_version")
    val modelVersion: String,
    
    @SerializedName("visualization_video")
    val visualizationVideo: String? = null,
    
    @SerializedName("csv_export")
    val csvExport: String? = null,
    
    @SerializedName("json_export")
    val jsonExport: String? = null
) : Parcelable

// API Response models
@Parcelize
data class AnalysisResponse(
    @SerializedName("analysis_id")
    val analysisId: String,
    
    val status: AnalysisStatus,
    val message: String,
    
    @SerializedName("estimated_processing_time")
    val estimatedProcessingTime: Float? = null,
    
    @SerializedName("created_at")
    val createdAt: Date
) : Parcelable

@Parcelize
data class StatusResponse(
    @SerializedName("analysis_id")
    val analysisId: String,
    
    val status: AnalysisStatus,
    val progress: Float,
    val message: String,
    
    @SerializedName("created_at")
    val createdAt: Date,
    
    @SerializedName("started_at")
    val startedAt: Date? = null,
    
    @SerializedName("completed_at")
    val completedAt: Date? = null,
    
    @SerializedName("error_message")
    val errorMessage: String? = null,
    
    @SerializedName("current_frame")
    val currentFrame: Int? = null,
    
    @SerializedName("total_frames")
    val totalFrames: Int? = null,
    
    @SerializedName("processing_stage")
    val processingStage: String? = null
) : Parcelable

@Parcelize
data class ResultsResponse(
    @SerializedName("analysis_id")
    val analysisId: String,
    
    val status: AnalysisStatus,
    
    val results: AnalysisResults? = null,
    
    @SerializedName("available_downloads")
    val availableDownloads: List<String> = emptyList()
) : Parcelable

@Parcelize
data class FileUploadResponse(
    val filename: String,
    val size: Long,
    val format: String,
    val duration: Float? = null,
    
    @SerializedName("upload_id")
    val uploadId: String,
    
    val timestamp: Date
) : Parcelable