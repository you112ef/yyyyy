package com.spermanalysis.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.spermanalysis.BuildConfig
import com.spermanalysis.R
import com.spermanalysis.databinding.ActivityMainBinding
import com.spermanalysis.ui.analysis.AnalysisDetailActivity
import com.spermanalysis.ui.analysis.AnalysisListAdapter
import com.spermanalysis.ui.settings.SettingsActivity
import com.spermanalysis.utils.FileUtils
import com.spermanalysis.viewmodel.MainViewModel
import kotlinx.coroutines.launch
import pub.devrel.easypermissions.EasyPermissions
import java.io.File

/**
 * Main Activity - Sperm Analysis App
 * Author: Youssef Shitiwi (يوسف شتيوي)
 * 
 * Main screen for video upload and analysis management
 */
class MainActivity : AppCompatActivity(), EasyPermissions.PermissionCallbacks {

    private lateinit var binding: ActivityMainBinding
    private val viewModel: MainViewModel by viewModels()
    private lateinit var analysisAdapter: AnalysisListAdapter

    companion object {
        private const val PERMISSIONS_REQUEST_CODE = 123
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
        )
    }

    // Activity result launchers
    private val videoPickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let { handleVideoSelected(it) }
    }

    private val cameraLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            viewModel.tempVideoUri?.let { uri ->
                handleVideoSelected(uri)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupUI()
        setupObservers()
        checkPermissions()
        
        // Load existing analyses
        viewModel.loadAnalyses()
    }

    private fun setupUI() {
        setSupportActionBar(binding.toolbar)
        
        // Setup RecyclerView
        analysisAdapter = AnalysisListAdapter { analysis ->
            val intent = Intent(this, AnalysisDetailActivity::class.java).apply {
                putExtra("analysis_id", analysis.id)
            }
            startActivity(intent)
        }
        
        binding.recyclerViewAnalyses.apply {
            layoutManager = LinearLayoutManager(this@MainActivity)
            adapter = analysisAdapter
        }

        // Setup FAB
        binding.fabAddAnalysis.setOnClickListener {
            showVideoSourceDialog()
        }

        // Setup toolbar menu
        binding.toolbar.setOnMenuItemClickListener { menuItem ->
            when (menuItem.itemId) {
                R.id.action_settings -> {
                    startActivity(Intent(this, SettingsActivity::class.java))
                    true
                }
                R.id.action_refresh -> {
                    viewModel.refreshAnalyses()
                    true
                }
                else -> false
            }
        }

        // Setup swipe refresh
        binding.swipeRefreshLayout.setOnRefreshListener {
            viewModel.refreshAnalyses()
        }

        // Developer info
        binding.textDeveloperInfo.text = "Developed by ${BuildConfig.DEVELOPER_NAME}"
    }

    private fun setupObservers() {
        // Analyses list
        viewModel.analyses.observe(this) { analyses ->
            analysisAdapter.submitList(analyses)
            binding.swipeRefreshLayout.isRefreshing = false
            
            // Show/hide empty state
            if (analyses.isEmpty()) {
                binding.layoutEmptyState.visibility = android.view.View.VISIBLE
                binding.recyclerViewAnalyses.visibility = android.view.View.GONE
            } else {
                binding.layoutEmptyState.visibility = android.view.View.GONE
                binding.recyclerViewAnalyses.visibility = android.view.View.VISIBLE
            }
        }

        // Loading state
        viewModel.isLoading.observe(this) { isLoading ->
            if (!binding.swipeRefreshLayout.isRefreshing) {
                binding.progressBar.visibility = if (isLoading) android.view.View.VISIBLE else android.view.View.GONE
            }
        }

        // Error messages
        viewModel.errorMessage.observe(this) { error ->
            error?.let {
                Toast.makeText(this, it, Toast.LENGTH_LONG).show()
                viewModel.clearError()
            }
        }

        // Upload progress
        viewModel.uploadProgress.observe(this) { progress ->
            if (progress > 0) {
                binding.progressBarUpload.visibility = android.view.View.VISIBLE
                binding.progressBarUpload.progress = progress
                
                if (progress >= 100) {
                    binding.progressBarUpload.visibility = android.view.View.GONE
                }
            }
        }
    }

    private fun checkPermissions() {
        if (!EasyPermissions.hasPermissions(this, *REQUIRED_PERMISSIONS)) {
            EasyPermissions.requestPermissions(
                this,
                "This app needs camera and storage permissions to record and analyze videos.",
                PERMISSIONS_REQUEST_CODE,
                *REQUIRED_PERMISSIONS
            )
        }
    }

    private fun showVideoSourceDialog() {
        if (!EasyPermissions.hasPermissions(this, *REQUIRED_PERMISSIONS)) {
            checkPermissions()
            return
        }

        val options = arrayOf(
            "Record Video",
            "Choose from Gallery",
            "Browse Files"
        )

        MaterialAlertDialogBuilder(this)
            .setTitle("Select Video Source")
            .setItems(options) { _, which ->
                when (which) {
                    0 -> recordVideo()
                    1 -> chooseFromGallery()
                    2 -> browseFiles()
                }
            }
            .show()
    }

    private fun recordVideo() {
        val intent = Intent(MediaStore.ACTION_VIDEO_CAPTURE).apply {
            // Create temp file for video
            val videoFile = FileUtils.createTempVideoFile(this@MainActivity)
            val videoUri = androidx.core.content.FileProvider.getUriForFile(
                this@MainActivity,
                "${packageName}.fileprovider",
                videoFile
            )
            
            viewModel.tempVideoUri = videoUri
            putExtra(MediaStore.EXTRA_OUTPUT, videoUri)
            putExtra(MediaStore.EXTRA_VIDEO_QUALITY, 1) // High quality
        }

        if (intent.resolveActivity(packageManager) != null) {
            cameraLauncher.launch(intent)
        } else {
            Toast.makeText(this, "No camera app available", Toast.LENGTH_SHORT).show()
        }
    }

    private fun chooseFromGallery() {
        videoPickerLauncher.launch("video/*")
    }

    private fun browseFiles() {
        val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
            type = "video/*"
            addCategory(Intent.CATEGORY_OPENABLE)
        }
        
        val chooser = Intent.createChooser(intent, "Select Video File")
        if (chooser.resolveActivity(packageManager) != null) {
            videoPickerLauncher.launch("video/*")
        }
    }

    private fun handleVideoSelected(uri: Uri) {
        lifecycleScope.launch {
            try {
                // Validate video file
                val fileInfo = FileUtils.getVideoFileInfo(this@MainActivity, uri)
                
                if (fileInfo.duration < 1000) { // Less than 1 second
                    Toast.makeText(this@MainActivity, "Video is too short (minimum 1 second)", Toast.LENGTH_LONG).show()
                    return@launch
                }
                
                if (fileInfo.size > 500 * 1024 * 1024) { // 500MB limit
                    Toast.makeText(this@MainActivity, "Video file is too large (maximum 500MB)", Toast.LENGTH_LONG).show()
                    return@launch
                }

                // Show analysis configuration dialog
                showAnalysisConfigDialog(uri)
                
            } catch (e: Exception) {
                Toast.makeText(this@MainActivity, "Error processing video: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun showAnalysisConfigDialog(videoUri: Uri) {
        // This would show a dialog for analysis configuration
        // For now, use default settings
        val analysisName = "Analysis ${System.currentTimeMillis()}"
        
        viewModel.startAnalysis(
            videoUri = videoUri,
            analysisName = analysisName,
            fps = 30.0f,
            pixelToMicron = 1.0f,
            confidenceThreshold = 0.3f,
            iouThreshold = 0.5f,
            minTrackLength = 10,
            enableVisualization = true,
            exportTrajectories = true
        )
    }

    override fun onPermissionsGranted(requestCode: Int, perms: MutableList<String>) {
        if (requestCode == PERMISSIONS_REQUEST_CODE) {
            Toast.makeText(this, "Permissions granted", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onPermissionsDenied(requestCode: Int, perms: MutableList<String>) {
        if (requestCode == PERMISSIONS_REQUEST_CODE) {
            Toast.makeText(this, "Some permissions were denied. App may not work correctly.", Toast.LENGTH_LONG).show()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        EasyPermissions.onRequestPermissionsResult(requestCode, permissions, grantResults, this)
    }

    override fun onResume() {
        super.onResume()
        // Refresh analyses when returning to the screen
        viewModel.refreshAnalyses()
    }
}