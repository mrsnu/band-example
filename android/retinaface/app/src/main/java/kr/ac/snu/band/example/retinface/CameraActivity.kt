/*
 * Copyright 2023 Seoul National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package kr.ac.snu.band.example.retinface

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.View
import android.view.ViewGroup
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import kr.ac.snu.band.example.retinface.databinding.ActivityCameraBinding
import org.mrsnu.band.BackendType
import org.mrsnu.band.Band
import org.mrsnu.band.Buffer
import org.mrsnu.band.BufferFormat
import org.mrsnu.band.ConfigBuilder
import org.mrsnu.band.CpuMaskFlag
import org.mrsnu.band.Device
import org.mrsnu.band.Engine
import org.mrsnu.band.ImageProcessorBuilder
import org.mrsnu.band.LogSeverity
import org.mrsnu.band.Model
import org.mrsnu.band.SchedulerType
import org.mrsnu.band.SubgraphPreparationType
import org.mrsnu.band.Tensor
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random


/** Activity that displays the camera and performs object detection on the incoming frames */
class CameraActivity : AppCompatActivity() {

    private lateinit var activityCameraBinding: ActivityCameraBinding

    private lateinit var bitmapBuffer: Bitmap

    private val executor = Executors.newSingleThreadExecutor()
    private val permissions = listOf(Manifest.permission.CAMERA)
    private val permissionsRequestCode = Random.nextInt(0, 10000)

    private var lensFacing: Int = CameraSelector.LENS_FACING_BACK

    private var setCurrFaceAsTarget = false
    private var imageRotationDegrees: Int = 0

    private var identityVec : FloatArray? = null
    private var identityBm : Bitmap? = null

    private val engine by lazy {
        Band.init()
        Band.setVerbosity(LogSeverity.INTERNAL)

        val builder = ConfigBuilder()
        builder.addPlannerLogPath("/data/local/tmp/log.json")
        builder.addSchedulers(arrayOf(SchedulerType.HETEROGENEOUS_EARLIEST_FINISH_TIME))
        builder.addMinimumSubgraphSize(7)
        builder.addSubgraphPreparationType(SubgraphPreparationType.MERGE_UNIT_SUBGRAPH)
        builder.addCPUMask(CpuMaskFlag.ALL)
        builder.addPlannerCPUMask(CpuMaskFlag.PRIMARY)
        builder.addWorkers(arrayOf(Device.CPU, Device.GPU, Device.DSP, Device.NPU))
        builder.addWorkerNumThreads(intArrayOf(1, 1, 1, 1))
        builder.addWorkerCPUMasks(
           arrayOf(
               CpuMaskFlag.ALL, CpuMaskFlag.ALL,
               CpuMaskFlag.ALL, CpuMaskFlag.ALL
           ))
        builder.addSmoothingFactor(0.1f)
        builder.addProfileDataPath("/data/local/tmp/profile.json")
        builder.addOnline(true)
        builder.addNumWarmups(1)
        builder.addNumRuns(1)
        builder.addAllowWorkSteal(true)
        builder.addAvailabilityCheckIntervalMs(30000)
        builder.addScheduleWindowSize(10)
        Engine(builder.build())
    }


    /** Models */
    private val detModel by lazy {
        // Load mapped byte buffer from asset
        val fileDescriptor = assets.openFd(DET_MODEL_PATH)
        val inputStream = fileDescriptor.createInputStream()
        val mappedBuffer = inputStream.channel.map(
            FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
        inputStream.close()
        val model = Model(BackendType.TFLITE, mappedBuffer)
        engine.registerModel(model)
        model
    }
    private val recModel by lazy {
        // Load mapped byte buffer from asset
        val fileDescriptor = assets.openFd(REC_MODEL_PATH)
        val inputStream = fileDescriptor.createInputStream()
        val mappedBuffer = inputStream.channel.map(
            FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
        inputStream.close()
        val model = Model(BackendType.TFLITE, mappedBuffer)
        engine.registerModel(model)
        model
    }

    /** Input/Output Tensors */
    private val detInputTensors by lazy {
        List<Tensor>(engine.getNumInputTensors(detModel)) { engine.createInputTensor(detModel, it) }
    }
    private val detOutputTensors by lazy {
        List<Tensor>(engine.getNumOutputTensors(detModel)) { engine.createOutputTensor(detModel, it) }
    }
    private val recInputTensors by lazy {
        List(MAX_NUM_FACES) {
            List<Tensor>(engine.getNumInputTensors(recModel)) { engine.createInputTensor(recModel, it) }
        }
    }
    private val recOutputTensors by lazy {
        List(MAX_NUM_FACES) {
            List<Tensor>(engine.getNumOutputTensors(recModel)) { engine.createOutputTensor(recModel, it) }
        }
    }

    /** Input Size */
    private val detInputSize by lazy {
        Size(detInputTensors[0].dims[2], detInputTensors[0].dims[1]) // Order of axis is: {1, height, width, 3}
    }
    private val recInputSize by lazy {
        Size(recInputTensors[0][0].dims[2], recInputTensors[0][0].dims[1]) // Order of axis is: {1, height, width, 3}
    }

    /** Helper classes */
    private val faceDet by lazy {
        FaceDetectionHelper(engine, detModel)
    }
    private val faceRec by lazy {
        FaceRecognitionHelper(engine)
    }

    /** UI elements */
    private val boxPredictions by lazy{
        listOf(
            activityCameraBinding.boxPrediction1,
            activityCameraBinding.boxPrediction2,
            activityCameraBinding.boxPrediction3,
            activityCameraBinding.boxPrediction4,
            activityCameraBinding.boxPrediction5,
        )
    }

    /** Preprocessor */
    private val detImageProcessor by lazy {
        val cropSize = minOf(bitmapBuffer.width, bitmapBuffer.height)
        val cropStart = Size((bitmapBuffer.width - cropSize) / 2, (bitmapBuffer.height - cropSize) / 2)
        val builder = ImageProcessorBuilder()
        builder.addColorSpaceConvert(BufferFormat.RGB)
        builder.addCrop(cropStart.width, cropStart.height, cropStart.width + cropSize - 1, cropStart.height + cropSize - 1)
        builder.addResize(detInputSize.width, detInputSize.height)
        builder.addRotate(-imageRotationDegrees)
        builder.addDataTypeConvert()
        builder.build()
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        activityCameraBinding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(activityCameraBinding.root)

        activityCameraBinding.cameraCaptureButton.setOnClickListener {
            if (!setCurrFaceAsTarget) {
                // Will be set to false after the next frame's highest confidence face is set as target
                setCurrFaceAsTarget = true
            }
        }
    }

    override fun onResume() {
        super.onResume()

        // Request permissions each time the app resumes, since they can be revoked at any time
        if (!hasPermissions(this)) {
            ActivityCompat.requestPermissions(
                this, permissions.toTypedArray(), permissionsRequestCode)
        } else {
            bindCameraUseCases()
        }
    }

    override fun onDestroy() {
        // Terminate all outstanding analyzing jobs (if there is any).
        executor.apply {
            shutdown()
            awaitTermination(1000, TimeUnit.MILLISECONDS)
        }
        super.onDestroy()
    }

    /** Declare and bind preview and analysis use cases */
    @SuppressLint("UnsafeExperimentalUsageError", "UnsafeOptInUsageError")
    private fun bindCameraUseCases() = activityCameraBinding.viewFinder.post {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener ({

            // Camera provider is now guaranteed to be available
            val cameraProvider = cameraProviderFuture.get()

            // Set up the view finder use case to display camera preview
            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(activityCameraBinding.viewFinder.display.rotation)
                .build()

            // Set up the image analysis use case which will process frames in real time
            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(activityCameraBinding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()

            var frameCounter = 0
            var lastFpsTimestamp = System.currentTimeMillis()

            imageAnalysis.setAnalyzer(executor, ImageAnalysis.Analyzer { image ->
                if (!::bitmapBuffer.isInitialized) {
                    // The image rotation and RGB image buffer are initialized only once
                    // the analyzer has started running
                    imageRotationDegrees = image.imageInfo.rotationDegrees
                    bitmapBuffer = Bitmap.createBitmap(
                        image.width, image.height, Bitmap.Config.ARGB_8888)
                }

                // get non-null image
                val realImage = image.image ?: return@Analyzer
                val inputBuffer = Buffer(realImage.planes, image.width, image.height, BufferFormat.NV21)
                image.close()

                // Perform face detection on the input image
                // and take predictions with highest confidence
                val detPredictions = detectFace(inputBuffer).sortedByDescending { it.confidence }.take(MAX_NUM_FACES)

                // If there is at least one face detected, perform face recognition
                if (detPredictions.isNotEmpty()) {
                    val identities = recognizeFaces(inputBuffer, detPredictions, image.width, image.height)

                    var maxSimilarityIndex = -1
                    if (setCurrFaceAsTarget) {
                        setTargetFace(identities[0])
                    } else if (identityVec != null) {
                        // Find the face with highest similarity to the target
                        var maxSimilarity = 0f
                        identities.forEachIndexed{idIndex, identity ->
                            val similarity = faceRec.similarity(identity, identityVec!!)
                            if (similarity > maxSimilarity && similarity > SIM_THRESHOLD) {
                                maxSimilarity = similarity
                                maxSimilarityIndex = idIndex
                            }
                        }
                    }
                    // Draw the bounding boxes and emphasize the face with highest similarity
                    reportPredictions(detPredictions, maxSimilarityIndex)
                }

                // Compute the FPS of the entire pipeline
                val frameCount = 10
                if (++frameCounter % frameCount == 0) {
                    frameCounter = 0
                    val now = System.currentTimeMillis()
                    val delta = now - lastFpsTimestamp
                    val fps = 1000 * frameCount.toFloat() / delta
                    Log.d(TAG, "FPS: ${"%.02f".format(fps)} with tensorSize: ${image.width} x ${image.height}")

                    // Display it on the screen
                    runOnUiThread {
                        activityCameraBinding.textFps.text = "FPS: %.2f".format(fps)
                        activityCameraBinding.textFps.visibility = View.VISIBLE
                        activityCameraBinding.textFps.bringToFront()
                    }

                    lastFpsTimestamp = now
                }
            })

            // Create a new camera selector each time, enforcing lens facing
            val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

            // Apply declared configs to CameraX using the same lifecycle owner
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this as LifecycleOwner, cameraSelector, preview, imageAnalysis)

            // Use the camera object to link our preview use case with the view
            preview.setSurfaceProvider(activityCameraBinding.viewFinder.surfaceProvider)

        }, ContextCompat.getMainExecutor(this))
    }

    private fun detectFace(inputBuffer: Buffer) : List<FaceDetectionHelper.BoundingBox> {
        // Preprocess the image for detection
        detImageProcessor.process(inputBuffer, detInputTensors[0])

        // Perform the object detection for the current frame
        return faceDet.predict(detInputTensors, detOutputTensors)
    }

    private fun recognizeFaces(inputBuffer: Buffer, predictions: List<FaceDetectionHelper.BoundingBox>, imgWidth: Int, imgHeight: Int) : List<FloatArray> {
        val recModels = ArrayList<Model>()

        // Crop same region as detection, and use it as offset for bbox
        val cropSize = minOf(bitmapBuffer.width, bitmapBuffer.height)
        val cropStart = Size((bitmapBuffer.width - cropSize) / 2, (bitmapBuffer.height - cropSize) / 2)

        predictions.forEachIndexed { index, prediction ->

            // Preprocess the image for recognition
            val builder = ImageProcessorBuilder()
            builder.addColorSpaceConvert(BufferFormat.RGB)
            builder.addRotate(-imageRotationDegrees)
            val bboxRect = RectF(
                prediction.location.left * cropSize,
                prediction.location.top * cropSize,
                prediction.location.right * cropSize,
                prediction.location.bottom * cropSize
            )
            bboxRect.left = max(bboxRect.left, 0f)
            bboxRect.top = max(bboxRect.top, 0f)
            bboxRect.right = min(bboxRect.right, imgWidth.toFloat())
            bboxRect.bottom = min(bboxRect.bottom, imgHeight.toFloat())
            builder.addCrop(
                cropStart.height + bboxRect.left.toInt(),
                cropStart.width + bboxRect.top.toInt(),
                cropStart.height + bboxRect.right.toInt(),
                cropStart.width + bboxRect.bottom.toInt()
            )
            builder.addResize(recInputSize.width, recInputSize.height)
            builder.addNormalize(
                0f,
                255f
            )  // addDataTypeConvert() is not required when using addNormalize()
            val recImageProcessor = builder.build()

            // Preprocess the image for recognition
            recImageProcessor.process(inputBuffer, recInputTensors[index][0])
            recImageProcessor.close()

            recModels.add(recModel)
        }

        // Perform the face recognition for the current frame's faces
        return faceRec.predict(recModels, recInputTensors, recOutputTensors)
    }

    private fun setTargetFace(identity: FloatArray) {
        // Set the face with highest confidence as target
        identityVec = identity
        identityBm = tensor2bitmap(recInputTensors[0][0], recInputSize.width, recInputSize.height, 255f)

        // Draw the target face on the screen
        runOnUiThread {
            activityCameraBinding.imageView?.setImageBitmap(identityBm)
            activityCameraBinding.imageView?.visibility = View.VISIBLE
            activityCameraBinding.imageView?.bringToFront()
        }
        setCurrFaceAsTarget = false
    }

    private fun tensor2bitmap(tensor: Tensor, width: Int, height: Int, scale: Float) : Bitmap {
        // 1f for int8, 255f for float32
        val colorArr = IntArray(width * height * 3)
        val rawBuffer = tensor.data.order(ByteOrder.nativeOrder()).rewind()
        val floatBuffer = (rawBuffer as ByteBuffer).asFloatBuffer()

        for (y in 0 until height) {
            for (x in 0 until width) {
                val index = y * width + x
                val r = (floatBuffer[index * 3 + 0]*scale).toInt()
                val g = (floatBuffer[index * 3 + 1]*scale).toInt()
                val b = (floatBuffer[index * 3 + 2]*scale).toInt()
                colorArr[index] = Color.argb(255, r, g, b)
            }
        }
        return Bitmap.createBitmap(colorArr, width, height, Bitmap.Config.ARGB_8888)
    }

    private fun reportPredictions(
        predictions: List<FaceDetectionHelper.BoundingBox>, maxSimilarityIndex : Int
    ) = activityCameraBinding.viewFinder.post {
        // TODO : Dynamic box drawing

        // Hide all the bounding boxes by default
        for (boxPrediction in boxPredictions) {
            boxPrediction?.visibility = View.GONE
        }
        activityCameraBinding.boxPrediction0?.visibility = View.GONE


        predictions.forEachIndexed {index, prediction ->
            val location = RectF(
                prediction.location.left * activityCameraBinding.viewFinder.width,
                prediction.location.top * activityCameraBinding.viewFinder.height,
                prediction.location.right * activityCameraBinding.viewFinder.width,
                prediction.location.bottom * activityCameraBinding.viewFinder.height
            )

            var boxUI = boxPredictions[index]
            if (maxSimilarityIndex != -1 && maxSimilarityIndex == index) {
                // If target face is found, emphasize it (boxPrediction0 uses different color: red)
                boxUI = activityCameraBinding.boxPrediction0
            }

            (boxUI?.layoutParams as ViewGroup.MarginLayoutParams).apply {
                topMargin = location.top.toInt()
                leftMargin = location.left.toInt()
                width = min(
                    activityCameraBinding.viewFinder.width,
                    location.right.toInt() - location.left.toInt()
                )
                height = min(
                    activityCameraBinding.viewFinder.height,
                    location.bottom.toInt() - location.top.toInt()
                )
            }
            boxUI.visibility = View.VISIBLE
        }

    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == permissionsRequestCode && hasPermissions(this)) {
            bindCameraUseCases()
        } else {
            finish() // If we don't have the required permissions, we can't run
        }
    }

    /** Convenience method used to check if all permissions required by this app are granted */
    private fun hasPermissions(context: Context) = permissions.all {
        ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
    }

    /** For debugging */
    private fun saveBitmap(bitmap: Bitmap, filename: String) {
        val file = File(applicationInfo.dataDir + "/img/" + filename)
        try{
            val out = FileOutputStream(file)
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)
            out.close()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }


    companion object {
        private val TAG = CameraActivity::class.java.simpleName

        private const val DET_MODEL_PATH = "retinaface-mbv2-int8.tflite"
        //private const val DET_MODEL_PATH = "retinaface-mbv2.tflite"
        private const val REC_MODEL_PATH = "arc-mbv2-int8.tflite"
        private const val SIM_THRESHOLD = 0.7f

        private const val MAX_NUM_FACES = 5
    }
}
