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

package kr.ac.snu.band.example.holistic_face_pose

import android.Manifest
import android.R
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
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
import kr.ac.snu.band.example.holistic_face_pose.databinding.ActivityCameraBinding

import org.mrsnu.band.BackendType
import org.mrsnu.band.Band
import org.mrsnu.band.Buffer
import org.mrsnu.band.BufferFormat
import org.mrsnu.band.ConfigBuilder
import org.mrsnu.band.CpuMaskFlag
import org.mrsnu.band.Device
import org.mrsnu.band.Engine
import org.mrsnu.band.ImageProcessorBuilder
import org.mrsnu.band.Model
import org.mrsnu.band.SchedulerType
import org.mrsnu.band.SubgraphPreparationType
import org.mrsnu.band.Tensor

import java.nio.channels.FileChannel
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.min
import kotlin.random.Random


/** Activity that displays the camera and performs object detection on the incoming frames */
class CameraActivity : AppCompatActivity() {

    private lateinit var activityCameraBinding: ActivityCameraBinding

    private lateinit var bitmapBuffer: Bitmap
    private lateinit var inputBuffer : Buffer

    private val executor = Executors.newSingleThreadExecutor()
    private val permissions = listOf(Manifest.permission.CAMERA)
    private val permissionsRequestCode = Random.nextInt(0, 10000)

    private var lensFacing: Int = CameraSelector.LENS_FACING_BACK
    private val isFrontFacing get() = lensFacing == CameraSelector.LENS_FACING_FRONT

    private var pauseAnalysis = false
    private var imageRotationDegrees: Int = 0
    private lateinit var faceCropSize: RectF
    private lateinit var poseCropSize: RectF

    // Engine
    private val engine by lazy {
        Band.init()

        val builder = ConfigBuilder()
        builder.addPlannerLogPath("/data/local/tmp/log.json")
        builder.addSchedulers(arrayOf<SchedulerType>(SchedulerType.HETEROGENEOUS_EARLIEST_FINISH_TIME))
        builder.addMinimumSubgraphSize(7)
        builder.addSubgraphPreparationType(SubgraphPreparationType.MERGE_UNIT_SUBGRAPH)
        builder.addCPUMask(CpuMaskFlag.ALL)
        builder.addPlannerCPUMask(CpuMaskFlag.PRIMARY)
        builder.addWorkers(arrayOf<Device>(Device.CPU, Device.GPU, Device.DSP, Device.NPU))
        builder.addWorkerNumThreads(intArrayOf(1, 1, 1, 1))
        builder.addWorkerCPUMasks(
            arrayOf<CpuMaskFlag>(
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

    // Models
    private val faceDetectorModel by lazy {
        // Load mapped byte buffer from asset
        val fileDescriptor = assets.openFd(FACE_DETECTOR_MODEL_PATH)
        val inputStream = fileDescriptor.createInputStream()
        val mappedBuffer = inputStream.channel.map(
            FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
        inputStream.close()
        val model = Model(BackendType.TFLITE, mappedBuffer)
        engine.registerModel(model)
        model
    }
    private val faceLandmarksModel by lazy {
        // Load mapped byte buffer from asset
        val fileDescriptor = assets.openFd(FACE_LANDMARKS_MODEL_PATH)
        val inputStream = fileDescriptor.createInputStream()
        val mappedBuffer = inputStream.channel.map(
            FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
        inputStream.close()
        val model = Model(BackendType.TFLITE, mappedBuffer)
        engine.registerModel(model)
        model
    }
    private val poseDetectorModel by lazy {
        // Load mapped byte buffer from asset
        val fileDescriptor = assets.openFd(POSE_DETECTOR_MODEL_PATH)
        val inputStream = fileDescriptor.createInputStream()
        val mappedBuffer = inputStream.channel.map(
            FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
        inputStream.close()
        val model = Model(BackendType.TFLITE, mappedBuffer)
        engine.registerModel(model)
        model
    }
    private val poseLandmarksModel by lazy {
        // Load mapped byte buffer from asset
        val fileDescriptor = assets.openFd(POSE_LANDMARKS_MODEL_PATH)
        val inputStream = fileDescriptor.createInputStream()
        val mappedBuffer = inputStream.channel.map(
            FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
        inputStream.close()
        val model = Model(BackendType.TFLITE, mappedBuffer)
        engine.registerModel(model)
        model
    }

    private val faceDetectorInputTensors by lazy {
        List<Tensor>(engine.getNumInputTensors(faceDetectorModel)) { engine.createInputTensor(faceDetectorModel, it) }    // 1 is the number of input tensors
    }
    private val faceLandmarksInputTensors by lazy {
        List<Tensor>(engine.getNumInputTensors(faceLandmarksModel)) { engine.createInputTensor(faceLandmarksModel, it) }    // 1 is the number of input tensors
    }
    private val faceDetectorOutputTensors by lazy {
        List<Tensor>(engine.getNumOutputTensors(faceDetectorModel)) { engine.createOutputTensor(faceDetectorModel, it) }    // 4 is the number of output tensors
    }
    private val faceLandmarksOutputTensors by lazy {
        List<Tensor>(engine.getNumOutputTensors(faceLandmarksModel)) { engine.createOutputTensor(faceLandmarksModel, it) }    // 4 is the number of output tensors
    }
    private val faceDetectorInputSize by lazy {
        Size(faceDetectorInputTensors[0].dims[2], faceDetectorInputTensors[0].dims[1]) // Order of axis is: {1, height, width, 3}
    }
    private val faceLandmarksInputSize by lazy {
        Size(faceLandmarksInputTensors[0].dims[2], faceLandmarksInputTensors[0].dims[1]) // Order of axis is: {1, height, width, 3}
    }

    private val poseDetectorInputTensors by lazy {
        List<Tensor>(engine.getNumInputTensors(poseDetectorModel)) { engine.createInputTensor(poseDetectorModel, it) }    // 1 is the number of input tensors
    }
    private val poseLandmarksInputTensors by lazy {
        List<Tensor>(engine.getNumInputTensors(poseLandmarksModel)) { engine.createInputTensor(poseLandmarksModel, it) }    // 1 is the number of input tensors
    }
    private val poseDetectorOutputTensors by lazy {
        List<Tensor>(engine.getNumOutputTensors(poseDetectorModel)) { engine.createOutputTensor(poseDetectorModel, it) }    // 4 is the number of output tensors
    }
    private val poseLandmarksOutputTensors by lazy {
        List<Tensor>(engine.getNumOutputTensors(poseLandmarksModel)) { engine.createOutputTensor(poseLandmarksModel, it) }    // 4 is the number of output tensors
    }
    private val poseDetectorInputSize by lazy {
        Size(poseDetectorInputTensors[0].dims[2], poseDetectorInputTensors[0].dims[1]) // Order of axis is: {1, height, width, 3}
    }
    private val poseLandmarksInputSize by lazy {
        Size(poseLandmarksInputTensors[0].dims[2], poseLandmarksInputTensors[0].dims[1]) // Order of axis is: {1, height, width, 3}
    }

    private val labels by lazy {
        assets.open(LABELS_PATH).bufferedReader().useLines { it.toList() }
    }
    private val helper by lazy {
        HolisticFaceHelper(engine, faceDetectorModel, faceLandmarksModel)
    }
    private val poseHelper by lazy {
        HolisticPoseHelper(engine, poseDetectorModel, poseLandmarksModel)
    }

    private val imageProcessor by lazy {
        val builder = ImageProcessorBuilder()
        builder.addColorSpaceConvert(BufferFormat.RGB)
        builder.addResize(faceDetectorInputSize.width, faceDetectorInputSize.height)
        builder.addDataTypeConvert()
        builder.build()
    }
    private val landmarksImageProcessor by lazy {
        val builder = ImageProcessorBuilder()
        builder.addColorSpaceConvert(BufferFormat.RGB)
        builder.addCrop(faceCropSize.left.toInt(), faceCropSize.top.toInt(),
            faceCropSize.right.toInt(), faceCropSize.bottom.toInt())
        builder.addResize(faceLandmarksInputSize.width, faceLandmarksInputSize.height)
        builder.addNormalize(0.0f, 255.0f)
        builder.addDataTypeConvert()
        builder.build()
    }

    private val poseDetectorImageProcessor by lazy {
        val cropSize = minOf(bitmapBuffer.width, bitmapBuffer.height)
        val cropStart = Size((bitmapBuffer.width - cropSize) / 2, (bitmapBuffer.height - cropSize) / 2)
        val builder = ImageProcessorBuilder()
        builder.addColorSpaceConvert(BufferFormat.RGB)
        // center crop
        builder.addCrop(cropStart.width, cropStart.height, cropStart.width + cropSize - 1, cropStart.height + cropSize - 1)
        builder.addResize(poseDetectorInputSize.width, poseDetectorInputSize.height)
        builder.addRotate(-imageRotationDegrees)
        builder.build()
    }
    private val poseLandmarkImageProcessor by lazy {
        val builder = ImageProcessorBuilder()
        builder.addColorSpaceConvert(BufferFormat.RGB)
        builder.addCrop(poseCropSize.left.toInt(), poseCropSize.top.toInt(),
            poseCropSize.right.toInt(), poseCropSize.bottom.toInt())
        builder.addResize(faceLandmarksInputSize.width, faceLandmarksInputSize.height)
        builder.addNormalize(0.0f, 255.0f)
        builder.addDataTypeConvert()
        builder.build()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        activityCameraBinding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(activityCameraBinding.root)

        activityCameraBinding.cameraCaptureButton.setOnClickListener {

            // Disable all camera controls
            it.isEnabled = false

            if (pauseAnalysis) {
                // If image analysis is in paused state, resume it
                pauseAnalysis = false
                activityCameraBinding.imagePredicted.visibility = View.GONE

            } else {
                // Otherwise, pause image analysis and freeze image
                pauseAnalysis = true
                val matrix = Matrix().apply {
                    postRotate(imageRotationDegrees.toFloat())
                    if (isFrontFacing) postScale(-1f, 1f)
                }
                val uprightImage = Bitmap.createBitmap(
                    bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height, matrix, true)
                activityCameraBinding.imagePredicted.setImageBitmap(uprightImage)
                activityCameraBinding.imagePredicted.visibility = View.VISIBLE
            }

            // Re-enable camera controls
            it.isEnabled = true
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

                    activityCameraBinding.faceBoxPrediction?.setCanvasSize(
                        Size(activityCameraBinding.viewFinder.width, activityCameraBinding.viewFinder.height))
                }

                // Early exit: image analysis is in paused state
                if (pauseAnalysis) {
                    image.close()
                    return@Analyzer
                }

                // get non-null image
                val realImage = image.image ?: return@Analyzer
                inputBuffer = Buffer(realImage.planes, image.width, image.height, BufferFormat.YV12)

                /* FACE PIPELINE */
                imageProcessor.process(inputBuffer, faceDetectorInputTensors[0])
                image.close()
                // Perform the face & pose pipeline for the current frame
                var predictions =
                    helper.detectorPredict(faceDetectorInputTensors, faceDetectorOutputTensors)
                makeBoxesSquare(predictions, PADDING_RATIO)
                val bestFace = predictions.maxByOrNull { it.score }
                reportPrediction2(bestFace)
                if (predictions.isNotEmpty()) {
                    if (bestFace != null) {
                        faceCropSize = RectF(
                                bestFace.box.left * bitmapBuffer.width,
                                bestFace.box.top * bitmapBuffer.height,
                                bestFace.box.right * bitmapBuffer.width,
                                bestFace.box.bottom * bitmapBuffer.height
                            )
                        landmarksImageProcessor.process(
                            inputBuffer,
                            faceLandmarksInputTensors[0]
                        )
                        val landmarks = helper.landmarksPredict(
                            faceLandmarksInputTensors,
                            faceLandmarksOutputTensors
                        )
                        faceReportLandmarks(landmarks, faceCropSize)
                    }
                }


                // POSE PIPELINE
                poseDetectorImageProcessor.process(inputBuffer, poseDetectorInputTensors[0])
                image.close()
                // Perform the face & pose pipeline for the current frame
                var posePredictions =
                    poseHelper.detectorPredict(poseDetectorInputTensors, poseDetectorOutputTensors)
                if(posePredictions.isNotEmpty()) {
                    posePredictions.sortByDescending { it.score }
                    val bestPerson = posePredictions[0]
                    poseCropSize = RectF(
                        bestPerson.box.left * bitmapBuffer.width,
                        bestPerson.box.top * bitmapBuffer.height,
                        bestPerson.box.right * bitmapBuffer.width,
                        bestPerson.box.bottom * bitmapBuffer.height
                    )
                    poseLandmarkImageProcessor.process(inputBuffer, poseLandmarksInputTensors[0])
                    var poseLandmarks = poseHelper.landmarksPredict(poseLandmarksInputTensors, poseLandmarksOutputTensors)
                    for( landmark in poseLandmarks ){
                        landmark.x *= poseCropSize.width()
                        landmark.y *= poseCropSize.height()
                    }
                    poseReportLandmarks(poseLandmarks, poseCropSize)
                    Log.d("HYUNSOO", "After: $poseLandmarks")
                }

                // Compute the FPS of the entire pipeline
                val frameCount = 10
                if (++frameCounter % frameCount == 0) {
                    frameCounter = 0
                    val now = System.currentTimeMillis()
                    val delta = now - lastFpsTimestamp
                    val fps = 1000 * frameCount.toFloat() / delta
                    Log.d(TAG, "FPS: ${"%.02f".format(fps)} with tensorSize: ${image.width} x ${image.height}")
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


    private fun makeBoxesSquare(boxes: List<HolisticFaceHelper.FaceBoxPrediction>, paddingRate: Float): List<HolisticFaceHelper.FaceBoxPrediction>{
        for (faceBox in boxes){
            val box = faceBox.box

            val width = box.width() * bitmapBuffer.width
            val height = box.height() * bitmapBuffer.height

            val centerX = ((box.left + box.right) / 2) * bitmapBuffer.width
            val centerY = ((box.top + box.bottom) / 2) * bitmapBuffer.height

            val size = if (height < width) height else width
            val paddedSize = (1f + paddingRate) * size

            box.top = (centerY - paddedSize / 2) / bitmapBuffer.height
            box.bottom = (centerY + paddedSize / 2) / bitmapBuffer.height
            box.left = (centerX - paddedSize / 2) / bitmapBuffer.width
            box.right = (centerX + paddedSize / 2) / bitmapBuffer.width
        }
        return boxes
    }

    private fun reportPrediction2(faceBoxPrediction: HolisticFaceHelper.FaceBoxPrediction?) {
        if (faceBoxPrediction != null)
            activityCameraBinding.faceBoxPrediction?.setRect(faceBoxPrediction.box)
        activityCameraBinding.faceBoxPrediction?.invalidate()
    }

    private fun faceReportLandmarks(landmarks: ArrayList<HolisticFaceHelper.Landmark>, boxCrop: RectF) {
        activityCameraBinding.faceLandmarksPrediction?.setLandmarks(landmarks, boxCrop)
        activityCameraBinding.faceLandmarksPrediction?.invalidate()
    }

    private fun poseReportLandmarks(landmarks: ArrayList<HolisticFaceHelper.Landmark>, boxCrop: RectF) {
        activityCameraBinding.poseLandmarksPrediction?.setLandmarks(landmarks, boxCrop)
        activityCameraBinding.poseLandmarksPrediction?.invalidate()
    }

    private fun reportPrediction(
        prediction: HolisticFaceHelper.FaceBoxPrediction?
    ) = activityCameraBinding.viewFinder.post {

        // Early exit: if prediction is not good enough, don't report it
        if (prediction == null || prediction.score < ACCURACY_THRESHOLD) {
            activityCameraBinding.boxPrediction.visibility = View.GONE
//            activityCameraBinding.textPrediction.visibility = View.GONE
            return@post
        }

        // Location has to be mapped to our local coordinates
        val location = mapOutputCoordinates(prediction.box)

        // Update the text and UI
//        activityCameraBinding.textPrediction.text = "${"%.2f".format(prediction.score)} ${prediction.label}"
        (activityCameraBinding.boxPrediction.layoutParams as ViewGroup.MarginLayoutParams).apply {
            topMargin = location.top.toInt()
            leftMargin = location.left.toInt()
            width = min(activityCameraBinding.viewFinder.width, location.right.toInt() - location.left.toInt())
            height = min(activityCameraBinding.viewFinder.height, location.bottom.toInt() - location.top.toInt())
        }

        // Make sure all UI elements are visible
        activityCameraBinding.boxPrediction.visibility = View.VISIBLE
//        activityCameraBinding.textPrediction.visibility = View.VISIBLE
    }

    /**
     * Helper function used to map the coordinates for objects coming out of
     * the model into the coordinates that the user sees on the screen.
     */
    private fun mapOutputCoordinates(location: RectF): RectF {
        // Step 1: map location to the preview coordinates
        val previewLocation = RectF(
            location.left * activityCameraBinding.viewFinder.width,
            location.top * activityCameraBinding.viewFinder.height,
            location.right * activityCameraBinding.viewFinder.width,
            location.bottom * activityCameraBinding.viewFinder.height
        )

        // Step 2: compensate for camera sensor orientation and mirroring
        val isFrontFacing = lensFacing == CameraSelector.LENS_FACING_FRONT
        val correctedLocation = if (isFrontFacing) {
            RectF(
                activityCameraBinding.viewFinder.width - previewLocation.right,
                previewLocation.top,
                activityCameraBinding.viewFinder.width - previewLocation.left,
                previewLocation.bottom)
        } else {
            previewLocation
        }

        // Step 3: compensate for 1:1 to 4:3 aspect ratio conversion + small margin
        val margin = 0.1f
        val requestedRatio = 4f / 3f
        val midX = (correctedLocation.left + correctedLocation.right) / 2f
        val midY = (correctedLocation.top + correctedLocation.bottom) / 2f
        return if (activityCameraBinding.viewFinder.width < activityCameraBinding.viewFinder.height) {
            RectF(
                midX - (1f + margin) * requestedRatio * correctedLocation.width() / 2f,
                midY - (1f - margin) * correctedLocation.height() / 2f,
                midX + (1f + margin) * requestedRatio * correctedLocation.width() / 2f,
                midY + (1f - margin) * correctedLocation.height() / 2f
            )
        } else {
            RectF(
                midX - (1f - margin) * correctedLocation.width() / 2f,
                midY - (1f + margin) * requestedRatio * correctedLocation.height() / 2f,
                midX + (1f - margin) * correctedLocation.width() / 2f,
                midY + (1f + margin) * requestedRatio * correctedLocation.height() / 2f
            )
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

    companion object {
        private val TAG = CameraActivity::class.java.simpleName

        private const val ACCURACY_THRESHOLD = 0.2f
        private const val MAX_NUM_FACES = 1
        private const val LABELS_PATH = "coco_ssd_mobilenet_v1_1.0_labels.txt"
        private const val PADDING_RATIO = 0.75f

        // Pose pipeline: SSD-MobilenetV2 + MoveNet Lightning
        // Face pipeline: RetinaFace-MobilenetV2 + FaceMesh
        private const val POSE_DETECTOR_MODEL_PATH = "ssd_mobilenet_v2_coco_quant_postprocess.tflite"
        private const val POSE_LANDMARKS_MODEL_PATH = "lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite"
        private const val FACE_DETECTOR_MODEL_PATH = "retinaface-mbv2-int8.tflite" // (1, 160, 160, 3) -> [(1, 1050, 2), (1, 1050, 4), (1, 1050, 10)]
        private const val FACE_LANDMARKS_MODEL_PATH = "face_landmark_192_full_integer_quant.tflite" // (1, 192, 192, 3) -> [(1, 1, 1, 1404), (1, 1, 1, 1)]
    }
}
