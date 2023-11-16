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
import android.graphics.Matrix
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
    private lateinit var inputBuffer : Buffer

    private val executor = Executors.newSingleThreadExecutor()
    private val permissions = listOf(Manifest.permission.CAMERA)
    private val permissionsRequestCode = Random.nextInt(0, 10000)

    private var lensFacing: Int = CameraSelector.LENS_FACING_BACK
    private val isFrontFacing get() = lensFacing == CameraSelector.LENS_FACING_FRONT

    private var setTargetIdentity = false
    private var imageRotationDegrees: Int = 0

    var detCnt = 0
    var recCnt = 0
    var identity_vec : FloatArray? = null
    var identity_bm : Bitmap? = null

    private val engine by lazy {
        Band.init()
        Band.setVerbosity(LogSeverity.INTERNAL)

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

    private val detInputSize by lazy {
        Size(detInputTensors[0].dims[2], detInputTensors[0].dims[1]) // Order of axis is: {1, height, width, 3}
    }

    private val recInputSize by lazy {
        Size(recInputTensors[0][0].dims[2], recInputTensors[0][0].dims[1]) // Order of axis is: {1, height, width, 3}
    }

    private val faceDet by lazy {
        FaceDetectionHelper(engine, detModel)
    }

    private val faceRec by lazy {
        FaceRecognitionHelper(engine)
    }

    /*
    private val imageProcessor by lazy {
        val cropSize = minOf(bitmapBuffer.width, bitmapBuffer.height)
        val cropStart = Size((bitmapBuffer.width - cropSize) / 2, (bitmapBuffer.height - cropSize) / 2)
        val builder = ImageProcessorBuilder()
        builder.addColorSpaceConvert(BufferFormat.RGB)
        // center crop
        builder.addCrop(cropStart.width, cropStart.height, cropStart.width + cropSize - 1, cropStart.height + cropSize - 1)
        builder.addResize(inputSize.width, inputSize.height)
        builder.addRotate(-imageRotationDegrees)
        builder.build()
    }
     */

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

            setTargetIdentity = true

            /*
            // Disable all camera controls
            it.isEnabled = false

            if (setTargetIdentity) {
                // If image analysis is in paused state, resume it
                setTargetIdentity =
                activityCameraBinding.imagePredicted.visibility = View.GONE

            } else {
                // Otherwise, pause image analysis and freeze image
                setTargetIdentity = true
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
             */
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


    fun saveBitmap(bitmap: Bitmap, filename: String) {
        val file = File(applicationInfo.dataDir + "/img/" + filename)
        try{
            val out = FileOutputStream(file)
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)
            out.close()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun tensor2bitmap(tensor: Tensor, width: Int, height: Int, scale: Float) : Bitmap {
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
                //Log.d("XXX", "index: $index | rgb: $r,$g,$b")
            }
        }

        return Bitmap.createBitmap(colorArr, width, height, Bitmap.Config.ARGB_8888)
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

                // Early exit: image analysis is in paused state
                if (setTargetIdentity) {
                    image.close()
                    return@Analyzer
                }

                // get non-null image
                val realImage = image.image ?: return@Analyzer
                inputBuffer = Buffer(realImage.planes, image.width, image.height, BufferFormat.NV21)

                // Process the image in Tensorflow
                detImageProcessor.process(inputBuffer, detInputTensors[0])
                image.close()

                // save image to /data/data/{packagename}/img/
                /*
                val filename = "det_${detCnt}.jpg"
                detCnt += 1
                val bitmap = tensor2bitmap(detInputTensors[0], detInputSize.width, detInputSize.height)
                saveBitmap(bitmap, filename)

                 */

                // Perform the object detection for the current frame
                val predictions = faceDet.predict(detInputTensors, detOutputTensors)

                Log.d(TAG, "predictions: $predictions")

                // Report only the top prediction
                /*
                val bestPrediction = predictions.maxByOrNull { it.confidence }
                reportPrediction(bestPrediction)
                 */

                // Report top5 predictions
                val top5Predictions = predictions.sortedByDescending { it.confidence }.take(MAX_NUM_FACES)
                reportPredictions(top5Predictions)

                val recModels = ArrayList<Model>()

                if (top5Predictions.isNotEmpty()) {

                    //top5Predictions.forEachIndexed{index, prediction ->

                    val index = 0
                    val prediction = top5Predictions[index]


                    val builder = ImageProcessorBuilder()
                    builder.addColorSpaceConvert(BufferFormat.RGB)
                    builder.addRotate(-imageRotationDegrees)

                    val cropSize = minOf(bitmapBuffer.width, bitmapBuffer.height)
                    val cropStart = Size((bitmapBuffer.width - cropSize) / 2, (bitmapBuffer.height - cropSize) / 2)
                    //builder.addCrop(cropStart.width, cropStart.height, cropStart.width + cropSize - 1, cropStart.height + cropSize - 1)

                    val bboxRect = RectF(
                        prediction.location.left * cropSize,
                        prediction.location.top * cropSize,
                        prediction.location.right * cropSize,
                        prediction.location.bottom * cropSize
                    )

                    // log prediction's location
                    //Log.d("XXX", "prediction: ${prediction.location.left}, ${prediction.location.top}, ${prediction.location.right}, ${prediction.location.bottom}")

                    bboxRect.left = max(bboxRect.left, 0f)
                    bboxRect.top = max(bboxRect.top, 0f)
                    bboxRect.right = min(bboxRect.right, image.width.toFloat())
                    bboxRect.bottom = min(bboxRect.bottom, image.height.toFloat())

                    builder.addCrop(
                        cropStart.height + bboxRect.left.toInt(),
                        cropStart.width + bboxRect.top.toInt(),
                        cropStart.height  + bboxRect.right.toInt(),
                        cropStart.width + bboxRect.bottom.toInt()
                    )

                    builder.addResize(recInputSize.width, recInputSize.height)
                    //builder.addDataTypeConvert()
                    builder.addNormalize(0f, 255f)
                    val recImageProcessor = builder.build()

                    // TODO: support multiple faces.

                    recImageProcessor.process(inputBuffer, recInputTensors[index][0])
                    recImageProcessor.close()
                    recModels.add(recModel)

                    // print few values in recInputTensors[i][0]

                    /*
                    val filename2 = "rec_${recCnt}.jpeg"
                    recCnt += 1
                    val bitmap2 = tensor2bitmap(recInputTensors[index][0], recInputSize.width, recInputSize.height)
                    saveBitmap(bitmap2, filename2)
                     */
                    val identities = faceRec.predict(recModels, recInputTensors, recOutputTensors)

                    if (setTargetIdentity) {
                        setTargetIdentity = false
                        identity_vec = identities[0]
                        identity_bm = tensor2bitmap(recInputTensors[index][0], recInputSize.width, recInputSize.height, 255f)

                        val filename2 = "identity.jpeg"
                        saveBitmap(tensor2bitmap(recInputTensors[index][0], recInputSize.width, recInputSize.height, 255f), filename2)

                        //
                        runOnUiThread(Runnable {
                            activityCameraBinding.imageView?.setImageBitmap(identity_bm)
                            activityCameraBinding.imageView?.visibility = View.VISIBLE
                            activityCameraBinding.imageView?.bringToFront()
                        })
                    } else if (identity_vec != null) {
                        var similarity: Float = (faceRec.dotProduct(identities[0], identity_vec!!) / (faceRec.norm(identities[0]) * faceRec.norm(identity_vec!!)) + 1f) / 2f
                        similarity = min(max(similarity, 0f), 1f)

                        Log.d("XXX", "similarity: $similarity")

                        /*
                        if (similarity > SIM_THRESHOLD) {
                            Log.d("XXX", "same!")
                        } else {
                            Log.d("XXX", "different!")
                        }
                         */
                    }
                    //}
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

    private fun reportPredictions(
        predictions: List<FaceDetectionHelper.BoundingBox>
    ) = activityCameraBinding.viewFinder.post {

        if (predictions.isEmpty()) {
            return@post
        }

        val boxPredictions = listOf(
            activityCameraBinding.boxPrediction1,
            activityCameraBinding.boxPrediction2,
            activityCameraBinding.boxPrediction3,
            activityCameraBinding.boxPrediction4,
            activityCameraBinding.boxPrediction5,
        )

        for (boxPrediction in boxPredictions) {
            boxPrediction?.visibility = View.GONE
        }

        predictions.forEachIndexed {index, prediction ->
            val location = mapOutputCoordinates(prediction.location)

            // Update the text and UI
            (boxPredictions[index]?.layoutParams as ViewGroup.MarginLayoutParams).apply {
                topMargin = location.top.toInt()
                leftMargin = location.left.toInt()
                width = min(activityCameraBinding.viewFinder.width, location.right.toInt() - location.left.toInt())
                height = min(activityCameraBinding.viewFinder.height, location.bottom.toInt() - location.top.toInt())
            }

            // Make sure all UI elements are visible
            boxPredictions[index]?.visibility = View.VISIBLE

        }

    }

    private fun reportPrediction(
        prediction: FaceDetectionHelper.BoundingBox?
    ) = activityCameraBinding.viewFinder.post {

        // Early exit: if prediction is not good enough, don't report it
        if (prediction == null) {
            activityCameraBinding.boxPrediction1?.visibility = View.GONE
            activityCameraBinding.boxPrediction2?.visibility = View.GONE
            activityCameraBinding.boxPrediction3?.visibility = View.GONE
            activityCameraBinding.boxPrediction4?.visibility = View.GONE
            activityCameraBinding.boxPrediction5?.visibility = View.GONE
            return@post
        }

        // Location has to be mapped to our local coordinates
        val location = mapOutputCoordinates(prediction.location)

        // Update the text and UI
        (activityCameraBinding.boxPrediction1?.layoutParams as ViewGroup.MarginLayoutParams).apply {
            topMargin = location.top.toInt()
            leftMargin = location.left.toInt()
            width = min(activityCameraBinding.viewFinder.width, location.right.toInt() - location.left.toInt())
            height = min(activityCameraBinding.viewFinder.height, location.bottom.toInt() - location.top.toInt())
        }

        // Make sure all UI elements are visible
        activityCameraBinding.boxPrediction1?.visibility = View.VISIBLE
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
        return previewLocation

        /*
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
         */



        /*
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
         */
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

        private const val DET_MODEL_PATH = "retinaface-mbv2-int8.tflite"
        private const val REC_MODEL_PATH = "arc-mbv2-int8.tflite"
        private const val SIM_THRESHOLD = 0.8f

        private const val MAX_NUM_FACES = 6
    }
}
