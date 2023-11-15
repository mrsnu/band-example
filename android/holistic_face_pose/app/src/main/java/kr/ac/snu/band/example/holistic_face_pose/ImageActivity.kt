package kr.ac.snu.band.example.holistic_face_pose

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
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
import java.io.File
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.Executors

class ImageActivity: AppCompatActivity() {
    private lateinit var activityCameraBinding: ActivityCameraBinding
    private lateinit var bitmapBuffer: Bitmap
    private lateinit var inputBuffer: Buffer
    private val executor = Executors.newSingleThreadExecutor()
    private lateinit var imageView: ImageView
    private var imageRotationDegrees: Int = 0

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

    private val imageProcessor by lazy {
        val cropSize = minOf(bitmapBuffer.width, bitmapBuffer.height)
        val cropStart = Size((bitmapBuffer.width - cropSize) / 2, (bitmapBuffer.height - cropSize) / 2)
        val builder = ImageProcessorBuilder()
//        builder.addColorSpaceConvert(BufferFormat.RGB)
        // center crop
        builder.addCrop(cropStart.width, cropStart.height, cropStart.width + cropSize - 1, cropStart.height + cropSize - 1)
        builder.addResize(faceDetectorInputSize.width, faceDetectorInputSize.height)
        builder.addRotate(-imageRotationDegrees)
        builder.build()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        activityCameraBinding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(activityCameraBinding.root)

//        imageView = findViewById(R.id.imageView)

        activityCameraBinding.cameraCaptureButton.setOnClickListener {
            // Replace camera-related logic with image loading logic
            val imageFile = assets.open("image.jpg")
            val loadedBitmap = BitmapFactory.decodeStream(imageFile)
            bitmapBuffer = Bitmap.createBitmap(loadedBitmap.width, loadedBitmap.height, Bitmap.Config.ARGB_8888)
            processLoadedImage(loadedBitmap)
        }
    }

    // Other methods...

    private fun processLoadedImage(loadedBitmap: Bitmap) {
        imageView.setImageBitmap(loadedBitmap)
        // Process the image in your Tensorflow pipeline
        // Update the logic to work with the loadedBitmap instead of the camera image


        val bytes = loadedBitmap.byteCount
        val buffer = ByteBuffer.allocate(bytes)
        loadedBitmap.copyPixelsToBuffer(buffer)
        inputBuffer = Buffer(buffer, loadedBitmap.width, loadedBitmap.height, BufferFormat.RGB)

        imageProcessor.process(inputBuffer, faceDetectorInputTensors[0])
        Log.d("HYUNSOO", "IMAGE PROCESSING")
//        // Perform the face & pose pipeline for the loaded image
//        val predictions = helper.predict(faceDetectorInputTensors, faceDetectorOutputTensors)
//
//        Log.d(TAG, "predictions: $predictions")
//
//        // Report only the top prediction
//        reportPrediction(predictions.maxByOrNull { it.score })

        // Other image processing logic...
    }

    // Other methods...
    companion object {
        private val TAG = CameraActivity::class.java.simpleName

        private const val ACCURACY_THRESHOLD = 0.5f
        private const val LABELS_PATH = "coco_ssd_mobilenet_v1_1.0_labels.txt"

        // Pose pipeline: SSD-MobilenetV2 + MoveNet Lightning
        // Face pipeline: RetinaFace-MobilenetV2 + FaceMesh
        private const val POSE_DETECTOR_MODEL_PATH = "ssd_mobilenet_v2_coco_quant_postprocess.tflite"
        private const val POSE_LANDMARKS_MODEL_PATH = "lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite"
        private const val FACE_DETECTOR_MODEL_PATH = "retinaface-mbv2-int8.tflite" // (1, 160, 160, 3) -> [(1, 1050, 2), (1, 1050, 4), (1, 1050, 10)]
        private const val FACE_LANDMARKS_MODEL_PATH = "face_landmark_192_full_integer_quant.tflite" // (1, 192, 192, 3) -> [(1, 1, 1, 1404), (1, 1, 1, 1)]
    }
}