package kr.ac.snu.band.example.holistic_face_pose

import android.annotation.SuppressLint
import android.content.res.AssetManager
import android.graphics.RectF
import android.media.Image
import android.util.Log
import android.util.Size
import androidx.camera.core.ImageProxy
import org.mrsnu.band.BackendType
import org.mrsnu.band.Band
import org.mrsnu.band.Buffer
import org.mrsnu.band.ColorSpaceConvert
import org.mrsnu.band.ConfigBuilder
import org.mrsnu.band.CpuMaskFlag
import org.mrsnu.band.Crop
import org.mrsnu.band.DTypeConvert
import org.mrsnu.band.DataType
import org.mrsnu.band.Device
import org.mrsnu.band.Engine
import org.mrsnu.band.ImageBuffer
import org.mrsnu.band.ImageFormat
import org.mrsnu.band.ImageOrientation
import org.mrsnu.band.RequestOption
import org.mrsnu.band.PipelineBuilder
import org.mrsnu.band.Model
import org.mrsnu.band.Normalize
import org.mrsnu.band.Resize
import org.mrsnu.band.Rotate
import org.mrsnu.band.SchedulerType
import org.mrsnu.band.SubgraphPreparationType
import org.mrsnu.band.Tensor
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

class HolisticHelper(assetManager: AssetManager) {
    data class ImageProperties (val width: Int, val height: Int, val rotationDegrees: Int, var isInitialized: Boolean)
    data class HolisticFace (
        var faceDetection: HolisticFaceHelper.FaceBoxPrediction?,
        var faceLandmarks: ArrayList<HolisticFaceHelper.Landmark>?
    )
    data class HolisticPose (
        var poseDetection: HolisticPoseHelper.PosePrediction?,
        var poseLandmarks: ArrayList<HolisticFaceHelper.Landmark>?
    )
    data class Holistic (
        var faceDetection: HolisticFaceHelper.FaceBoxPrediction?,
        var faceLandmarks: ArrayList<HolisticFaceHelper.Landmark>?,
        var poseDetection: HolisticPoseHelper.PosePrediction?,
        var poseLandmarks: ArrayList<HolisticFaceHelper.Landmark>?
    )

    private lateinit var faceCropSize: RectF
    private var cameraImageProperties = ImageProperties(0, 0, 0, false)
    private var faceDetection: HolisticFaceHelper.FaceBoxPrediction? = null
    private var faceLandmarks: ArrayList<HolisticFaceHelper.Landmark>? = null
    private var poseDetection: HolisticPoseHelper.PosePrediction? = null
    private var poseLandmarks: ArrayList<HolisticFaceHelper.Landmark>? = null

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
                CpuMaskFlag.BIG, CpuMaskFlag.ALL,
                CpuMaskFlag.ALL, CpuMaskFlag.ALL
            ))
        builder.addNumWarmups(5)
        builder.addNumRuns(50)
        builder.addAvailabilityCheckIntervalMs(30000)
        builder.addScheduleWindowSize(10)
        Engine(builder.build())
    }

    // Models
    private val faceDetectorModel by lazy {
        // Load mapped byte buffer from asset
        val fileDescriptor = assetManager.openFd(FACE_DETECTOR_MODEL_PATH)
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
        val fileDescriptor = assetManager.openFd(FACE_LANDMARKS_MODEL_PATH)
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
        val fileDescriptor = assetManager.openFd(POSE_DETECTOR_MODEL_PATH)
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
        val fileDescriptor = assetManager.openFd(POSE_LANDMARKS_MODEL_PATH)
        val inputStream = fileDescriptor.createInputStream()
        val mappedBuffer = inputStream.channel.map(
            FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
        inputStream.close()
        val model = Model(BackendType.TFLITE, mappedBuffer)
        engine.registerModel(model)
        model
    }

    private val option by lazy {
        RequestOption()
    }

    private val detectorModels by lazy{
        arrayOf(faceDetectorModel, poseDetectorModel)
    }

    private val faceLandmarksInputTensors by lazy {
        List<Tensor>(engine.getNumInputTensors(faceLandmarksModel)) { engine.createInputTensor(faceLandmarksModel, it) }    // 1 is the number of input tensors
    }
    private val faceLandmarksOutputTensors by lazy {
        List<Tensor>(engine.getNumOutputTensors(faceLandmarksModel)) { engine.createOutputTensor(faceLandmarksModel, it) }    // 4 is the number of output tensors
    }
    private val faceLandmarksInputSize by lazy {
        Size(faceLandmarksInputTensors[0].dims[2], faceLandmarksInputTensors[0].dims[1]) // Order of axis is: {1, height, width, 3}
    }

    private val poseLandmarksInputTensors by lazy {
        List<Tensor>(engine.getNumInputTensors(poseLandmarksModel)) { engine.createInputTensor(poseLandmarksModel, it) }    // 1 is the number of input tensors
    }
    private val poseLandmarksOutputTensors by lazy {
        List<Tensor>(engine.getNumOutputTensors(poseLandmarksModel)) { engine.createOutputTensor(poseLandmarksModel, it) }    // 4 is the number of output tensors
    }
    private val poseLandmarksInputSize by lazy {
        Size(poseLandmarksInputTensors[0].dims[2], poseLandmarksInputTensors[0].dims[1]) // Order of axis is: {1, height, width, 3}
    }

    private val detectorInputTensors by lazy {
        arrayOf(
            List<Tensor>(engine.getNumInputTensors(detectorModels[FACE])) { engine.createInputTensor(detectorModels[FACE], it) },
            List<Tensor>(engine.getNumInputTensors(detectorModels[POSE])) { engine.createInputTensor(detectorModels[POSE], it) }
        )
    }
    private val detectorInputSizes by lazy{
        arrayOf(
            Size(detectorInputTensors[FACE][0].dims[2], detectorInputTensors[FACE][0].dims[1]), // Order of axis is: {1, height, width, 3}
            Size(detectorInputTensors[POSE][0].dims[2], detectorInputTensors[POSE][0].dims[1]) // Order of axis is: {1, height, width, 3}
        )
    }
    private val detectorOutputTensors by lazy {
        arrayOf(
            List<Tensor>(engine.getNumOutputTensors(detectorModels[FACE])) { engine.createOutputTensor(detectorModels[FACE], it) },
            List<Tensor>(engine.getNumOutputTensors(detectorModels[POSE])) { engine.createOutputTensor(detectorModels[POSE], it) }
        )
    }

    private val faceHelper by lazy {
        HolisticFaceHelper(engine, faceDetectorModel, faceLandmarksModel, option)
    }
    private val poseHelper by lazy {
        val labels = assetManager.open(LABEL_PATH).bufferedReader().useLines { it.toList() }
        HolisticPoseHelper(engine, poseDetectorModel, poseLandmarksModel, labels)
    }

    private val faceDetectorImageProcessor by lazy {
        val cropSize = minOf(cameraImageProperties.width, cameraImageProperties.height)
        val cropStart = Size((cameraImageProperties.width - cropSize) / 2, (cameraImageProperties.height - cropSize) / 2)

        val builder = PipelineBuilder()
        builder.add(ColorSpaceConvert(ImageFormat.RGB))
        builder.add(Resize(detectorInputSizes[FACE].width, detectorInputSizes[FACE].height))
        builder.add(DTypeConvert(DataType.FLOAT32))
        builder.add(Normalize(127.5f, 127.5f))
        builder.build()
    }

    private val poseDetectorImageProcessor by lazy {
        val cropSize = minOf(cameraImageProperties.width, cameraImageProperties.height)
        val cropStart = Size((cameraImageProperties.width - cropSize) / 2, (cameraImageProperties.height - cropSize) / 2)

        val builder = PipelineBuilder()
        builder.add(ColorSpaceConvert(ImageFormat.RGB))
        // center crop
//        builder.addCrop(cropStart.width, cropStart.height, cropStart.width + cropSize - 1, cropStart.height + cropSize - 1)
        builder.add(Resize(detectorInputSizes[POSE].width, detectorInputSizes[POSE].height))
        builder.build()
    }

    @SuppressLint("UnsafeOptInUsageError")
    fun predict(image: ImageProxy, rotationDegrees: Int): Holistic?{
        cameraImageProperties = ImageProperties(image.width, image.height, rotationDegrees, true)

        val realImage = image.image ?: return null
        val inputBuffer = ImageBuffer(realImage.planes, cameraImageProperties.width, cameraImageProperties.height, ImageFormat.YV12, ImageOrientation.TOP_LEFT)

        predict(inputBuffer)

        return Holistic(faceDetection, faceLandmarks, poseDetection, poseLandmarks)
    }

    private fun predict(inputBuffer: Buffer){
        // 1. Detect face and person first
        // Process
        val faceDetectionInputBuffer = faceDetectorImageProcessor.run(inputBuffer)
        val poseDetectionInputBuffer = poseDetectorImageProcessor.run(inputBuffer)

        faceDetectionInputBuffer.copyToTensor(detectorInputTensors[FACE][0])
        poseDetectionInputBuffer.copyToTensor(detectorInputTensors[POSE][0])

        // new empty list of RequestOption
        val options = mutableListOf<RequestOption>()

        // Inference
        var requests = engine.requestAsyncBatch(detectorModels.toMutableList(), detectorInputTensors.toMutableList(), options)
        engine.wait(requests[FACE], detectorOutputTensors[FACE].toMutableList())
        engine.wait(requests[POSE], detectorOutputTensors[POSE].toMutableList())
        // Post-process
        var faceRet = faceHelper.detectorPostProcess(detectorOutputTensors[FACE].toMutableList())
        var poseRet = poseHelper.detectorPostProcess(detectorOutputTensors[POSE].toMutableList())
        makeBoxesSquare(faceRet, PADDING_RATIO)
        faceDetection = faceRet.maxByOrNull { it.score }
        if(poseRet.isNotEmpty()){
            poseDetection = poseRet.maxByOrNull { it.score }
            poseDetection!!.box.left = max((poseDetection!!.box.left - poseDetection!!.box.width() * POSE_PADDING_RATIO), 0.01f)
            poseDetection!!.box.top = max((poseDetection!!.box.top - poseDetection!!.box.height() * POSE_PADDING_RATIO), 0.01f)
            poseDetection!!.box.right = min((poseDetection!!.box.right + poseDetection!!.box.width() * POSE_PADDING_RATIO), 0.99f)
            poseDetection!!.box.bottom = min((poseDetection!!.box.bottom + poseDetection!!.box.height() * POSE_PADDING_RATIO), 0.99f)
        }
        // 2-1. Face landmarks
        if (faceDetection != null) {
            faceCropSize = RectF(
                faceDetection!!.box.left * cameraImageProperties.width,
                faceDetection!!.box.top * cameraImageProperties.height,
                faceDetection!!.box.right * cameraImageProperties.width,
                faceDetection!!.box.bottom * cameraImageProperties.height
            )

            val builder = PipelineBuilder()
            builder.add(ColorSpaceConvert(ImageFormat.RGB))
            builder.add(
                Crop(faceCropSize.left.toInt(), faceCropSize.top.toInt(),
                faceCropSize.right.toInt(), faceCropSize.bottom.toInt()
            )
            )
            builder.add(Resize(faceLandmarksInputSize.width, faceLandmarksInputSize.height))
            builder.add(Normalize(0.0f, 255.0f))
            val landmarksImageProcessor = builder.build()
            Log.d("HYUNSOO", "done till here3")
            val landmarkInputBuffer = landmarksImageProcessor.run(
                inputBuffer
            )
            landmarkInputBuffer.copyToTensor(faceLandmarksInputTensors[0])
            Log.d("HYUNSOO", "done till here4")
            faceLandmarks = faceHelper.landmarksPredict(
                faceLandmarksInputTensors,
                faceLandmarksOutputTensors,
                faceLandmarksInputSize
            )
            Log.d("HYUNSOO", "done till here5")
        }

        // 2-2. Pose landmarks
        var poseCropSize: RectF
        if(poseRet.isNotEmpty()) {
            poseCropSize = RectF(--
                poseDetection!!.box.left * cameraImageProperties.width,
                poseDetection!!.box.top * cameraImageProperties.height,
                poseDetection!!.box.right * cameraImageProperties.width,
                poseDetection!!.box.bottom * cameraImageProperties.height
            )

            val builder = PipelineBuilder()
            builder.add(ColorSpaceConvert(ImageFormat.RGB))
            builder.add(Crop(
                poseCropSize.left.toInt(), poseCropSize.top.toInt(),
                poseCropSize.right.toInt(), poseCropSize.bottom.toInt()
            ))
            builder.add(Resize(poseLandmarksInputSize.width, poseLandmarksInputSize.height))
            builder.add(Rotate(-cameraImageProperties.rotationDegrees))
            val poseLandmarkImageProcessor = builder.build()
            val poseLandmarkInputBuffer = poseLandmarkImageProcessor.run(inputBuffer)
            poseLandmarkInputBuffer.copyToTensor(poseLandmarksInputTensors[0])
            poseLandmarks = poseHelper.landmarksPredict(poseLandmarksInputTensors, poseLandmarksOutputTensors)
        }
    }

    private fun makeBoxesSquare(boxes: List<HolisticFaceHelper.FaceBoxPrediction>, paddingRate: Float): List<HolisticFaceHelper.FaceBoxPrediction>{
        for (faceBox in boxes){
            val box = faceBox.box

            val width = box.width() * cameraImageProperties.width
            val height = box.height() * cameraImageProperties.height

            val centerX = ((box.left + box.right) / 2) * cameraImageProperties.width
            val centerY = ((box.top + box.bottom) / 2) * cameraImageProperties.height

            val size = if (height < width) height else width
            val paddedSize = (1f + paddingRate) * size

            box.top = (centerY - paddedSize / 2) / cameraImageProperties.height
            box.bottom = (centerY + paddedSize / 2) / cameraImageProperties.height
            box.left = (centerX - paddedSize / 2) / cameraImageProperties.width
            box.right = (centerX + paddedSize / 2) / cameraImageProperties.width
        }
        return boxes
    }


    companion object {
        private const val FACE = 0
        private const val POSE = 1
        // Pose pipeline: SSD-MobilenetV2 + MoveNet Lightning
        // Face pipeline: RetinaFace-MobilenetV2 + FaceMesh
        private const val LABEL_PATH = "coco_ssd_mobilenet_v1_1.0_labels.txt"
        private const val POSE_DETECTOR_MODEL_PATH = "coco_ssd_mobilenet_v1_1.0_quant.tflite"
        private const val POSE_LANDMARKS_MODEL_PATH = "lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite"
        private const val FACE_DETECTOR_MODEL_PATH = "retinaface_mbv2_shrink-int8.tflite" // (1, 160, 160, 3) -> [(1, 1050, 2), (1, 1050, 4), (1, 1050, 10)]
        private const val FACE_LANDMARKS_MODEL_PATH = "face_landmark_192_full_integer_quant.tflite" // (1, 192, 192, 3) -> [(1, 1, 1, 1404), (1, 1, 1, 1)]

        private const val PADDING_RATIO = 0.75f
        private const val POSE_PADDING_RATIO = 0.5f
    }

}