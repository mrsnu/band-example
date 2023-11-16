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
    private lateinit var poseCropSize: RectF
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

    private val faceHelper by lazy {
        HolisticFaceHelper(engine, faceDetectorModel, faceLandmarksModel)
    }
    private val poseHelper by lazy {
        val labels = assetManager.open(LABEL_PATH).bufferedReader().useLines { it.toList() }
        HolisticPoseHelper(engine, poseDetectorModel, poseLandmarksModel, labels)
    }

    private val imageProcessor by lazy {
        val builder = ImageProcessorBuilder()
        builder.addColorSpaceConvert(BufferFormat.RGB)
        builder.addResize(faceDetectorInputSize.width, faceDetectorInputSize.height)
        builder.addRotate(-cameraImageProperties.rotationDegrees)
        builder.addDataTypeConvert()
        builder.build()
    }
    private val landmarksImageProcessor by lazy {
        val cropSize = minOf(faceCropSize.width(), faceCropSize.height())
        val cropStart = Size(((faceCropSize.width() - cropSize) / 2).toInt(),
            ((faceCropSize.height() - cropSize) / 2).toInt()
        )

        val builder = ImageProcessorBuilder()
        builder.addColorSpaceConvert(BufferFormat.RGB)
        builder.addCrop(cropStart.width, cropStart.height,
            (cropStart.width + cropSize - 1).toInt(), (cropStart.height + cropSize - 1).toInt()
        )
        builder.addResize(faceLandmarksInputSize.width, faceLandmarksInputSize.height)
        builder.addNormalize(0.0f, 255.0f)
        // TODO: should I add rotate?
        builder.addDataTypeConvert()
        builder.build()

//        val cropSize = minOf(faceCropSize.width(), faceCropSize.height())
//        val cropStart = Size(((faceCropSize.width() - cropSize) / 2).toInt(),
//            ((faceCropSize.height() - cropSize) / 2).toInt()
//        )
//
//        val builder = ImageProcessorBuilder()
//        builder.addColorSpaceConvert(BufferFormat.RGB)
//        builder.addCrop(faceCropSize.left.toInt(), faceCropSize.top.toInt(),
//            faceCropSize.right.toInt(), faceCropSize.bottom.toInt()
//        )
//        builder.addResize(faceLandmarksInputSize.width, faceLandmarksInputSize.height)
//        builder.addNormalize(0.0f, 255.0f)
//        // TODO: should I add rotate?
//        builder.addDataTypeConvert()
//        builder.build()
    }

    private val poseDetectorImageProcessor by lazy {
        val builder = ImageProcessorBuilder()
        builder.addColorSpaceConvert(BufferFormat.RGB)
        builder.addResize(poseDetectorInputSize.width, poseDetectorInputSize.height)
        builder.addRotate(-cameraImageProperties.rotationDegrees)
        builder.build()
    }
    private val poseLandmarkImageProcessor by lazy {
        val builder = ImageProcessorBuilder()
        builder.addColorSpaceConvert(BufferFormat.RGB)
        builder.addCrop(
            poseCropSize.left.toInt(), poseCropSize.top.toInt(),
            poseCropSize.right.toInt(), poseCropSize.bottom.toInt()
        )
        builder.addResize(poseLandmarksInputSize.width, poseLandmarksInputSize.height)
        builder.addRotate(-cameraImageProperties.rotationDegrees)
        builder.build()
    }

    @SuppressLint("UnsafeOptInUsageError")
    fun predict(image: ImageProxy): Holistic?{
        if(!cameraImageProperties.isInitialized){
            cameraImageProperties = ImageProperties(image.width, image.height, image.imageInfo.rotationDegrees, true)
        }

        val realImage = image.image ?: return null
        val inputBuffer = Buffer(realImage.planes, cameraImageProperties.width, cameraImageProperties.height, BufferFormat.YV12)

        predictFace(inputBuffer)
        predictPose(inputBuffer)

        return Holistic(faceDetection, faceLandmarks, poseDetection, poseLandmarks)
    }

    private fun predictFace(inputBuffer: Buffer): HolisticFace {
        /* FACE PIPELINE */
        imageProcessor.process(inputBuffer, faceDetectorInputTensors[0])

        // Perform the face & pose pipeline for the current frame
        var predictions =
            faceHelper.detectorPredict(faceDetectorInputTensors, faceDetectorOutputTensors)
        makeBoxesSquare(predictions, PADDING_RATIO)
        faceDetection = predictions.maxByOrNull { it.score }

        if (faceDetection != null) {
            faceCropSize = RectF(
                faceDetection!!.box.left * cameraImageProperties.width,
                faceDetection!!.box.top * cameraImageProperties.height,
                faceDetection!!.box.right * cameraImageProperties.width,
                faceDetection!!.box.bottom * cameraImageProperties.height
            )
            landmarksImageProcessor.process(
                inputBuffer,
                faceLandmarksInputTensors[0]
            )
            faceLandmarks = faceHelper.landmarksPredict(
                faceLandmarksInputTensors,
                faceLandmarksOutputTensors,
                faceLandmarksInputSize
            )
        }
        return HolisticFace(
            faceDetection,
            faceLandmarks
        )
    }

    private fun predictPose(inputBuffer: Buffer): HolisticPose {
        // POSE PIPELINE
        poseDetectorImageProcessor.process(inputBuffer, poseDetectorInputTensors[0])

        // Perform the face & pose pipeline for the current frame
        var posePredictions =
            poseHelper.detectorPredict(poseDetectorInputTensors, poseDetectorOutputTensors)
        if(posePredictions.isNotEmpty()) {
            poseDetection = posePredictions.maxByOrNull { it.score }
            poseDetection!!.box.left = max((poseDetection!!.box.left - poseDetection!!.box.width() * POSE_PADDING_RATIO), 0.01f)
            poseDetection!!.box.top = max((poseDetection!!.box.top - poseDetection!!.box.height() * POSE_PADDING_RATIO), 0.01f)
            poseDetection!!.box.right = min((poseDetection!!.box.right + poseDetection!!.box.width() * POSE_PADDING_RATIO), 0.99f)
            poseDetection!!.box.bottom = min((poseDetection!!.box.bottom + poseDetection!!.box.height() * POSE_PADDING_RATIO), 0.99f)
            poseCropSize = RectF(
                poseDetection!!.box.left * cameraImageProperties.width,
                poseDetection!!.box.top * cameraImageProperties.height,
                poseDetection!!.box.right * cameraImageProperties.width,
                poseDetection!!.box.bottom * cameraImageProperties.height
            )
            poseLandmarkImageProcessor.process(inputBuffer, poseLandmarksInputTensors[0])
            poseLandmarks = poseHelper.landmarksPredict(poseLandmarksInputTensors, poseLandmarksOutputTensors)
        }

        return HolisticPose(
            poseDetection,
            poseLandmarks
        )
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
        // Pose pipeline: SSD-MobilenetV2 + MoveNet Lightning
        // Face pipeline: RetinaFace-MobilenetV2 + FaceMesh
        private const val LABEL_PATH = "coco_ssd_mobilenet_v1_1.0_labels.txt"
        private const val POSE_DETECTOR_MODEL_PATH = "coco_ssd_mobilenet_v1_1.0_quant.tflite"
        private const val POSE_LANDMARKS_MODEL_PATH = "lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite"
        private const val FACE_DETECTOR_MODEL_PATH = "retinaface-mbv2-int8.tflite" // (1, 160, 160, 3) -> [(1, 1050, 2), (1, 1050, 4), (1, 1050, 10)]
        private const val FACE_LANDMARKS_MODEL_PATH = "face_landmark_192_full_integer_quant.tflite" // (1, 192, 192, 3) -> [(1, 1, 1, 1404), (1, 1, 1, 1)]

        private const val PADDING_RATIO = 0.75f
        private const val POSE_PADDING_RATIO = 0.5f
    }

}