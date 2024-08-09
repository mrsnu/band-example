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

import android.R.attr.height
import android.R.attr.width
import android.graphics.RectF
import android.media.FaceDetector.Face
import android.util.Log
import android.util.Size
import org.mrsnu.band.Engine
import org.mrsnu.band.Model
import org.mrsnu.band.Request
import org.mrsnu.band.Tensor
import org.mrsnu.band.RequestOption
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min


/**
 * Helper class used to communicate between our app and the Band detection model
 */
class HolisticFaceHelper(private val engine: Engine, private val faceDetectorModel: Model, private val faceLandmarksModel: Model, private val option: RequestOption) {

    // retinaface detector:  (1, 640, 640, 3) -> [(16800, 16)]
    // landmarks: (1, 192, 192, 3) -> [(1, 1, 1, 1404), (1, 1, 1, 1)]



////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /** Abstraction object that wraps a prediction output in an easy to parse way */
    data class Landmark(var x: Float, var y: Float, var z: Float)
    data class FaceBoxPrediction(val box: RectF, val score: Float)

    fun detectorPostProcess(outputTensors: List<Tensor>): List<FaceBoxPrediction> {
        // post process face detector output
        var faceBoxes = ArrayList<FaceBoxPrediction>()

        val bboxBuffer = FloatArray(DET_NUM_RESULTS * 4)
        val lmrkBuffer = FloatArray(DET_NUM_RESULTS * 10)
        val clssBuffer = FloatArray(DET_NUM_RESULTS * 2)
        val bboxByteBuffer = outputTensors[0].data.order(ByteOrder.nativeOrder()).rewind()
        (bboxByteBuffer as ByteBuffer).asFloatBuffer().get(bboxBuffer)
        val clssByteBuffer = outputTensors[1].data.order(ByteOrder.nativeOrder()).rewind()
        (clssByteBuffer as ByteBuffer).asFloatBuffer().get(clssBuffer)
        val lmrkByteBuffer = outputTensors[2].data.order(ByteOrder.nativeOrder()).rewind()
        (lmrkByteBuffer as ByteBuffer).asFloatBuffer().get(lmrkBuffer)

        val priors = priorBox(listOf(640f, 640f), listOf(listOf(16f, 32f), listOf(64f, 128f), listOf(256f, 512f)), listOf(8f, 16f, 32f))
        val variances = listOf(0.1f, 0.2f)

        // Decode bbox np
        val boxes = decodeBbox(bboxBuffer, priors, variances)

        for (i in 0 until DET_NUM_RESULTS){
            val conf = clssBuffer[i*2 + 1]
            if(conf > SCORE_THRESH){
                faceBoxes.add(
                    FaceBoxPrediction(
                        box = RectF(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]),
                        score = conf
                    )
                )
            }
        }

        // FILTER_BOXES_BY_SIZE
        faceBoxes = filterBoxesBySize(faceBoxes)
        // NMS
        faceBoxes = nms(faceBoxes, IOU_THRESHOLD)

        return faceBoxes
    }


    private fun decodeBbox(bboxBuffer: FloatArray, priors: Array<FloatArray>, variances: List<Float>): Array<FloatArray> {
        val decodedBoxes = Array(DET_NUM_RESULTS) { FloatArray(4) }

        for (i in 0 until DET_NUM_RESULTS) {
            val centerX = priors[i][0] + bboxBuffer[i * 4 + 0] * variances[0] * priors[i][2]
            val centerY = priors[i][1] + bboxBuffer[i * 4 + 1] * variances[0] * priors[i][3]
            val width = priors[i][2] * exp(bboxBuffer[i * 4 + 2] * variances[1])
            val height = priors[i][3] * exp(bboxBuffer[i * 4 + 3] * variances[1])

            decodedBoxes[i][0] = (centerX - width / 2)  // xmin
            decodedBoxes[i][1] = (centerY - height / 2) // ymin
            decodedBoxes[i][2] = (centerX + width / 2)  // xmax
            decodedBoxes[i][3] = (centerY + height / 2) // ymax
        }

        return decodedBoxes
    }

    private fun priorBox(imageSizes: List<Float>, minSizes: List<List<Float>>, steps: List<Float>, ): Array<FloatArray> {
        val featureMaps = steps.map { step ->
            listOf(
                ceil(imageSizes[0] / step),
                ceil(imageSizes[1] / step)
            )
        }

        val anchors = Array(DET_NUM_RESULTS) {FloatArray(4)}
        var cnt = 0
        for ((k, f) in featureMaps.withIndex()) {
            for (i in 0 until f[0].toInt()) {
                for (j in 0 until f[1].toInt()) {
                    for (minSize in minSizes[k]) {
                        val sKx = minSize / imageSizes[1]
                        val sKy = minSize / imageSizes[0]
                        val cx = (j + 0.5) * steps[k] / imageSizes[1]
                        val cy = (i + 0.5) * steps[k] / imageSizes[0]
                        anchors[cnt][0] = cx.toFloat()
                        anchors[cnt][1] = cy.toFloat()
                        anchors[cnt][2] = sKx
                        anchors[cnt++][3] = sKy
                    }
                }
            }
        }

        return anchors
    }


    fun detectorPredict(inputTensors : List<Tensor>, outputTensors: List<Tensor>): List<FaceBoxPrediction> {
        // inference
        engine.requestSync(faceDetectorModel, inputTensors, outputTensors, option)
        return detectorPostProcess(outputTensors)
    }

    private fun filterBoxesBySize(boxes: ArrayList<FaceBoxPrediction>): ArrayList<FaceBoxPrediction> {
        val returnBoxes = ArrayList<FaceBoxPrediction>()
        for (box in boxes){
            if (box.box.width() * box.box.height() < SIZE_RATIO){
                returnBoxes.add(box)
            }
        }
        return returnBoxes
    }

    private fun boxArea(box: RectF): Float {
        return box.width() * box.height()
    }
    private fun boxIntersection(boxA: RectF, boxB: RectF): Float {
        if (boxA.right <= boxB.left || boxB.right <= boxA.left) return 0f
        if (boxA.bottom <= boxB.top || boxB.bottom <= boxA.top) return 0f

        return (min(boxA.right, boxB.right) - max(boxA.left, boxB.left)) *
                (min(boxA.bottom, boxB.bottom) - max(boxA.top, boxB.top))
    }
    private fun boxUnion(boxA: RectF, boxB: RectF): Float {
        return boxArea(boxA) + boxArea(boxB) - boxIntersection(boxA, boxB)
    }
    private fun boxIou(boxA: RectF, boxB: RectF): Float{
        return boxIntersection(boxA, boxB) / boxUnion(boxA, boxB)
    }
    private fun nms(boxes: MutableList<FaceBoxPrediction>, iouThreshold: Float): ArrayList<FaceBoxPrediction> {
        val nmsBoxes = ArrayList<FaceBoxPrediction>()

        while(boxes.size > 0){
            boxes.sortByDescending { it.score }
            val currBoxes = ArrayList<FaceBoxPrediction>(boxes)
            val maxBox = currBoxes[0]
            nmsBoxes.add(maxBox)
            boxes.clear()
            assert(boxes.size == 0)

            for( i in 1 until currBoxes.size){
                val currBox = currBoxes[i]
                if(boxIou(maxBox.box, currBox.box) < iouThreshold) {
                    boxes.add(currBox)
                }
            }
        }
        return nmsBoxes
    }

    fun landmarksPredict(inputTensors : List<Tensor>, outputTensors: List<Tensor>, inputSize: Size): ArrayList<Landmark> {
        // inference
        engine.requestSync(faceLandmarksModel, inputTensors, outputTensors, option)

        // post process face landmarks output
        val outputBuffer = FloatArray(LND_NUM_RESULTS * LND_LEN_RESULT)
        val byteBuffer = outputTensors[0].data.order(ByteOrder.nativeOrder()).rewind()
        (byteBuffer as ByteBuffer).asFloatBuffer().get(outputBuffer)

        val faceLandmarksPrediction = ArrayList<Landmark>()

        // Get detected faces
        for(i in 0 until LND_NUM_RESULTS){
            faceLandmarksPrediction.add(
                Landmark(
                    (outputBuffer[i * LND_LEN_RESULT + 0] / inputSize.width) ?: 0f,
                    (outputBuffer[i * LND_LEN_RESULT + 1] / inputSize.height) ?: 0f,
                    outputBuffer[i * LND_LEN_RESULT + 0] ?: 0f
                )
            )
        }
        return faceLandmarksPrediction
    }

    companion object {
        const val DET_NUM_RESULTS = 16800
        const val DET_LEN_RESULT = 16
        const val SCORE_THRESH = 0.2f
        const val SIZE_RATIO = 0.3f
        const val IOU_THRESHOLD = 0.6f

        const val LND_NUM_RESULTS = 468
        const val LND_LEN_RESULT = 3
    }
}
