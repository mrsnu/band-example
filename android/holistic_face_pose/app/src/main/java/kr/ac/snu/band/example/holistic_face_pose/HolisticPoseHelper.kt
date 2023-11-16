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

import android.graphics.RectF
import android.util.Log
import android.util.Size

import org.mrsnu.band.Engine
import org.mrsnu.band.Model
import org.mrsnu.band.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Helper class used to communicate between our app and the Band detection model
 */
class HolisticPoseHelper(private val engine: Engine, private val poseDetectorModel: Model, private val poseLandmarksModel: Model, private val labels: List<String>) {

    data class PosePrediction(val box: RectF, val label: String, val score: Float)

    fun detectorPredict(inputTensors: List<Tensor>, outputTensors: List<Tensor>): ArrayList<PosePrediction> {
    engine.requestSync(poseDetectorModel, inputTensors, outputTensors)
    val outputBuffer = mutableMapOf<Int, FloatArray>(
        0 to FloatArray(4 * OBJECT_COUNT),
        1 to FloatArray(OBJECT_COUNT),
        2 to FloatArray(OBJECT_COUNT),
        3 to FloatArray(1)
    )

    outputTensors.forEachIndexed { index, tensor ->
        // byteBuffer to floatArray
        val byteBuffer = tensor.data.order(ByteOrder.nativeOrder()).rewind()
        (byteBuffer as ByteBuffer).asFloatBuffer().get(outputBuffer[index])
    }

    var objects = ArrayList<PosePrediction>(OBJECT_COUNT)
    (0 until OBJECT_COUNT).map {
        val locationBuffer = outputBuffer[0]
        val labelBuffer = outputBuffer[1]
        val scoreBuffer = outputBuffer[2]

        val pose = PosePrediction(
            // The locations are an array of [0, 1] floats for [top, left, bottom, right]
            box = RectF(
                locationBuffer?.get(4 * it + 1) ?: 0f,
                locationBuffer?.get(4 * it) ?: 0f,
                locationBuffer?.get(4 * it + 3) ?: 0f,
                locationBuffer?.get(4 * it + 2) ?: 0f
            ),

            // SSD Mobilenet V1 Model assumes class 0 is background class
            // in label file and class labels start from 1 to number_of_classes + 1,
            // while outputClasses correspond to class index from 0 to number_of_classes
            label = labels[1 + (labelBuffer?.get(it)?.toInt() ?: 0)],

            // Score is a single value of [0, 1]
            score = scoreBuffer?.get(it) ?: 0f
        )
        if (pose.score >= SCORE_THRESH && pose.label == "person"){
            objects.add(pose)
        }

    }
    objects = filterBoxesBySize(objects)
    return objects
    }

    fun landmarksPredict(inputTensors: List<Tensor>, outputTensors: List<Tensor>): ArrayList<HolisticFaceHelper.Landmark> {
        // inference
        engine.requestSync(poseLandmarksModel, inputTensors, outputTensors)

        // post-process
        val outputBuffer = FloatArray(LND_NUM_RESULTS * LND_LEN_RESULT)
        val byteBuffer = outputTensors[0].data.order(ByteOrder.nativeOrder()).rewind()
        (byteBuffer as ByteBuffer).asFloatBuffer().get(outputBuffer)

        val poseLandmarksPrediction = ArrayList<HolisticFaceHelper.Landmark>()

        for(i in 0 until LND_NUM_RESULTS){
            poseLandmarksPrediction.add(
                HolisticFaceHelper.Landmark(
                    outputBuffer[i * LND_LEN_RESULT + 1] ?: 0f,
                    outputBuffer[i * LND_LEN_RESULT + 0] ?: 0f,
                    outputBuffer[i * LND_LEN_RESULT + 2] ?: 0f // confidence
                )
            )
        }
        return poseLandmarksPrediction
    }

    private fun filterBoxesBySize(boxes: ArrayList<PosePrediction>): ArrayList<PosePrediction> {
        val returnBoxes = ArrayList<PosePrediction>()
        for (box in boxes){
            if (box.box.width() * box.box.height() < HolisticFaceHelper.SIZE_RATIO){
                returnBoxes.add(box)
            }
        }
        return returnBoxes
    }


    companion object {
        const val SCORE_THRESH = 0.5f
        const val OBJECT_COUNT = 10

        const val LND_NUM_RESULTS = 17
        const val LND_LEN_RESULT = 3
    }
}
