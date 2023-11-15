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
import android.util.Size

import org.mrsnu.band.Engine
import org.mrsnu.band.Model
import org.mrsnu.band.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Helper class used to communicate between our app and the Band detection model
 */
class HolisticPoseHelper(private val engine: Engine, private val poseDetectorModel: Model, private val poseLandmarksModel: Model) {

    data class ObjectPrediction(val box: RectF, val label: String, val score: Float)

////////////////////////////////////////////////////////////////////////////////////////////////////////////
    fun detectorPredict(inputTensors: List<Tensor>, outputTensors: List<Tensor>): ArrayList<ObjectPrediction> {
        // inference
        engine.requestSync(poseDetectorModel, inputTensors, outputTensors)

        // post-process
        val boxResults = FloatArray(DET_NUM_RESULTS * DET_LEN_RESULT)
        val boxByteBuffer = outputTensors[0].data.order(ByteOrder.nativeOrder()).rewind()
        (boxByteBuffer as ByteBuffer).asFloatBuffer().get(boxResults)

        val confResults = FloatArray(DET_NUM_RESULTS)
        val confByteBuffer = outputTensors[2].data.order(ByteOrder.nativeOrder()).rewind()
        (confByteBuffer as ByteBuffer).asFloatBuffer().get(confResults)

        val classResults = FloatArray(DET_NUM_RESULTS)
        val classByteBuffer = outputTensors[1].data.order(ByteOrder.nativeOrder()).rewind()
        (classByteBuffer as ByteBuffer).asFloatBuffer().get(classResults)

        var objects = ArrayList<ObjectPrediction>()
        for (i in 0 until DET_NUM_RESULTS){
            val confidence = confResults[i]
            if(confidence > SCORE_THRESH && classResults[i].toInt() == 0){
                objects.add(
                    ObjectPrediction(
                        box = RectF(
                            boxResults[i * DET_LEN_RESULT + 1] ?: 0f,
                            boxResults[i * DET_LEN_RESULT + 0] ?: 0f,
                            boxResults[i * DET_LEN_RESULT + 3] ?: 0f,
                            boxResults[i * DET_LEN_RESULT + 2] ?: 0f
                        ),
                        score = confidence,
                        label = "person"
                    )
                )
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
                    outputBuffer[i * LND_LEN_RESULT + 0] ?: 0f,
                    outputBuffer[i * LND_LEN_RESULT + 1] ?: 0f,
                    outputBuffer[i * LND_LEN_RESULT + 2] ?: 0f
                )
            )
        }
        return poseLandmarksPrediction
    }

    private fun filterBoxesBySize(boxes: ArrayList<ObjectPrediction>): ArrayList<ObjectPrediction> {
        val returnBoxes = ArrayList<ObjectPrediction>()
        for (box in boxes){
            if (box.box.width() * box.box.height() < HolisticFaceHelper.SIZE_RATIO){
                returnBoxes.add(box)
            }
        }
        return returnBoxes
    }


    companion object {
        const val DET_NUM_RESULTS = 20
        const val DET_LEN_RESULT = 4
        const val SCORE_THRESH = 0.5f

        const val LND_NUM_RESULTS = 17
        const val LND_LEN_RESULT = 3
    }
}
