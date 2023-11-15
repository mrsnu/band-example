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

import org.mrsnu.band.Engine
import org.mrsnu.band.Model
import org.mrsnu.band.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Helper class used to communicate between our app and the Band detection model
 */
class HolisticFacePoseHelper(private val engine: Engine, private val model: Model, private val labels: List<String>) {

    /** Abstraction object that wraps a prediction output in an easy to parse way */
    data class ObjectPrediction(val location: RectF, val label: String, val score: Float)

    fun predict(inputTensor : List<Tensor>, outputTensor: List<Tensor>): List<ObjectPrediction> {
        engine.requestSync(model, inputTensor, outputTensor)

        val outputBuffer = mutableMapOf<Int, FloatArray>(
            0 to FloatArray(4 * OBJECT_COUNT),
            1 to FloatArray(OBJECT_COUNT),
            2 to FloatArray(OBJECT_COUNT),
            3 to FloatArray(1)
        )

        outputTensor.forEachIndexed { index, tensor ->
            // byteBuffer to floatArray
            val byteBuffer = tensor.data.order(ByteOrder.nativeOrder()).rewind()
            (byteBuffer as ByteBuffer).asFloatBuffer().get(outputBuffer[index])
        }

        return (0 until OBJECT_COUNT).map {
            val locationBuffer = outputBuffer[0]
            val labelBuffer = outputBuffer[1]
            val scoreBuffer = outputBuffer[2]

            ObjectPrediction(
                // The locations are an array of [0, 1] floats for [top, left, bottom, right]
                location = RectF(
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
        }
    }

    companion object {
        const val OBJECT_COUNT = 10
    }
}
