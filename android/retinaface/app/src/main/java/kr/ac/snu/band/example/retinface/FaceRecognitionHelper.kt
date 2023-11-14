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

import android.util.Log
import org.mrsnu.band.Engine
import org.mrsnu.band.Model
import org.mrsnu.band.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * Helper class used to communicate between our app and the Band detection model
 */
class FaceRecognitionHelper(private val engine: Engine) {

    fun predict(models : List<Model>, inputTensor : List<List<Tensor>>, outputTensor: List<List<Tensor>>): List<FloatArray> {
        // TODO : support multiple faces
        val i = 0

        //engine.requestSync(model, inputTensor, outputTensor)
        //val requests = engine.requestAsyncBatch(models, inputTensor)
        //engine.wait(requests[i], outputTensor[i])


        engine.requestSync(models[i], inputTensor[i], outputTensor[i])
        val outBuffer = outputTensor[i][0].data.order(ByteOrder.nativeOrder())
        val identity = FloatArray(OUTPUT_SHAPE[1])
        (outBuffer as ByteBuffer).asFloatBuffer()[identity]

        // log hash val of identity

        return listOf(identity)
    }

    fun dotProduct(a : FloatArray, b : FloatArray) : Float {
        assert (a.size == b.size)
        var sum = 0f
        for (i in a.indices) {
            sum += a[i] * b[i]
        }
        return sum
    }

    fun norm(a : FloatArray) : Float {
        var sum = 0f
        for (i in a.indices) {
            sum += a[i] * a[i]
        }
        return sqrt(sum.toDouble()).toFloat()
    }

    companion object {
        val OUTPUT_SHAPE : List<Int> = listOf(1, 512)
    }
}
