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

import android.graphics.RectF

import org.mrsnu.band.Engine
import org.mrsnu.band.Model
import org.mrsnu.band.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

/**
 * Helper class used to communicate between our app and the Band detection model
 */
class FaceDetectionHelper(private val engine: Engine, private val model: Model) {

    fun predict(inputTensor: List<Tensor>, outputTensor: List<Tensor>): List<BoundingBox> {
        engine.requestSync(model, inputTensor, outputTensor)

        var rawResults = outputTensor[0].data.order(ByteOrder.nativeOrder()).rewind()
        rawResults = (rawResults as ByteBuffer).asFloatBuffer()

        val boxes = ArrayList<BoundingBox>()

        for (i in 0 until OUTPUT_SHAPE[0]) {
            val offset = i * OUTPUT_SHAPE[1]
            val confidence = rawResults[offset + OUTPUT_SHAPE[1] - 1]
            if (confidence > SCORE_THRESHOLD) {
                boxes.add(BoundingBox(
                    RectF(
                        rawResults[offset + 0],
                        rawResults[offset + 1],
                        rawResults[offset + 2],
                        rawResults[offset + 3]
                    ),
                    confidence
                ))
            }

        }
        return nms(boxes)

    }

    class BoundingBox(val location: RectF, val confidence: Float) {
        fun iou (other: BoundingBox) : Float {
            return intersection(other) / union(other)
        }

        private fun intersection (other: BoundingBox) : Float {
            if (location.left < other.location.left || location.right > other.location.right ||
                location.top < other.location.top || location.bottom > other.location.bottom) {
                return 0f
            }
            return (min(location.right, other.location.right) - max(location.left, other.location.left)) *
                    (min(location.bottom, other.location.bottom) - max(location.top, other.location.top))
        }

        private fun union (other: BoundingBox) : Float {
            val intersection: Float = intersection(other)
            val area1 = (location.right - location.left) * (location.bottom - location.top)
            val area2 =
                (other.location.right - other.location.left) * (other.location.bottom - other.location.top)
            return area1 + area2 - intersection
        }
    }

    private fun nms(boxes: ArrayList<BoundingBox>): ArrayList<BoundingBox> {
        val nmsBoxes = ArrayList<BoundingBox>()
        val prevBoxes = ArrayList<BoundingBox>(boxes)

        while (prevBoxes.size > 0) {
            prevBoxes.sortBy { it.confidence }
            val currBoxes = ArrayList<BoundingBox>(prevBoxes)
            val maxBox = currBoxes[0];
            nmsBoxes.add(maxBox)
            prevBoxes.clear()

            for (i in 1 until currBoxes.size) {
                val detection = currBoxes[i]
                if (maxBox.iou(detection) < IOU_THRESHOLD)
                    prevBoxes.add(detection)
            }
        }
        return nmsBoxes
    }


    companion object {
        val OUTPUT_SHAPE : List<Int> = listOf(16800, 16)
        const val SCORE_THRESHOLD : Float = 0.75f
        const val IOU_THRESHOLD : Float = 0.3f
    }
}
