package kr.ac.snu.band.example.holistic_face_pose

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.util.Log
import android.util.Size
import android.view.View

class FaceDetectionView(context: Context?, attrs: AttributeSet): View(context, attrs) {
    private var rect = RectF(0f, 0f, 0f, 0f)
    private var canvasSize = Size(0, 0)
    private var boxPaint = Paint()
    init {
        boxPaint.color = Color.BLUE
        boxPaint.style = Paint.Style.STROKE
        boxPaint.strokeWidth = 50f
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        Log.d("HYUNSOO", "Drawn box")
        canvas.drawRect(RectF(
            rect.left * canvasSize.width,
            rect.top * canvasSize.height,
            rect.right * canvasSize.width,
            rect.bottom * canvasSize.height
        ), boxPaint)
    }

    fun setCanvasSize(outputSize: Size){
        canvasSize = outputSize
    }

    fun setRect(_rect: RectF){
        rect = _rect
    }
}