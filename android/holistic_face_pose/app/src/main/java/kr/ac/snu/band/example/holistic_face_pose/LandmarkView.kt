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

class LandmarkView(context: Context?, attrs: AttributeSet): View(context, attrs) {
    private var points = ArrayList<HolisticFaceHelper.Landmark>()
    private var canvasSize = Size(0, 0)
    private var shiftRect = RectF(0f, 0f, 0f, 0f)
    private var pointPaint = Paint()
    private var boxPaint = Paint()
    private var option = "pose"

    init {
        pointPaint.color = Color.RED
        pointPaint.style = Paint.Style.FILL
        pointPaint.strokeWidth = 10f

        pointPaint.color = Color.RED
        boxPaint.style = Paint.Style.STROKE
        boxPaint.strokeWidth = 10f
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        var i = 0
        val resizedRect = RectF(shiftRect.left * canvasSize.width, shiftRect.top * canvasSize.height,
            shiftRect.right * canvasSize.width, shiftRect.bottom * canvasSize.height)
        canvas.drawRect(resizedRect, boxPaint)

        if(option == "face"){
            for (point in points) {
                val x = point.x * resizedRect.width() + resizedRect.left
                val y = point.y * resizedRect.height() + resizedRect.top
                canvas.drawCircle(x, y, 5f, pointPaint)
                i += 1
            }
        }
        else if (option == "pose"){
            for (point in points) {
                val x = point.x * resizedRect.width() + resizedRect.left
                val y = point.y * resizedRect.height() + resizedRect.top
                canvas.drawCircle(x, y, 5f, pointPaint)
                i += 1
            }
        }
    }

    fun setCanvasSize(outputSize: Size, drawOption: String){
        canvasSize = outputSize
        option = drawOption
        if(option == "pose"){
            boxPaint.color = Color.BLUE
            pointPaint.color = Color.BLUE
        }
    }

    fun setLandmarks(_points: ArrayList<HolisticFaceHelper.Landmark>, _shiftRect: RectF){
        points = _points
        shiftRect = _shiftRect
    }
}