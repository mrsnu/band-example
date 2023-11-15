package kr.ac.snu.band.example.holistic_face_pose

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.util.Log
import android.view.View

class LandmarkView(context: Context?, attrs: AttributeSet): View(context, attrs) {
    private var points = ArrayList<HolisticFaceHelper.Landmark>()
    private var shiftRect = RectF(0f, 0f, 0f, 0f)
    private var pointPaint = Paint()
    init {
        pointPaint.color = Color.RED
        pointPaint.style = Paint.Style.FILL
        pointPaint.strokeWidth = 50f
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        var i = 0
        canvas.drawRect(shiftRect, pointPaint)
        for (point in points) {
            canvas.drawCircle(point.x + shiftRect.width(), point.y + shiftRect.height(), 5f, pointPaint)
            i += 1
        }
    }

    fun setLandmarks(_points: ArrayList<HolisticFaceHelper.Landmark>, _shiftRect: RectF){
        points = _points
        shiftRect = _shiftRect
    }
}