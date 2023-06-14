/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <memory>
#include <vector>
#include <memory>

#include "Common.h"
#include "FrameBuffer.h"
#include "FrameBufferUtilsInterface.h"

namespace band {
// Returns the minimal buffer size for a plane in bytes based on the given
// format and dimensions.
BAND_API int GetBufferByteSize(FrameBuffer::Dimension dimension,
                               FrameBuffer::Format format);

// Rotates the `from_box` in `from_orientation` to `to_orientation` within an
// image of size `from_dimension`.
BAND_API BoundingBox OrientBoundingBox(const BoundingBox& from_box,
                                       FrameBuffer::Orientation
                                       from_orientation,
                                       FrameBuffer::Orientation to_orientation,
                                       FrameBuffer::Dimension from_dimension);

// Same as OrientBoundingBox but from normalized coordinates.
BAND_API BoundingBox OrientAndDenormalizeBoundingBox(
    float from_left, float from_top, float from_right, float from_bottom,
    FrameBuffer::Orientation from_orientation,
    FrameBuffer::Orientation to_orientation,
    FrameBuffer::Dimension from_dimension);

// Rotates `(from_x, from_y)` coordinates from an image of dimension
// `from_dimension` and orientation `from_orientation` into `(to_x, to_y)`
// coordinates with orientation `to_orientation`.
BAND_API void OrientCoordinates(int from_x, int from_y,
                                FrameBuffer::Orientation from_orientation,
                                FrameBuffer::Orientation to_orientation,
                                FrameBuffer::Dimension from_dimension,
                                int* to_x,
                                int* to_y);

// Returns whether the conversion from from_orientation to to_orientation
// requires 90 or 270 degrees rotation.
BAND_API bool RequireDimensionSwap(FrameBuffer::Orientation from_orientation,
                                   FrameBuffer::Orientation to_orientation);

// Structure to express parameters needed to achieve orientation conversion.
struct BAND_API OrientParams {
  // Counterclockwise rotation angle in degrees. This is expressed as a
  // multiple of 90 degrees.
  int rotation_angle_deg;

  // Flipping operation. It must come after the rotation.
  enum class FlipType { kNo, kHorizontal, kVertical };

  FlipType flip;
};

// Returns rotation angle and the need for horizontal flipping or vertical
// flipping.
OrientParams GetOrientParams(FrameBuffer::Orientation from_orientation,
                             FrameBuffer::Orientation to_orientation);

struct BAND_API FrameBufferOperation {
  virtual ~FrameBufferOperation() {
  }

  // Use Enum-based type checking to avoid RTTI
  enum class Type {
    kCropResize = 0,
    kUniformCropResize = 1,
    kConvert = 2,
    kOrient = 3
  };

  virtual Type IsType() const = 0;
};

// The parameters needed to crop / resize.
//
// The coordinate system has its origin at the upper left corner, and
// positive values extend down and to the right from it.
//
// After the operation, the `crop_origin` will become the new origin.
// `crop_width` and `crop_height` defines the desired cropping region. After
// cropping, a resize is performed based on the `resize_width` and
// `resize_height`.
//
// To perform just cropping, the `crop_width` and `crop_height` should be the
// same as `resize_width` `and resize_height`.
struct BAND_API CropResizeOperation : public FrameBufferOperation {
  CropResizeOperation(int crop_origin_x, int crop_origin_y,
                      FrameBuffer::Dimension crop_dimension,
                      FrameBuffer::Dimension resize_dimension)
    : crop_origin_x(crop_origin_x),
      crop_origin_y(crop_origin_y),
      crop_dimension(crop_dimension),
      resize_dimension(resize_dimension) {
  }

  virtual Type IsType() const override { return Type::kCropResize; }

  int crop_origin_x;
  int crop_origin_y;
  FrameBuffer::Dimension crop_dimension;
  FrameBuffer::Dimension resize_dimension;
};

// The parameters needed to crop / resize / pad.
//
// The coordinate system has its origin at the upper left corner, and
// positive values extend down and to the right from it.
//
// After the operation, the `crop_origin` will become the new origin.
// `crop_width` and `crop_height` defines the desired cropping region. After
// cropping, a resize is performed based on the `resize_width` and
// `resize_height`.
//
// To perform just cropping, the `crop_width` and `crop_height` should be the
// same as `resize_width` `and resize_height`.
//
// The cropped region is resized uniformly (respecting the aspect ratio) to best
// match the size of the given `output_dimension` in both x and y dimensions.
// The resized region is aligned to the upper left pixel of the output buffer.
// The unfilled area of the output buffer remains untouched.
struct BAND_API UniformCropResizeOperation : public FrameBufferOperation {
  UniformCropResizeOperation(int crop_origin_x, int crop_origin_y,
                             FrameBuffer::Dimension crop_dimension,
                             FrameBuffer::Dimension output_dimension)
    : crop_origin_x(crop_origin_x),
      crop_origin_y(crop_origin_y),
      crop_dimension(crop_dimension),
      output_dimension(output_dimension) {
  }

  virtual Type IsType() const override { return Type::kUniformCropResize; }

  int crop_origin_x;
  int crop_origin_y;
  FrameBuffer::Dimension crop_dimension;
  FrameBuffer::Dimension output_dimension;
};

// The parameters needed to convert to the specified format.
struct BAND_API ConvertOperation : public FrameBufferOperation {
  explicit ConvertOperation(FrameBuffer::Format to_format)
    : to_format(to_format) {
  }

  virtual Type IsType() const override { return Type::kConvert; }
  FrameBuffer::Format to_format;
};

// The parameters needed to change the orientation.
struct OrientOperation : public FrameBufferOperation {
  explicit OrientOperation(FrameBuffer::Orientation to_orientation)
    : to_orientation(to_orientation) {
  }

  virtual Type IsType() const override { return Type::kOrient; }
  FrameBuffer::Orientation to_orientation;
};

// Image processing utility. This utility provides both basic image buffer
// manipulations (e.g. rotation, format conversion, resizing, etc) as well as
// capability for chaining pipeline executions. The actual buffer processing
// engine is configurable to allow optimization based on platforms.
//
// Examples:
//
//  // Create an instance of FrameBufferUtils with Halide processing engine.
//  std::unique_ptr<FrameBufferUtils> utils =
//  FrameBufferUtils::Create(kHalide);
//
//  // Perform single basic operation by each individual call.
//  std::unique_ptr<FrameBuffer> input = FrameBuffer::Create(...);
//  std::unique_ptr<FrameBuffer> output = FrameBuffer::Create(...);
//  utils->Orient(*input, output.get());
//  utils->Resize(*input, output.get());
//
//  // Chaining processing operations.
//  const std::vector<FrameBufferOperation> operations = {
//      ConvertOperation(FrameBuffer::Format::kNV21),
//      CropResizeOperation(/*crop_origin_x=*/20, /*crop_origin_y=*/20,
//                          /*crop_width=*/10, /*crop_height=*/10,
//                          /*resize_width=*/10, /*resize_height=*/10),
//      OrientOperation(FrameBuffer::Orientation::kLeftTop)};
//  utils->Execute(*input, operations, output.get());
class BAND_API FrameBufferUtils {
public:
  // Counter-clockwise rotation in degree.
  enum class RotationDegree { k0 = 0, k90 = 1, k180 = 2, k270 = 3 };

  // Underlying process engine used for performing operations.
  enum class ProcessEngine {
    kLibyuv,
  };

  // Factory method FrameBufferUtils instance. The processing engine is
  // defined by `engine`.
  static std::unique_ptr<FrameBufferUtils> Create(ProcessEngine engine) {
    return std::make_unique<FrameBufferUtils>(engine);
  }

  explicit FrameBufferUtils(ProcessEngine engine);

  // Performs cropping operation.
  //
  // The coordinate system has its origin at the upper left corner, and
  // positive values extend down and to the right from it. After cropping,
  // (x0, y0) becomes (0, 0). The new width and height are
  // (x1 - x0 + 1, y1 - y0 + 1).
  //
  // The `output_buffer` should have metadata populated and its backing buffer
  // should be big enough to store the operation result. If the `output_buffer`
  // size dimension does not match with crop dimension, then a resize is
  // automatically performed.
  bool Crop(const FrameBuffer& buffer, int x0, int y0, int x1, int y1,
            FrameBuffer* output_buffer);

  // Performs resizing operation.
  //
  // The resize dimension is determined based on output_buffer's size metadata.
  //
  // The output_buffer should have metadata populated and its backing buffer
  // should be big enough to store the operation result.
  bool Resize(const FrameBuffer& buffer, FrameBuffer* output_buffer);

  // Performs rotation operation.
  //
  // The rotation is specified in counter-clockwise direction.
  //
  // The output_buffer should have metadata populated and its backing buffer
  // should be big enough to store the operation result.
  bool Rotate(const FrameBuffer& buffer, RotationDegree rotation,
              FrameBuffer* output_buffer);

  // Performs horizontal flip operation.
  //
  // The `output_buffer` should have metadata populated and its backing buffer
  // should be big enough to store the operation result.
  bool FlipHorizontally(const FrameBuffer& buffer,
                        FrameBuffer* output_buffer);

  // Performs vertical flip operation.
  //
  // The `output_buffer` should have metadata populated and its backing buffer
  // should be big enough to store the operation result.
  bool FlipVertically(const FrameBuffer& buffer,
                      FrameBuffer* output_buffer);

  // Performs buffer format conversion.
  //
  // The `output_buffer` should have metadata populated and its backing buffer
  // should be big enough to store the operation result.
  bool Convert(const FrameBuffer& buffer, FrameBuffer* output_buffer);

  // Performs buffer orientation conversion. Depends on the orientations, this
  // method may perform rotation and optional flipping operations.
  //
  // If `buffer` and `output_buffer` has the same orientation, then a copy
  // operation will performed.
  //
  // The `output_buffer` should have metadata populated and its backing buffer
  // should be big enough to store the operation result.
  bool Orient(const FrameBuffer& buffer, FrameBuffer* output_buffer);

  // Performs the image processing operations specified, in that order.
  //
  // The `output_buffer` should have metadata populated and its backing buffer
  // should be big enough to store the operation result.
  bool Execute(const FrameBuffer& buffer,
               const std::vector<FrameBufferOperation*>& operations,
               FrameBuffer* output_buffer);

  // Performs a chain of operations to convert `buffer` to desired metadata
  // (width, height, format, orientation) defined by `output_buffer` and
  // optional cropping (`bounding_box`).
  //
  // Internally, a chain of operations is constructed. For performance
  // optimization, operations are performed in the following order: crop,
  // resize, convert color space format, and rotate.
  //
  // The `output_buffer` should have metadata populated and its backing buffer
  // should be big enough to store the operation result. Insufficient backing
  // buffer size may cause garbage result or crash. Use `GetBufferByteSize` to
  // calculate the minimal buffer size.
  //
  // If the `buffer` is already in desired format, then an extra copy will be
  // performed.
  //
  // If `uniform_resizing` is set to true, the source region is resized
  // uniformly (respecting the aspect ratio) to best match the dimension of the
  // given `output_buffer` in both x and y dimensions. The resized region is
  // aligned to the upper left pixel of the output buffer. The unfilled area of
  // the output buffer remains untouched. Default `uniform_resizing` to false;
  //
  // The input param `bounding_box` is defined in the `buffer` coordinate space.
  bool Preprocess(const FrameBuffer& buffer,
                  BoundingBox bounding_box,
                  FrameBuffer* output_buffer,
                  bool uniform_resizing = false);

  bool Preprocess(const FrameBuffer& buffer,
                  FrameBuffer* output_buffer,
                  bool uniform_resizing = false);

private:
  // Returns the new FrameBuffer size after the operation is applied.
  FrameBuffer::Dimension GetSize(const FrameBuffer& buffer,
                                 const FrameBufferOperation& operation);

  // Returns the new FrameBuffer orientation after command is processed.
  FrameBuffer::Orientation GetOrientation(
      const FrameBuffer& buffer, const FrameBufferOperation& operation);

  // Returns the new FrameBuffer format after command is processed.
  FrameBuffer::Format GetFormat(const FrameBuffer& buffer,
                                const FrameBufferOperation& operation);

  // Returns Plane struct based on one dimension buffer and its metadata. If
  // an error occurred, it will return an empty vector.
  std::vector<FrameBuffer::Plane> GetPlanes(const uint8* buffer,
                                            FrameBuffer::Dimension dimension,
                                            FrameBuffer::Format format);

  // Executes command with params.
  bool Execute(const FrameBuffer& buffer,
               const FrameBufferOperation& operation,
               FrameBuffer* output_buffer);

  // Execution engine conforms to FrameBufferUtilsInterface.
  std::unique_ptr<FrameBufferUtilsInterface> utils_;
};
} // namespace Band