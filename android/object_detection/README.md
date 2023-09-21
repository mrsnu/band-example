# Camera + Band (Object Detection)
This is a starter project for camera-based Android applicafion with Band. It is based on the [CameraX](https://developer.android.com/training/camerax)

This example implements an Activity that performs real-time processing on the live camera frames. It expects to perform the following operations:

1. Initializes camera preview and image analysis frame streams using CameraX
2. Loads models and initializes the Band engine
3. Performs inference on the transformed frames and reports the object predicted on the screen

## How to use this example as a template
1. Clone this folder and change the application ID in `build.gradle` to your own application ID
(You can find the application ID in the `build.gradle` file of your app module. For example, `com.example.myapp`). 
2. Update the package name in `app/src/main/AndroidManifest.xml` to your own package name
3. Change the package name in `app/src/main/*` to your own package name and update the directory structure accordingly (e.g., `src/main/java/com/example/myapp`)
4. Change the application name in `app/src/main/res/values/strings.xml` to your own application name
5. Start developing your own application!

## Reference
This example is heavily based on [Android Camera Samples](https://github.com/android/camera-samples)
