<!-- ABOUT THE PROJECT -->
## Unreal Engine 4 Example for Band

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#dependency">Dependency</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


### Dependency

* [Band](https://github.com/mrsnu/tflite)
* [Band UE4 Plugin](https://github.com/mrsnu/band-ue)
* [Android Camera Plugin](https://github.com/snuhcs/android-camera-ue)
* [Unreal Engine 4](https://www.unrealengine.com/en-US/)

<!-- GETTING STARTED -->
### Getting Started

#### Prerequisites

1. Unreal Engine version 4.27.2 or above (preferred to build from source)
2. Android Studio version 4.0.0


#### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/mrsnu/band-example.git
   cd band-example
   ```
2. Update submodules
   ```sh
   git submodule update --init --recursive
   ```
3. Re-generate visual studio project of root .uproject

#### How to start
1. Launch BandExample.uproject in Unreal Engine
2. Open one of the example maps in Content/Maps:

- **Band_Classification_BP**: Example of using Band plugin with Blueprint-based mobile augmented reality classificaiton application
- **Band_Detection_BP**: Example of using Band plugin with Blueprint-based mobile augmented reality object detection application
- **Band_Detection_C++**: Example of using Band plugin with C++-based mobile augmented reality object detection application

3. Build and run the application on Android device

<!-- CONTACT -->
### Contact

Jingyu Lee - jingyu.lee@hcs.snu.ac.kr


### Citation

If you find our work useful, please cite our paper below!

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Android Camera](https://forums.unrealengine.com/t/plugin-android-camera/69320)
* [Tensorflow Lite Support](https://github.com/tensorflow/tflite-support)
