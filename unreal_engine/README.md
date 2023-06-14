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
* [Band UE4 Plugin](https://github.com/mrsnu/ue4-plugin)
* [Unreal Engine 4](https://www.unrealengine.com/en-US/)

<!-- GETTING STARTED -->
### Getting Started

#### Prerequisites

1. Unreal Engine version 4.27.2 or above (preferred to build from source)
2. Android Studio version 4.0.0


#### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/mrsnu/ue4-example.git
   cd ue4-example
   ```
2. Update submodules
   ```sh
   git submodule update --init --recursive
   ```
3. Re-generate visual studio project of root .uproject

#### How to visualize timeline trace
1. Launch android applciation in editor
2. Run preparation script
   ``` sh
   cd Scripts
   ./PrepareInsights.bat
   ```
3. Run [Unreal Insights](https://docs.unrealengine.com/4.27/en-US/TestingAndOptimization/PerformanceAndProfiling/UnrealInsights/) to attach live session with band traces

<!-- CONTACT -->
### Contact

Jingyu Lee - jingyu.lee@hcs.snu.ac.kr


### Citation

If you find our work useful, please cite our paper below!

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Android Camera](https://forums.unrealengine.com/t/plugin-android-camera/69320)
* [Tensorflow Lite Support](https://github.com/tensorflow/tflite-support)
