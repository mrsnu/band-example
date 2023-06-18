// Fill out your copyright notice in the Description page of Project Settings.

#include "PoseEstimation_Actor.h"

#include "BandInterfaceComponent.h"
#include "BandBlueprintLibrary.h"

// Sets default values
APoseEstimation_Actor::APoseEstimation_Actor() {
  // Set this actor to call Tick() every frame.  You can turn this off to
  // improve performance if you don't need it.
  PrimaryActorTick.bCanEverTick = true;

  AndroidCamera =
      CreateDefaultSubobject<UAndroidCameraComponent>(TEXT("AndroidCamera"));
  BandInterpreter =
      CreateDefaultSubobject<UBandInterfaceComponent>(TEXT("BandInterpreter"));
}

// Called when the game starts or when spawned
void APoseEstimation_Actor::BeginPlay() {
  DetectorInputTensors = BandInterpreter->AllocateInputTensors(DetectorModel);
        DetectorOutputTensors =
        BandInterpreter->AllocateOutputTensors(DetectorModel);

  BandInterpreter->OnEndInvoke.AddStatic()
  
  
  Super::BeginPlay();
}

// Called every frame
void APoseEstimation_Actor::Tick(float DeltaTime) {
  Super::Tick(DeltaTime);
}

TArray<UBandTensor*>& APoseEstimation_Actor::GetLandmarkInputTensors(
    size_t Index) {
  if (LandmarkInputTensors.Num() <= Index) {
    for (size_t i = LandmarkInputTensors.Num(); i <= Index; ++i) {
      LandmarkInputTensors.Add(BandInterpreter->AllocateInputTensors(LandmarkModel));
    }
  }

  return LandmarkInputTensors[Index];
}

TArray<UBandTensor*>& APoseEstimation_Actor::GetLandmarkOutputTensors(
    size_t Index) {
  if (LandmarkOutputTensors.Num() <= Index) {
    for (size_t i = LandmarkOutputTensors.Num(); i <= Index; ++i) {
      LandmarkOutputTensors.Add(BandInterpreter->AllocateOutputTensors(LandmarkModel));
    }
  }

  return LandmarkOutputTensors[Index];
}