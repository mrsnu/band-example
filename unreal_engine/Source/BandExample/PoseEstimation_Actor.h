// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "BandLabel.h"
#include "BandModel.h"
#include "CoreMinimal.h"
#include "AndroidCameraComponent.h"
#include "BandInterfaceComponent.h"
#include "GameFramework/Actor.h"

#include "PoseEstimation_Actor.generated.h"

UCLASS()
class BANDEXAMPLE_API APoseEstimation_Actor : public AActor {
  GENERATED_BODY()
 public:
  // Sets default values for this actor's properties
  APoseEstimation_Actor();

 protected:
  // Called when the game starts or when spawned
  virtual void BeginPlay() override;

 public:
  // Called every frame
  virtual void Tick(float DeltaTime) override;

  // Returns the n'th input and output tensors for the landmark model.
  TArray<UBandTensor*>& GetLandmarkInputTensors(size_t Index);
  TArray<UBandTensor*>& GetLandmarkOutputTensors(size_t Index);

  
  UPROPERTY(BlueprintReadOnly, Transient)
  UAndroidCameraComponent* AndroidCamera = nullptr;
  UPROPERTY(BlueprintReadOnly, Transient)
  UBandInterfaceComponent* BandInterpreter = nullptr;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Band")
  UBandModel* DetectorModel = nullptr;
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Band")
  UBandModel* LandmarkModel = nullptr;
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Band")
  UBandLabel* Label = nullptr;

  TArray<UBandTensor*> DetectorInputTensors;
  TArray<UBandTensor*> DetectorOutputTensors;

  TArray<TArray<UBandTensor*>> LandmarkInputTensors;
  TArray<TArray<UBandTensor*>> LandmarkOutputTensors;
};
