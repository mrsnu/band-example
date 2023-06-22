// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "BandLabel.h"
#include "BandEnum.h"
#include "BandModel.h"
#include "CoreMinimal.h"
#include "AndroidCameraComponent.h"
#include "BandBoundingBox.h"
#include "GameFramework/Actor.h"

#include "DetectionActor.generated.h"

class UImage;
class UBandUIBase;

UCLASS()
class BANDEXAMPLE_API ADetectionActor : public AActor
{
	GENERATED_BODY()

public:
	// Sets default values for this actor's properties
	ADetectionActor();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

public:
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	void OnFrameAvailable(const UAndroidCameraFrame* Frame);
        UFUNCTION()
	void OnTextureAvailable(UTexture2D* Texture);
	void OnEndRequest(int32 JobId, EBandStatus Status);

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	TSoftClassPtr<UBandUIBase> WidgetClass;
	
	UPROPERTY(BlueprintReadOnly, Transient)
	UBandUIBase* Widget = nullptr;
	UPROPERTY(BlueprintReadOnly, Transient)
	UAndroidCameraComponent* AndroidCamera = nullptr;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Band")
	UBandModel* DetectorModel = nullptr;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Band")
	UBandLabel* Label = nullptr;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	TArray<UBandTensor*> DetectorInputTensors;
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	TArray<UBandTensor*> DetectorOutputTensors;

private:
	UImage* CameraImage = nullptr;
};
