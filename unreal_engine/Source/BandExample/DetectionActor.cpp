// Fill out your copyright notice in the Description page of Project Settings.

#include "DetectionActor.h"
#include "BandBlueprintLibrary.h"
#include "BandUIBase.h"
#include "Components/Image.h"

// Sets default values
ADetectionActor::ADetectionActor()
{
	// Set this actor to call Tick() every frame.  You can turn this off to
	// improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = false;
	AndroidCamera =
		CreateDefaultSubobject<UAndroidCameraComponent>(TEXT("AndroidCamera"));
}

// Called when the game starts or when spawned
void ADetectionActor::BeginPlay()
{
	DetectorInputTensors =
		DetectorModel->AllocateInputTensors();
	DetectorOutputTensors =
		DetectorModel->AllocateOutputTensors();

	TSubclassOf<UBandUIBase> WidgetClassType = WidgetClass.LoadSynchronous();
	Widget = CreateWidget<UBandUIBase>(GetWorld(), WidgetClassType);
	Widget->AddToViewport();

	CameraImage = Cast<UImage>(
		Widget->GetWidgetFromName("CameraImage"));
	GetGameInstance()->GetSubsystem<UBandSubSystem>()
	                 ->OnEndInvoke.AddUObject(
		                 this, &ADetectionActor::OnEndRequest);

	AndroidCamera->StartCamera(640, 640, 30);
	AndroidCamera->OnFrameAvailable.AddUObject(
		this, &ADetectionActor::OnFrameAvailable);

	Super::BeginPlay();
}

void ADetectionActor::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	if (AndroidCamera)
	{
		AndroidCamera->OnFrameAvailable.RemoveAll(this);
	}

	Super::EndPlay(EndPlayReason);
}

// Called every frame
void ADetectionActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
}

void ADetectionActor::OnFrameAvailable(const UAndroidCameraFrame* Frame)
{
	DetectorInputTensors[0]->FromCameraFrame(Frame, 0.f, 1.f);
	FBandModule::Get().RequestAsync(DetectorModel, DetectorInputTensors);
}

void ADetectionActor::OnTextureAvailable(UTexture2D* Texture)
{
	AsyncTask(ENamedThreads::GameThread, [&]()
	{
		CameraImage->SetBrushFromTexture(Texture);
	});
}

void ADetectionActor::OnEndRequest(int32 JobId, EBandStatus Status)
{
	if (Status == EBandStatus::Ok)
	{
		check(FBandModule::Get().GetOutputs(JobId, DetectorOutputTensors) ==
			Status);
		FString ClassLabel = UBandBlueprintLibrary::GetLabel(
			DetectorOutputTensors, Label);
		Widget->BoundingBoxes = UBandBlueprintLibrary::GetDetectedBoxes(
			DetectorOutputTensors, EBandDetector::SSDMNetV2, Label);
	}
}
