#pragma once
#include "Blueprint/UserWidget.h"
#include "BandBoundingBox.h"
#include "BandUIBase.generated.h"

UCLASS()
class BANDEXAMPLE_API UBandUIBase : public UUserWidget
{
public:
  GENERATED_BODY()

  UPROPERTY(BlueprintReadWrite)
  TArray<FBandBoundingBox> BoundingBoxes;
};
