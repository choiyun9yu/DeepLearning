# Segmentation

## 1. Segmentation Model의 종류

## 1-1. Semantic Segmentation
- 이미지에서 사로 다른 범주에 해당하는 영역을 구분할 수 있지만 같은 범주에 해당하는 객체는 개별적으로 구분 불가

## 1-2. Instance Segmentation
- 같은 범주에 해당하는 객체 간의 구분은 간으하지만 셀수 없는 영역인 stuff 영역에 대한 분류 불가

## 1-3. Panoptic Segmentation
- Sementic 과 Instance 를 합친 방법으로 셀 수 있는 intance 객체에 대해서는 Instance Segmentation 방식을 사용하고,
  하늘, 바닥과 같은 셀 수 없는 stuff 영역에 대해서는 Sementic Segmentation 방식을 사용하여 이미지 내부에 존재하는 모든 객체와 영역에 대해 분류 가능 
