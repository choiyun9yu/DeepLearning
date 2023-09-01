# Button Crack Detection

## 1. Data Download

    import gdown

    # file download
    file_id = "16L1zxYt01jb3ed42-W6XIzQyB2jKtRJh"
    output = "데이터 저장 경로/Fabric_Data.zip" # 저장 위치 및 저장할 파일 이름
    gdown.download(id=file_id, output=output, quiet=False)

    # zip file unzip
    $ unzip "/content/drive/MyDrive/data/Fabric_Data.zip"

## 2. Data Labeling
### 2-1. Install locally with pip

    # Requires Python >=3.8
    $ pip install label-studio

    # Start the server at http://localhost:8080
    $ label-studio

## 2-2. How to use label-studio


### 3. 