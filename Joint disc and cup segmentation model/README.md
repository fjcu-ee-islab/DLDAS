## Joint disc and cup segmentation model training anaconda environment
```
conda env create -f jdcseg.yaml
conda activate jdcseg
```


Downloads [training & test data](https://drive.google.com/file/d/1VFoLdrJ6S63qk4ha-GutqQWbDc7KD5Ex/view?usp=sharing)
## Joint disc and cup segmentation model training & test
### Drishti-GS few labeled data 
downloads training & test data put in ```./data/dgs/few```
```
bash train_dgs_few.sh
```
### Drishti-GS normal labeled data 
downloads training & test data put in ```./data/dgs/paper```
```
bash train_dgs.sh
```
### RIM-ONEv3 few labeled data 
downloads training & test data put in ```./data/rim/few```
```
bash train_rim_few.sh
```
### RIM-ONEv3 normal labeled data 
downloads training & test data put in ```./data/rim/paper```
```
bash train_rim.sh
```
### JSRT few labeled data 
downloads training & test data put in ```./data/jsrt/few```
```
bash train_jsrt_few.sh
```
### JSRT normal labeled data 
downloads training & test data put in ```./data/jsrt/paper```
```
bash train_jsrt.sh
```
## Joint disc and cup segmentation model load best model weight& test
### Drishti-GS few labeled data 
downloads [best model weight](https://drive.google.com/file/d/1b9Xn7zX923hvX_iwzS-7bYvO8030cKUH/view?usp=sharing) put in ```./model```
```
bash test_best_few_dgs.sh
```
### Drishti-GS normal paper data 
downloads [best model weight](https://drive.google.com/file/d/1f_MwgwO-jK99IzILox3t9RvsQCEZST6p/view?usp=sharing) put in ```./model```
```
bash test_best_paper_dgs.sh
```
### RIM-ONEv3 few labeled data 
downloads [best model weight](https://drive.google.com/file/d/1JiXuv9QRc9z9uEwwmt8Ifhc-aHKmcLrX/view?usp=sharing) put in ```./model```
```
bash train_dgs_few.sh
```
### RIM-ONEv3 normal labeled data 
downloads [best model weight](https://drive.google.com/file/d/1TPHP9aQQSfYYYQU6L0FmsjU_moOip0bt/view?usp=sharing) put in ```./model```
```
bash train_dgs_few.sh
```
### JSRT few labeled data 
downloads [best model weight](https://drive.google.com/file/d/10xHs6Y1duHAowU7VHcY_UR7C_wpCW0Nw/view?usp=sharing) put in ```./model```
```
bash train_dgs_few.sh
```
### JSRT norma labeled data 
downloads [best model weight](https://drive.google.com/file/d/1rVpPvVaVVwzkj9kHVvIL8a2qwAk48Awm/view?usp=sharing) put in ```./model```
```
bash train_dgs_few.sh
```
## Joint disc and cup segmentation model test npy
### Drishti-GS few labeled data 
```
cd npy_test
bash npy2img_dgs_few.sh
```
### Drishti-GS normal paper data 
```
cd npy_test
bash npy2img_dgs.sh
```
### RIM-ONEv3 few labeled data 
```
cd npy_test
bash npy2img_rim_few.sh
```
### RIM-ONEv3 normal labeled data 
```
cd npy_test
bash npy2img_rim.sh
```
### JSRT few labeled data 
```
cd npy_test
bash npy2img_jsrt_few.sh
```
### JSRT norma labeled data 
```
cd npy_test
bash npy2img_jsrt.sh
```

