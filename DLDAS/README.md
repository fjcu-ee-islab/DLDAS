## DLDAS anaconda environment
cuda version:9.0\
cudnn version:7.4.2\
```
conda env create -f DLDAS_env.yaml
conda activate dldas
```
Downloads [training data](https://drive.google.com/file/d/1V2GUO2K60UbTYjsG3sHZOy1Ft_stk7_G/view?usp=sharing)
## Train Muti-Domain Texture Generative Adversarial Network
downloads [weight of domain-invariant perceptual loss](https://drive.google.com/file/d/1mtecxy-Mt_uPjJza7uGymh5ZFpGFXyDg/view?usp=sharing).\
```mkdir ./models``` and put in ```./models```
### RIM-ONEv3 content sythesis training
Downloads Data put in ```./dataset/rim```\
Content data put in ```./dataset/rim/trainA``` and ```./dataset/rim/testA```  
Domain Style data put in ```./dataset/rim/trainB``` and random choice 30 samples copy to ```./dataset/rim/testB``` 
```
bash train_rim.sh
```
### Drishti-GS content sythesis training
Downloads Data put in ```./dataset/dgs```\
Content data put in ```./dataset/dgs/trainA``` and ```./dataset/dgs/testA```  
Domain Style data put in ```./dataset/dgs/trainB``` and random choice 30 samples copy to ```./dataset/dgs/testB``` 
```
bash train_dgs.sh
```
### JSRT content sythesis training
Downloads Data put in ```./dataset/jsrt```\
Content data put in ```./dataset/jsrt/trainA``` and ```./dataset/jsrt/testA```  
Domain Style data put in ```./dataset/jsrt/trainB``` and random choice 30 samples copy to ```./dataset/jsrt/testB``` 
```
bash train_jsrt.sh
```
## Synthesis Texture & Structure Data
### RIM-ONEv3 content sythesis
downloads [RIM-ONEv3 sythesize generator pretrained model](https://drive.google.com/file/d/1WYmTcpvP4NqtNB1vQfWOet7hMQphtDvK/view?usp=sharing).\
```mkdir -p output/rim_paper/checkpoints``` and put in ```./output/rim_paper/checkpoints```\
Content data label put in ```./dataset/rim/testA_label```
```
synthesis_rim.sh
```
output to ```./rim_paper_aug```
### Drishti-GS content sythesis
downloads [Drishti-GS sythesize generator pretrained model](https://drive.google.com/file/d/1QjhBvf4Xs7wrisoMSljDN9RC6SUlZePn/view?usp=sharing).\
```mkdir -p output/dgs_paper/checkpoints``` and put in ```./output/dgs_paper/checkpoints```\
Content data label put in ```./dataset/dgs/testA_label```
```
synthesis_dgs.sh
```
output to ```./dgs_paper_aug```
### JSRT content sythesis
downloads [JSRT sythesize generator pretrained model](https://drive.google.com/file/d/1ODqrEh31oDIi_Y7T7LfpESetvkdZ9dmO/view?usp=sharing).\
```mkdir -p output/jsrt_paper/checkpoints``` and put in ```./output/jsrt_paper/checkpoints```\
Content data label put in ```./dataset/jsrt/testA_label```
```
synthesis_jsrt.sh
```
output to ```./jsrt_paper_aug```

