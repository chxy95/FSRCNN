# FSRCNN
Reproduction of the paper 《Accelerating the Super-Resolution Convolutional Neural Network》（CVPR 2016） by Pytorch and Matlab.
## Dependence
Matlab 2016  
Pytorch 1.0.0  
## Explanation
Some Matlab codes provided by the paper author, url: http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html.  
The main reason for using two languages to do the project is because the different implementation of bicubic interpolation, which causes the broader difference of the results when using PSNR standard. 
## Usage
Use ./data_pro/generate_train.m to generate train.h5.  
Use ./data_pro/generate_test.m to generate test.h5.  
Train by train.py:
```
python train.py
```
Convert the Pytorch model .pkl to Matlab matrix .mat. (weights.pkl -> weights.mat)  
```
python convert.py
```
Use ./test/demo_FSRCNN.m to get the result.
## Result
Use the ./model/weights.mat can get the result:  
Set5 Average：reconstruction PSNR = 32.52dB VS bicubic PSNR = 30.39dB  
Set14 Average: reconstruction PSNR = 29.07dB VS bicubic PSNR = 27.54dB  
Image example:  
<<<<<<< HEAD
<img src="https://raw.githubusercontent.com/chxy95/FSRCNN/master/images/Comparison.png" width="500"/>  
=======
<img src="https://raw.githubusercontent.com/chxy95/FSRCNN/master/images/Comparison1.png" width="500"/>  
>>>>>>> 99a7acbcd94c320957c19aa7b6789ded82e27ed7
<img src="https://raw.githubusercontent.com/chxy95/FSRCNN/master/images/Comparison2.png" width="500"/>
