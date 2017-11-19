# build regression
## feature extraction
- resnet152
## Loss
- L1_Smooth_loss

# How to use?
- train
 main(mode="train") 
- validate
 main(mode="validate",resume='./model_best.pth.tar')
- test
 main(mode="test", resume='./models/checkpoint.pth.tar') 
- test on one image
 ui_test(filename, model_dir)
```bash
$ python train_regression.py 
```
