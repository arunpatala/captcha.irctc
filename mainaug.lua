data = require 'data'
dir = 'data/'
X,Y = data.storeXY(dir,50,170,'captchaImage.')
X,Y = data.loadXY(dir)
Xt,Yt,Xv,Yv = data.split(X,Y,50)
models = require 'models'
net,ct = models.augnet(34)
net = net:cuda()
ct = ct:cuda()
batchSize = 16
train = require 'train'
sgd_config = {
	   learningRate = 0.1,
	      momentum = 0.9,
}
trans = require 'transforms'
proc = trans.Compose({
	        trans.CropHW(50,110),
	        trans.CenterCropHW(40,100),
	        trans.Rotation(1)
	})
train.sgd(net,ct,Xt,Yt,Xv,Yv,20,sgd_config,batchSize,prep)
