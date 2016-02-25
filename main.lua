data = require 'data'
dir = 'captcha/'
X,Y = data.storeXY(dir,50,170,'captchaImage.')
X,Y = data.loadXY(dir)
Xt,Yt,Xv,Yv = data.split(X,Y,1000)
models = require 'models'
net,ct = models.cnnModel()
net = net:cuda()
ct = ct:cuda()
batchSize = 16
train = require 'train'
sgd_config = {
	   learningRate = 0.1,
	      momentum = 0.9,
}
train.sgd(net,ct,Xt,Yt,Xv,Yv,20,sgd_config,batchSize)
