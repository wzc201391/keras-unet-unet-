from data import trainGenerator, testGenerator, saveResult
from model import *
from data import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
#'''#train'''
#data_gen_args = dict(rotation_range=0.2,
#                     width_shift_range=0.05,
#                     height_shift_range=0.05,
#                     shear_range=0.05,
#                     zoom_range=0.05,
#                     horizontal_flip=True,
#                     fill_mode='nearest')
#myGene = trainGenerator(10,r'./data/membrane/train','image3','label3',data_gen_args,save_to_dir = None)#这里10代表每批次读取多少图
#
#from keras import backend as K
#K.clear_session()
##model = unet()
#model = Nest_Net(256,256,1)
#model_checkpoint = ModelCheckpoint('nestnet_membrane_2020-05-30.hdf5', monitor='loss',verbose=1, save_best_only=True)
#model.fit_generator(myGene,steps_per_epoch=350,epochs=30,callbacks=[model_checkpoint])
#
#model.save('nestnet_filed_mode2020-05-30-3.h5')

'''#test'''
model = load_model('nestnet_filed_mode2020-05-30-3.h5')

testGene = testGenerator("data/membrane/DSC_0260_slide/DSC_0260slide")
results = model.predict_generator(testGene,15)#40为测试图片数量
saveResult("data/membrane/DSC_0260_slide/DSC_0260slide",results)

