from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(1,'data/retina/train','image','label',data_gen_args,save_to_dir = None)

model = unet()
model.load_weights("unet_retina_new.hdf5")
#model_checkpoint = ModelCheckpoint('unet_retina_new2.hdf5', monitor='loss',verbose=1, save_best_only=True)
#model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])
#
testGene = testGenerator("data/retina/test/image/")
results = model.predict_generator(testGene,20,verbose=1)
saveResult("data/retina/view",results)

#ssh -i ~/.ssh/id_rsa kajal@34.67.147.119