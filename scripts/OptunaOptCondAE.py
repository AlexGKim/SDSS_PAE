#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
tf.__version__
tf.config.list_physical_devices('GPU')


# In[20]:


#import tensorflow_datasets as tfds
#import tensorflow_probability as tfp
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import logging
import sys
from sklearn import preprocessing
import optuna
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Layer, Reshape, LeakyReLU, BatchNormalization, Dense, Flatten, Input,Dropout
from optuna.trial import TrialState
import tensorflow_addons as tfa

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

study_folder  = '/global/cscratch1/sd/vboehm/OptunaStudies/'
study_name    = "AE_normed_new"  # Unique identifier of the study.
study_name    = os.path.join(study_folder, study_name)
storage_name  = "sqlite:///{}.db".format(study_name)
SEED          = 512
EPOCHS        = 40
NUM_HOURS     = 6
N_TRIALS      = 500

cond_on         = 'type'
fixed_num_bins  = 1000
dim             = fixed_num_bins

optimizers      = {'Adam': tf.keras.optimizers.Adam, 'SGD':tf.keras.optimizers.SGD , 'RMSprop':tf.keras.optimizers.RMSprop}

param_history   = {'batchsize':[], 'lr_init':[]}

def dense_cond_block(x,z,num, non_lin=True):
    x = tf.concat([x,z], axis=1)
    x = Flatten()(x)
    x = Dense(num)(x)
    if non_lin:
        x = LeakyReLU()(x)
    return Reshape((num,1))(x)

def dense_block(x,num, non_lin=True,spec_norm=False):
    x = Flatten()(x)
    if spec_norm:
        x = tfa.layers.SpectralNormalization(Dense(num))(x)
    else:
        x = Dense(num)(x)
    if non_lin:
        x = LeakyReLU()(x)
    return x



def lossFunction(y_true,y_pred,mask,inverse):
        loss = tf.math.square(y_true-y_pred)*inverse
        loss = tf.reduce_mean(tf.boolean_mask(loss,mask))
        return loss
    
from tensorflow.python.keras.engine import data_adapter


class CustomModel(tf.keras.Model):
    def compile(self, optimizer, my_loss,metrics, run_eagerly):
        super().compile(optimizer,metrics, run_eagerly)
        self.my_loss = my_loss

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        input_data = data_adapter.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(data, training=True)
            loss_value = self.my_loss(input_data[0][0],y_pred,input_data[0][1],input_data[0][2])

        grads = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {"training_loss": loss_value}


def make_scheduler(length, initial_lr,factor=1.2):
    def scheduler(epoch, lr):
        if epoch < length:
            lr = initial_lr
            return lr
        else:
            return lr * tf.math.exp(-factor)
    return scheduler
                             
def training_cycle(BATCH_SIZE, n_epochs, lr_anneal, lr_initial, reduce_fac): 
    scheduler = make_scheduler(lr_anneal, lr_initial, reduce_fac)
    callback  = tf.keras.callbacks.LearningRateScheduler(scheduler)
    history   = lstm_ae.fit(x=(train_data,train_mask,train_noise, train_types, train_params), batch_size=BATCH_SIZE, epochs=n_epochs, callbacks=[callback],verbose=0)
    return history

def custom_metric(y_true, y_pred):
    loss = (y_true[0]-y_pred)**2*y_true[2]
    valid_loss = np.mean(loss[np.where(y_true[1])])
    return valid_loss


# In[16]:


def objective(trial):
    input        = Input(shape=(dim,1))
    input_mask   = Input(shape=(dim,1))
    input_noise  = Input(shape=(dim,1))
    input_type   = Input(shape=(1,1))
    input_params = Input(shape=(1,1))

    if cond_on=='type':
        z = input_type
    if cond_on=='redshift':
        z = input_params

    n_layers   = trial.suggest_int('n_layers', 2, 5)
    latent_dim = trial.suggest_int('latent_dim', 8, 14)
                                               
    x = input
    out_features = []
    for ii in range(n_layers-1):
        if ii>0:
            out_features.append(trial.suggest_int('n_units_l{}'.format(ii), latent_dim, min(dim,2*out_features[-1])))
            p = trial.suggest_float("dropout_encoder_l{}".format(ii), 1e-5, 0.3, log=True)
            x = Dropout(p)(x)
        else:
            out_features.append(trial.suggest_int('n_units_l{}'.format(ii), latent_dim,dim))
        x = dense_block(x,out_features[ii], spec_norm=True)
    x = dense_block(x,latent_dim,non_lin=False, spec_norm=True)
    x = Reshape((latent_dim,1))(x)
    for ii in range(n_layers-1):
        x = dense_cond_block(x,z,out_features[-1-ii])
        if ii ==0:
            pass
        else:
            p = trial.suggest_float("dropout_decoder_l{}".format(ii), 1e-5, 0.3, log=True)
            x = Dropout(p)(x)
    x = dense_cond_block(x,z,dim, non_lin=False)

    lr_initial  = trial.suggest_float("lr_init", 5e-4, 2e-3, log=False)
    lr_end      = trial.suggest_float("lr_final", 5e-6, lr_initial, log=True)
    batchsize   = trial.suggest_int("batchsize", 16, 256)
    decay_steps = trial.suggest_int("decay_steps",2000,40000//batchsize*20,log=True)
    if batchsize in param_history["batchsize"]:
        if lr_initial in param_history['lr_init']:
            raise optuna.exceptions.TrialPruned()  
    
    param_history['batchsize'].append(batchsize)
    param_history['lr_init'].append(lr_initial)  
                                           
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
                                        
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    lr_initial,
    decay_steps,
    lr_end,
    power=0.5, cycle=True)
                                               
    optim = optimizers[optimizer_name]
                                               
    lstm_ae = CustomModel(inputs=[input,input_mask,input_noise, input_type, input_params], outputs=x)
    lstm_ae.compile(optimizer=optim(learning_rate=learning_rate_fn), my_loss=lossFunction, metrics=[],run_eagerly=False)
                                        

    lstm_ae.fit(x=(train['spec'],train['mask'],train['noise'], np.expand_dims(train['subclass'],-1), np.expand_dims(train['z'],-1)), batch_size=batchsize, epochs=EPOCHS,verbose=0)

    res_valid   = lstm_ae.predict((valid['spec'],valid['mask'],valid['noise'], valid['subclass'], valid['z']))
    recon_error = custom_metric((valid['spec'],valid['mask'],valid['noise'], valid['subclass'], valid['z']),res_valid)
    
    return recon_error



RUN             = '1'
EPOCHS          = 20

seeds           = {'1':512, '2':879, '3':9981, '4': 20075, '5': 66, '6': 276, '7': 936664}

conditional     = False
cond_on         = 'type'

root_model_data = '/global/cscratch1/sd/vboehm/Datasets/sdss/by_model/'
root_models     = '/global/cscratch1/sd/vboehm/Models/SDSS_AE/'
root_encoded    = '/global/cscratch1/sd/vboehm/Datasets/encoded/sdss/'
root_decoded    = '/global/cscratch1/sd/vboehm/Datasets/decoded/sdss/'


wlmin, wlmax    = (3388,8318)
fixed_num_bins  = 1000
min_SN          = 50
min_z           = 0.05
max_z           = 0.36
label           = 'galaxies_quasars_bins%d_wl%d-%d'%(fixed_num_bins,wlmin,wlmax)
label_          = label+'_minz%s_maxz%s_minSN%d'%(str(int(min_z*100)).zfill(3),str(int(max_z*100)).zfill(3),min_SN)

train,valid,test,le = pickle.load(open(os.path.join(root_model_data,'combined_%s_new.pkl'%label_),'rb'))                                             



le = preprocessing.LabelEncoder()
le.fit(train['subclass'])
train['subclass'] = le.transform(train['subclass'])
valid['subclass'] = le.transform(valid['subclass'])
test['subclass']  = le.transform(test['subclass'])
print(le.classes_, le.transform(le.classes_))                    

time = NUM_HOURS*60*60-600
study = optuna.create_study(direction='minimize',study_name=study_name, storage=storage_name,load_if_exists=True,  sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
study.optimize(objective, n_trials=N_TRIALS, timeout=time)


pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
