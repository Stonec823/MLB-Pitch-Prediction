import numpy as np
from numpy import array
from numpy import hstack
from numpy import vstack

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import Hyperband
import kerastuner


# Load raw pitch data with pybaseball
from pybaseball import statcast
df = statcast(start_dt='2021-04-01', end_dt='2021-10-03')


# Include only regular season games
df = df.loc[df['game_type']=='R'].copy()
df.reset_index(drop=True, inplace=True)

# Create unique ID for every plate appearance in every game
df['pa_id'] = df['game_pk'].astype(str)+"-"+df['at_bat_number'].astype(str)

# Fill in the occasional missing value;
# WOBA values should only be missing for pitches that are NOT the final pitch
df['release_speed'].fillna(0, inplace=True)
df['woba_value'].fillna(0, inplace=True)

# Create dictionary to re-group pitch types
pitch_dict = {'FF':0, 'FA':0,
              'FT':1, 'SI':1,
              'FC':2,
              'CU':3,'KC':3,'CS':3,'EP':3,
              'SL':4,
              'CH':5,'FS':5,'FO':5,'SC':5,
              'KN':6,
              'PO':np.nan}

# Map old pitch types to new mapping
df['pitch_type_map'] = df['pitch_type'].map(pitch_dict)
df.dropna(subset=['pitch_type_map'], inplace = True)
df['pitch_type_copy'] = df['pitch_type_map']

df = pd.get_dummies(df, columns=['pitcher','p_throws','stand','pitch_type_map',
                                 'balls','strikes'])


# Only include plate appearances that last at least 3 pitches
data = df[df.groupby('pa_id')['pa_id'].transform('size') >= 3].copy()
data.reset_index(drop=True, inplace=True)


# Function to split data into training and test data after preparing features
def data_splitter(data_seq, train_test_cutoff):
    
    # Take the following variables from the raw data:
    # Pitch velocity, pitcher, pitcher and batter handedness, pitch type, balls and strikes
    # Stack all data and return training and test sets based off cut-off date
        
    train = data_seq.loc[data['game_date'] <= train_test_cutoff].copy()
    train.reset_index(drop=True, inplace=True)
    test = data.loc[data_seq['game_date'] > train_test_cutoff].copy()
    test.reset_index(drop=True, inplace=True)
    
    train.sort_values(by=['pa_id','pitch_number'],inplace=True, ignore_index=True)
    test.sort_values(by=['pa_id','pitch_number'],inplace=True, ignore_index=True)
    
    y_train = np.array(train['pitch_type_copy'])
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = np.array(test['pitch_type_copy'])
    y_test = tf.keras.utils.to_categorical(y_test)

    scaler = StandardScaler()

    pa_id_train = train.pa_id.values.reshape((len(train['pa_id']),1))
    pa_id_test = test.pa_id.values.reshape((len(test['pa_id']),1))

    pitch_velo_train = train.release_speed.values.reshape((len(train['release_speed']),1))
    pitch_velo_train = scaler.fit_transform(pitch_velo_train)
    pitch_velo_test = test.release_speed.values.reshape((len(test['release_speed']),1))
    pitch_velo_test = scaler.transform(pitch_velo_test)
    
    pitcher_train = array(train.loc[:,train.columns.str.startswith('pitcher_')])
    pitcher_test = array(test.loc[:, test.columns.str.startswith('pitcher_')])

    p_throws_train = array(train.loc[:,train.columns.str.startswith('p_throws_')])
    p_throws_test = array(test.loc[:, test.columns.str.startswith('p_throws_')])

    stand_train = array(train.loc[:,train.columns.str.startswith('stand_')])
    stand_test = array(test.loc[:, test.columns.str.startswith('stand_')])
    
    pitch_type_train = array(train.loc[:,train.columns.str.startswith('pitch_type_map_')])
    pitch_type_test = array(test.loc[:, test.columns.str.startswith('pitch_type_map_')])

    balls_train = array(train.loc[:,train.columns.str.startswith('balls_')])
    strikes_train = array(train.loc[:, train.columns.str.startswith('strikes_')])
    balls_test = array(test.loc[:, test.columns.str.startswith('balls_')])
    strikes_test = array(test.loc[:, test.columns.str.startswith('strikes_')])

    #Combine all features
    x_train = hstack((pa_id_train, pitch_velo_train, pitch_type_train, balls_train, strikes_train,
                      pitcher_train, p_throws_train, stand_train))
    x_test = hstack((pa_id_test, pitch_velo_test, pitch_type_test, balls_test, strikes_test, 
                     pitcher_test, p_throws_test, stand_test))
    
    # Store number of features that are not known before pitch is thrown
    future_vars = pitch_velo_train.shape[1]+pitch_type_train.shape[1]
    
    return y_train, y_test, x_train, x_test, future_vars 


# Function to take raw data in order and convert it into a 3D array
def seq_prepper(data, sequences, y, n_future_only, n_steps_in, n_steps_out):
    
    # Returns: encoder inputs, decoder inputs and outputs, and wOBA for each PA
    
    input_, output, decoder_input = list(), list(), list()
    woba_tracker = list()
    running_ix = 0
    out_end_ix = 0
    # Loop over each plate appearance and build a sequence
    while out_end_ix < len(sequences):
        temp_pa_id = sequences[running_ix,0]
        obs = len(sequences[sequences[:,0] == temp_pa_id])
        pa_woba = data.loc[data['pa_id'] == temp_pa_id, 'woba_value'].max()
        end_ix = running_ix + obs - n_steps_out
        out_end_ix = running_ix + obs
        # Break loop if index becomes longer than length of original dataset
        if out_end_ix > len(sequences): break
        
        seq_all = sequences[running_ix:end_ix, 1:]            
        seq_out = y[end_ix:out_end_ix, :]
        seq_train_decoder = sequences[end_ix:out_end_ix, n_future_only+1:]
        # Make balls-strikes unknown after first pitch in sequence...
        # Equivalent to setting get_dummies columns to 0
        seq_train_decoder[1:, :7] = 0        
        # Pad sequences to be uniform in length
        padded_all = tf.keras.preprocessing.sequence.pad_sequences([seq_all], 
                                                                   maxlen = n_steps_in,
                                                                   padding = 'pre',
                                                                   truncating = 'pre',
                                                                   value = -100.,
                                                                   dtype='float32')       
        padded_all = array(padded_all).reshape((n_steps_in, (sequences.shape[1]-1)))
        
        input_.append(padded_all)
        output.append(seq_out)
        decoder_input.append(seq_train_decoder)
        woba_tracker.append(pa_woba)
            
        running_ix += obs
            
    input0 = np.array(input_)
    output0 = np.array(output)
    decoder_input = np.array(decoder_input)
    
    input0 = input0.astype(np.float32)
    output0 = np.array(output0).astype(np.float32)
    decoder_input = decoder_input.astype(np.float32)
    
    woba = np.array(woba_tracker)
    
    return input0, output0, decoder_input, woba


# Create training and test sets
# July 31 is the cutoff date between training and test data
cutoff_date = '2021-07-31'
y_train, y_test, x_train, x_test, n_future_vars = data_splitter(data, cutoff_date)

# Format data for encoder-decoder
# Predict two steps ahead, and pad sequences to max length based on the longest PA in training set
n_steps_out = 2
n_steps_in = df.loc[df['game_date']<=cutoff_date].groupby(['pa_id']).size().max() - n_steps_out

encoder_input, decoder_output, decoder_input, woba = seq_prepper(data, 
                                                                 x_train, 
                                                                 y_train,
                                                                 n_future_vars, 
                                                                 n_steps_in, 
                                                                 n_steps_out)

encoder_test, decoder_otest, decoder_itest, woba_test = seq_prepper(data, 
                                                                    x_test, 
                                                                    y_test,
                                                                    n_future_vars, 
                                                                    n_steps_in, 
                                                                    n_steps_out)


# In this section we build an Encoder-Decoder
num_input_features = encoder_input.shape[2] # Number of encoder features
num_decoder_inputs = decoder_input.shape[2] # Number of decoder features
n_ptypes = decoder_output.shape[2] # Number of pitch types

batch_size = encoder_input.shape[0] # Number of sequences
input_sequence_length = n_steps_in # Length of the sequence used by the encoder


def model_builder(hp):

    # First branch of the net is an lstm which finds an embedding for the past
    past_inputs = tf.keras.Input(shape=(None,num_input_features), 
                                 name='past_inputs')
    # Mask inputs with values of -100
    masking = tf.keras.layers.Masking(mask_value= -100.,
                                      input_shape=(n_steps_in, num_input_features))(past_inputs)
    # Number of units to search for in each LSTM/dense layer
    n_units = hp.Int('units', min_value=36, max_value=96, step=12)  
    # L2 regularizer for the encoder and decoder
    regularizer = tf.keras.regularizers.l2(0.01)
    # Encode the past
    encoder1 = tf.keras.layers.LSTM(n_units, return_state=True,
                                    return_sequences=True, 
                                    kernel_regularizer=regularizer)  
    encoder_outputs1, state_h1, state_c1 = encoder1(masking)
    
    enc_drop = tf.keras.layers.Dropout(rate=0.1)(encoder_outputs1)
    
    encoder2 = tf.keras.layers.LSTM(n_units,return_sequences=True,
                                    return_state=True, 
                                    kernel_regularizer=regularizer)
    encoder_outputs2, state_h2, state_c2 = encoder2(enc_drop)
    
    future_inputs = tf.keras.Input(
        shape=(None, num_decoder_inputs), name='future_inputs')
    # Decoder
    decoder_lstm = tf.keras.layers.LSTM(n_units, 
                                        return_sequences=True, 
                                        return_state=True, 
                                        kernel_regularizer=regularizer)   
    x, dec_h, dec_c = decoder_lstm(future_inputs,
                                   initial_state=[state_h2, state_c2])
    # Attention layer
    att = tf.keras.layers.Attention()([x,encoder_outputs2])
    out = tf.keras.layers.Concatenate(axis=-1)([x, att])
    decoder_dense_a = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_ptypes, activation='softmax')) 
    output_a = decoder_dense_a(out)
    model = tf.keras.models.Model(
        inputs=[past_inputs, future_inputs], outputs=output_a)
    # Learning rates to try out
    learning_rate = hp.Choice('learning_rate', values=[1e-1,1e-2,1e-3])
    
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss = 'categorical_crossentropy'
    model.compile(loss=loss, optimizer=optimizer)
    
    return model


# Hyperband tuner with early stopping built to minimize loss function
tuner = Hyperband(model_builder,
                     objective=kerastuner.Objective('val_loss', direction='min'),
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='kt314')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


# Search over 30 epochs
tuner.search([encoder_input,decoder_input],
             decoder_output, 
             epochs=30, 
             validation_split=0.2, 
             callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]


print(f"""Optimal number of layers: {best_hps.get('units')} \n
Optimal learning rate: {best_hps.get('learning_rate')}.
""")


# Find optimal number of epochs with the optimal hyperparameters
model = tuner.hypermodel.build(best_hps)
history = model.fit([encoder_input,decoder_input],
                    decoder_output, 
                    epochs=30, 
                    validation_split=0.2, 
                    verbose=0)

val_loss_per_epoch = history.history['val_loss']
best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))


hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
history2 = hypermodel.fit([encoder_input,decoder_input],
                          decoder_output, 
                          epochs=best_epoch, 
                          validation_split=0.2, 
                          verbose=0)


hypermodel.summary()


# ## Loss During Training  

preds = hypermodel.predict([encoder_test,decoder_itest])


# Plot loss and validation loss for each epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss','validation loss'])
plt.title('Training Progression')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


m = tf.keras.metrics.CategoricalCrossentropy()
m.update_state(decoder_otest, preds)
print('Loss on test data: ', float(m.result()))


# # Classification Analysis

# Find accuracy for setup and result pitch
for pitch_seq in [1,2]:    
    m = tf.keras.metrics.CategoricalAccuracy()
    m.update_state(decoder_otest[:,pitch_seq-1,:], preds[:,pitch_seq-1,:])
    print('Prediction accuracy for pitch {}:'.format(pitch_seq), float(m.result()))


# Build confusion matrices for the setup and result pitch
cm1 = confusion_matrix(decoder_otest[:,0,:].argmax(axis=1), 
                       preds[:,0,:].argmax(axis=1))
cm2 = confusion_matrix(decoder_otest[:,1,:].argmax(axis=1), 
                       preds[:,1,:].argmax(axis=1))

labels = ['Four-Seam', 'Two-Seam', 'Cutter', 'Curveball', 'Slider', 'Changeup', 'Knuckleball']

disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=labels)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=labels)

disp1.plot(cmap=plt.cm.Reds, xticks_rotation='vertical')
plt.title('Setup Pitch')

plt.show()


disp2.plot(cmap=plt.cm.Reds, xticks_rotation='vertical')
plt.title('Result Pitch')

plt.show()


print('Four-seam fastball rate on setup:', cm1[0,:].sum()/cm1.sum())
print('Four-seam fastball rate on result:', cm2[0,:].sum()/cm2.sum())


# Process of plotting roc-auc curve belonging to all classes.
# The code below is lifted from https://www.kaggle.com/muhammetvarl/mlp-multiclass-classification-roc-auc
from itertools import cycle
sns.set_theme()

def roc_auc_plot(y_true, y_preds, future_pitch_index, n_classes):
    # Return plots of ROC-AUC curves for et-up and result pitch
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:,future_pitch_index,i], y_preds[:,future_pitch_index,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr['micro'], tpr['micro'], _ = roc_curve(y_true[:,future_pitch_index,:].ravel(), y_preds[:,future_pitch_index,:].ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    lw = 2 # line_width

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr['micro'], tpr['micro'],
        label="micro-average ROC curve (area = {0:0.2f})"
            ''.format(roc_auc['micro']),
        color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr['macro'], tpr['macro'],
        label="macro-average ROC curve (area = {0:0.2f})"
            ''.format(roc_auc['macro']),
            color='navy', linestyle=':', linewidth=4)
    
    remap = ['FF', 'FT', 'FC', 'CU', 'SL', 'CH', 'KN']
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'seagreen', 'darkred', 'lawngreen'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class ' + remap[i]+ ' (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    if future_pitch_index==0: pitch_seq = 'Setup'
    else: pitch_seq = 'Result'
    
    plt.plot([0,1],[0,1], lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Multi-Class Curve for ' + pitch_seq)
    plt.legend(loc='lower right')
    return plt.show()


roc_auc_plot(decoder_otest,preds,0,7)


roc_auc_plot(decoder_otest,preds,1,7)


# Doing some stuff related to wOBA
# Find where the prediction matches reality by using an indicator
pred_indicator = np.where(decoder_otest.argmax(axis=2) == preds.argmax(axis=2),1,0)


pred_woba = hstack((pred_indicator,woba_test.reshape(woba_test.shape[0],1)))


print('Number of plate appearances in training set: ', encoder_input.shape[0])
print('Number of plate appearances in test set: ', encoder_test.shape[0])


print('PA with both pitches correct:', pred_woba[:,2][(pred_woba[:,0]==1)&(pred_woba[:,1]==1)].shape[0])
print('PA with only set-up correct:', pred_woba[:,2][(pred_woba[:,0]==1)&(pred_woba[:,1]==0)].shape[0])
print('PA with only result correct:', pred_woba[:,2][(pred_woba[:,0]==0)&(pred_woba[:,1]==1)].shape[0])
print('PA with neither correct:', pred_woba[:,2][(pred_woba[:,0]==0)&(pred_woba[:,1]==0)].shape[0])


print('wOBA with both pitches correct:', pred_woba[:,2][(pred_woba[:,0]==1)&(pred_woba[:,1]==1)].mean())
print('wOBA with only set-up correct:', pred_woba[:,2][(pred_woba[:,0]==1)&(pred_woba[:,1]==0)].mean())
print('wOBA with only result correct:', pred_woba[:,2][(pred_woba[:,0]==0)&(pred_woba[:,1]==1)].mean())
print('wOBA with neither correct:', pred_woba[:,2][(pred_woba[:,0]==0)&(pred_woba[:,1]==0)].mean())