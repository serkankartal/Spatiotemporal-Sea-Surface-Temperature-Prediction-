from numpy import concatenate
import numpy as np
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
import math
import os

from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error

img_width=205
img_height=50
total_Test_case=12
activation = "tanh"

model_name="mlp" #svm ,tree,knn
MODEL_DIR="./models/"+model_name
config="config"

np.random.seed(1)
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset
def CALLFunc(train_v,test_v, mask,train_per=80, n_mins=5):

    total_Test_case = 12 #0.08 için
    test_start_year = 2020
    test_start_month = 1
    n_mins = 6


    all_data = concatenate((train_v, test_v), axis=0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(all_data)
    scaled = scaler.transform(train_v)
    scaled_test=scaler.transform(test_v)

    # frame as supervised learning
    n_features = train_v.shape[1]
    train = series_to_supervised(scaled, n_mins, 1).values
    test = series_to_supervised(scaled_test, n_mins, 1).values
    # drop columns we don't want to predict

    # split into input and outputs
    n_obs = n_mins * n_features
    train_X, train_y = train[:, :n_obs], train[:, -1]
    test_X, test_y = test[:, :n_obs], test[:, -1]

    print(train_y, test_y)
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design svm model,
    global config

    layer_count=10
    model = Sequential()
    model.add(Dense(layer_count, activation=activation, input_dim=train_X.shape[1]))
    config = config + "_" + str(layer_count)
    model.add(Dense(layer_count, activation=activation, input_dim=train_X.shape[1]))
    config = config + "_" + str(layer_count)
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    if not os.path.exists(MODEL_DIR+ "/"+config):
        os.makedirs(MODEL_DIR+"/"+config)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=11),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=0.0000001
        ),
        keras.callbacks.CSVLogger('./logs/' + model_name + '_' + config + '.csv', append=True, separator=';'),

    ]
    model.fit(train_X, train_y, epochs=100, batch_size=512, callbacks=callbacks,validation_data=(test_X, test_y),verbose=2, shuffle=False)

    # fit model

    # make a prediction
    predict_y = model.predict(test_X)
    yhat = predict_y.reshape(predict_y.shape[0], 1)
    test_X = test_X.reshape((test_X.shape[0], n_obs))
    print(yhat.shape, test_X.shape)
    # invert scaling for forecast
    inv_yhat = concatenate((test[:, -n_features: -1], yhat), axis=1)
    inv_yhat_t = scaler.inverse_transform(inv_yhat)
    inv_yhat_t[:,0]=np.around(inv_yhat_t[:,0],0).astype(int)
    inv_yhat_t[:,1]=np.around(inv_yhat_t[:,1],0).astype(int)
    inv_yhat_t[:,2]=np.around(inv_yhat_t[:,2],0).astype(int)
    inv_yhat_t[:,3]=np.around(inv_yhat_t[:,3],0).astype(int)
    inv_yhat = inv_yhat_t[:, -1]


    # invert scaling for actual concatenate((scaled_test[:,:-1], test_y), axis=1)
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test[:, -n_features: -1], test_y), axis=1)
    inv_y_t = scaler.inverse_transform(inv_y)
    inv_y_t[:,0]=np.around(inv_y_t[:,0],0).astype(int)
    inv_y_t[:,1]=np.around(inv_y_t[:,1],0).astype(int)
    inv_y_t[:,2]=np.around(inv_y_t[:,2],0).astype(int)
    inv_y_t[:,3]=np.around(inv_y_t[:,3],0).astype(int)
    inv_y = inv_y_t[:, -1]


    image_errors=np.zeros((12,3))

    for i in range(total_Test_case):
        if test_start_month==13:
            test_start_month=1
            test_start_year=test_start_year+1
        tempdata_o=inv_yhat_t[(np.where(inv_yhat_t[:,0]==test_start_year))]
        tempdata_o=tempdata_o[(np.where(tempdata_o[:,1]==test_start_month))]
        img_org = np.zeros((img_height,img_width), dtype=np.float)

        tempdata=inv_y_t[(np.where(inv_y_t[:,0]==test_start_year))]
        tempdata=tempdata[(np.where(tempdata[:,1]==test_start_month))]
        img_predicted = np.zeros((img_height,img_width), dtype=np.float)
        for row in range(img_height):
            for col in range(img_width):
                if mask[row, col]:
                    t=tempdata[np.where(tempdata[:,2]==row)]
                    img_org[row][col]=t[np.where(t[:,3]==col)][0][4]

                    t=tempdata_o[np.where(tempdata_o[:,2]==row)]
                    img_predicted[row][col]=t[np.where(t[:,3]==col)][0][4]


        # print(year,"/",month,"/",i)
        max_ref=30#np.quantile(img_org, 0.98) #np.amax([np.amax(img_predicted), np.amax(img_org)])
        min_ref=15#np.quantile(img_org, 0.5) #np.amin([np.amin(img_predicted), np.amin(img_org)])+20

        MSE_score = math.sqrt(mean_squared_error(np.squeeze(img_org[mask]), np.squeeze(img_predicted[mask])))
        MAE_score = mean_absolute_error(np.squeeze(img_org[mask]), np.squeeze(img_predicted[mask]))
        MAPE_score = mean_absolute_percentage_error(np.squeeze(img_org[mask]), np.squeeze(img_predicted[mask]))*100
        print("******** %d image*********"%i)
        print('Image RMSE: %.3f' % MSE_score)
        print('Image MAE: %.3f' % MAE_score)
        print('Image MAPE: %.3f' % MAPE_score)

        fig, (ax1, ax2, ax3) = pyplot.subplots(3, 1, figsize=(28, 26))

        # nir in first subplot
        nir = ax1.imshow(img_org, cmap=pyplot.get_cmap("jet"))
        ax1.set_title("Real Temperature")
        nir.set_clim(vmin=min_ref, vmax=max_ref)
        fig.colorbar(nir, ax=ax1)

        # Now red band in the second subplot
        red = ax2.imshow(img_predicted, cmap=pyplot.get_cmap("jet"))
        ax2.set_title("Predicted Temperature RMSE:{:.2f}".format(MSE_score) + " MAE:{:.2f}".format(MAE_score)+ " MAPE:{:.2f}".format(MAPE_score))
        red.set_clim(vmin=min_ref, vmax=max_ref)
        fig.colorbar(red, ax=ax2)

        # differences mae
        mse = np.sqrt(((np.squeeze(img_org) - np.squeeze(img_predicted)) ** 2))
        mae = np.abs(((np.squeeze(img_org) - np.squeeze(img_predicted))))
        dif = ax3.imshow(mae, cmap=pyplot.get_cmap("jet"))
        ax3.set_title("Map for MAE:{:.2f}".format(MAE_score))
        dif.set_clim(vmin=min_ref, vmax=max_ref)
        fig.colorbar(dif, ax=ax3)

        pyplot.savefig(MODEL_DIR + "/"+config+"/" + str(test_start_year) +"_"+str(test_start_month)+ ".png")
        pyplot.close('all')

        image_errors[i]=[MSE_score,MAE_score,MAPE_score]
        test_start_month=test_start_month+1

        # pyplot.show()

    print("******** image Average *********")
    print('Image RMSE: %.3f' % (np.sum(image_errors[:,0])/i))
    print('Image MAE: %.3f' % (np.sum(image_errors[:,1])/i))
    print('Image MAPE: %.3f' % (np.sum(image_errors[:,2])/i))

    with open(MODEL_DIR+"/"+config+"/avgerrors.txt","wt")as rf:
        rf.write('Image Average RMSE: %.3f \n' % (np.sum(image_errors[:,0])/i))
        rf.write('Image Average MAE: %.3f\n' % (np.sum(image_errors[:,1])/i))
        rf.write('Image Average MAPE: %.3f\n' % (np.sum(image_errors[:,2])/i))

    image_errors=np.round(image_errors,2)
    np.savetxt(MODEL_DIR + "/"+config+"/errors.csv", image_errors, delimiter=",")
