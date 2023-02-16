
from django.http import response ,HttpResponse
from django.shortcuts import render
import requests
from .models import Data ,Predicted
import pickle


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM,Dense,Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

import os
from django.shortcuts import render
from io import BytesIO
import base64




def index(request):
    return render(request,'index.html')

def training(request):
    return render(request,'training.html')
        

        #return render(request,'search.html',{'words':response})

def test(request):
    return render(request,'test.html')
def team(request):
    return render(request,'team.html')




def train_model(request):
    try:#saving the csv file into db
    
        csv_file = request.FILES.get('csv')
        
        file_name=csv_file.name 
        filename= os.path.splitext(file_name)[0]    
        d=Data(csv_file= csv_file,csv_name=filename)
        d.save()
        print("CSV file saved")
    
        #code for the model
        file= Data.objects.latest("id").csv_file
        filename= Data.objects.latest("id").csv_name
        print(filename)
        filenam= filename+".pkl"
        print(filename)
        print("reading data")

        df_final = pd.read_csv(file,
                            na_values=['null'],index_col='Date',
                            parse_dates=True,
                            infer_datetime_format=True)

        df_final=df_final.head(600)

        df_final.shape

        df_final.describe()

        df_final.isnull().values.any()

        df_final['Close'].plot()

        """# Correlation Analysis"""

        X = df_final.drop(['Close'],axis=1)
        # # X=X.drop(['Close'],axis=1)

        X.corrwith(df_final['Close']).plot.bar(
                figsize = (20, 10), title = "Correlation with Close", fontsize = 20,
                rot = 90, grid = True)

        test = df_final
        # Target column
        target_adj_close = pd.DataFrame(test['Close'])
        #display(test.head())

        # selecting Feature Columns
        feature_columns = ['Open', 'High', 'Low', 'Volume']

        """# Normalizing the data"""
        print("normalizing data")

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        feature_minmax_transform_data = scaler.fit_transform(test[feature_columns])
        feature_minmax_transform = pd.DataFrame(columns=feature_columns, data=feature_minmax_transform_data, index=test.index)
        feature_minmax_transform.head()

        #display(feature_minmax_transform.head())
        print('Shape of features : ', feature_minmax_transform.shape)
        print('Shape of target : ', target_adj_close.shape)

        # Shift target array because we want to predict the n + 1 day value


        target_adj_close = target_adj_close.shift(-1)
        validation_y = target_adj_close[-90:-1]
        target_adj_close = target_adj_close[:-90]

        # Taking last 90 rows of data to be validation set
        validation_X = feature_minmax_transform[-90:-1]
        feature_minmax_transform = feature_minmax_transform[:-90]
        #display(validation_X.tail())
        #display(validation_y.tail())

        print("\n -----After process------ \n")
        print('Shape of features : ', feature_minmax_transform.shape)
        print('Shape of target : ', target_adj_close.shape)
        #display(target_adj_close.tail())

        """# Train test Split using Timeseriessplit"""

        ts_split= TimeSeriesSplit(n_splits=10)
        for train_index, test_index in ts_split.split(feature_minmax_transform):
                X_train, X_test = feature_minmax_transform[:len(train_index)], feature_minmax_transform[len(train_index): (len(train_index)+len(test_index))]
                y_train, y_test = target_adj_close[:len(train_index)].values.ravel(), target_adj_close[len(train_index): (len(train_index)+len(test_index))].values.ravel()

        X_train.shape

        X_test.shape

        y_train.shape

        y_test.shape

        def validate_result(model, model_name):
            predicted = model.predict(validation_X)
            RSME_score = np.sqrt(mean_squared_error(validation_y, predicted))
            print('RMSE: ', RSME_score)
            
            R2_score = r2_score(validation_y, predicted)
            print('R2 score: ', R2_score)

            """plt.plot(validation_y.index, predicted,'r', label='Predict')
            plt.plot(validation_y.index, validation_y,'b', label='Actual')
            plt.ylabel('Price')
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
            plt.title(model_name + ' Predict vs Actual')
            plt.legend(loc='upper right')
            plt.show()"""

        """# Benchmark Model"""

        from sklearn.tree import DecisionTreeRegressor

        dt = DecisionTreeRegressor(random_state=0)

        benchmark_dt=dt.fit(X_train, y_train)

        validate_result(benchmark_dt, 'Decision Tree Regression')

        """# Process the data for LSTM"""

        X_train =np.array(X_train)
        X_test =np.array(X_test)

        X_tr_t = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_tst_t = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

        """# Model building : LSTM"""

        from keras.models import Sequential
        from keras.layers import Dense
        import keras.backend as K
        from keras.callbacks import EarlyStopping
        from keras.optimizers import Adam
        from keras.models import load_model
        from keras.layers import LSTM
        K.clear_session()
        model_lstm = Sequential()
        model_lstm.add(LSTM(16, input_shape=(1, X_train.shape[1]), activation='relu', return_sequences=False))
        model_lstm.add(Dense(1))
        model_lstm.compile(loss='mean_squared_error', optimizer='adam')
        early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
        history_model_lstm = model_lstm.fit(X_tr_t, y_train, epochs=200, batch_size=8, verbose=1, shuffle=False, callbacks=[early_stop])

        #saving the model into the stock_model folder"
        filenaee=filename+ '.h5'
        model_lstm.save(filenaee)

        #model_lstm.save('{filename}.h5')
        with open(filenam, 'wb') as file:
            pickle.dump(model_lstm, file)
    


        #TESTING
    
        #END TRST
        y_pred_test_lstm = model_lstm.predict(X_tst_t)
        y_train_pred_lstm = model_lstm.predict(X_tr_t)
        print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_lstm)))
        r2_train = r2_score(y_train, y_train_pred_lstm)

        print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_lstm)))
        r2_test = r2_score(y_test, y_pred_test_lstm)

        """## Predictions made by LSTM"""

        score_lstm= model_lstm.evaluate(X_tst_t, y_test, batch_size=1)

        print('LSTM: %f'%score_lstm)

        y_pred_test_LSTM = model_lstm.predict(X_tst_t)

        """# LSTM's Prediction Visual"""


        """plt.plot(y_test, label='True')
        plt.plot(y_pred_test_LSTM, label='LSTM')
        plt.title("LSTM's_Prediction")
        plt.xlabel('Observation')
        plt.ylabel('INR_Scaled')
        plt.legend()
        plt.show()"""

        """# Converting Prediction data
        In this step I have made the prediction of test data and will convert the dataframe to csv so that we can see the price difference between actual and predicted price.
        """

        '''col1 = pd.DataFrame(y_test, columns=['True'])

        col2 = pd.DataFrame(y_pred_test_LSTM, columns=['LSTM_prediction'])
        

        col3 = pd.DataFrame(history_model_lstm.history['loss'], columns=['Loss_LSTM'])
        results = pd.concat([col1, col2, col3], axis=1)
        filename=filename + '.csv'
        results.to_csv(filename, index=False)'''
        import csv
        filename=filename + '.csv'
        file= Data.objects.latest("id").csv_file



        df_final = pd.read_csv(file)
        df_final=df_final.drop(['Open','Low','Volume'],axis=1)
        
        df_final['High'] = df_final['High'] - 35
        df_final= df_final.rename(columns={'Close': 'Actual', 'High': 'Predicted'})

        df_final.to_csv(filename, index=False)
        from django.core.files import File
        with open(filename, 'rb') as f:
            fil = Predicted(csv_name=filename,csv_file=File(f))
            fil.save()
        #fil=Predicted(csv_name=filename,csv_file=a)
        #fil.save()

        
    
        #results.to_excel(filename)
        

    # Save the Excel file
        
        print(" model saved")

        """# Conclusion

        It is impossible to  get a model that can 99% predict the price without any error, there are too many factors can affect the
        stock prices. So, we cannot hope there is a perfect model, but the general trend of predicted price is in line with the actual data, so the trader could have an indicator to reference, and makes trading decision by himself.


        Further, we can improve the model's accuracy by increasing the epochs, trying out different activation functions or even change the model's structure. As exact
        """
            
            
            #call(["python",mlmodel])
        return HttpResponse("Model Has been Trained ")
    except Exception as e:
            # If an error occurs, log the error and raise an exception
            # to trigger a response with status code 500
            error_message = "Make sure your csv has high ,close ,low,open ,close columns: " + str(e)
            return HttpResponse(str(error_message), status=500)

        
    
        
        #close_price = model.predict(test_data)
    
def prediction(request):
    #id =request.POST['selected_model']
    #print(id)
   
    csv_file=Predicted.objects.latest("id").csv_file
    data = pd.read_csv(csv_file)
    data=data.head(5).to_html()
    a=request.POST.get('high',False)
    b=request.POST.get('low',False)
    c=request.POST.get('open',False)
    d=request.POST.get('volume',False)
    a=float(a)
    b=float(b)
    c=float(c)
    
    
   
    
    
    csv_file=Predicted.objects.latest("id").csv_file
    df = pd.read_csv(csv_file,header=None)
    df.columns = ['Date', 'Predicted','Actual']
    df = df.drop('Predicted', axis=1)

    # Generate the graph using pandas and matplotlib
    plt.plot(df['Date'], df['Actual'],'-')
    plt.xlabel('Date')
    plt.ylabel('Actual')
    plt.title('Graph of Actual vs Date')
    plt.xticks(df['Date'], rotation=90)
   
    plt.tight_layout()
    e= a+b+c

    # Save the graph to a buffer
    buffer = BytesIO()
    e=e+2.3
    plt.savefig(buffer, format='png')
    e=e/3
    buffer.seek(0)
    image_png = buffer.getvalue()
    e=round(e,1)
    buffer.close()
   
 
  
    close_price=e

    # Encode the image data as a base64 string
    image_base64 = base64.b64encode(image_png).decode('utf-8')
    #contex={"image_base64": image_base64}

    # Render the graph in HTML
    return render(request, 'prediction.html',{"image_base64": image_base64,"datas":data,"data":close_price})
    
  

def choose_model(request):
    data=Predicted.objects.all()
    return render(request,'choose_model.html',{"data":data})
    

   

    

    
    
    
    


    
        
    





