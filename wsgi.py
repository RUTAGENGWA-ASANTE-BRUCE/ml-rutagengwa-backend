from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from flask import Flask, render_template, jsonify, Response, request,send_from_directory
from flask_cors import CORS
import cv2
import pickle
import os
import time
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import random
import numpy as np
app = Flask(__name__)
CORS(app)
datagen = ImageDataGenerator(
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')

root_folder=app.root_path
datasets_folder=os.path.join(root_folder,'datasets')
test_images_folder=os.path.join(root_folder,'TEST_IMAGES_FOLDER')
models_folder=os.path.join(root_folder,'models')
IMG_SIZE=70





def feature(name,images_array):
        return {"name":name,"images_array":images_array}

def feature_frontend(name,images_array):
        return {"name":name,"images_array":images_array}

def resizeImage(myFrame):
    new_array=cv2.resize(myFrame,(IMG_SIZE,IMG_SIZE))
#     return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)
    return new_array
def add_feature(feature_name):
    pickle_out=open(os.path.join(datasets_folder,f"{feature_name}.pickle"),"wb")
    pickle.dump(np.array([]),pickle_out)
    pickle_out.close()
    return feature_frontend(feature_name,[])

def add_feature_image(feature_name,frame):
    new_training_data=[]
    model_features=[]
    all_data=[]
    timeStamped_name=f"{feature_name}_{time.time()}.jpg"
    cv2.imwrite(os.path.join(test_images_folder,timeStamped_name),frame)
    for dataset in os.listdir(datasets_folder):
        model_features.append(dataset[0:-7])
    pickle_in=open(os.path.join(datasets_folder,f"{feature_name}.pickle"),"rb")
    initial_data=pickle.load(pickle_in)
    pickle_in.close()
      
    for singleData in initial_data:
        all_data.append(singleData)
    all_data.append([resizeImage(frame),model_features.index(feature_name)])
  
    random.shuffle(all_data)
    pickle_out=open(os.path.join(datasets_folder,f"{feature_name}.pickle"),"wb")
    pickle.dump(all_data,pickle_out)
    pickle_out.close()
    
    return timeStamped_name;

def get_training_data():
    all_features=[]
    all_labels=[]
    for dataset in os.listdir(datasets_folder):
        pickle_in=open(os.path.join(datasets_folder,dataset),"rb")
        dataset_data=pickle.load(pickle_in)
        for features, labels in dataset_data:
            all_labels.append(labels)
            all_features.append(features)
    X=np.array(all_features)
    X=X/255.0
    X=np.array(X)
    Y=np.array(all_labels)
    return {"X":X,"Y":Y}

def training_the_model(X,Y):
    output_labels=0
    for dataset in os.listdir(datasets_folder):
        output_labels=output_labels+1
        print(f"model_ouput_class: {dataset}: {output_labels}")
    
    dense_layers=[1]
    layer_sizes=[64]
    conv_layers=[2]

    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for conv_layer in conv_layers:
                NAME=f"BUCE_ML_{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{int(time.time())}"
                print(NAME)

                model=Sequential()

                model.add(Conv2D(layer_size, (3, 3), activation='relu', input_shape=X.shape[1:]))
#                 model.add(Activation('relu'))
                model.add(MaxPooling2D((2, 2)))


                for l in range(conv_layer-1):

                    model.add(Conv2D(layer_size, (3, 3),activation='relu'))
#                     model.add(Activation('relu'))
                    model.add(MaxPooling2D((2, 2)))
                    model.add(Conv2D(layer_size, (3, 3),activation='relu'))

                model.add(Flatten())

                for l in range(dense_layer):
                    model.add(Dense(layer_size))
                    model.add(Dense(layer_size, activation='relu'))
#                     model.add(Activation('relu'))
    #                 model.add(Dropout(0.2))



                model.add(Flatten())
                model.add(Dense(64, activation='relu'))
                model.add(Dense(output_labels))


                model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

                model.fit(X,Y,batch_size=32,epochs=10,validation_split=0.1)
                test_loss, test_acc = model.evaluate(X,  Y, verbose=2)
                print(test_acc)

    model.save(os.path.join(root_folder,"BRUCE_ML_MODEL.model"))
    return f"Model Trained Successfully. Test accuracy: {test_acc*100}%"

def predict(myFrame):
    class_names=[]
    for dataset in os.listdir(datasets_folder):
        class_names.append(dataset[0:-7])
    model=tf.keras.models.load_model(os.path.join(root_folder,"BRUCE_ML_MODEL.model"))
    new_array=resizeImage(frame);
    x=np.expand_dims(new_array/255,0)
    prediction=model.predict([x])
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class
        
    #print(value)


def generate_features():
    model_features=[]
    for dataset in os.listdir(datasets_folder):
        model_features.append(dataset[0:-7])
        print(dataset)
    training_data_frontend=[]
    print(f"==========model_features")
    print(model_features)
    for category in model_features:
        string_image_urls=[]
        for image in os.listdir(test_images_folder):
            if(category==image[:len(category)]): 
                string_image_urls.append(image)
        training_data_frontend.append(feature_frontend(category,string_image_urls))

    return training_data_frontend

# def prepare(myFrame):
#     IMG_SIZE=70
#     new_array=cv2.resize(myFrame,(IMG_SIZE,IMG_SIZE))
#     return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)

# model=tf.keras.models.load_model("Paper_Scissor_Rock_CNN.model")


camera = cv2.VideoCapture(0)
#success, frame = camera.read()  # default value at start
success = False  # default value at start
frame   = None   # default value at start



trainig_data_frontend=generate_features()
def gen_image(img_array,imageName):
    
    ret, buffer = cv2.imencode('.jpg', img_array)
    image = buffer.tobytes()   # use other variable instead of `frame`
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n'
           b'\r\n' + image + b'\r\n')  # concat frame one by one and show result
            

def gen_frames():
    global success
    global frame
    
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, buffer = cv2.imencode('.jpg', frame)
            image = buffer.tobytes()   # use other variable instead of `frame`
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'\r\n' + image + b'\r\n')  # concat frame one by one and show result
            time.sleep(0.04) # my `Firefox` needs this to have time to display image.
                             # And this gives stream with 25 FPS (Frames Per Second) (1s/0.04s = 25)

# @app.route('/get_value')
# def get_value():
#     #if frame is not None:
#     if success:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         new_array=prepare(gray);
#         prediction=model.predict([new_array])
#         predicted_class = class_names[np.argmax(prediction)]
#         value=predicted_class
#         print(value)
#     else:
#         value = ""
        
#     #print(value)
#     return jsonify(value)        


@app.route('/')
def index():
    return "Hello world"


@app.route('/add_category',methods =["POST"])
def add_category():
    body=request.get_json()
    category_name = body["category_name"]
    newFeature=add_feature(category_name)
  
    return newFeature
    
@app.route('/add_category_image',methods =["POST"])
def add_category_image():
    body=request.get_json()
    category_name = body["category_name"]
    new_image=add_feature_image(category_name,frame)
    return new_image
    
@app.route('/get_categories',methods =["GET"])
def get_categories():
    categories=generate_features()
    return categories

@app.route('/train_model',methods =["GET"])
def train_model():
    training_data=get_training_data()
    message=training_the_model(training_data['X'],training_data['Y'])
    return message

@app.route('/predict_class',methods=["GET"])
def predict_class():
    predicted_class=predict(frame)
    return predicted_class
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/test_images/<path:name>")
def download_file(name):
    folder = os.path.join(app.root_path,'TEST_IMAGES_FOLDER') 
    return send_from_directory(folder, name, as_attachment=True)

if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0',port=8000)
    except KeyboardInterrupt:
        print("Stopped by `Ctrl+C`")
    finally:
        camera.release()
