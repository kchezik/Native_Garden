#For importing data
import sys
import os, re
sys.path.append("..")
import pandas as pd
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import random

#For analyzing similarity
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

#For website display
import flask
from werkzeug.utils import secure_filename
from PIL import ExifTags, Image

from Nat_Gard_Build import app

for f in os.listdir("./Nat_Gard_Build/static/tmp/"):
    print(f)
    if re.search("plot*", f):
            os.remove(os.path.join("./Nat_Gard_Build/static/tmp/", f))

#Get pre-processed image embeddings and plant data.
pic_embs = pickle.load(open(os.path.join(app.config['DATA_FOLDER'],'data.pkl'),'rb'))
plant_dat = pd.read_csv(os.path.join(app.config['DATA_FOLDER'],'USDA_Lady_Plants.csv'), index_col=0)

#Retrieve Model
model = models.resnet18(pretrained=True)
layer = model._modules.get('avgpool')
model.eval()

#Transforms
scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

#Function to munge the image data into a vector.
def get_vector(image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.squeeze())
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature bvector
    return my_embedding

#Function to process the new image.
def process_img(imgurl, pic_embs, plant_dat):

    #Image to vector processes.
    new_img = get_vector(imgurl)

    #Similarity by cosine
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    #Get cosine similarity
    pic_cos = {}
    for plant in pic_embs:
        pic_cos[plant] = cos(new_img.unsqueeze(0),pic_embs[plant].unsqueeze(0)).item()
    #Create a dataframe with File names and Similarity scores.
    df = pd.DataFrame(columns=['File','Similarity'])
    df['File'] = list(pic_cos.keys())
    df['Similarity'] = list(pic_cos.values())
    #From the File name create a USDA label
    USDA_ID = {}
    for i in df.index:
        USDA_ID[i] = df['File'][i].split('.')[0]
    #Add the USDA label and join to plant data on that label
    df['USDA_ID'] = pd.Series(USDA_ID)    
    sim_df = pd.merge(plant_dat,df)
    return(sim_df)


# Test for allowed file types for upload.
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

#
@app.route('/',  methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
        # Get method type
    method = flask.request.method
    #print(method)

    if method == 'GET':
            return flask.render_template('index.html')
        
    if method == 'POST':
        # No file found in the POST submission
        if 'file' not in flask.request.files:
            #print("FAIL")
            return flask.redirect(flask.request.url)

        # File was found
        # State selection
        select = flask.request.form['dropdown']
        # Reduce plant options to those native to the state.
        IDs = plant_dat.USDA_ID.unique()
        
        # Add the .jpg ending to the USDA IDs in selected within the state.
        File = []
        for i in range(len(IDs)):
            File.append(IDs[i]+'.jpg')
        # Filter the dictionary of picture embeddings by plants within the state.
        all_embs={}
        for i in File:
            if(i in pic_embs.keys()):
                all_embs[i] = pic_embs[i]
        print(select)

        file = flask.request.files['file']
        if file and allowed_file(file.filename):

            img_file = flask.request.files.get('file')
            #secure file name so stop hackers
            img_name = secure_filename(img_file.filename)

            # Write image to tmp folder so it can be shown on the next page 
            imgurl=os.path.join(app.config['UPLOAD_FOLDER'], img_name)
            file.save(imgurl)

            #
            randomnumber = random.random() 
            filename1 = "plot{}.jpg".format(randomnumber)
            targetname = os.path.join("./Nat_Gard_Build/static/tmp/",filename1)

            sim_img = process_img(img_file, all_embs, plant_dat)
            #Plot all similarities
            plt.subplots(figsize=(9,6))
            g = sns.violinplot(x="Plant_Type", y="Similarity",
                palette="YlGnBu", data=sim_img, 
                order=['Herb','Fern','Subshrub','Shrub','Grass/Grass-like','Vine','Cactus/Succulent'])
            g.set_xlabel("USDA: Plant Habitat",fontsize=15)
            g.set_ylabel("% Similarity to Input Image",fontsize=15)
            sns.despine()
            
            plt.savefig(targetname, format="jpg", dpi = 200)
            
            #
            randomnumber = random.random() 
            filename2 = "plot{}.jpg".format(randomnumber)
            targetname = os.path.join("./Nat_Gard_Build/static/tmp/",filename2)

            sim_img = sim_img[sim_img['State']==select].sort_values('Similarity',ascending=False)
            #Plot state similarities
            plt.subplots(figsize=(9,6))
            h = sns.violinplot(x="Plant_Type", y="Similarity",
                palette="YlGnBu", data=sim_img,
                order=['Herb','Fern','Subshrub','Shrub','Grass/Grass-like','Vine','Cactus/Succulent'])
            h.set_xlabel("USDA: Plant Habitat",fontsize=15)
            h.set_ylabel("% Similarity to Input Image",fontsize=15)
            sns.despine()

            plt.savefig(targetname, format="jpg", dpi = 200)

            #Return
            return flask.render_template('results.html',
		matches=sim_img['File'][:19],
		original=img_name,
		sci_nm = sim_img['Scientific_Name'][:19],
		wtr = sim_img['Water_Req'][:19],
		sn = sim_img['Sun_Req'][:19],
		ids = sim_img['USDA_ID'][:19],
        cmn = sim_img['Common_Name'][:19],
        state = select, 
        imgfilename1 = filename1,
        imgfilename2 = filename2)

        return flask.redirect(flask.request.url)
