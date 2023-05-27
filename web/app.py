from pathlib import Path
import base64
import re
import os
import streamlit as st
from PIL import Image
import requests
import random
import pickle
import numpy as np
import io
# from google_images_search import GoogleImagesSearch
import warnings
warnings.filterwarnings('ignore')


# set page layout
st.set_page_config(
    page_title="Image Classification App",
    page_icon=":brain:",
    layout="centered",
    initial_sidebar_state="expanded",

)


# Markdown cannot be directly rendered from local machine using st.markdown , we need to use st.image for that and these funcitons below
# parse the markdown file and find the markdown way of mentioning images ![]<> and convert it into html and then it is seen.
def markdown_images(markdown):
    # example image markdown:
    # ![Test image](images/test.png "Alternate text")
    images = re.findall(
        r'(!\[(?P<image_title>[^\]]+)\]\((?P<image_path>[^\)"\s]+)\s*([^\)]*)\))', markdown)
    return images


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def img_to_html(img_path, img_alt):
    img_format = img_path.split(".")[-1]
    img_html = f'<img src="data:image/{img_format.lower()};base64,{img_to_bytes(img_path)}" alt="{img_alt}" style="max-width: 100%;">'

    return img_html


def markdown_insert_images(markdown):
    images = markdown_images(markdown)

    for image in images:
        image_markdown = image[0]
        image_alt = image[1]
        image_path = image[2]
        if os.path.exists(image_path):
            markdown = markdown.replace(
                image_markdown, img_to_html(image_path, image_alt))
    return markdown


def get_file_content_as_string(path):
    with open(path, 'r') as f:
        readme_text = f.read()
    readme_text = markdown_insert_images(readme_text)
    return readme_text


# function to show the developer information
def about():
    st.sidebar.markdown("# A B O U T")
    # st.sidebar.image('profile.png',width=180)
    # st.sidebar.markdown("## Meher")
    # st.sidebar.markdown('* ####  Connect via [LinkedIn]()')
    # st.sidebar.markdown('* ####  Connect via [Github]()')
    # st.sidebar.markdown('* ####  xxxxxxx@gmail.com')
    st.sidebar.markdown("""

### Mentors

- Bharadwaja M Chittapragada
- Charu Samir Shah

### Members

- Aishini Bhattacharjee
- Abhijith D Kumble
- Lasya Reddy
- Mahitha Kankatala

### [GitHub](https://github.com/MeherRushi/I02-BrainTumorClassification)

## Acknowledgements

Special thanks to Ashish Bharath and the seniors for guiding us during the project.
    """)


# defining each page

# This will be the landing page: we will start with project report
def project_report():
    # This is load the markdown page for the entire home page
    # This is for the  home page
    st.title('Brain Tumor Classification')
    st.caption('A basic Brain Tumor Detector and Classifier :brain:')
    # Main_image = st.image('images/t.png',caption='Source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset')
    readme_text = st.markdown(get_file_content_as_string(
        'report.md'), unsafe_allow_html=True)

    success_text = st.sidebar.success('To continue, select "Run the app" ')
    info_text = st.sidebar.info(
        'To see the Benchmarking results of all the models used, go to "Benchmarking Results"')
    # option = st.sidebar.selectbox('',('Project Report','Run the app', 'Benchmarking', 'Source code'))
    about()

# This will be run the app page


def run_app():

    # further instructions
    warning_text = st.sidebar.warning(
        'Go to "Project Report" to read more about the app')
    info_text = st.sidebar.info(
        'To see the Benchmarking results of all the models used, go to "Benchmarking Results"')

    # display the developer information
    about()



    # define the fucntion to make the class label predictions
    def predict():

        # divider for the asthetics of the page
        st.write("-"*34)



    # ask for user preference
    st.header('The Brain Tumor Classifier App')
    st.caption('You can upload images to this app to predict the prescence and type of Brain Tumor (glioma, meningioma, pituitary)')


    st.markdown('### Choose your preferred method for uploading images')
    genre = st.radio("", ('Upload a Brain MRI Scan image yourself',
                     'Get a random Brain MRI image from the test set automatically'))

    # for aesthetic purposes
    st.write("-"*34)
    # for model selection
    models_list = ["VGG16", "MobileNet", "Inception",
                   "3-Layered CNN (Batch Norm)", "3-Layered CNN (DropOut)", "Multiclass Logistic Regression"]
    st.markdown('### Choose your preferred Model for the classification task')
    network = st.selectbox('Select the Model to be used for prediction', models_list)
    st.write("-"*34)

    # MODELS = {
    # "VGG16": VGG16,
    # "MobileNet": NobileNet,
    # "Inception": InceptionV3,    # TensorFlow ONLY
    # "3-Layered CNN (Batch Norm)": cnn3bn, 
    # "3-Layered CNN (DropOut)": cnn3do,
    # "Multiclass Logistic Regression": mlr
    # }



    # code get the uploaded image from the user
    if genre == 'Upload a Brain MRI Scan image yourself':

        # display the upload image interface
        uploaded_file = st.file_uploader(
            "Choose an image", type=["jpg", "jpeg", "png"])

        # check for errors in the file
        if uploaded_file is not None:

            try:
                # columns to get the tabular format with 2 columns
                col1, col2 = st.columns([1, 1])

                # display the image in the left column
                with col1:

                    # preprocess the image
                    image = Image.open(uploaded_file).convert('RGB')
                    image = image.resize((512, 512))



                    # display the image to the screen
                    st.image(image, width=250)

                # display the feed-back further instructions in the right column
                with col2:

                    # further feedback and instructions
                    st.success(
                        'Successfully uploaded the image. Please see the category predictions below')
                    st.write("-"*34)
                    st.info(
                        'To get a random image from the test set, press the 2nd option above..!')

            # display the error message of the format is not an image format
            except Exception as e:
                st.error('Please upload an image in jpg or png format..!')

        # Display the status while the model is predicting the class labels
        status = st.spinner(text="getting the predictions..")




    # This is the code to get the random image from google.
    if genre == 'Get a random Brain MRI image from the test set automatically':

        # display the status to the screen
        status = st.spinner(
            text="Getting a random Brain MRI image from the test set...")

        # Display the downloaded image in a tabular format
        col1, col2 = st.columns([1, 1])

        # Display the image in the left column
        with col1:

            # this is to get the image from the given url
            try:

                # first get the urls for the images
                results = st.session_state.gis.results()
                im = np.random.choice(results, 1)[0]

                # get the images from the obtained urls
                image_file = requests.get(im.url, headers=hdr).content
                image_file = io.BytesIO(image_file)
                image = Image.open(image_file).convert('RGB')

                # some preprocessing that needs to be done
                image = image.resize((512, 512))

                # display the image to the screen
                st.image(image, width=250)


            # error messages in case is something goes wrong
            except Exception as e:
                st.error("Could not get the image due to some technical issue..!")
                st.warning("Please retry with another image..!")

        # This is to display the further instructions in the right column
        with col2:

            # This is the button to try the prediction with another image
            if st.button('Try with another image'):
                genre = 'Get a random product image from google automatically'

            st.success(
                "Got a random product image from the web. Please see the predictions below..!")
            st.warning("Choose 'upload an image' to input your custom image..!")


        status = st.text("getting the predictions..")


def benchmarking():
    st.header('Benchmarking the models for Brain Tumor Classification')
    st.markdown(
        'Dataset Used: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset')
    st.markdown('''
We used a dataset of 7022 MRI images of the human brain, which are labeled into 4 classes: 
- glioma 
- meningioma 
- pituitary and
- no tumor 

And The following are the results we obtained on using different models:

| Model    | Accuracy |
| -------- | ------- |
| Logistic Regression  | 85.59%    |
| CNN (3 - layered with Drop-Out) | 91.35%     |
| CNN (3 - layered with Batch Normalization)    | 92.42%    |
| MobileNet    | 89.02%    |
| VGG-16    | ~99.4%    |
| Inception-v3    | 96.87%    |

    ''')
    success_text = st.sidebar.success('To continue, select "Run the app" ')
    info_text = st.sidebar.info('Go to "Project Report" to read more about the app')
    # option = st.sidebar.selectbox('',('Project Report','Run the app', 'Benchmarking', 'Source code'))
    about()


# This is for the side menu for selecting the sections of the app
st.sidebar.markdown('# M E N U')

page_bg_img = '''
<style>
.stApp{
background-image: url("https://images.unsplash.com/photo-1567095751004-aa51a2690368?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80");
background-size: cover;

}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)


page_names_to_funcs = {
    "Project Report": project_report,
    "Run the App": run_app,
    "Benchmarking Results": benchmarking,
}

selected_page = st.sidebar.selectbox(
    "Choose the app mode", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()

