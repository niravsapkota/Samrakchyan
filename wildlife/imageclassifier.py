import base64
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import os
import numpy
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import cv2  


animals = ['Agouti','Ocelot','Red Squirrel','Bird Spec','White Tailed Deer','Mouflon','Red Deer','Roe Deer','Wild Boar','European Hare']
animal_desc = {
    "Red Deer":"The red deer inhabits most of Europe, the Caucasus Mountains region, Anatolia, Iran, and parts of western Asia. Red deer are ruminants, characterized by a four-chambered stomach. Genetic evidence indicates that the red deer, as traditionally defined, is a species group, rather than a single species, though exactly how many species the group includes remains disputed. The red deer is the fourth-largest extant deer species. Only the stags (male) have antlers, which start growing in the spring and are shed each year, usually at the end of winter.",
    "Agouti":"The agouti is a rodent from Central and South America rainforests that looks a bit like a really large guinea pig. Its coarse hair is covered with an oily (and stinky!) substance that acts like a raincoat. Agoutis have five toes on their front feet and three toes on their hind feet; the first toe is very small. The tail is very short or nonexistent and hairless.",
    "Wild Boar":"Wild boar have stocky, powerful bodies with a double layer of grey-brown fur – the top layer harsh, bristly hair; the under layer much softer. Mature males have tusks that protrude from the mouth. Piglets are a lighter ginger-brown, with stripes on their coat for camouflage.Boar are omnivores and will eat a wide range of plant and animal matter. The majority of their diet is made up of roots, bulbs, seeds, nuts and green plants.",
    "Roe Deer":"The roe deer, also known as the roe, is a species of deer. The male of the species is sometimes referred to as a roebuck. The roe is a small deer, reddish and grey-brown, and well-adapted to cold environments. This species can utilize a large number of habitats, including open agricultural areas and above the tree line, but a requisite factor is access to food and cover. It retreats to dense woodland, especially among conifers, or bramble scrub when it must rest, but it is very opportunistic and a hedgerow may be good enough.",
    "Red Deer":"Red deer is a popular deer that is native to North America, Europe, Asia, and northwestern Africa and was introduced into New Zealand. The red deer has long been hunted for both sport and food. A large animal, the red deer stands about 1.2 metres (4 feet) tall at the shoulder. Its coat is reddish brown, darkening to grayish brown in winter, with lighter underparts and a light rump. The hart has long, regularly branched antlers bearing a total of 10 or more tines.x",
    "Ocelot":"Ocelots are one of the more beautiful feline species. Their coat is short and soft, forming two whorls on the shoulder, the hairline on the neck being directed towards the crown. Ground colour varies from whitish or tawny yellow to reddish grey. Markings run into chain-like streaks and blotches, forming elongate spots bordered with black enclosing an area darker than the ground colour.",
    "Red Squirrel":"The red squirrel is a species of tree squirrel in the genus Sciurus common throughout Europe and Asia. The red squirrel is an arboreal, primarily herbivorous rodent. The red squirrel has a typical head-and-body length of 19 to 23 cm, a tail length of 15 to 20 cm, and a mass of 250 to 340 g. The long tail helps the squirrel to balance and steer when jumping from tree to tree and running along branches and may keep the animal warm during sleep.",
    "Mouflon":"Mouflon are wild sheep, a species regarded as one of the two original ancestors of modern-day sheep. Their coat is reddish-brown and short-haired, and a dark stripe runs along their back, with lighter-colored patches on the side. They are very wary animals. The males have large horns of a sickle shape, prized by many trophy hunters. Females have horns too, but much smaller ones than those of males. The adult males develop a large ruff of coarse long hair on their chest, which is white at the throat, becoming black towards the forelegs.",
    "European Hare":"The European hare, also known as the brown hare, is a species of hare native to Europe and parts of Asia. It is among the largest hare species and is adapted to temperate, open country. Hares are herbivorous and feed mainly on grasses and herbs, supplementing these with twigs, buds, bark and field crops, particularly in winter. Their natural predators include large birds of prey, canids and felids. They rely on high-speed endurance running to escape predation, having long, powerful limbs and large nostrils.",
    "White Tailed Deer":"White-tailed deer, the smallest members of the North American deer family, are found from southern Canada to South America. In the heat of summer they typically inhabit fields and meadows using clumps of broad-leaved and coniferous forests for shade. During the winter they generally keep to forests, preferring coniferous stands that provide shelter from the harsh elements. White-tailed deer are herbivores, leisurely grazing on most available plant foods. Their stomachs allow them to digest a varied diet, including leaves, twigs, fruits and nuts, grass, corn, alfalfa, and even lichens and other fungi.",
    "Bird Spec":"Birds are a group of warm-blooded vertebrates constituting the class Aves, characterised by feathers, toothless beaked jaws, the laying of hard-shelled eggs, a high metabolic rate, a four-chambered heart, and a strong yet lightweight skeleton. Birds have wings whose development varies according to species. Some of the examples of birds are eagles, vultures, pigeons etc."
    
}

"""
mobile = tf.keras.applications.mobilenet.MobileNet()

def prepare_image(file):
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

def disp_image(file_url):
    preprocessed_image = prepare_image(file_url)
    predictions = mobile.predict(preprocessed_image)
    results = imagenet_utils.decode_predictions(predictions)
    return results
"""

def loadModel(pathname):
    model = load_model(pathname)
    return model

def runTest(image_path):
    
    img = load_img(image_path,target_size=(224,224))
    imgarray = img_to_array(img)
    imga = tf.expand_dims(imgarray,0)
    predictions = loadModel(f"{os.getcwd()}/wildlife/ml_models/resnet50.h5").predict(imga)
    predictions_result = predictions[0]
    sorted_predictions = np.sort(predictions[0])


    index1 = np.where(predictions[0] == sorted_predictions[-1])
    index2 = np.where(predictions[0] == sorted_predictions[-2])
    index3 = np.where(predictions[0] == sorted_predictions[-3])
    print('hey this is score')
    animal1 = animals[index1[0][0]]
    animal2 = animals[index2[0][0]]
    animal3 = animals[index3[0][0]]

    return [
        {
            'name':f"{animal1}",
            'desc':animal_desc[animal1],
            'percentage':round(predictions_result[index1][0]*100,3)
        },
        {
            'name': f"{animal2}",
            'desc':animal_desc[animal3],
            'percentage':round(predictions_result[index2][0]*100,3)
            
        },
        {
            'name': f"{animal3}",
            'desc':animal_desc[animal3],
            'percentage':round(predictions_result[index3][0]*100,3)
            
        }
    ]
    
    

model_path = f"{os.getcwd()}/wildlife/ml_models/retinanet.h5"
model = models.load_model(model_path, backbone_name='resnet50')
labels_to_names = {
    0:'Agouti',
    1:'Roe_Deer',
    2:'Wild_Boar',
    3:'Ocelot',
    4:'Red_Squirrel',
    5:'Bird_spec',
    6:'White_Tailed_Deer',
    7:'Mouflon',
    8:'Red_Deer',
    9:'European_Hare'
}

def imageDetection(filepath):
    image = read_image_bgr(filepath)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    # correct for image scale
    boxes /= scale

    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    # plt.figure(figsize=(15, 15))
    # plt.axis('off')
    # result = plt.imshow(draw)
    # print(type(result))
    # return result
    # plt.show()
    # print(image.mime_type)
    # img_result = base64.b64encode(image)
    # final_result = img_result.decode('utf-8').split(';')[0]
    # print(type(final_result) == type(img_result))
    # return img_result
    
    # w, h = 512, 512
    # data = np.zeros((h, w, 3), dtype=np.uint8)
    # data[0:256, 0:256] = [255, 0, 0] # red patch in upper left
    # img = Image.fromarray(image, 'RGB')
    # img.save('my.png')
    # img.show()
    # return 'hello'
    
    img = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.png', img)
    result = base64.b64encode(buffer).decode('utf-8')
    # result.save('my-img.jpeg')
    return result