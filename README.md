# HelloSivi_Assignment1
Approach:
   1)First it reads an image.
   2)Then it tries to detect human face.
   3)if a human face is detected it tries to recognize it with a pre-trained model file.
   4)If human face is recognized it print the human as output else it print non-human.
   5)I used vgg_face pretrained weights for transfer learning and i also used sigmoid activation function layer in output layer.
   
Repositories is structured as follows:
    1)main.py: It contain all the codes for above model.
    2)result: It contain resulting images of test images after applying apencv.
    3)test: It contain test images

# HelloSivi_Assignment2
Approach:
    1)First it reads an image.
    2)Then it resize it into 200*200 pixels.
    3)Then i used Kmean Clustring for identifying dominant colours and eliminating neutral colours.
    4)Then i predict the input image after fitting.
    
Repositories is structured as follows:
    1)main.py: It contain all the codes for above model.
    2)result: It contain resulting images of test images after applying apencv.
    3)test: It contain test images
