# Convoluted_neural_networks
Using CNN to diagnose presence of Lung &amp; colon cancer
Introduction

Convolutional Neural Networks

CNNs are a type of artificial neural network (ANN) commonly used for image recognition and computer vision tasks. They are designed to automatically and adaptively learn spatial hierarchies of features from input images by performing a series of convolution and pooling operations. In a CNN, the input image is passed through a series of convolutional layers, where each layer applies a set of learnable filters to the input and produces a feature map. The feature maps are then downsampled using pooling operations to reduce their size while retaining important information. Finally, the feature maps are fed into fully connected layers to produce the output, which can be used for classification, object detection, or other image processing tasks. The ability of CNNs to learn spatially invariant features from raw pixel data makes them highly effective in various image-related applications, including object detection, image recognition, face recognition, and medical image analysis.

There are different types of CNNs like LeNet, AlexNet, ResNet etc. Our CNN model does not resemble any well-known CNN architectures , as it is a custom architecture designed specifically for the task at hand. However, it follows a common pattern of gradually increasing the number of filters while decreasing the spatial dimensions of the feature maps through the use of convolutional and pooling layers. 

Report Objectives

The main objective of the project is to evaluate the effectiveness of the proposed algorithm of Mangal et al.'s Convolutional Neural Network (CNN) in identifying the common types of lung and colon cancer in the human body. The algorithm is designed with an understanding of the patterns of neurons and their connectivity in the human brain, which contributes to its low preprocessing requirements compared to other classification algorithms. Unlike the primitive method of hand-engineered filters, the algorithm has the capability to learn features. The ConvNet takes input in the form of image attribute weights (learnable weights and biases) for multiple features in the image, allowing it to distinguish between different images.

Dataset description

The realm of Artificial Intelligence called Machine Learning has resulted in remarkable progress in various fields, particularly in medicine. For successful computer model training, Machine Learning algorithms require extensive datasets. Though some medical image datasets exist, more image datasets are required from diverse medical entities, particularly cancer pathology. Scarcer still are image datasets that are prepared and suitable for Machine Learning. The dataset we have used for our project is  an image dataset (LC25000) comprising 25,000 colored images distributed among five categories. Each category of 5,000 images shows the following histologic entities: colon adenocarcinoma, benign colonic tissue, lung adenocarcinoma, lung squamous cell carcinoma, and benign lung tissue. 





**CNN Architecture and Training Strategy**
To classify the LC25000 dataset, the following deep CNN architecture was
constructed:
**Input Layer**: The input layer loads data and feeds it
to the first convolutional layer. The input size is an image of 70x70 pixels
with 3 color channels for RGB.
**Convolution Layer**: The model contains six
convolutional layers with filter size 3x3, stride set to 1, and padding kept
the same. The number of filters in each layer is 32, 64, 128, 256, 512, and
1024, respectively. All convolutional layers are followed by ReLU activation
for non-linear operations.
**Pooling Layer**: Pooling operation is used for downsampling
the output images received from the convolution layer. There is one pooling
layer after each convolutional layer with a pooling size of 2 and padding set
to valid. All the pooling layers use the most common max pooling operation.
**Flatten Layer**: The flatten layer is used to convert
the output from the convolutional layer into a 1D tensor to connect a dense
layer or fully connected layer.
**Fully Connected Layer**: Two dense layers are used in
this model. The first one contains 512 neurons with a ReLU activation function,
and the second one contains 5 neurons with a sigmoid activation function
depending on the input class.

 
Layer1
Layer2
Layer3
Layer4
Layer5
Layer6
Layer7
Layer8
Type
CONV+POOL
CONV+POOL
CONV+POOL
CONV+POOL
CONV+POOL
CONV+POOL
FC
FC
Channels
32
64
128
256
512
1024
512
5
Filter size
3x3
3x3
3x3
3x3
3x3
3x3
 
 
Convolution Strides
2x2
2x2
2x2
2x2
2x2
2x2
 
 
Pooling Size
1x1
1x1
1x1
1x1
1x1
1x1
 
 
Pooling Strides
2x2
2x2
2x2
2x2
2x2
2x2
 
 



CNN Implementation

Overall, this model can be used for image classification tasks with five possible classes.
Out of the five classes of tissues, Lung adenocarcinoma ('lung_aca' ), Lung squamous cell carcinoma ('lung_scc'),  and Colon adenocarcinoma ('colon_aca’) are the ones that cause cancer of the lung or colon. Following code snippets show the implementation.


Model: "sequential"
_________________________________________________________________
 Layer (type)                     			Output Shape              Param #   
=================================================================
 conv2d (Conv2D)            			 (None, 70, 70, 32)        896       
                                                                 
 activation (Activation)     			(None, 70, 70, 32)        0         
                                                                 
 max_pooling2d (MaxPooling2D ) 		(None, 35, 35, 32)        0                                                               
                                                                 
 conv2d_1 (Conv2D)          			 (None, 35, 35, 64)        18496     
                                                                 
 activation_1 (Activation)   			(None, 35, 35, 64)         0         
                                                                 
 max_pooling2d_1 (MaxPooling  2D) 	(None, 17, 17, 64)         0         
                                                             
                                                                 
 conv2d_2 (Conv2D)           			(None, 17, 17, 128)       73856     
                                                                 
 activation_2 (Activation)   			(None, 17, 17, 128)       0         
                                                                 
 max_pooling2d_2 (MaxPooling  2D) 	(None, 8, 8, 128)           0         
                                                            
                                                                 
 conv2d_3 (Conv2D)           			(None, 8, 8, 256)         295168    
                                                                 
 activation_3 (Activation)  			 (None, 8, 8, 256)         0         
                                                                 
 max_pooling2d_3 (MaxPooling2D)     	(None, 4, 4, 256)          0         
                                                          
                                                                 
 conv2d_4 (Conv2D)           			(None, 4, 4, 512)         1180160   
                                                                 
 activation_4 (Activation)   			(None, 4, 4, 512)         0         
                                                                 
 max_pooling2d_4 (MaxPooling 2D)  	(None, 2, 2, 512)         0         
                                                            
                                                                 
 conv2d_5 (Conv2D)           			(None, 2, 2, 1024)        4719616   
                                                                 
 activation_5 (Activation)   			(None, 2, 2, 1024)        0         
                                                                 
 max_pooling2d_5 (MaxPooling 2D)	 (None, 1, 1, 1024)       0         
                                                             
                                                                 
 flatten (Flatten)                            		 (None, 1024)              0         
                                                                 
 dense (Dense)              			 (None, 512)                524800    
                                                                 
 dense_1 (Dense)             (None, 5)                 2565      
                                                                 
=================================================================
Total params: 6,815,557
Trainable params: 6,815,557
Non-trainable params: 0
___________________________________________________________________________


Comparing related works and our Results


While attempting to implement the CNN architecture by Mangal et al., we modified it according to our needs. They have used three layers with two fully connected layers while we used six layers and two fully connected layers. We used 10 epochs at a initial learning rate of 0.1, while Mangal et al., have used 100 epochs and a learning rate of 10-4. Mangal’s algorithm took a considerably large amount of run time of 45 minutes per iteration (total 3.125 days). To reduce the run time for our algorithm, we chose the initial learning rate of 10-1 which resulted in a total run time of 8 hours.

The project is implemented with a sigmoid function in the final layer which is different from the paper referred. The main difference between the sigmoid and softmax activation functions lies in their output ranges and how they handle multi-class classification problems.
The sigmoid activation function squashes the output of each neuron in the range [0,1]. Therefore, each neuron in the final layer of the neural network model equipped with sigmoid activation can independently output a probability value indicating the likelihood of the input belonging to a certain class. In the given code, the final layer has 5 neurons, with each neuron outputting a probability value indicating the probability of the input belonging to that class. This approach is suitable for multi-label classification problems, where each sample can belong to multiple classes simultaneously.
On the other hand, the softmax activation function is also a squashing function, but it normalizes the output of each neuron to be in the range [0,1] and ensures that the sum of the outputs across all neurons in the layer is 1. Therefore, the output of each neuron in the final layer of a neural network equipped with the softmax activation function represents the probability of the input belonging to that class, conditioned on the input only belonging to one class. In other words, softmax activation is suitable for multi-class classification problems where each sample can only belong to one class at a time.
In the given code, the sigmoid activation function is used in the final layer, which means that the model is designed for multi-label classification problems. If the problem was a multi-class classification problem where each sample could only belong to one class, softmax activation would be more appropriate.

Following images show the final results of our algorithm.
We have tried to show the prediction by taking one image from all the five classes. 






The convergence is seen in the first two epochs and the validation accuracy is diverging from  training accuracy in 2-3 epochs but, is converging again in epoch 3 to epoch 6. The accuracy has increased once it reached the final epoch. Hence, one can increase the number of epochs and more hyper parameter tuning for better results.


The loss has decreased gradually from the first epoch to the 10th epoch and are diverged widely. after the first 2 epochs.
we can see that the training loss and accuracy are decreasing and increasing, respectively, with each epoch, which is a good sign that the model is learning from the data. However, the validation loss and accuracy are not improving significantly after a certain number of epochs. This indicates that the model is starting to overfit the training data and not generalizing well to the unseen validation data.
To combat overfitting, one can use techniques such as regularization, dropout, early stopping, or data augmentation in future works.


Model Evaluation

Training loss: 0.38
Testing loss: 0.40
Training accuracy: 0.83
Testing accuracy: 0.82





Conclusion

Medical image processing is done in two ways. At the patient level, it means determination of the accuracy of image classification for each patient. At the image level, the accuracy of cancer image classification is computed. Since the LC25000 dataset did not provide any patient information, we used the image classification method to evaluate model performance. We used Google's Colab to train the CNN models using the TensorFlow framework. It took us approximately 1 hour to train each model. The model implemented by us ran for 10 epochs resulting in a final loss of 53% and accuracy of 78%. Our implementation of CNN was a success with a good accuracy score with much lesser run time with little loss of accuracy as compared to Mangal et al.’s accuracy score of around 96%.

