# MVCNN: Multi-view Convolutional Neural Networks for 3D Shape Recognition

It is a pytorch implementation of [Multi-view Convolutional Neural Networks for 3D Shape Recognition ][1]

Dataset can be download from here: [dataset](https://drive.google.com/open?id=0B4v2jR3WsindMUE3N2xiLVpyLW8). 

## abstract 
A longstanding question in computer vision concerns the representation of 3D shapes for recognition: should 3D shapes be represented with descriptors operating on their native 3D formats, such as voxel grid or polygon mesh, or can they be effectively represented with view-based descriptors? We address this question in the context of learning to recognize 3D shapes from a collection of their rendered views on 2D images. We first present a standard CNN architecture trained to recognize the shapes’ rendered views independently of each other, and show that a 3D shape can be recognized even from a single view at an accuracy far higher than using state-of-the-art 3D shape descriptors. Recognition rates further increase when multiple views of the shapes are provided. In addition, we present a novel CNN architecture that combines information from multiple views of a 3D shape into a single and compact shape descriptor offering even better recognition performance. The same architecture can be applied to accurately recognize human hand-drawn sketches of shapes. We conclude that a collection of 2D views can be highly informative for 3D shape recognition and is amenable to emerging CNN architectures and their derivatives.

## What is implement in this tensorflow project is

- MVCNN with Alex Net 

- MVCNN with VGG Net 


the structure is something like this:
![model](https://user-images.githubusercontent.com/10870023/31384689-d171baec-ad74-11e7-985a-ebfdf3c2a2aa.png)


## What will be implemented

- ResNext MVCNN
- Densenet MVCNN
- Dynamic View CNN


[1]: http://vis-www.cs.umass.edu/mvcnn/




