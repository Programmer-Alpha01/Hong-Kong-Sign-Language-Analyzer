# Hong Kong Sign Language Analyzer

## Introdution

In Hong Kong, there are about 246,200 deaf or hearing-impairing people using Hong Kong Sign Language (HKSL) to communicate. However, most of the citizens lack education in sign language, also the ratio of Hong Kong Sign Language interpreters in Hong Kong to people with hearing difficulty is around 1:16413 which leads to most citizens not understanding the meaning of Sign Language and cannot communicate with impaired people. At present, people have invented AI interpreters to solve this problem. AI interpreters are mainly lens-type interpreters and wearable interpreters. However, lens-type interpreters lack development in Hong Kong and wearable sign language interpreters have limits such as heavy-weight, short battery life and inconvenient to carry, those issues led to Hong Kong sign language translators not being generalized. To solve this issue, we designed a sign language video interpreter and a lens-type interpreter to translate the actions with the LSTM network and transfer the sign language action meaning back, to break the barrier of sign language interpreter generalization. 

This project is used to ananlyze Hong Kong sign language. By using mediapipe landmark to extract human movement and input to trained LSTM model to prodicte Hong Kong sign language. 

## Current state
Currently, the model can analyze 4 type of Hong Kong Sign Language: Good Morning, Good Afternoon, Good Night, Thank You

The advanced version will be updated in the future.

## Notice
Please note that the analysis results provided by the application model are not 100% accurate.




# Programme file explanation
## Analyzer
| File Name  | Explanation|
|-|-|
|Realtime sign language analyzer.py|This analyzer can analyze sign language movement in real time|
|Sign language video analyzer.py|This analyzer can analyze sign language movement in by input videos into a folder, but it can analyze one Sign Language only|


## Data pre-processing : Video to Numpy data
Extracting the NumPy data directly without any pre-processing will cause the NumPy data inhomogeneous. Therefore the model input requires homogeneous data to proceed, unifying the video frame rate and length for the model input is crucial. The key points are then detected and converted to NumPy data by frame.

For large amounts of data to train the model, an automatic conversion program is necessary to ensure efficiency. The following part will explain the function and process of the program.  
| File Name  | Explanation|
|-|-|
|Change_frame_rate.py|This code is used to change video frame rate |
|Change_video_length.py|This code is used to change video length|
|Video to NumPY.py|This code is used to convert video into NumPy data massively|
|Encoder.py|This encoder is used to convert video into NumPy data. |

## Training
| File Name  | Explanation|
|-|-|
|LSTM(Training).py|The code is used for model training|


## Trained Model
![image](https://github.com/user-attachments/assets/aa80db9c-4652-4ac7-b8a2-a53ebc96b7a4)
In the architecture of the LSTM model above, is designed for sequence prediction tasks, with
7 layers.

The first three layers are the LSTM layers are activated by ‘ReLU’ function, in which the first 
layer has 128 units, the second and third layers have 256 layers respectively. The fourth to the 
sixth layers are the Dense layer, which are used to classify the LSTM output. The fourth layer 
has 256 units, the fifth has 128 units, and the sixth layer has 64 units. The decision to use 
‘ReLU’ activation function is to ensure high computational efficiency and high speed of 
convergence.

The seventh layer it is the output layer. This layer uses the ‘SoftMax’ activation function to 
predict the possible actions. The shape and parameter are 6 and 260 which regarded to the four 
3-dimensional datasets.

Finally, the ‘Adam’ optimizer is implemented in batch size 64, and run for 100 epochs. The 
total params of this model is 1,224,900, including 1,224,900 trainable params and 0 non-trainable params.
| File Name  | Explanation|
|-|-|
|trained_model_LSTM.h5|A trained LSTM model|

# Trained Model conclusion
## Trained Model Accuracy and loss
![image](https://github.com/user-attachments/assets/22b76b0c-ca30-4921-ba33-cf79b066e72f)









