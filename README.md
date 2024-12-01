# Pre-Training a Deep Learning Model for Human Activity Recognition Using the Source Device of Sensor Data As a Label
This project proposes a new pre-training technique, where the pre-trained model predicts which recording device the data originates from. By using this as a pseudo- label, the project aims to create a pre-trained model that can be fine-tuned to com- plete HAR tasks. This proposed technique is useful as publicly available datasets all give information about the data recording device.  
The project evaluates the effectiveness of the pre-trained model on different datasets by fine-tuning the model with various amounts of downstream data and comparing the model’s F1-score to a fully-supervised model and other state of the art techniques. The evaluation finds that while the proposed technique performs well on certain datasets, it fails to consistently perform better other established training techniques. 

## Going through the messy repo
First of all, I apologize for the messy repository. I never thought that others would have to see this.
I included all the failed code and commits so that you can glimpse the chaos. While I have improved significantly since then, it provides insight into my working process.
To get a picture of the final results, the folder "DISS Deliverables" contains scripts that were used to produce the data found in my dissertation.

## Acknowledgments
The preprocessing and model architecture were initially based on the repository [SelfHAR](https://github.com/iantangc/SelfHAR)
