# NLP-Project---Emotion-Detection-in-the-Text
This is a repository for the NLP Semester Project where we dealt with emotion detection in the text using BERT and RoBERTa Pre-trained encoder models and fine-tuned the models for the better performance and compared the model performances of BERT and RoBERTa using different metrics.


The Project is chosen to do the comparative Analysis of the performances of BERT and RoBERTa Pre-trained encoder models in recognizing/detecting the emotions underlying in the text.


The dataset is been collected from Hugging Face Hub named go_emotions.
The dataset contains 28 emotions including Neutral.
The emotions considered are ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'] 
As, it's difficult to handle all the 28 emotions, they have been categorized in a optimised way to 7 categories and made the process easier to understand and handle.
The categories are these:
{
"anger": ["anger", "annoyance", "disapproval"],
"disgust": ["disgust"],
"fear": ["fear", "nervousness"],
"joy": ["joy", "amusement", "approval", "excitement", "gratitude",  "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
"sadness": ["sadness", "disappointment", "embarrassment", "grief",  "remorse"],
"surprise": ["surprise", "realization", "confusion", "curiosity"]
}
Due to GPU Constraints, the Project has been done in 2 files, one is dedicated to BERT and other is dedicated to RoBERTa.
The file named - Sireesha_BERT_Emotion_Detection_Text.ipynb is for BERT model and Sireesha_RoBERTa_Emotion_Detection_Text.ipynb is for RoBERTa Model.
We have used the libraries like beautifulsoup4, emoji, transformers, matplotlib, numpy, pandas, sklearn, torchvision, json, and other sub-libraries from these to proceed with the project.
The dataset named go_emotions has been used in which it contains train, test, dev data files, emotion mapping json file, emotions.txt file, all the labels in csv format.
We have categorised the data and mapped the emotions and stored in a file before pre-processing the data.
In Data Preprocessing, we have dealt with Contraction mapping, punctuation handling, Mispelled word handling, cleaning the text, removing spaces, handling special characters etc.
The configuration details of batch_size = 32, no of epochs =3, learning rate of 2e-5, and maximum length of 200 have been used.
Tokenizers like RobertaTokenizer from pre-trained roberta-base, BertTokenizer from pre-trained bert-base-uncased have been used.
A Custom dataset has been used to get the ids, masks, targets, token_type ids. The Data Loader has been used to load the dataset.
Then, a Customized model has been created for both BERT and RoBERTa where attention mask and other information has been used and then optimised the model using AdamW optimiser with the provided learning rate and weight decay.
Then, the model is trained for 3 epochs and loss is calculated to monitor the performance.
The model is then validated using the validation data and analysed the perfomances of both the models.
Metrics like Accuracy score, F1 score, precision, recall, ROC curves and AUC scores have been used to measure the performances.
