### the code is for embedding generation for our model

### environment setup
conda create --name test python=3.7
conda activate test
pip install -r requirements.txt


### run the code e.g.
python main -config ./config/config_pos.yml

the output embedding will be saved as bio_embedding_2000_pos.txt, and the parameters will be saved in out_pos
(other dataset is the same, change the name accordingly)



###
All the saved checkpoint in our model for the paper can be found in tmp_rst (https://drive.google.com/drive/folders/1LlwPxJOZoPYRpH2WUtCLplW1DeSPnif4?usp=sharing)


