### the code is for eval the embedding performance, to compare with deepwalk, we use their evaluator

### environment setup (need to clone deepwalk git)
git clone https://github.com/phanein/deepwalk.git
cd deepwalk
pip install -r requirements.txt
python setup.py install



### run the code e.g.
copy the graph data (e.g. POS.mat) as well as the embedding (e.g. bio_embedding_2000_pos.txt) to ./deepwalk/example_graphs folder

then run with

python example_graphs/scoring.py --emb example_graphs/bio_embedding_2000_pos.txt
--network example_graphs/POS.mat
--num-shuffle 20 --all

(other dataset and deepwalk baseline can be run in the same fashion, change the name accordingly)


###
All the saved checkpoint in our model for the paper can be found in tmp_rst


