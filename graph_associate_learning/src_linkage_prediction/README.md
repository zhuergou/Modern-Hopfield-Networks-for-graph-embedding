
### run the code e.g.
python main -config ./config/config_blog_2000.yml

the output embedding will be saved as bio_embedding_2000_blogcatalog.txt, and the parameters will be saved in out_blog_2000
(other dataset is the same, change the name accordingly)


## for the linkage prediction
run python linkage_prediction.py --network ../data/blogcatalog.mat --embedding bio_embedding_2000_blogcatalog.txt
(for other datasets, change the name accordingly)


