# CombiGCN

This is our improvement Tensorflow based on code of https://github.com/kuandeng/LightGCN


## Introduction
By identifying the similarity weight of users through their interaction history, a key concept of CF, we endeavor to build a user-user weighted connection graph based on their similarity weight of them. We propose a recommendation framework CombiGCN that combine user-user weighted connection graph and user-item interaction graph.

## Environment Requirement
The code has been tested running under Python 3.10.9 The required packages are as follows:
* tensorflow == 2.11.0
* numpy == 1.24.3
* scipy == 1.9.0
* sklearn == 1.2.0

## Examples to run a 3-layer CombiGCN

### Ciao dataset
* Command
```
python CombiGCN.py --dataset ciao --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 2048 --epoch 10000
```

### Epinions dataset
* Command
```
python CombiGCN.py --dataset epinions --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 2048 --epoch 10000
```

### Foursquare dataset
* Command
```
python CombiGCN.py --dataset foursquare --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 8192 --epoch 10000

```

NOTE : the duration of training and testing depends on the running environment.
## Dataset
We provide three processed datasets: Gowalla, Yelp2018 and Amazon-book.
* `train.txt`
  * Train file.
  * Each line is a user with her/his positive interactions with items: userID\t a list of itemID\n.

* `test.txt`
  * Test file (positive instances).
  * Each line is a user with her/his positive interactions with items: userID\t a list of itemID\n.
  * Note that here we treat all unobserved interactions as the negative instances when reporting performance.
  
* `user_list.txt`
  * User file.
  * Each line is a triplet (org_id, remap_id) for one user, where org_id and remap_id represent the ID of the user in the original and our datasets, respectively.
  
* `item_list.txt`
  * Item file.
  * Each line is a triplet (org_id, remap_id) for one item, where org_id and remap_id represent the ID of the item in the original and our datasets, respectively.


=======
