This folder contains the main body of code for the following paper:

Yuzhe Ma, Xiaojin Zhu, and Justin Hsu. Data Poisoning against Differentially-Private Learners: Attacks and Defenses. In The 28th International Joint Conference on Artificial Intelligence (IJCAI), 2019

(1). The subfolder /data contains the dataset for the 2D grid example in Figure 2 of the paper. To reproduce it, first run trajectory.py and then run plot2D.py.

To run trajectory.py, open the terminal and navigate into the path of the directory, use command

python trajectory.py labelaversion 0.1 deep-DPV 10

The parameters above correspond to (attack goal, attack method, privacy parameter of the victim learner, number of points to be poisoned). One will need to change the parameters (accordingly for plot2D.py) to experiment with different attack settings.

Running script trajectory.py is going to generate a subfolder /eps0.1, which contains the trajectories of the poisoned points. Then one can reproduce Figure 2(a) by using command

python plot2D.py labelaversion 0.1 deep-DPV 10

(2) J.py is used to generate Figure 2(d)-(f). Simply use command

python J.py labelaversion 0.1 deep-DPV 10

Note that running J.py may take several hours.

