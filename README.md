# VM2: Learning Approximate Nash Equilibria for Jass
This repository contains the sources of the VM2 titled "_Learning Approximate Nash Equilibria for Jass_".


## Setup
Build all the docker image using docker-compose

```bash
$ docker-compose build
```

Afterwards run all tests to verify local setup

```bash
$ docker-compose up test
```

And finally start the container hosting the baselines with

```bash
$ docker-compose up -d baselines
```
## Online outcome sampling
The counterfactual regret minimising method called online outcome sampling (OOS) is designed to be applied directly 
to the game in a online setting. This means that the method does not require any offline computations and can therefore
be directly used to play. The OOS implementation and the respective agent is located at `lib/cfr`.

## MuZero
MuZero is a model based reinforcement learning method using deep learning and therefore requires training.
The data is generated either through self-play or reanalysing existing data (tfrecord format as described in [jass-ml-py](https://github.com/thomas-koller/jass-ml-py/tree/master/jass/features) repo.
All MuZero specific code can be found at `lib/mu_zero`.

### Training
The MuZero training process is implemented in a distributed manner.
The docker-compose service `trainer` is the master container which will gather all the data, train the networks
and evaluate them asynchronously on different metrics.
In the folder `resources/data_collectors` there are different compose files for different machines to host data collectors.
They all assume that the master container is running on the `ws03` machine (IP: 10.180.39.13).
If this would not be the case, the IP in the files must be adapted.
To start the training process first run 

```bash
$ docker-compose up trainer
```

and wait until the flask server started hosting. Then start the data collectors on the respective machines

```bash
$ docker-compose -f resources/data_collectors/(gpu03|e01|...).yml up
```

The collectors should then register them on the master container and start to collect data.
Once the replay buffer has been filled, the optimization procedure will start and the corresponding metrics will
be logged to wandb.ai at the configured location.
All settings and hyperparameters can be configured in the file `scripts/settings.json` 


## Evaluate
To reproduce the evaluation results as shown in the report run
```bash
$ docker-compose up evaluate
```
which will take quite a long time.
The evaluation script also offers the option to run only specific evaluations among other arguments.
