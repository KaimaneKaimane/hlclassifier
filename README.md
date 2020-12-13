# hlclassifier
Solution for the assignment found at https://github.com/Rocketloop/assignment-machine-learning-nlp

Restrictions:
* Local system was running Windows 10
* Solution was computed on a machine without cuda support for GPU training and a maximum of 16GB RAM
* Tested on Windows 10 on a WSL2 Ubuntu setup

# Simple API

I added a simple API to use the trained model for predictions

To build the docker image:
```console
docker build . -f ./docker/Dockerfile -t hlclassifier/base:latest --cache-from hlclassifier/base:latest --build-arg PROJECT=hlclassifier
```

Train/Retrain Model:
```console
docker-compose -f docker-compose-train.yml up
```

Start API:
```console
docker-compose -f docker-compose.yml up
```

How to use the API:
Some sample requests:
```console
curl 127.0.0.1:8080/status
curl 127.0.0.1:8080/predict -H "Content-Type: application/json" -X POST -d '{"headline": "The president said what?!?"}'
curl 127.0.0.1:8080/predict -H "Content-Type: application/json" -X POST -d '{"headline": "Living it up in the city!"}''
```
