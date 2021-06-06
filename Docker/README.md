# Dockerfile for generating the HangulFontsDataset

This folder contains scripts for building a Docker image which can generate the HangulFontsDataset locally. Alternatively, you can download a premade image using
```bash

```

1. To generate the Docker image locally run the follow from this folder
```bash
docker build -t "hfd" . 
```

2. To download the premade image from [dockerhub](https://hub.docker.com/r/jesselivezey/hfd)
```bash
docker pull jesselivezey/hfd
```

To generate the HangulFontsDataset after running 1 or 2, from the `scripts` folder run
```bash
bash generate_hfd.sh
```
