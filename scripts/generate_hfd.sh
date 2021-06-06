docker run -it -v ${PWD}/..:/home/hfd/ -u `id -u $USER` hfd python /home/hfd/scripts/process_all_fonts.py /home/hfd 24
docker run -it -v ${PWD}/..:/home/hfd/ -u `id -u $USER` hfd python /home/hfd/scripts/add_median_images.py /home/hfd/h5s 24
docker run -it -v ${PWD}/..:/home/hfd/ -u `id -u $USER` hfd python /home/hfd/scripts/add_variables.py /home/hfd/h5s 24
