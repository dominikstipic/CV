docker build --tag demo .
docker run --rm -it \
    --name gideon \
    --net host \
    --privileged \
    --ipc host \
    --device /dev/video0 \
    --device /dev/video1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/course-materials:/home/opencv/course-materials \
    -e DISPLAY=$DISPLAY \
    demo 