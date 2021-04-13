cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1
CXX=${CXX:-g++}

$CXX -O2 -fno-inline -I.\
     -I$PWD/include \
     -o inference -std=c++17 \
     $PWD/src/inference.cpp \
     $PWD/src/common.cpp  \
     -lvart-runner \
     -lopencv_videoio  \
     -lopencv_imgcodecs \
     -lopencv_highgui \
     -lopencv_imgproc \
     -lopencv_core \
     -lglog \
     -lxir \
     -lunilog \
     -lpthread

