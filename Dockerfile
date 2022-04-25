FROM tensorflow/tensorflow:2.7.0-gpu

ARG DEV

WORKDIR /tmp

RUN apt-get update && apt-get install build-essential wget git -y

RUN curl -sSL https://cmake.org/files/v3.5/cmake-3.5.2-Linux-x86_64.tar.gz | tar -xzC /opt\
    && apt-get update\
    && wget http://www.cmake.org/files/v3.5/cmake-3.5.2.tar.gz\
    && tar xf cmake-3.5.2.tar.gz\
    && cd cmake-3.5.2\
    &&./configure\
    && make\
    && make install

# RUN apt install libeigen3-dev

RUN git clone https://gitlab.com/libeigen/eigen.git && cd eigen && git checkout 3.4.0
RUN cmake eigen && make install
RUN ln -s /usr/local/include/eigen3/Eigen /usr/local/include/Eigen

RUN wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.6.0.tar.gz
RUN tar -C /usr/local -xzf libtensorflow-gpu-linux-x86_64-2.6.0.tar.gz
RUN rm libtensorflow-gpu*

WORKDIR /repos

RUN git clone https://github.com/thomas-koller/jass-kit-py.git\
    && cd jass-kit-py && pip install -e . && cd ..

RUN git clone --recurse-submodules \
    https://github.com/thomas-koller/jass-kit-cpp.git\
    && cd jass-kit-cpp && pip install . && cd ..

# required in order for linker to find jass headers
RUN cd jass-kit-cpp && cmake . && make install && cd ..

COPY .github .github
RUN git clone --recurse-submodules \
    https://$(cat .github)@github.com/thomas-koller/jass-ml-cpp.git \
    && cd jass-ml-cpp && pip install . && cd ..

RUN git clone https://$(cat .github)@github.com/thomas-koller/jass-ml-py.git\
    && cd jass-ml-py && pip install -e . && cd ..

COPY requirements.txt requirements.txt
COPY requirements-dev.txt requirements-dev.txt

# install latex for plots and dev requirements
RUN if [[ -z "$DEV" ]];\
    then echo "No DEV mode";\
    else ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && echo "Etc/UTC" > /etc/timezone \
    && apt-get update \
    && apt-get upgrade -y \
    && apt-get install texlive-full -y \
    && rm -rf /var/lib/apt/lists/*; fi

RUN if [[ -z "$DEV" ]];\
    then echo "No DEV mode";\
    else pip install -r requirements-dev.txt; fi

RUN pip install -r requirements.txt

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /app

COPY .wandbkey .wandbkey

# add user

RUN chown -hR 1000 /repos

RUN adduser user --uid 1000
RUN adduser user sudo
USER user

ENV XLA_PYTHON_CLIENT_MEM_FRACTION=.7

ENTRYPOINT ["sh", "/entrypoint.sh"]