# FROM unifyai/multicuda:base_and_requirements
FROM unifyai/ivy:latest
RUN pip3 install --upgrade pip
RUN pip3 install imageio
RUN pip3 install pyvis
RUN pip3 install transformers
RUN pip3 install flax
RUN pip3 install kornia
RUN pip3 install cython
RUN pip3 install ipython==8.12.0
RUN apt-get update
RUN apt-get -y install clang

