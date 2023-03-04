FROM gcr.io/world-fishing-827/github.com/globalfishingwatch/gfw-bash-pipeline:latest-python3.7 as radenv

COPY ./radenv.yaml ./

ARG MINICONDA="Miniconda3-py39_4.12.0-Linux-x86_64.sh"
ENV PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update

RUN wget https://repo.anaconda.com/miniconda/${MINICONDA} \
    && mkdir /root/.conda \
    && bash ${MINICONDA}  -b \
    && rm -f ${MINICONDA}  \
    && conda install mamba -n base -c conda-forge \
    && conda init bash \
    && mamba env create -f radenv.yaml \
    && echo "source activate rad" >> ~/.bashrc

FROM radenv as project-base

# Install Java 1.8
RUN apt-get update -y
RUN apt-get install -y software-properties-common
RUN apt-add-repository 'deb http://security.debian.org/debian-security stretch/updates main'
RUN apt-get update -y
RUN apt-get install -y \
    openjdk-8-jdk openjdk-8-jre

# ENV LC_ALL "en_US.UTF-8"
# SNAP wants the current folder '.' included in LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH ".:$LD_LIBRARY_PATH"
WORKDIR /usr/lib/jvm
RUN ln -s java-8-openjdk-amd64 java-8-openjdk
ENV JAVA_HOME "/usr/lib/jvm/java-8-openjdk"

# Install ESA-SNAP dependencies
RUN apt-get install -y \
    gcc libc6 libc6-dev libc-dbg \
    gfortran libgfortran5 \
    musl musl-dev musl-tools \
    fonts-dejavu fonts-dejavu-core fonts-dejavu-extra \
    gawk maven wget zip

FROM project-base AS project-exec

ENV VIRTUAL_ENV=/opt/venv

RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip


COPY . /opt/code
WORKDIR /opt/code pip install .[all]

ENTRYPOINT [ "${VIRTUALENV}/bin/python" ]

FROM project-base AS project-dev

RUN apt-get install -y sudo

ARG USER
ARG UID
ARG GID
RUN groupadd -g ${GID} -o ${USER}
RUN useradd -m -u ${UID} -g ${GID} -o -s /bin/bash ${USER}
RUN usermod -aG sudo ${USER}
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN PATH="${HOME}.local/bin:${PATH}"

RUN apt-get install -y emacs-nox tree

USER ${USER}

CMD /bin/bash
