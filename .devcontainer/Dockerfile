##### Base Stage #####

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 as base

LABEL MAINTAINER="yueyang.zhang@tum.de"

SHELL [ "/bin/bash", "-c" ]

# setup environment
ENV LANG=en_US.UTF-8
ENV LC_NUMERIC=en_US.UTF-8
ENV LC_ALL=C
ENV DEBIAN_FRONTEND=noninteractive

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
  ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
  apt-get update && \
  apt-get install -q -y --no-install-recommends tzdata && \
  rm -rf /var/lib/apt/lists/*

# USE German mirror to speed up things
RUN cp /etc/apt/sources.list /etc/apt/sources.list.old \
  && sed -i -e 's/http:\/\/archive\.ubuntu\.com\/ubuntu\// \
  http:\/\/de.archive\.ubuntu\.com\/ubuntu/' /etc/apt/sources.list
RUN apt-key update && apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
  terminator \
  vim \
  git \
  bash-completion \ 
  software-properties-common \
  ca-certificates \
  curl \
  wget \
  ssh \
  dirmngr \
  gnupg2 \
  lsb-release \
#  python3.7\
  python3.9 \
  python3.9-dev \
  python3-pip \
  build-essential libffi-dev libssl-dev \
  && rm -rf /var/lib/apt/lists/*

# 将 Python 默认指向 Python 3.9
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Fix ssl certificate issue
RUN apt-get update
RUN apt-get install ca-cacert \
  && update-ca-certificates \
  && rm -rf /var/lib/apt/lists/*


##### ROS Stage #####

FROM base as ros

# install ros
RUN sh -c 'echo "deb [arch=amd64] http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-get update
RUN apt-get install -y ros-noetic-desktop-full
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN source ~/.bashrc

RUN apt-get update && apt-get install --no-install-recommends -y \
  python3-rosdep \
  python3-rosinstall \
  python3-rosinstall-generator \
  python3-wstool \
  python3-vcstools \
  python3-catkin-tools\
  && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init \
  && rosdep update

ENV ROS_PACKAGE_PATH=/opt/ros/noetic/share

##### User Stage #####

FROM ros as user

ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
  && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && apt-get update \
  && apt-get install -q -y sudo \
  && rm -rf /var/lib/apt/lists/* \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

CMD ["bash"]

RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

RUN sudo mkdir -p /home/$USERNAME/Downloads/openssl/ssl \
  && sudo cp -r /etc/ssl/* /home/$USERNAME/Downloads/openssl/ssl/ \
  && sudo rm /etc/ros/rosdep/sources.list.d/20-default.list \
  && sudo rosdep init \
  && rosdep update

##### Runtime Stage #####

FROM user as runtime

ENV NVIDIA_VISIBLE_DEVICES \
  ${NVIDIA_VISIBLE_DEVICES:-all}

ENV NVIDIA_DRIVER_CAPABILITIES compute,graphics,utility,video,display

# detailed fix for the ssl certificate issue
RUN sudo mkdir -p /home/$USERNAME/Downloads/openssl/ssl \
  && sudo cp -r /etc/ssl/* /home/$USERNAME/Downloads/openssl/ssl/ \
  && sudo rm /etc/ros/rosdep/sources.list.d/20-default.list \
  && sudo rosdep init \
  && rosdep update

##### Dependencies Stage #####

FROM runtime as dep

RUN sudo sh -c 'echo "deb [arch=amd64] https://hsr-user:jD3k4G2e@packages.hsr.io/ros/ubuntu `lsb_release -cs` main" > /etc/apt/sources.list.d/tmc.list'
RUN sudo sh -c 'echo "deb [arch=amd64] https://hsr-user:jD3k4G2e@packages.hsr.io/tmc/ubuntu `lsb_release -cs` multiverse main" >> /etc/apt/sources.list.d/tmc.list'
RUN sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list' 
RUN wget https://hsr-user:jD3k4G2e@packages.hsr.io/tmc.key -O - | sudo apt-key add - 
RUN wget https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc -O - | sudo apt-key add - 
RUN wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add - 
RUN sudo sh -c 'mkdir -p /etc/apt/auth.conf.d' 
RUN sudo sh -c '/bin/echo -e "machine packages.hsr.io\nlogin hsr-user\npassword jD3k4G2e" >/etc/apt/auth.conf.d/auth.conf'
RUN sudo sh -c '/bin/echo -e "Package: ros-noetic-laser-ortho-projector\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-laser-scan-matcher\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-laser-scan-sparsifier\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-laser-scan-splitter\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-ncd-parser\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-polar-scan-matcher\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-scan-to-cloud-converter\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-scan-tools\nPin: version 0.3.3*\nPin-Priority: 1001" > /etc/apt/preferences' 

RUN sudo apt-get update
RUN sudo apt-get install ros-noetic-tmc-desktop-full -q -y --no-install-recommends

RUN echo -e "\nexport LOCALHOST_IP=\$(hostname -I | awk '{print \$1}')\necho \"Localhost IP: \$LOCALHOST_IP\"\n\nexport ROS_IP=\$LOCALHOST_IP" >> ~/.bashrc


RUN echo 'export ROS_HOME=~/.ros' >> ~/.bashrc
RUN echo 'alias sim_mode='\''export ROS_MASTER_URI=http://localhost:11311 export PS1="\[\033[44;1;37m\]<local>\[\033[0m\]\w$ "'\''' >> ~/.bashrc
RUN echo 'alias hsrb_mode='\''export ROS_MASTER_URI=http://10.152.246.155:11311 export PS1="\[\033[41;1;37m\]<hsrb>\[\033[0m\]\w$ "'\''' >> ~/.bashrc
RUN echo "alias normal_mode='source ~/.bashrc'" >> ~/.bashrc

RUN source home/$USERNAME/.bashrc

RUN echo 'source /workspaces/adv_robocup/hsrb_ws/devel/setup.bash' >> ~/.bashrc
RUN echo "alias sss='source devel/setup.bash'" >> ~/.bashrc
RUN sudo apt install -qqy lsof \
  net-tools\
  lshw\
  iputils-ping\
  python-is-python3


# TODO: pytorch, segment_anything, etc.

RUN sudo apt install -qqy ros-noetic-moveit
RUN pip3 install --upgrade pip numpy==1.24.3 scipy==1.10.1 protobuf
RUN pip install networkx==3.1
RUN pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install git+https://github.com/facebookresearch/segment-anything.git
RUN pip install opencv-python pycocotools onnxruntime onnx
RUN pip3 install --upgrade --force-reinstall matplotlib==3.7.2
RUN pip install ftfy regex tqdm
RUN pip install git+https://github.com/openai/CLIP.git

# RUN pip install -r /workspaces/adv_robocup/hsrb_ws/src/ultralytics_ros/requirements.txt
RUN pip install opencv-python smash

WORKDIR /workspaces/ros/ros_snn_project/src
RUN rosdep install --from-paths . --ignore-src --rosdistro noetic -y


RUN pip install git+https://github.com/BindsNET/bindsnet.git
# RUN python3 -c "import bindsnet; print('BindsNET version:', bindsnet.__version__)"