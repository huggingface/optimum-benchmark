ARG IMAGE

FROM $IMAGE

# Create a non-root user
ARG USER_ID
ARG GROUP_ID
ENV PATH="/home/user/.local/bin:${PATH}"

RUN addgroup --gid $GROUP_ID group
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

# For ROCm, the user needs to be in the video and render groups, check with /opt/rocm/
RUN if [ -d /opt/rocm/ ]; then usermod -a -G video user; fi
RUN if [ -d /opt/rocm/ ]; then usermod -a -G render user; fi

USER user
WORKDIR /home/user

